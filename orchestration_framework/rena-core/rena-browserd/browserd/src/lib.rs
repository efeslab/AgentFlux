use error_stack::ResultExt;
use thiserror::Error;
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::info;

pub mod agent_protocol;
pub mod app_manager;
pub mod app_registry;
pub mod config;
pub mod container;
pub mod container_comm;
pub mod discovery;
pub mod event_processor;
pub mod executor;
pub mod handler;
pub mod inference_engine;
pub mod orchestrator;
pub mod trace;

use crate::container_comm::server::GrpcContainerComm;

const DISCOVERY_TOP_K: u16 = 5;
const RUN_AGENT_TIMEOUT: u64 = 180;
const LIST_TOOLS_TIMEOUT: u64 = 10;

#[derive(Error, Debug)]
pub enum Error {
    #[error("container_comm error")]
    ContainerComm,
    #[error("app manager error")]
    AppManager,
    #[error("discovery error")]
    Discovery,
    #[error("orchestrator error")]
    Orchestrator,
    #[error("executor error")]
    Executor,
    #[error("config error")]
    Config,
}

type Result<T> = error_stack::Result<T, Error>;

#[derive(Clone)]
pub struct AppClient<T, D, O>
where
    T: agent_protocol::AgentClient + Clone,
    D: discovery::Discovery + Clone,
    O: orchestrator::Orchestrator + Clone,
{
    pub discovery: D,
    pub app_manager: app_manager::AppManager<T>,
    pub orchestrator: O,
    pub executor: executor::Executor<T>,
}

impl<T, D, O> AppClient<T, D, O>
where
    T: agent_protocol::AgentClient + Clone,
    D: discovery::Discovery + Clone,
    O: orchestrator::Orchestrator + Clone,
{
    pub async fn query(
        &mut self,
        query_input: &str,
        tool_filter: &Option<Vec<String>>,
    ) -> Result<(String, trace::Trace)> {
        let (mut trace, start) = trace::Trace::start("query_browserd");
        let mut inner_traces = Vec::new();

        let (apps, discovery_trace) = trace::instrument(
            || async {
                self.discovery
                    .discover(query_input, DISCOVERY_TOP_K)
                    .await
                    .change_context(Error::Discovery)
            },
            "discovery",
            |apps: &Vec<discovery::AppBundle>| {
                Some(format!(
                    "ranked_apps: {}",
                    apps.iter()
                        .map(|a| a.config.name.clone())
                        .collect::<Vec<String>>()
                        .join(", ")
                ))
            },
            |_| None,
        )
        .await?;
        inner_traces.push(discovery_trace);

        let (_, run_apps_trace) = trace::instrument(
            || async {
                self.app_manager
                    .run_apps(apps.clone(), &RUN_AGENT_TIMEOUT)
                    .await
                    .change_context(Error::AppManager)
            },
            "run_apps",
            |_| None,
            |_| None,
        )
        .await?;
        inner_traces.push(run_apps_trace);

        let (tools, list_tools_trace) = trace::instrument(
            || async {
                self.app_manager
                    .list_tools(
                        apps.into_iter().map(|a| a.config.name.clone()).collect(),
                        &LIST_TOOLS_TIMEOUT,
                    )
                    .await
                    .change_context(Error::AppManager)
            },
            "list_tools",
            |tools: &Vec<app_manager::Tool>| {
                Some(format!(
                    "tools: {}",
                    tools
                        .iter()
                        .map(|t| t.tool_name.clone())
                        .collect::<Vec<String>>()
                        .join(", ")
                ))
            },
            |_| None,
        )
        .await?;
        inner_traces.push(list_tools_trace);

        let tools = match tool_filter {
            Some(tool_filter) => tools
                .into_iter()
                .filter(|t| {
                    let name = format!("{}_{}", t.app_name, t.tool_name);
                    for filter in tool_filter.clone() {
                        if name.starts_with(&filter) {
                            return true;
                        }
                    }
                    false
                })
                .collect::<Vec<_>>(),
            None => tools,
        };

        let mut message_params = vec![inference_engine::MessageParam {
            role: inference_engine::Role::User,
            content: inference_engine::MessageParamContent::Text(query_input.to_string()),
        }];

        loop {
            let (orchestrator_output, orchestrator_trace) = trace::instrument(
                || async {
                    self.orchestrator
                        .orchestrate(&mut message_params, tools.clone())
                        .await
                        .change_context(Error::Orchestrator)
                },
                "create_execution_plan",
                |orchestrator_output| {
                    Some(format!(
                        "nodes: {}",
                        orchestrator_output
                            .0
                            .as_ref()
                            .map(|plan| plan.nodes.len())
                            .unwrap_or(0)
                    ))
                },
                |orchestrator_output| Some(orchestrator_output.1.clone()),
            )
            .await?;
            inner_traces.push(orchestrator_trace);

            println!("Orchestrator output: {:?}", orchestrator_output);

            let execution_plan = match orchestrator_output.0 {
                Some(execution_plan) => execution_plan,
                None => {
                    info!("no execution plan");
                    let _ = self.orchestrator.inference_engine()
                        .print_inference_calls(message_params.clone(), tools.clone().into_iter().map(Into::into).collect());
                    break;
                }
            };

            let (_, executor_trace) = trace::instrument(
                || async {
                    self.executor
                        .execute(&mut message_params, execution_plan)
                        .await
                        .change_context(Error::Executor)
                },
                "execute_execution_plan",
                |_| None,
                |inner_traces| Some(inner_traces.clone()),
            )
            .await?;
            inner_traces.push(executor_trace);
        }

        trace.end(start, None, Some(inner_traces));

        Ok((format!("{:?}", message_params), trace))
    }
}

pub struct App {
    pub container_comm: GrpcContainerComm,
}

impl App {
    pub async fn new<T, D, O>(
        container_comm: GrpcContainerComm,
        agent_client: T,
        discovery: D,
        orchestrator: O,
    ) -> Result<(Self, AppClient<T, D, O>)>
    where
        T: agent_protocol::AgentClient + Clone,
        D: discovery::Discovery + Clone,
        O: orchestrator::Orchestrator + Clone,
    {
        let app_manager = app_manager::AppManager::new(agent_client);

        let app_client = AppClient {
            discovery,
            app_manager: app_manager.clone(),
            orchestrator,
            executor: executor::Executor::new(app_manager),
        };

        Ok((Self { container_comm }, app_client))
    }

    pub async fn run(self, cancel: CancellationToken) -> Result<()> {
        let Self { container_comm } = self;

        select! {
            res = container_comm.run(cancel.child_token()) => res.change_context(Error::ContainerComm),
            _ = cancel.cancelled() => Ok(()),
        }
    }
}
