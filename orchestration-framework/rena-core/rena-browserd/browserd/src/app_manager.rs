use error_stack::{Report, ResultExt};
use futures::future::try_join_all;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::broadcast::Sender;
use tokio::sync::Mutex;
use tokio::time::Duration;

use crate::discovery::AppBundle;
use crate::inference_engine::{ToolParam, ToolResultBlockParam, ToolUseBlock};

#[derive(Clone, Debug)]
pub enum AppStatus {
    NotRunning(Sender<Option<()>>),
    Scheduled(Sender<Option<()>>),
    Running,
}

use crate::agent_protocol::AgentClient;

#[derive(Error, Debug)]
pub enum Error {
    #[error("agent client error")]
    AgentClient,
}

pub type Result<T> = error_stack::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Tool {
    pub app_name: String,
    pub tool_name: String,
    pub input_schema: HashMap<String, serde_json::Value>,
    pub description: String,
}

impl From<(&str, ToolParam)> for Tool {
    fn from((app_name, tool): (&str, ToolParam)) -> Self {
        Self {
            app_name: app_name.to_string(),
            tool_name: tool.name,
            input_schema: tool.input_schema,
            description: tool.description,
        }
    }
}

// TODO(sean): abstract AppManager and AppManagerClient
#[derive(Clone)]
pub struct AppManager<T>
where
    T: AgentClient,
{
    inner: T,
    app_statuses: Arc<Mutex<HashMap<String, AppStatus>>>,
}

impl<T> AppManager<T>
where
    T: AgentClient,
{
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            app_statuses: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn run_apps(&self, apps: Vec<AppBundle>, timeout: &u64) -> Result<()> {
        let futs = apps
            .into_iter()
            .map(|app| self.run_app(app.clone(), timeout));

        try_join_all(futs).await?;

        Ok(())
    }

    pub async fn list_tools(&self, app_names: Vec<String>, timeout: &u64) -> Result<Vec<Tool>> {
        let futs = app_names.into_iter().map(|app_name| async move {
            if let AppStatus::NotRunning(_) | AppStatus::Scheduled(_) =
                self.app_status(&app_name).await
            {
                return Err(Report::new(Error::AgentClient)
                    .attach_printable(format!("app {} is not running or scheduled", app_name)));
            }

            self.inner
                .list_tools(&app_name, timeout)
                .await
                .change_context(Error::AgentClient)
                .and_then(|tools| {
                    Ok(tools
                        .into_iter()
                        .map(|tool| (app_name.as_str(), tool).into())
                        .collect::<Vec<_>>())
                })
        });

        let tools = try_join_all(futs)
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        Ok(tools)
    }

    pub async fn call_tool(
        &self,
        tool_use_block: &ToolUseBlock,
        timeout: &u64,
    ) -> Result<ToolResultBlockParam> {
        self.inner
            .call_tool(tool_use_block, timeout)
            .await
            .change_context(Error::AgentClient)
    }

    async fn app_status(&self, app_name: &str) -> AppStatus {
        let mut m = self.app_statuses.lock().await;
        match m.get(app_name) {
            Some(status) => status.clone(),
            None => {
                let tx = Sender::new(1);
                m.insert(app_name.to_string(), AppStatus::Scheduled(tx.clone()));
                AppStatus::NotRunning(tx)
            }
        }
    }

    async fn run_app(&self, app: AppBundle, timeout: &u64) -> Result<()> {
        let app_status = self.app_status(&app.config.name).await;

        match app_status {
            AppStatus::NotRunning(tx) => {
                let result = self.inner.run_app(&app, timeout).await;

                if tx.receiver_count() > 0 {
                    tx.send(result.as_ref().ok().cloned())
                        .change_context(Error::AgentClient)
                        .attach_printable("notify run agent listener failed")?;
                }

                let container_id = result.change_context(Error::AgentClient)?;
                self.app_statuses
                    .lock()
                    .await
                    .insert(app.config.name.clone(), AppStatus::Running);

                container_id
            }
            AppStatus::Scheduled(tx) => {
                tokio::time::timeout(Duration::from_secs(180), tx.subscribe().recv())
                    .await
                    .change_context(Error::AgentClient)
                    .attach_printable(format!("run agent timeout. agent: {}", app.config.name))?
                    .change_context(Error::AgentClient)?
                    .ok_or(Error::AgentClient)
                    .attach_printable(format!("run agent failed. agent: {}", app.config.name))?
            }
            AppStatus::Running => {}
        };

        Ok(())
    }
}
