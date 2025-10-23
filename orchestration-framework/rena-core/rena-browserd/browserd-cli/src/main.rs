use clap::{arg, Parser, Subcommand};
use dotenv::dotenv;
use error_stack::{FutureExt, ResultExt};
use std::process::ExitCode;
use tokio::signal::unix::{signal, SignalKind};
use tokio::{select, task::JoinSet};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use browserd::{
    agent_protocol, app_registry, config, container, container_comm, discovery, inference_engine,
    orchestrator, App,
};

pub mod common;
pub mod eval;
pub mod query;
pub mod repl;

#[derive(Debug, Subcommand)]
pub enum SubCommand {
    /// Launch a REPL to query browserd interactively
    Repl,
    /// Query browserd with a specific config
    Query(query::Args),
    /// Run evaluation suite against browserd
    Eval(eval::Args),
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long, default_value = "config.toml")]
    pub config: String,

    #[clap(subcommand)]
    pub cmd: Option<SubCommand>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Args = Args::parse();
    setup_logger();

    match dotenv() {
        Ok(_) => info!("loaded .env file"),
        Err(_) => warn!(".env file not found"),
    }

    let token = CancellationToken::new();
    let exit_token = token.clone();
    tokio::spawn(async move {
        let mut sigint = signal(SignalKind::interrupt()).expect("failed to capture SIGINT");
        let mut sigterm = signal(SignalKind::terminate()).expect("failed to capture SIGTERM");

        select! {
            _ = sigint.recv() => {}
            _ = sigterm.recv() => {}
        }

        info!("signal received, exiting gracefully");

        exit_token.cancel();
    });

    match inner_main(args, token).await {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {:?}", e);
            ExitCode::FAILURE
        }
    }
}

async fn inner_main(args: Args, cancel: CancellationToken) -> common::Result<()> {
    let config = common::init_config::<config::Config>(&args.config)
        .change_context(common::Error::Config)?;

    let (container_comm, container_comm_client) = container_comm::server::GrpcContainerComm::new(
        &config.container_comm_host,
        config.container_comm_port,
    );

    let agent_client = agent_protocol::grpc_transport::GrpcTransportAgent::new(
        container_comm_client,
        container::docker::Docker::new(
            bollard::Docker::connect_with_local_defaults()
                .change_context(common::Error::Browserd)?,
        )
        .await,
        &config.runtime_path,
    );

    let registry = app_registry::Registry::new(&config.registry)
        .await
        .change_context(common::Error::Browserd)?;

    let config_for_eval = config.clone();
    let discovery = match config.clone().discovery {
        discovery::Config::Static { apps } => discovery::StaticDiscovery::new(apps, registry).await,
    };

    match config.inference_engine {
        inference_engine::Config::Anthropic {
            model,
            max_tokens,
            api_key,
        } => {
            run(
                container_comm,
                agent_client,
                discovery,
                inference_engine::anthropic::Anthropic::new(&model, &max_tokens, &api_key)
                    .change_context(common::Error::Browserd)?,
                cancel,
                args.cmd,
                config_for_eval,
            )
            .await
        }
        inference_engine::Config::Ollama { model, max_tokens } => {
            run(
                container_comm,
                agent_client,
                discovery,
                inference_engine::ollama::Ollama::new(&model, max_tokens),
                cancel,
                args.cmd,
                config_for_eval,
            )
            .await
        }
        inference_engine::Config::OpenAI {
            model,
            max_tokens,
            api_key,
            url,
        } => {
            run(
                container_comm,
                agent_client,
                discovery,
                inference_engine::openai::OpenAI::new(&model, max_tokens, &api_key, &url)
                    .change_context(common::Error::Browserd)?,
                cancel,
                args.cmd,
                config_for_eval,
            )
            .await
        }
    }
}

async fn run<T, D, H>(
    container_comm: container_comm::server::GrpcContainerComm,
    agent_client: T,
    discovery: D,
    inference_engine: H,
    token: CancellationToken,
    cmd: Option<SubCommand>,
    browserd_config: config::Config,
) -> common::Result<()>
where
    T: agent_protocol::AgentClient + Clone + Send + Sync + 'static,
    D: discovery::Discovery + Clone + Send + Sync + 'static,
    H: inference_engine::HTTPClient + Clone + Send + Sync + 'static,
{
    let (browserd, browserd_client) = App::new(
        container_comm,
        agent_client,
        discovery,
        orchestrator::BasicOrchestrator::new(inference_engine),
    )
    .await
    .change_context(common::Error::Browserd)?;

    let mut join_set = JoinSet::new();
    join_set.spawn(
        browserd
            .run(token.child_token())
            .change_context(common::Error::Browserd),
    );

    match cmd {
        Some(SubCommand::Repl) | None => {
            join_set.spawn(repl::run(browserd_client, token.child_token()))
        }
        Some(SubCommand::Query(args)) => {
            join_set.spawn(query::run(browserd_client, args, token.child_token()))
        }
        Some(SubCommand::Eval(args)) => join_set.spawn(eval::run(
            browserd_client,
            args,
            browserd_config,
            token.child_token(),
        )),
    };

    match join_set.join_next().await {
        Some(Ok(task_result)) => {
            token.cancel();
            while let Some(_) = join_set.join_next().await {}

            task_result.change_context(common::Error::Browserd)?;
        }
        _ => Err(common::Error::Browserd)?,
    }

    Ok(())
}

fn setup_logger() {
    tracing_subscriber::fmt().compact().init();
}
