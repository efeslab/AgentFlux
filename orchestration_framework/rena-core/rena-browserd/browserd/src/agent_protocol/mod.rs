use async_trait::async_trait;
use thiserror::Error;

use crate::discovery::AppBundle;
use crate::inference_engine::{ToolParam, ToolResultBlockParam, ToolUseBlock};

pub mod container;
pub mod container_comm;
pub mod grpc_transport;

pub type ToolWithContainerId = (ToolParam, uuid::Uuid);

#[derive(Error, Debug)]
pub enum Error {
    #[error("invalid runtime error")]
    InvalidRuntime,
    #[error("container client error")]
    ContainerClient,
    #[error("transport error")]
    Transport,
    #[error("timeout error")]
    Timeout,
    #[error("invalid event error")]
    InvalidEvent,
    #[error("run app error")]
    RunApp,
    #[error("list tools error")]
    ListTools,
    #[error("agent client error")]
    AgentClient,
    #[error("inference engine error")]
    InferenceEngine,
    #[error("discovery error")]
    Discovery,
}

pub type Result<T> = error_stack::Result<T, Error>;

#[async_trait]
pub trait AgentClient {
    async fn run_app(&self, app_bundle: &AppBundle, timeout: &u64) -> Result<()>;
    async fn list_tools(&self, app_name: &str, timeout: &u64) -> Result<Vec<ToolParam>>;
    async fn call_tool(
        &self,
        tool_use_block: &ToolUseBlock,
        timeout: &u64,
    ) -> Result<ToolResultBlockParam>;
}
