use async_trait::async_trait;
use error_stack::{Report, ResultExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast::Receiver;
use tokio::sync::Mutex;

use super::{container, container_comm, AgentClient, Error, Result};
use crate::container::ContainerClient;
use crate::container_comm::client::ContainerCommClient;
use crate::container_comm::{ContainerStatus, Event, Payload};
use crate::discovery::AppBundle;
use crate::inference_engine::{ToolParam, ToolResultBlockParam, ToolUseBlock};

#[derive(Clone)]
pub struct GrpcTransportAgent<T, C>
where
    T: ContainerCommClient + Send + Sync,
    C: ContainerClient + Send + Sync,
{
    pub transport: container_comm::ContainerComm<T>,
    pub container_client: container::Container<C>,
    pub app_container_map: Arc<Mutex<HashMap<String, uuid::Uuid>>>,
    pub tool_container_map: Arc<Mutex<HashMap<String, uuid::Uuid>>>,
}

impl<T, C> GrpcTransportAgent<T, C>
where
    T: ContainerCommClient + Send + Sync,
    C: ContainerClient + Send + Sync,
{
    pub fn new(container_comm_client: T, container_client: C, runtime_path: &str) -> Self {
        Self {
            transport: container_comm::ContainerComm::new(container_comm_client),
            container_client: container::Container::new(container_client, runtime_path),
            app_container_map: Arc::new(Mutex::new(HashMap::new())),
            tool_container_map: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl<T, C> AgentClient for GrpcTransportAgent<T, C>
where
    T: ContainerCommClient + Send + Sync,
    C: ContainerClient + Send + Sync,
{
    async fn run_app(&self, app_bundle: &AppBundle, timeout: &u64) -> Result<()> {
        let event_stream = self.transport.subscribe().await;

        let container_id = self
            .container_client
            .run(
                &app_bundle.config.runtime,
                &self.transport.port(),
                &app_bundle.config.mounts,
            )
            .await?;

        let connected = tokio::time::timeout(
            tokio::time::Duration::from_secs(*timeout),
            is_connected(event_stream, &container_id),
        )
        .await
        .change_context(Error::Timeout)
        .attach_printable("container status timeout")??;

        if !connected {
            return Err(Report::from(Error::Transport).attach_printable("container not connected"));
        }

        self.transport
            .run_app(&container_id, &app_bundle, timeout)
            .await?;

        self.app_container_map
            .lock()
            .await
            .insert(app_bundle.config.name.clone(), container_id);

        Ok(())
    }

    async fn list_tools(&self, app_name: &str, timeout: &u64) -> Result<Vec<ToolParam>> {
        let container_id = self
            .app_container_map
            .lock()
            .await
            .get(app_name)
            .ok_or_else(|| Report::new(Error::Transport).attach_printable("app not found"))?
            .to_owned();

        let tools = self
            .transport
            .list_tools(&container_id, timeout)
            .await?
            .iter()
            .map(|tool| {
                tool.clone()
                    .try_into()
                    .map_err(|_| Report::new(Error::Transport))
            })
            .collect::<Result<Vec<ToolParam>>>()?;

        self.tool_container_map
            .lock()
            .await
            .extend(tools.iter().map(|tool| (tool.name.clone(), container_id)));

        Ok(tools)
    }

    async fn call_tool(
        &self,
        tool_use_block: &ToolUseBlock,
        timeout: &u64,
    ) -> Result<ToolResultBlockParam> {
        let container_id = self
            .tool_container_map
            .lock()
            .await
            .get(tool_use_block.name.as_str())
            .ok_or_else(|| {
                Report::new(Error::Transport).attach_printable(format!(
                    "tool not found. tool_name: {}",
                    tool_use_block.name
                ))
            })?
            .to_owned();

        let tool_result = self
            .transport
            .call_tool(
                &(container_id, tool_use_block)
                    .try_into()
                    .map_err(|_| Report::new(Error::InvalidEvent))?,
                timeout,
            )
            .await?;

        Ok(tool_result.into())
    }
}

async fn is_connected(
    mut event_stream: Receiver<Event>,
    container_id: &uuid::Uuid,
) -> Result<bool> {
    loop {
        let container_status = match event_stream
            .recv()
            .await
            .change_context(Error::Transport)?
            .payload
        {
            Payload::ContainerStatus(container_status) => container_status,
            _ => continue,
        };

        match container_status {
            ContainerStatus::Connected(event_contianer_id) => {
                if event_contianer_id != *container_id {
                    continue;
                }

                return Ok(true);
            }
            ContainerStatus::Disconnected(event_contianer_id) => {
                if event_contianer_id != *container_id {
                    continue;
                }

                return Ok(false);
            }
        }
    }
}
