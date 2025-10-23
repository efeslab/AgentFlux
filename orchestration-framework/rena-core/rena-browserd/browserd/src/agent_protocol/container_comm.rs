use error_stack::ResultExt;
use tokio::sync::broadcast::Receiver;
use tracing::info;

use super::{Error, Result};
use crate::container_comm::{
    client, CallToolRequest, CallToolResponse, Event, ListToolsRequest, ListToolsResponse, Payload,
    RunAppRequest, RunAppResponse,
};
use crate::discovery::AppBundle;
use crate::inference_engine::ToolParam;

#[derive(Clone)]
pub struct ContainerComm<T>
where
    T: client::ContainerCommClient,
{
    pub inner: T,
}

impl<T> ContainerComm<T>
where
    T: client::ContainerCommClient,
{
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    pub async fn subscribe(&self) -> Receiver<Event> {
        self.inner.subscribe().await
    }

    pub async fn run_app(
        &self,
        container_id: &uuid::Uuid,
        app_bundle: &AppBundle,
        timeout: &u64,
    ) -> Result<()> {
        let event_id = uuid::Uuid::new_v4();
        info!(
            "sending run_app req to container with event_id: {}",
            event_id
        );

        let req_event = Event {
            id: event_id.as_bytes().to_vec(),
            payload: Payload::RunAppRequest(RunAppRequest {
                container_id: container_id.clone(),
                app_bundle: app_bundle.clone(),
            }),
        };

        match self
            .inner
            .publish(req_event, &container_id.as_bytes().to_vec(), true, timeout)
            .await
            .change_context(Error::Transport)?
        {
            Some(res_event) => match res_event.payload {
                Payload::RunAppResponse(RunAppResponse { success, error }) => {
                    if !success {
                        return Err(error_stack::Report::new(Error::RunApp)
                            .attach_printable(error.unwrap_or_default()));
                    }

                    return Ok(());
                }
                _ => return Err(error_stack::Report::new(Error::InvalidEvent)),
            },
            None => Err(error_stack::Report::new(Error::Transport)),
        }
    }

    pub fn port(&self) -> u16 {
        self.inner.port()
    }

    pub async fn list_tools(
        &self,
        container_id: &uuid::Uuid,
        timeout: &u64,
    ) -> Result<Vec<ToolParam>> {
        let event_id = uuid::Uuid::new_v4();
        info!(
            "sending list_tools req to container with event_id: {}",
            event_id
        );

        let req_event = Event {
            id: event_id.as_bytes().to_vec(),
            payload: Payload::ListToolsRequest(ListToolsRequest {
                container_id: container_id.clone(),
            }),
        };

        match self
            .inner
            .publish(req_event, &container_id.as_bytes().to_vec(), true, timeout)
            .await
            .change_context(Error::Transport)?
        {
            Some(res_event) => match res_event.payload {
                Payload::ListToolsResponse(ListToolsResponse { tools, error }) => {
                    if let Some(error) = error {
                        return Err(
                            error_stack::Report::new(Error::ListTools).attach_printable(error)
                        );
                    }

                    return Ok(tools);
                }
                _ => return Err(error_stack::Report::new(Error::InvalidEvent)),
            },
            None => Err(error_stack::Report::new(Error::Transport)),
        }
    }

    pub async fn call_tool(
        &self,
        req: &CallToolRequest,
        timeout: &u64,
    ) -> Result<CallToolResponse> {
        let event_id = uuid::Uuid::new_v4();
        info!(
            "sending {} req with input {} to container with event_id: {}",
            req.tool_name, req.tool_input, event_id
        );

        let req_event = Event {
            id: event_id.as_bytes().to_vec(),
            payload: Payload::CallToolRequest(CallToolRequest {
                container_id: req.container_id.clone(),
                id: req.id.clone(),
                tool_name: req.tool_name.clone(),
                tool_input: req.tool_input.clone(),
            }),
        };

        match self
            .inner
            .publish(
                req_event,
                &req.container_id.as_bytes().to_vec(),
                true,
                timeout,
            )
            .await
            .change_context(Error::Transport)?
        {
            Some(res_event) => match res_event.payload {
                Payload::CallToolResponse(call_tool_res) => Ok(call_tool_res),
                _ => return Err(error_stack::Report::new(Error::InvalidEvent)),
            },
            None => Err(error_stack::Report::new(Error::Transport)),
        }
    }
}
