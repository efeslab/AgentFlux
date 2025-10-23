use thiserror::Error;

use crate::discovery::{AppBundle, AppConfig, AppFile};
use crate::inference_engine::{ToolParam, ToolUseBlock};
use crate::trace;

pub mod client;
pub mod server;

pub mod proto {
    pub mod common {
        tonic::include_proto!("common");
    }

    pub mod browserd {
        tonic::include_proto!("browserd");
    }
}

type Result<T> = error_stack::Result<T, Error>;
type ExpectRes = bool;

#[derive(Error, Debug)]
pub enum Error {
    #[error("grpc server failed")]
    GrpcServer,
    #[error("failed to send event")]
    SendEvent,
    #[error("container not connected error")]
    ContainerNotConnected,
    #[error("container_comm error")]
    ContainerComm,
    #[error("container_comm_client error")]
    ContainerCommClient,
    #[error("timeout")]
    Timeout,
    #[error("invalid event")]
    InvalidEvent,
}

#[derive(Debug, Clone)]
pub enum ContainerStatus {
    Connected(uuid::Uuid),
    Disconnected(uuid::Uuid),
}

#[derive(Debug, Clone)]
pub struct RunAppRequest {
    pub container_id: uuid::Uuid,
    pub app_bundle: AppBundle,
}

#[derive(Debug, Clone)]
pub struct ListToolsRequest {
    pub container_id: uuid::Uuid,
}

#[derive(Debug, Clone)]
pub struct CallToolRequest {
    pub container_id: uuid::Uuid,
    pub id: String,
    pub tool_name: String,
    pub tool_input: String,
}

impl TryFrom<(uuid::Uuid, &ToolUseBlock)> for CallToolRequest {
    type Error = Error;

    fn try_from(
        (container_id, value): (uuid::Uuid, &ToolUseBlock),
    ) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            container_id: container_id.clone(),
            id: value.id.clone(),
            tool_name: value.name.clone(),
            tool_input: serde_json::to_string(&value.input).map_err(|_| Error::InvalidEvent)?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RunAppResponse {
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ListToolsResponse {
    pub tools: Vec<ToolParam>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CallToolResponse {
    pub id: String,
    pub tool_results: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Payload {
    ContainerStatus(ContainerStatus),
    RunAppRequest(RunAppRequest),
    ListToolsRequest(ListToolsRequest),
    CallToolRequest(CallToolRequest),
    RunAppResponse(RunAppResponse),
    ListToolsResponse(ListToolsResponse),
    CallToolResponse(CallToolResponse),
}

impl TryFrom<Payload> for proto::browserd::event::Payload {
    type Error = Error;

    fn try_from(payload: Payload) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            msg: Some(match payload {
                Payload::ContainerStatus(_) => return Err(Error::InvalidEvent),
                Payload::RunAppRequest(run_app_req) => {
                    proto::browserd::event::payload::Msg::AppReq(proto::browserd::AppRequest {
                        container_id: run_app_req.container_id.as_bytes().to_vec(),
                        request: Some(proto::browserd::app_request::Request::RunAppReq(
                            proto::browserd::RunAppRequest {
                                app_bundle: Some(run_app_req.app_bundle.into()),
                            },
                        )),
                    })
                }
                Payload::ListToolsRequest(list_tools_req) => {
                    proto::browserd::event::payload::Msg::AppReq(proto::browserd::AppRequest {
                        container_id: list_tools_req.container_id.as_bytes().to_vec(),
                        request: Some(proto::browserd::app_request::Request::ListToolsReq(
                            proto::browserd::ListToolsRequest {},
                        )),
                    })
                }
                Payload::CallToolRequest(call_tool_req) => {
                    proto::browserd::event::payload::Msg::AppReq(proto::browserd::AppRequest {
                        container_id: call_tool_req.container_id.as_bytes().to_vec(),
                        request: Some(proto::browserd::app_request::Request::CallToolReq(
                            proto::browserd::CallToolRequest {
                                id: call_tool_req.id,
                                tool_name: call_tool_req.tool_name,
                                tool_input: call_tool_req.tool_input,
                            },
                        )),
                    })
                }
                _ => return Err(Error::InvalidEvent),
            }),
        })
    }
}

impl TryFrom<proto::browserd::event::Payload> for Payload {
    type Error = Error;

    fn try_from(
        payload: proto::browserd::event::Payload,
    ) -> std::result::Result<Self, Self::Error> {
        let msg = payload.msg.ok_or(Error::InvalidEvent)?;
        let app_response = match msg {
            proto::browserd::event::payload::Msg::AppRes(app_res) => {
                app_res.response.ok_or(Error::InvalidEvent)?
            }
            _ => return Err(Error::InvalidEvent),
        };

        match app_response {
            proto::browserd::app_response::Response::RunAppRes(run_app_res) => {
                Ok(Payload::RunAppResponse(RunAppResponse {
                    success: run_app_res.success,
                    error: run_app_res.error,
                }))
            }
            proto::browserd::app_response::Response::ListToolsRes(list_tools_res) => {
                Ok(Payload::ListToolsResponse(ListToolsResponse {
                    tools: list_tools_res
                        .tools
                        .into_iter()
                        .map(|t| t.try_into().map_err(|_| Error::InvalidEvent))
                        .collect::<std::result::Result<Vec<ToolParam>, Error>>()?,
                    error: list_tools_res.error,
                }))
            }
            proto::browserd::app_response::Response::CallToolRes(call_tool_res) => {
                Ok(Payload::CallToolResponse(CallToolResponse {
                    id: call_tool_res.id,
                    tool_results: call_tool_res.tool_results,
                    error: call_tool_res.error,
                }))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Event {
    pub id: Vec<u8>,
    pub payload: Payload,
}

impl TryFrom<Event> for proto::browserd::Event {
    type Error = error_stack::Report<Error>;

    fn try_from(event: Event) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            id: event.id,
            payload: Some(event.payload.try_into()?),
        })
    }
}

impl TryFrom<proto::browserd::Event> for Event {
    type Error = error_stack::Report<Error>;

    fn try_from(event: proto::browserd::Event) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            id: event.id,
            payload: event.payload.ok_or(Error::InvalidEvent)?.try_into()?,
        })
    }
}

impl From<AppConfig> for proto::browserd::AppConfig {
    fn from(app_config: AppConfig) -> Self {
        Self {
            name: app_config.name,
            agent_protocol: app_config.agent_protocol.into(),
            command: app_config.command,
            args: app_config.args,
            scripts: app_config.scripts.unwrap_or_default(),
            env: app_config.env.unwrap_or_default(),
        }
    }
}

impl From<AppFile> for proto::common::AppFile {
    fn from(app_file: AppFile) -> Self {
        Self {
            relative_path: app_file.relative_path,
            content: app_file.content,
        }
    }
}

impl From<AppBundle> for proto::browserd::AppBundle {
    fn from(app_bundle: AppBundle) -> Self {
        Self {
            config: Some(app_bundle.config.into()),
            files: app_bundle.files.map(|files| proto::common::AppFileList {
                files: files.into_iter().map(Into::into).collect(),
            }),
        }
    }
}

impl From<Vec<trace::Trace>> for proto::browserd::TraceList {
    fn from(traces: Vec<trace::Trace>) -> Self {
        Self {
            traces: traces.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<proto::browserd::TraceList> for Vec<trace::Trace> {
    fn from(trace_list: proto::browserd::TraceList) -> Self {
        trace_list.traces.into_iter().map(Into::into).collect()
    }
}

impl From<trace::Trace> for proto::browserd::Trace {
    fn from(trace: trace::Trace) -> Self {
        Self {
            label: trace.label,
            latency: trace.latency,
            metadata: trace.metadata,
            inner_traces: trace.inner_traces.map(Into::into),
        }
    }
}

impl From<proto::browserd::Trace> for trace::Trace {
    fn from(trace: proto::browserd::Trace) -> Self {
        Self {
            label: trace.label,
            latency: trace.latency,
            metadata: trace.metadata,
            inner_traces: trace.inner_traces.map(Into::into),
        }
    }
}
