use async_trait::async_trait;
use thiserror::Error;

use crate::container_comm::Event;

pub type Result<T> = error_stack::Result<T, Error>;
pub type ContainerId = Vec<u8>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("inference_engine handler failed: {0}")]
    InferenceEngineHandler(String),
    #[error("invalid event")]
    InvalidEvent,
    #[error("inferece engine error")]
    InferenceEngine,
    #[error("app request handler error")]
    AppReqHandler,
}

#[async_trait]
pub trait Handler {
    async fn handle(&self, req: Event) -> Result<Option<(Event, ContainerId)>>;
}
