use async_trait::async_trait;
use error_stack::ResultExt;
use tokio::sync::mpsc::Sender;

use super::{Error, Event, ExpectRes, Result};

#[async_trait]
pub trait ContainerCommClient {
    async fn publish(
        &self,
        event: Event,
        container_id: &Vec<u8>,
        expect_res: bool,
        timeout: &u64,
    ) -> Result<Option<Event>>;
    async fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event>;
    fn port(&self) -> u16;
}

#[derive(Clone)]
pub struct GrpcContainerCommClient {
    pub to_container_comm: Sender<(
        Vec<u8>,
        Event,
        ExpectRes,
        tokio::sync::oneshot::Sender<Result<Option<Event>>>,
    )>,
    pub from_container_comm: tokio::sync::broadcast::Sender<Event>,
    pub container_comm_port: u16,
}

#[async_trait]
impl ContainerCommClient for GrpcContainerCommClient {
    async fn publish(
        &self,
        event: Event,
        container_id: &Vec<u8>,
        expect_res: bool,
        timeout: &u64,
    ) -> Result<Option<Event>> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.to_container_comm
            .send((container_id.clone(), event, expect_res, tx))
            .await
            .change_context(Error::ContainerCommClient)?;

        let res = tokio::time::timeout(tokio::time::Duration::from_secs(*timeout), rx)
            .await
            .change_context(Error::Timeout)?
            .change_context(Error::ContainerCommClient)?;

        match res {
            Ok(res_event) => Ok(res_event),
            _ => Err(error_stack::Report::from(Error::ContainerCommClient)),
        }
    }

    async fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
        return self.from_container_comm.subscribe();
    }

    fn port(&self) -> u16 {
        self.container_comm_port
    }
}
