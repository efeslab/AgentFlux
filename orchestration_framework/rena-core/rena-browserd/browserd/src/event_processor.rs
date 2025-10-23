use thiserror::Error;
use tokio::select;
use tokio::sync::broadcast::Receiver;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::container_comm::client::ContainerCommClient;
use crate::container_comm::Event;
use crate::handler::Handler;

type Result<T> = error_stack::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("event stream error")]
    EventStream,
    #[error("container_comm_client error")]
    ContainerCommClient,
}

pub struct ContainerEventProcessor<T, C>
where
    T: Handler + Clone + Send + 'static,
    C: ContainerCommClient + Clone + Send + 'static,
{
    label: String,
    event_stream: Receiver<Event>,
    handler: T,
    container_comm_client: C,
}

impl<T, C> ContainerEventProcessor<T, C>
where
    T: Handler + Clone + Send + 'static,
    C: ContainerCommClient + Clone + Send + 'static,
{
    pub fn new(
        label: &str,
        event_stream: Receiver<Event>,
        handler: T,
        container_comm_client: C,
    ) -> Self {
        Self {
            label: label.to_string(),
            event_stream,
            handler,
            container_comm_client,
        }
    }

    pub async fn run(mut self, token: CancellationToken) -> Result<()> {
        loop {
            select! {
              data = self.event_stream.recv() => {
                  let event = data.map_err(|_| Error::EventStream)?;
                  let handler = self.handler.clone();
                  let container_comm_client = self.container_comm_client.clone();

                  // NOTE(sean): when one event is handling, other events should be processed
                  tokio::spawn(async move {
                    match handler.handle(event).await {
                      Ok(Some((event, container_id))) => {
                        match container_comm_client
                          .publish(event, &container_id, false, &10)
                          .await {
                            Ok(_) => {},
                            Err(e) => info!("failed to publish handler event to container: {:?}", e),
                          }
                      }
                      // non-relevant event to handler
                      Ok(None) => {}
                      Err(e) =>{
                          info!("handler error: {:?}", e);
                      }
                    }
                  });
              }
              _ = token.cancelled() => {
                  info!("exiting {} handler task", self.label);

                  return Ok(());
              }
            }
        }
    }
}
