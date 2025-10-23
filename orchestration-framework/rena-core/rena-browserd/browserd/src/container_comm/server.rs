use error_stack::ResultExt;
use futures::Stream;
use std::sync::{Arc, Mutex};
use std::{collections::HashMap, pin::Pin};
use tokio::select;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;
use tonic::{async_trait, Request, Response, Status};
use tracing::info;

use super::{proto::browserd, ContainerStatus, Error, Event, ExpectRes, Payload, Result};
use crate::container_comm::client::GrpcContainerCommClient;

pub struct GrpcContainerComm {
    url: String,
    to_clients: tokio::sync::broadcast::Sender<Event>,
    from_clients: Option<
        Receiver<(
            Vec<u8>,
            Event,
            ExpectRes,
            tokio::sync::oneshot::Sender<Result<Option<Event>>>,
        )>,
    >,
    containers: Arc<Mutex<HashMap<String, Sender<browserd::Event>>>>,
    requests: Arc<Mutex<HashMap<String, tokio::sync::oneshot::Sender<Result<Option<Event>>>>>>,
}

impl GrpcContainerComm {
    pub fn new(host: &str, port: u16) -> (Self, GrpcContainerCommClient) {
        let (to_container_comm, from_clients) = tokio::sync::mpsc::channel(1000);
        let (from_container_comm, _) = tokio::sync::broadcast::channel(1000);
        let client = GrpcContainerCommClient {
            to_container_comm,
            from_container_comm: from_container_comm.clone(),
            container_comm_port: port,
        };

        (
            Self {
                url: format!("{}:{}", host, port),
                to_clients: from_container_comm,
                from_clients: Some(from_clients),
                containers: Arc::new(Mutex::new(HashMap::new())),
                requests: Arc::new(Mutex::new(HashMap::new())),
            },
            client,
        )
    }

    pub async fn run(mut self, token: CancellationToken) -> Result<()> {
        let containers = self.containers.clone();
        let requests = self.requests.clone();
        let from_clients_listener_token = token.clone();
        let mut from_clients = self.from_clients.take().expect("from_clients not set");

        // TODO(sean): need to be aware of spawned task failure to notify browserd
        // probably using a joinset, waiting if task finish propagate result
        tokio::spawn(async move {
            loop {
                select! {
                    // listening for events from browser to be broadcasted to a container
                    data = from_clients.recv() => {
                        let (container_id, event, expect_res, res_chan) = match data {
                            Some((container_id, event, expect_res, res_chan)) => {
                                (container_id, event, expect_res, res_chan)
                            }
                            None => continue,
                        };

                        match send(&container_id, containers.clone(), event.clone()).await {
                            Ok(_) => match expect_res {
                                true => {
                                    requests.lock().unwrap().insert(
                                        uuid::Uuid::from_slice(&event.id).unwrap().to_string(),
                                        res_chan,
                                    );
                                }
                                false => {
                                    res_chan.send(Ok(None)).unwrap();
                                }
                            },
                            Err(e) => {
                                res_chan
                                    .send(Err(Error::SendEvent).attach_printable(e))
                                    .unwrap();
                            }
                        }
                    }
                    _ = from_clients_listener_token.cancelled() => {
                        info!("exiting container comm task from_browser listener");
                        break;
                    }
                }
            }
        });

        let addr = self.url.parse().expect("local address must be valid");
        let server = tonic::transport::Server::builder()
            .add_service(
                browserd::browserd_server::BrowserdServer::new(self)
                    .max_decoding_message_size(256 * 1024 * 1024) // 256MB
                    .max_encoding_message_size(256 * 1024 * 1024) // 256MB
            )
            .serve(addr);

        select! {
            result = server => {
                result.change_context(Error::GrpcServer)?;
            },
            _ = token.cancelled() => {
                info!("exiting container comm task grpc server");
                return Ok(());
            }
        }

        Ok(())
    }
}

#[async_trait]
impl browserd::browserd_server::Browserd for GrpcContainerComm {
    type ConnectStream =
        Pin<Box<dyn Stream<Item = std::result::Result<browserd::Event, Status>> + Send + 'static>>;

    async fn connect(
        &self,
        req: Request<tonic::Streaming<browserd::Event>>,
    ) -> std::result::Result<Response<Self::ConnectStream>, Status> {
        info!("received a connection from a container");

        let container_id: Arc<Mutex<Option<uuid::Uuid>>> = Arc::new(Mutex::new(None));
        let (tx, rx) = tokio::sync::mpsc::channel(1000);

        // NOTE(sean): to get the container_id, we send a GetContainerIdRequest to the connected container
        let get_container_id_event_id = uuid::Uuid::new_v4();
        let get_container_id_event = browserd::Event {
            id: get_container_id_event_id.as_bytes().to_vec(),
            payload: Some(browserd::event::Payload {
                msg: Some(browserd::event::payload::Msg::ContainerReq(
                    browserd::ContainerRequest {
                        request: Some(browserd::container_request::Request::GetContainerIdReq(
                            browserd::GetContainerIdRequest {},
                        )),
                    },
                )),
            }),
        };
        match tx.send(get_container_id_event).await {
            Ok(_) => {}
            Err(_) => return Err(Status::internal("failed to send GetContainerIdRequest")),
        }

        let mut input_stream = req.into_inner();
        let to_clients = self.to_clients.clone();
        let containers = self.containers.clone();
        let requests = self.requests.clone();
        tokio::spawn(async move {
            loop {
                match input_stream.next().await {
                    Some(Ok(event)) => {
                        let event_id = uuid::Uuid::from_slice(&event.id).unwrap();

                        // if container_id not in containers, container should answer
                        // our GetContainerIdRequest with a  GetContainerIdResponse
                        if container_id.lock().unwrap().is_none() {
                            match event.clone().payload {
                                Some(payload) => match payload.msg {
                                    Some(msg) => match msg {
                                        browserd::event::payload::Msg::ContainerRes(
                                            container_res,
                                        ) => {
                                            match container_res.response {
                                                Some(browserd::container_response::Response::GetContainerIdRes(
                                                get_container_id_res,
                                                )) => {
                                                    let id =
                                                        uuid::Uuid::from_slice(&get_container_id_res.container_id).unwrap();

                                                    *container_id.lock().unwrap() = Some(id.clone());
                                                    containers
                                                        .lock()
                                                        .unwrap()
                                                        .insert(id.clone().to_string(), tx.clone());
                                                    info!("container connected. container_id: {}", id);

                                                    let _ = to_clients.send(Event {
                                                        id: id.as_bytes().to_vec(),
                                                        payload: Payload::ContainerStatus(ContainerStatus::Connected(id.clone())),
                                                    });

                                                    // because browser does not need to receive GetContainerIdRes
                                                    // (it's detail of communication layer)
                                                    continue;
                                                }
                                                _ => continue,
                                            }
                                        }
                                        _ => continue,
                                    },
                                    None => continue,
                                },
                                None => continue,
                            }
                        }

                        info!(
                            "received event from container. container_id: {}, event_id: {}",
                            container_id.lock().unwrap().as_ref().unwrap(),
                            event_id
                        );

                        // NOTE(sean): if it's res_event, send it to its res_chan, otherwise send it to clients
                        let res_chan = {
                            let mut m = requests.lock().unwrap();
                            m.remove(&uuid::Uuid::from_slice(&event.id).unwrap().to_string())
                        };

                        match res_chan {
                            Some(res_chan) => {
                                let _ = res_chan.send(Ok(Some(event.try_into().unwrap())));
                            }
                            None => {
                                let _ = to_clients.send(event.try_into().unwrap());
                            }
                        };
                    }
                    Some(Err(e)) => {
                        let id_ = match container_id.lock().unwrap().as_ref() {
                            Some(id) => id.to_string(),
                            None => "unknown".to_string(),
                        };
                        info!(
                            "failed to receive event from container. container_id {}, error: {:?}",
                            id_, e
                        );
                        // TODO(sean): given one Err(Event) received from container is it
                        // possible to continue receiving events from the container?
                        // if not we should break here instead of continue
                        // continue;
                        break;
                    }
                    None => {
                        let id = match container_id.lock().unwrap().as_ref() {
                            Some(id) => {
                                containers.lock().unwrap().remove(&id.to_string());
                                let _ = to_clients.send(Event {
                                    id: id.as_bytes().to_vec(),
                                    payload: Payload::ContainerStatus(
                                        ContainerStatus::Disconnected(id.clone()),
                                    ),
                                });

                                id.to_string()
                            }
                            None => "unknown".to_string(),
                        };

                        info!("container disconnected. container_id: {}", id);

                        break;
                    }
                }
            }
        });

        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx).map(|event| Ok(event)),
        )))
    }
}

async fn send(
    container_id: &Vec<u8>,
    containers: Arc<Mutex<HashMap<String, Sender<browserd::Event>>>>,
    event: Event,
) -> Result<()> {
    let container_id = uuid::Uuid::from_slice(container_id).unwrap().to_string();

    // NOTE(sean): assuming container must exists,
    // hence browser (upper level) should make sure container exists
    let tx = containers
        .lock()
        .map_err(|_| Error::ContainerComm)?
        .get(&container_id)
        .ok_or(Error::ContainerNotConnected)?
        .clone();

    tx.send(event.try_into()?)
        .await
        .change_context(Error::SendEvent)?;

    Ok(())
}
