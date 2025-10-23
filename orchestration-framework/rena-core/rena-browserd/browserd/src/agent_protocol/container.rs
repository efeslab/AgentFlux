use error_stack::{Report, ResultExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast::Sender;
use tokio::sync::Mutex;
use tokio::time::{timeout, Duration};
use tracing::info;

use super::{Error, Result};
use crate::container::{self, ContainerClient, Image, Mount};
use crate::discovery::Runtime;

#[derive(Clone)]
pub enum RuntimeImageStatus {
    NotBuilt(Sender<Option<Image>>),
    Building(Sender<Option<Image>>),
    Built(Image),
}

#[derive(Clone)]
pub struct Container<T>
where
    T: ContainerClient,
{
    inner: T,
    runtime_path: String,
    runtime_image_statuses: Arc<Mutex<HashMap<Runtime, RuntimeImageStatus>>>,
}

impl<T> Container<T>
where
    T: ContainerClient,
{
    pub fn new(inner: T, runtime_path: &str) -> Self {
        Self {
            inner,
            runtime_path: runtime_path.to_string(),
            runtime_image_statuses: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn run(
        &self,
        runtime: &Runtime,
        container_comm_port: &u16,
        mounts: &Option<Vec<Mount>>,
    ) -> Result<uuid::Uuid> {
        let container_id = uuid::Uuid::new_v4();
        let image_status = image_status(runtime, &self.inner, &self.runtime_image_statuses).await?;

        build_image(
            runtime,
            &self.runtime_path,
            &image_status,
            &self.inner,
            &self.runtime_image_statuses,
        )
        .await?;

        let container_creation_result = create_container(
            &self.inner,
            &runtime.to_string(),
            &container_id.to_string(),
            container_comm_port,
            mounts,
        )
        .await?;

        // Start the container first
        self.inner
            .start_container(&container_creation_result.container)
            .await
            .change_context(Error::ContainerClient)?;

        // Execute copy operations after container is started
        if !container_creation_result.copy_operations.is_empty() {
            self.inner
                .execute_copy_operations(&container_id.to_string(), &container_creation_result.copy_operations)
                .await
                .change_context(Error::ContainerClient)?;
        }

        Ok(container_id)
    }
}

async fn create_container<T>(
    container_client: &T,
    image_name: &str,
    container_name: &str,
    container_comm_port: &u16,
    mounts: &Option<Vec<Mount>>,
) -> Result<container::ContainerCreationResult>
where
    T: ContainerClient,
{
    let image = container_client
        .get_image(image_name)
        .await
        .change_context(Error::ContainerClient)?
        .ok_or(Error::ContainerClient)?;

    let cmd = vec![
        "python3".to_string(),
        "rena_runtime/cli.py".to_string(),
        container_name.to_string(),
        // format!("host.docker.internal:{}", container_comm_port),
        format!("172.17.0.1:{}", container_comm_port),
    ];

    let container_creation_result = container_client
        .create_container(container_name, &image, cmd, None, None, mounts)
        .await
        .change_context(Error::ContainerClient)?;

    Ok(container_creation_result)
}

async fn image_status<T>(
    runtime: &Runtime,
    container_client: &T,
    runtime_image_statuses: &Arc<Mutex<HashMap<Runtime, RuntimeImageStatus>>>,
) -> Result<RuntimeImageStatus>
where
    T: ContainerClient,
{
    let mut m = runtime_image_statuses.lock().await;
    let image_status = match m.get(runtime) {
        Some(status) => status.clone(),
        None => match container_client
            .get_image(&runtime.to_string())
            .await
            .change_context(Error::ContainerClient)?
        {
            Some(image) => RuntimeImageStatus::Built(image),
            None => {
                let tx = Sender::new(1);
                m.insert(runtime.clone(), RuntimeImageStatus::Building(tx.clone()));
                RuntimeImageStatus::NotBuilt(tx)
            }
        },
    };

    Ok(image_status)
}

async fn build_image<T>(
    runtime: &Runtime,
    runtime_path: &str,
    image_status: &RuntimeImageStatus,
    container_client: &T,
    runtime_image_statuses: &Arc<Mutex<HashMap<Runtime, RuntimeImageStatus>>>,
) -> Result<()>
where
    T: ContainerClient,
{
    match image_status {
        RuntimeImageStatus::NotBuilt(tx) => {
            info!("building image for: {}", runtime.to_string());

            let result = container_client
                .build_image(
                    runtime.to_string().as_str(),
                    runtime_path,
                    "./rena-runtime",
                    &format!("./rena-runtime/Dockerfile.{}", runtime.to_string()),
                    true,
                    true,
                )
                .await;

            if tx.receiver_count() > 0 {
                tx.send(result.as_ref().ok().cloned())
                    .map_err(|e| Report::from(e))
                    .change_context(Error::ContainerClient)
                    .attach_printable("notify runtime image build listeners failed.")?;
            }

            let image = result.change_context(Error::ContainerClient)?;
            runtime_image_statuses
                .lock()
                .await
                .insert(runtime.clone(), RuntimeImageStatus::Built(image));
        }
        RuntimeImageStatus::Building(tx) => {
            timeout(Duration::from_secs(180), tx.subscribe().recv())
                .await
                .change_context(Error::ContainerClient)
                .attach_printable("runtime image build timeout")?
                .change_context(Error::ContainerClient)?
                .ok_or(Error::ContainerClient)
                .attach_printable("runtime image build failed")?;
        }
        RuntimeImageStatus::Built(_) => {}
    }

    Ok(())
}
