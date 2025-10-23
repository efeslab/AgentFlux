use async_trait::async_trait;
use bollard::exec::{CreateExecOptions, StartExecOptions, StartExecResults};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use tracing::info;

use bollard::container::{
    Config, CreateContainerOptions, ListContainersOptions, LogOutput, RemoveContainerOptions,
    StartContainerOptions,
};
use bollard::image::{BuildImageOptions, ListImagesOptions};
use bollard::models::{self, HostConfig};
use error_stack::ResultExt;
use tokio_stream::StreamExt;

use super::{ContainerClient, ContextTarPath, Error, Mount, Result, ContainerCreationResult};

#[derive(Clone)]
pub struct Docker {
    pub inner: bollard::Docker,
}

impl Docker {
    pub async fn new(inner: bollard::Docker) -> Self {
        Self { inner }
    }
}

impl From<&Mount> for models::Mount {
    fn from(mount: &Mount) -> Self {
        match mount {
            Mount::Bind { source, target } => models::Mount {
                source: Some(source.clone()),
                target: Some(target.clone()),
                typ: Some(models::MountTypeEnum::BIND),
                ..Default::default()
            },
            Mount::Copy { source, target } => {
                // For copy mounts, create a temporary bind mount for copying
                let temp_target = format!("/temp_copy_{}", target.replace("/", "_"));
                models::Mount {
                    source: Some(source.clone()),
                    target: Some(temp_target.clone()),
                    typ: Some(models::MountTypeEnum::BIND),
                    ..Default::default()
                }
            }
        }
    }
}

#[async_trait]
impl ContainerClient for Docker {
    async fn get_image(&self, image_name: &str) -> Result<Option<super::Image>> {
        let mut filters = HashMap::new();
        let filter_values = vec![image_name.to_string()];
        filters.insert("reference".to_string(), filter_values);

        let options = Some(ListImagesOptions {
            all: false,
            filters,
            ..Default::default()
        });

        let images = self
            .inner
            .list_images(options)
            .await
            .change_context(Error::GetImage)?;

        match images.first() {
            Some(image) => Ok(Some(super::Image {
                id: image.id.clone(),
                name: image_name.to_string(),
            })),
            None => Ok(None),
        }
    }

    async fn build_image(
        &self,
        name: &str,
        host_context_path: &str,
        container_context_path: &str,
        dockerfile_path: &str,
        rm: bool,
        forcerm: bool,
    ) -> Result<super::Image> {
        let mut context_builder = tar::Builder::new(
            File::create(ContextTarPath::from(name)).change_context(Error::BuildImage)?,
        );

        context_builder
            .append_dir_all(container_context_path, host_context_path)
            .change_context(Error::BuildImage)?;
        context_builder.finish().change_context(Error::BuildImage)?;

        let mut runtime_image_context = Vec::new();
        File::open(ContextTarPath::from(name))
            .change_context(Error::BuildImage)?
            .read_to_end(&mut runtime_image_context)
            .change_context(Error::BuildImage)?;
        std::fs::remove_file(ContextTarPath::from(name)).change_context(Error::BuildImage)?;

        let mut stream = self.inner.build_image(
            BuildImageOptions {
                dockerfile: dockerfile_path,
                t: name,
                rm,
                forcerm,
                ..Default::default()
            },
            None,
            Some(runtime_image_context.into()),
        );

        while let Some(res) = stream.next().await {
            match res {
                Ok(_) => {}
                Err(e) => {
                    info!("Error: {:?}", e);
                    return Err(error_stack::Report::new(Error::BuildImage).attach_printable(e));
                }
            }
        }

        Ok(self.get_image(name).await?.ok_or(Error::BuildImage)?)
    }

    async fn get_container(&self, container_name: &str) -> Result<Option<super::Container>> {
        let mut filters = HashMap::new();
        let filter_values = vec![container_name.to_string()];
        filters.insert("name".to_string(), filter_values);

        let options = Some(ListContainersOptions {
            all: true,
            filters,
            ..Default::default()
        });

        let containers = self
            .inner
            .list_containers(options)
            .await
            .change_context(Error::GetContainer)?;

        match containers.first() {
            Some(container) => {
                let container = container.clone();
                Ok(Some(super::Container {
                    image: match container.image {
                        Some(name) => self.get_image(&name).await?,
                        None => None,
                    },
                    id: container.id,
                    name: container_name.to_string(),
                    status: container.status.ok_or(Error::GetContainer)?,
                    command: container.command.ok_or(Error::GetContainer)?,
                }))
            }
            None => Ok(None),
        }
    }

    async fn create_container(
        &self,
        container_name: &str,
        image: &super::Image,
        command: Vec<String>,
        env: Option<HashMap<String, String>>,
        port: Option<u16>,
        mounts: &Option<Vec<Mount>>,
    ) -> Result<ContainerCreationResult> {
        let image_name = image.name.clone();
        let env: Option<Vec<String>> = match env {
            Some(env) => Some(env.iter().map(|(k, v)| format!("{}={}", k, v)).collect()),
            None => None,
        };

        let ports = match port {
            Some(port) => {
                let mut exposed_ports = HashMap::new();
                exposed_ports.insert(format!("{}/tcp", port), HashMap::new());
                Some(exposed_ports)
            }
            None => None,
        };

        // Handle copy mounts by creating temporary bind mounts
        let mut docker_mounts = Vec::new();
        let mut copy_operations = Vec::new();
        
        if let Some(mounts) = mounts {
            for mount in mounts {
                match mount {
                    Mount::Bind { source, target } => {
                        docker_mounts.push(models::Mount {
                            source: Some(source.clone()),
                            target: Some(target.clone()),
                            typ: Some(models::MountTypeEnum::BIND),
                            ..Default::default()
                        });
                    },
                    Mount::Copy { source, target } => {
                        // Create temporary bind mount for copying
                        let temp_target = format!("/temp_copy_{}", target.replace("/", "_"));
                        docker_mounts.push(models::Mount {
                            source: Some(source.clone()),
                            target: Some(temp_target.clone()),
                            typ: Some(models::MountTypeEnum::BIND),
                            ..Default::default()
                        });
                        copy_operations.push((temp_target, target.clone()));
                    }
                }
            }
        }

        let host_config = Some(HostConfig {
            // Enable host networking
            network_mode: Some("host".to_string()),
            // Use processed mounts
            mounts: Some(docker_mounts),
            ..Default::default()
        }); 

        self.inner
            .create_container(
                Some(CreateContainerOptions {
                    name: container_name,
                    platform: None,
                }),
                Config {
                    image: Some(image_name),
                    cmd: Some(command),
                    env,
                    exposed_ports: ports,
                    host_config,
                    ..Default::default()
                },
            )
            .await
            .change_context(Error::CreateContainer)?;

        let container = self
            .get_container(container_name)
            .await?
            .ok_or(Error::CreateContainer)?;

        // Return container creation result with copy operations
        Ok(ContainerCreationResult {
            container,
            copy_operations,
        })
    }

    async fn execute_copy_operations(
        &self,
        container_name: &str,
        copy_operations: &[(String, String)],
    ) -> Result<()> {
        for (temp_source, final_target) in copy_operations {
            let copy_command = vec![
                "sh".to_string(),
                "-c".to_string(),
                format!("mkdir -p {} && cp -r {}/. {}", 
                    final_target, temp_source, final_target),
            ];
            
            self.run_process(container_name, copy_command).await?;
            
            // Clean up temp mount (optional - container will be removed anyway)
            let cleanup_command = vec!["umount".to_string(), temp_source.clone()];
            let _ = self.run_process(container_name, cleanup_command).await;
        }
        
        Ok(())
    }

    async fn remove_container(&self, container_name: &str) -> Result<()> {
        if self.get_container(container_name).await?.is_none() {
            return Ok(());
        }

        self.inner
            .remove_container(
                container_name,
                Some(RemoveContainerOptions {
                    force: true,
                    ..Default::default()
                }),
            )
            .await
            .change_context(Error::RemoveContainer)?;

        Ok(())
    }

    async fn start_container(&self, container: &super::Container) -> Result<()> {
        self.inner
            .start_container(&container.name, None::<StartContainerOptions<String>>)
            .await
            .change_context(Error::StartContainer)?;

        Ok(())
    }

    async fn run_process(&self, container_name: &str, command: Vec<String>) -> Result<String> {
        let res = self
            .inner
            .create_exec(
                container_name,
                CreateExecOptions {
                    cmd: Some(command),
                    attach_stdout: Some(true),
                    attach_stderr: Some(true),
                    ..Default::default()
                },
            )
            .await
            .change_context(Error::RunProcess)?;

        let result = self
            .inner
            .start_exec(
                &res.id,
                Some(StartExecOptions {
                    detach: false,
                    tty: false,
                    ..Default::default()
                }),
            )
            .await
            .change_context(Error::RunProcess)?;

        match result {
            StartExecResults::Attached { output, .. } => {
                let mut stdout_buffer = Vec::new();
                let mut stderr_buffer = Vec::new();

                tokio::pin!(output);

                while let Some(log_result) = output.next().await {
                    match log_result {
                        Ok(LogOutput::StdOut { message }) => {
                            stdout_buffer.extend_from_slice(&message);
                        }
                        Ok(LogOutput::StdErr { message }) => {
                            stderr_buffer.extend_from_slice(&message);
                        }
                        Ok(_) => {}
                        Err(_) => {
                            return Err(error_stack::Report::new(Error::RunProcess));
                        }
                    }
                }

                let stdout_str = String::from_utf8_lossy(&stdout_buffer);
                let _ = String::from_utf8_lossy(&stderr_buffer);

                Ok(stdout_str.to_string())
            }
            StartExecResults::Detached => Err(error_stack::Report::new(Error::RunProcess)),
        }
    }
}
