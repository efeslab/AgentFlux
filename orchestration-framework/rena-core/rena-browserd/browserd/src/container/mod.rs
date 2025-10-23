use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

pub mod docker;

#[derive(Debug, Clone)]
pub struct Image {
    pub id: String,
    pub name: String,
}

pub struct Container {
    pub id: Option<String>,
    pub name: String,
    pub image: Option<Image>,
    pub status: String,
    pub command: String,
}

pub struct ContainerCreationResult {
    pub container: Container,
    pub copy_operations: Vec<(String, String)>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum Mount {
    #[serde(rename = "bind")]
    Bind { source: String, target: String },
    #[serde(rename = "copy")]
    Copy { source: String, target: String },
}

pub struct ContextTarPath(String);

impl From<&str> for ContextTarPath {
    fn from(value: &str) -> Self {
        Self(format!("./{}.tar", value))
    }
}

impl AsRef<Path> for ContextTarPath {
    fn as_ref(&self) -> &Path {
        Path::new(&self.0)
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("get image failed")]
    GetImage,
    #[error("build image failed")]
    BuildImage,
    #[error("remove image failed")]
    RemoveImage,
    #[error("get container failed")]
    GetContainer,
    #[error("create container failed")]
    CreateContainer,
    #[error("start container failed")]
    StartContainer,
    #[error("remove container failed")]
    RemoveContainer,
    #[error("run process in container failed")]
    RunProcess,
}

type Result<T> = error_stack::Result<T, Error>;

#[async_trait]
pub trait ContainerClient {
    async fn get_image(&self, image_name: &str) -> Result<Option<Image>>;
    async fn build_image(
        &self,
        name: &str,
        host_context_path: &str,
        container_context_path: &str,
        dockerfile_path: &str,
        rm: bool,
        forcerm: bool,
    ) -> Result<Image>;
    async fn get_container(&self, container_name: &str) -> Result<Option<Container>>;
    async fn create_container(
        &self,
        container_name: &str,
        image: &Image,
        command: Vec<String>,
        env: Option<HashMap<String, String>>,
        port: Option<u16>,
        mounts: &Option<Vec<Mount>>,
    ) -> Result<ContainerCreationResult>;
    async fn remove_container(&self, container_name: &str) -> Result<()>;
    async fn start_container(&self, container: &Container) -> Result<()>;
    async fn run_process(&self, container_name: &str, command: Vec<String>) -> Result<String>;
    async fn execute_copy_operations(
        &self,
        container_name: &str,
        copy_operations: &[(String, String)],
    ) -> Result<()>;
}
