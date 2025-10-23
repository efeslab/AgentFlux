use error_stack::{Report, ResultExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use thiserror::Error;

use crate::app_registry::Registry;
use crate::container::Mount;

#[derive(Error, Debug)]
pub enum Error {
    #[error("registry error")]
    Registry,
    #[error("apps config error")]
    AppsConfig,
}

pub type Result<T> = error_stack::Result<T, Error>;

#[derive(Serialize, Deserialize, Clone, Debug, std::cmp::Eq, std::cmp::PartialEq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    Python,
    NodeJs,
}

impl From<Runtime> for i32 {
    fn from(runtime: Runtime) -> Self {
        match runtime {
            Runtime::Python => 0,
            Runtime::NodeJs => 1,
        }
    }
}

impl Display for Runtime {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Runtime::Python => write!(f, "rena_python_runtime"),
            Runtime::NodeJs => write!(f, "rena_nodejs_runtime"),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "lowercase")]
pub enum AgentProtocol {
    MCP,
}

impl From<AgentProtocol> for i32 {
    fn from(agent_protocol: AgentProtocol) -> Self {
        match agent_protocol {
            AgentProtocol::MCP => 0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppConfig {
    pub name: String,
    pub path: Option<String>,
    pub agent_protocol: AgentProtocol,
    pub runtime: Runtime,
    pub command: String,
    pub args: Vec<String>,
    pub env: Option<HashMap<String, String>>,
    pub scripts: Option<Vec<String>>,
    pub mounts: Option<Vec<Mount>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppFile {
    pub relative_path: String,
    pub content: Vec<u8>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppBundle {
    pub config: AppConfig,
    pub files: Option<Vec<AppFile>>,
}

impl From<AppConfig> for AppBundle {
    fn from(app_config: AppConfig) -> Self {
        Self {
            config: app_config.clone(),
            files: None,
        }
    }
}

impl From<(AppConfig, Vec<AppFile>)> for AppBundle {
    fn from((app_config, app_files): (AppConfig, Vec<AppFile>)) -> Self {
        Self {
            config: app_config.clone(),
            files: Some(app_files),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum Config {
    #[serde(rename = "static")]
    Static { apps: Vec<AppConfig> },
}

pub trait Discovery {
    fn discover(
        &mut self,
        query: &str,
        top_k: u16,
    ) -> impl std::future::Future<Output = Result<Vec<AppBundle>>> + Send;
}

#[derive(Clone)]
pub struct StaticDiscovery {
    pub apps: Vec<AppConfig>,
    pub registry: Registry,
}

impl StaticDiscovery {
    pub async fn new(apps: Vec<AppConfig>, registry: Registry) -> Self {
        Self { apps, registry }
    }
}

impl Discovery for StaticDiscovery {
    async fn discover(&mut self, _: &str, _: u16) -> Result<Vec<AppBundle>> {
        let futs = self.apps.iter().map(|app_config| {
            let app_config = app_config.clone();
            async {
                match &app_config.path {
                    Some(path) => self
                        .registry
                        .pull(&path)
                        .await
                        .change_context(Error::Registry)?
                        .ok_or(Report::new(Error::Registry).attach_printable("app not found"))
                        .map(|app_files| (app_config, app_files).into()),
                    None => Ok(app_config.into()),
                }
            }
        });

        let app_bundles = futures::future::join_all(futs)
            .await
            .into_iter()
            .collect::<Result<Vec<AppBundle>>>()
            .change_context(Error::Registry)?;

        Ok(app_bundles)
    }
}
