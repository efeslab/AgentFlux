use serde::{Deserialize, Serialize};

use crate::app_registry;
use crate::discovery;
use crate::inference_engine;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub runtime_path: String,
    pub container_comm_host: String,
    pub container_comm_port: u16,
    pub inference_engine: inference_engine::Config,
    pub discovery: discovery::Config,
    pub registry: Option<app_registry::Config>,
}
