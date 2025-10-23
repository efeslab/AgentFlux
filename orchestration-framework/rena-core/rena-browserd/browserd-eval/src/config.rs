use serde::{Deserialize, Serialize};

use crate::evaluators;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EvalConfig {
    pub label: String,
    pub query: String,
    pub ideal: String,
    pub evaluator: String,
    pub tool_filter: Option<Vec<String>>,
    pub runs: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub name: String,
    pub description: Option<String>,
    #[serde(default = "default_runs")]
    pub runs: usize,
    pub evaluators: Vec<evaluators::Config>,
    pub evals: Vec<EvalConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_explainer: Option<browserd::inference_engine::Config>,
}

fn default_runs() -> usize {
    1
}
