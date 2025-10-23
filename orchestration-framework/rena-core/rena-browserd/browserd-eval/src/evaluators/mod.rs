use async_trait::async_trait;
use browserd::inference_engine::Config as InferenceEngineConfig;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod llm;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum Config {
    #[serde(rename = "basic_match")]
    BasicMatch { id: String },
    #[serde(rename = "llm")]
    LLM {
        id: String,
        base_prompt: String,
        inference_engine: InferenceEngineConfig,
    },
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("llm evaluator error")]
    LLM,
}

pub type Result<T> = error_stack::Result<T, Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub passed: bool,
    pub reason: Option<String>,
    pub eval_score: f64,
}

#[async_trait]
pub trait Evaluator {
    async fn evaluate(&self, query: &str, response: &str, ideal: &str) -> Result<EvaluationResult>;
}

pub struct BasicMatchEvaluator;

#[async_trait]
impl Evaluator for BasicMatchEvaluator {
    async fn evaluate(&self, _: &str, response: &str, ideal: &str) -> Result<EvaluationResult> {
        let passed = response.contains(ideal);
        Ok(EvaluationResult {
            passed,
            reason: None,
            eval_score: if passed { 100.0 } else { 0.0 },
        })
    }
}
