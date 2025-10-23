use async_trait::async_trait;
use browserd::inference_engine::{
    HTTPClient, InferenceEngine, MessageContentBlock, MessageParam, MessageParamContent, Role,
};
use error_stack::ResultExt;
use serde::{Deserialize, Serialize};
use tracing::info;

use super::{Error, EvaluationResult, Evaluator, Result};

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMResponse {
    passed: bool,
    reason: String,
    eval_score: f64,
}

pub struct LLMEvaluator<H>
where
    H: HTTPClient,
{
    inner: InferenceEngine<H>,
    base_prompt: String,
}

impl<H> LLMEvaluator<H>
where
    H: HTTPClient,
{
    pub fn new(inference_engine: H, base_prompt: &str) -> Self {
        Self {
            inner: InferenceEngine::new(inference_engine),
            base_prompt: base_prompt.to_string(),
        }
    }
}

#[async_trait]
impl<H> Evaluator for LLMEvaluator<H>
where
    H: HTTPClient + Send + Sync,
{
    async fn evaluate(&self, query: &str, response: &str, ideal: &str) -> Result<EvaluationResult> {
        let message_params = vec![MessageParam {
            role: Role::User,
            content: MessageParamContent::Text(format!(
                r#"{}

Evaluate the actual response against the ideal response. Provide a quality score from 0-100 where:
- 90-100: Excellent (correct, complete, well-formatted)
- 70-89: Good (correct but minor issues)
- 50-69: Acceptable (partially correct)
- 30-49: Poor (significant issues)
- 0-29: Failed (incorrect or harmful)

Respond with ONLY a JSON object in this format: {{"passed": <boolean>, "reason": <string>, "eval_score": <number>}}
Do NOT wrap the JSON in markdown code blocks or any other formatting. Return only the raw JSON object.

Query: {}
Ideal Response: {}
Actual Response: {}"#,
                self.base_prompt, query, ideal, response
            )),
        }];

        let (message, _) = self
            .inner
            .generate(message_params, vec![])
            .await
            .change_context(Error::LLM)?;

        let text = message
            .content
            .first()
            .ok_or(Error::LLM)
            .attach_printable("llm response is empty")?;

        info!("evaluator response: {:?}", text);

        match text {
            MessageContentBlock::Text(text) => {
                let json_str = if let Some(captures) =
                    regex::Regex::new(r"(?s)```(?:json)?\s*(\{[^`]+\})\s*```")
                        .expect("failed to create markdown regex")
                        .captures(&text.text)
                {
                    captures.get(1).map(|m| m.as_str()).unwrap_or(&text.text)
                } else {
                    regex::Regex::new(r"(?s)(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})")
                        .expect("failed to create regex")
                        .find(&text.text)
                        .ok_or(Error::LLM)
                        .attach_printable(format!(
                            "llm response is not a json object. response: {}",
                            text.text
                        ))?
                        .as_str()
                };

                let llm_response = serde_json::from_str::<LLMResponse>(json_str)
                    .change_context(Error::LLM)
                    .attach_printable(format!(
                        "failed to deserialize llm response. response: {}",
                        json_str
                    ))?;

                Ok(EvaluationResult {
                    passed: llm_response.passed,
                    reason: Some(llm_response.reason),
                    eval_score: llm_response.eval_score,
                })
            }
            _ => Err(Error::LLM).attach_printable(format!(
                "llm response is not a text block. response: {:?}",
                text
            )),
        }
    }
}
