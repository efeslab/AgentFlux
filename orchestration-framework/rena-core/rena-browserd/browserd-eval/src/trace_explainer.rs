use async_trait::async_trait;
use browserd::inference_engine::{
    HTTPClient, InferenceEngine, MessageContentBlock, MessageParam, MessageParamContent, Role,
};
use browserd::trace::Trace;
use error_stack::ResultExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

use crate::{EvalResult, RunOutcome, SingleRunResult};

#[derive(Error, Debug)]
pub enum Error {
    #[error("trace explainer error")]
    TraceExplainer,
    #[error("inference engine error")]
    InferenceEngine,
}

pub type Result<T> = error_stack::Result<T, Error>;

#[derive(Debug, Serialize, Deserialize)]
struct ExplanationResponse {
    explanation: String,
}

#[async_trait]
pub trait TraceExplainer {
    async fn explain_single_run(
        &self,
        query: &str,
        ideal: &str,
        run_result: &SingleRunResult,
    ) -> Result<String>;

    async fn explain_aggregated(&self, eval_result: &EvalResult) -> Result<String>;
}

pub struct LLMTraceExplainer<H>
where
    H: HTTPClient,
{
    inference_engine: InferenceEngine<H>,
}

impl<H> LLMTraceExplainer<H>
where
    H: HTTPClient,
{
    pub fn new(http_client: H) -> Self {
        Self {
            inference_engine: InferenceEngine::new(http_client),
        }
    }

    fn format_trace_for_llm(trace: &Trace, depth: usize) -> String {
        let indent = "  ".repeat(depth);
        let mut result = format!(
            "{}• {} ({}ms)",
            indent,
            trace.label,
            trace.latency.unwrap_or(0)
        );

        if let Some(metadata) = &trace.metadata {
            result.push_str(&format!(" - {}", metadata.replace("\n", " ")));
        }

        result.push('\n');

        if let Some(inner_traces) = &trace.inner_traces {
            for inner_trace in inner_traces {
                result.push_str(&Self::format_trace_for_llm(inner_trace, depth + 1));
            }
        }

        result
    }
}

#[async_trait]
impl<H> TraceExplainer for LLMTraceExplainer<H>
where
    H: HTTPClient + Send + Sync,
{
    async fn explain_single_run(
        &self,
        query: &str,
        ideal: &str,
        run_result: &SingleRunResult,
    ) -> Result<String> {
        let trace_text = Self::format_trace_for_llm(&run_result.trace, 0);

        let prompt = format!(
            r#"You are an expert at analyzing AI agent execution traces. Your task is to provide a concise, human-readable explanation of what happened during this specific run.

Query: {}
Ideal Response: {}
Actual Response: {}
Passed: {}
Evaluation Score: {}/100
Total Latency: {}ms
Token Usage: {} input, {} output

Execution Trace:
{}

Provide a brief narrative (2-3 sentences) explaining:
1. What the system did to answer the query
2. Key steps and their performance
3. Why it passed/failed if relevant

Respond with ONLY the explanation text, no JSON or other formatting."#,
            query,
            ideal,
            run_result.response,
            run_result.passed,
            run_result.eval_score,
            run_result.latency_ms.unwrap_or(0.0),
            run_result.input_tokens.unwrap_or(0),
            run_result.output_tokens.unwrap_or(0),
            trace_text
        );

        let message_params = vec![MessageParam {
            role: Role::User,
            content: MessageParamContent::Text(prompt),
        }];

        let (message, _) = self
            .inference_engine
            .generate(message_params, vec![])
            .await
            .change_context(Error::InferenceEngine)?;

        let explanation = message
            .content
            .first()
            .ok_or(Error::TraceExplainer)
            .attach_printable("LLM response is empty")?;

        match explanation {
            MessageContentBlock::Text(text) => {
                info!("Generated single run explanation: {}", text.text);
                Ok(text.text.clone())
            }
            _ => Err(Error::TraceExplainer).attach_printable("LLM response is not a text block"),
        }
    }

    async fn explain_aggregated(&self, eval_result: &EvalResult) -> Result<String> {
        let metrics = &eval_result.aggregated_metrics;

        let mut all_steps = Vec::new();
        let mut latency_patterns = Vec::new();

        for run in &eval_result.runs {
            match run {
                RunOutcome::Success(single_run) => {
                    let trace_text = Self::format_trace_for_llm(&single_run.trace, 0);
                    all_steps.push(trace_text);
                    if let Some(latency) = single_run.latency_ms {
                        latency_patterns.push(latency);
                    }
                }
                RunOutcome::Failed { .. } => {
                    info!("Skipping failed run for aggregated analysis");
                }
            }
        }

        let prompt = format!(
            r#"You are an expert at analyzing AI agent execution patterns. Analyze these evaluation results and provide insights about patterns across multiple runs.

Query: {}
Ideal Response: {}
Total Runs: {}
Success Rate: {:.1}%
Average Evaluation Score: {:.1}/100

Performance Statistics:
- Latency: mean={:.1}ms, min={:.1}ms, max={:.1}ms, std_dev={:.1}ms
- Tokens: mean_input={:.1}, mean_output={:.1}, total_used={}

Sample Execution Traces (first 3 runs):
{}

Provide a brief analysis (3-4 sentences) covering:
1. Overall pattern of execution across runs
2. Performance characteristics and consistency
3. Key insights about success/failure patterns
4. Token usage efficiency

Respond with ONLY the analysis text, no JSON or other formatting."#,
            eval_result.query,
            eval_result.ideal,
            metrics.total_runs,
            metrics.success_rate,
            metrics.eval_score_stats.mean,
            metrics
                .latency_stats
                .as_ref()
                .map(|s| s.mean_ms)
                .unwrap_or(0.0),
            metrics
                .latency_stats
                .as_ref()
                .map(|s| s.min_ms)
                .unwrap_or(0.0),
            metrics
                .latency_stats
                .as_ref()
                .map(|s| s.max_ms)
                .unwrap_or(0.0),
            metrics
                .latency_stats
                .as_ref()
                .map(|s| s.std_dev_ms)
                .unwrap_or(0.0),
            metrics
                .token_stats
                .as_ref()
                .map(|s| s.mean_input_tokens)
                .unwrap_or(0.0),
            metrics
                .token_stats
                .as_ref()
                .map(|s| s.mean_output_tokens)
                .unwrap_or(0.0),
            metrics
                .token_stats
                .as_ref()
                .map(|s| s.total_tokens)
                .unwrap_or(0),
            all_steps
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n---\n")
        );

        let message_params = vec![MessageParam {
            role: Role::User,
            content: MessageParamContent::Text(prompt),
        }];

        let (message, _) = self
            .inference_engine
            .generate(message_params, vec![])
            .await
            .change_context(Error::InferenceEngine)?;

        let explanation = message
            .content
            .first()
            .ok_or(Error::TraceExplainer)
            .attach_printable("LLM response is empty")?;

        match explanation {
            MessageContentBlock::Text(text) => {
                info!("Generated aggregated explanation: {}", text.text);
                Ok(text.text.clone())
            }
            _ => Err(Error::TraceExplainer).attach_printable("LLM response is not a text block"),
        }
    }
}

pub async fn generate_trace_explanations(
    evaluator_inference_engine: Option<&browserd::inference_engine::Config>,
    eval_results: &mut Vec<EvalResult>,
) -> Result<()> {
    let Some(engine_config) = evaluator_inference_engine else {
        info!("No inference engine configured for trace explanations, skipping");
        return Ok(());
    };

    info!(
        "Generating trace explanations for {} eval results",
        eval_results.len()
    );

    let explainer: Box<dyn TraceExplainer + Send + Sync> = match engine_config {
        browserd::inference_engine::Config::Ollama { model, max_tokens } => {
            Box::new(LLMTraceExplainer::new(
                browserd::inference_engine::ollama::Ollama::new(model, *max_tokens),
            ))
        }
        browserd::inference_engine::Config::Anthropic {
            model,
            max_tokens,
            api_key,
        } => Box::new(LLMTraceExplainer::new(
            browserd::inference_engine::anthropic::Anthropic::new(model, max_tokens, api_key)
                .change_context(Error::InferenceEngine)?,
        )),
        browserd::inference_engine::Config::OpenAI {
            model,
            max_tokens,
            api_key,
            url,
        } => Box::new(LLMTraceExplainer::new(
            browserd::inference_engine::openai::OpenAI::new(model, *max_tokens, api_key, url)
                .change_context(Error::InferenceEngine)?,
        )),
    };

    for eval_result in eval_results.iter_mut() {
        info!("Generating explanations for eval: {}", eval_result.query);

        if eval_result.eval_error.is_some() {
            info!(
                "Skipping trace explanation for failed eval: {}",
                eval_result.query
            );
            continue;
        }

        for (idx, run) in eval_result.runs.iter_mut().enumerate() {
            match run {
                RunOutcome::Success(single_run) => {
                    match explainer
                        .explain_single_run(&eval_result.query, &eval_result.ideal, single_run)
                        .await
                    {
                        Ok(explanation) => {
                            single_run.trace_explanation = Some(explanation);
                        }
                        Err(e) => {
                            info!("Failed to generate explanation for run {}: {:?}", idx, e);
                        }
                    }
                }
                RunOutcome::Failed { .. } => {
                    info!("Skipping trace explanation for failed run {}", idx);
                }
            }
        }

        match explainer.explain_aggregated(eval_result).await {
            Ok(explanation) => {
                eval_result.trace_explanation = Some(explanation);
            }
            Err(e) => {
                info!("Failed to generate aggregated explanation: {:?}", e);
            }
        }
    }

    info!("Completed generating trace explanations");
    Ok(())
}
