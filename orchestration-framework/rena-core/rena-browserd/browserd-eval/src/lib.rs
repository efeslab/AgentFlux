use browserd::inference_engine;
use browserd::trace::Trace;
use browserd::{
    agent_protocol::AgentClient, discovery::Discovery, orchestrator::Orchestrator,
    AppClient as BrowserdClient,
};
use error_stack::{Report, ResultExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::config::Config;
use crate::evaluators::Evaluator;

pub mod config;
pub mod evaluators;
pub mod reporter;
pub mod trace_explainer;

#[derive(Error, Debug)]
pub enum Error {
    #[error("config error")]
    Config,
    #[error("evaluator error")]
    Evaluator,
    #[error("browserd client error")]
    BrowserdClient,
}

pub type Result<T> = error_stack::Result<T, Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunOutcome {
    Success(SingleRunResult),
    Failed {
        error_message: String,
        error_type: String,
    },
}

fn extract_tokens_from_trace(trace: &Trace) -> (Option<u64>, Option<u64>) {
    fn extract_from_metadata(metadata: &str) -> (Option<u64>, Option<u64>) {
        let input_regex = regex::Regex::new(r"input_tokens:\s*(\d+)").ok();
        let output_regex = regex::Regex::new(r"output_tokens:\s*(\d+)").ok();

        let input_tokens = input_regex
            .and_then(|re| re.captures(metadata))
            .and_then(|cap| cap.get(1))
            .and_then(|m| m.as_str().parse::<u64>().ok());

        let output_tokens = output_regex
            .and_then(|re| re.captures(metadata))
            .and_then(|cap| cap.get(1))
            .and_then(|m| m.as_str().parse::<u64>().ok());

        (input_tokens, output_tokens)
    }

    let mut total_input_tokens: Option<u64> = None;
    let mut total_output_tokens: Option<u64> = None;

    if let Some(metadata) = &trace.metadata {
        let (input, output) = extract_from_metadata(metadata);
        if let Some(input_val) = input {
            total_input_tokens = Some(total_input_tokens.unwrap_or(0) + input_val);
        }
        if let Some(output_val) = output {
            total_output_tokens = Some(total_output_tokens.unwrap_or(0) + output_val);
        }
    }

    if let Some(inner_traces) = &trace.inner_traces {
        for inner_trace in inner_traces {
            let (input, output) = extract_tokens_from_trace(inner_trace);
            if let Some(input_val) = input {
                total_input_tokens = Some(total_input_tokens.unwrap_or(0) + input_val);
            }
            if let Some(output_val) = output {
                total_output_tokens = Some(total_output_tokens.unwrap_or(0) + output_val);
            }
        }
    }

    (total_input_tokens, total_output_tokens)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleRunResult {
    pub response: String,
    pub trace: Trace,
    pub passed: bool,
    pub reason: Option<String>,
    pub eval_score: f64,
    pub latency_ms: Option<f64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub trace_explanation: Option<String>,
    pub num_inference_calls: Option<u64>,
    pub concurrency_score: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub total_runs: usize,
    pub passed_runs: usize,
    pub failed_runs: usize,
    pub success_rate: f64,
    pub latency_stats: Option<LatencyStats>,
    pub eval_score_stats: EvalScoreStats,
    pub token_stats: Option<TokenStats>,
    pub inference_call_stats: Option<InferenceCallStats>,
    pub concurrency_score_stats: Option<ConcurrencyScoreStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub std_dev_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStats {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_tokens: u64,
    pub mean_input_tokens: f64,
    pub mean_output_tokens: f64,
    pub mean_total_tokens: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalScoreStats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCallStats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyScoreStats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub label: String,
    pub query: String,
    pub ideal: String,
    pub evaluator: String,
    pub runs: Vec<RunOutcome>,
    pub aggregated_metrics: AggregatedMetrics,
    pub trace_explanation: Option<String>,
    pub eval_error: Option<String>,
}

pub struct App<T, D, O>
where
    T: AgentClient + Clone,
    D: Discovery + Clone,
    O: Orchestrator + Clone,
{
    pub inner: BrowserdClient<T, D, O>,
}

impl<T, D, O> App<T, D, O>
where
    T: AgentClient + Clone,
    D: Discovery + Clone,
    O: Orchestrator + Clone,
{
    pub fn new(inner: BrowserdClient<T, D, O>) -> Self {
        Self { inner }
    }

    pub async fn run(
        &self,
        config: &Config,
        _browserd_config: &browserd::config::Config,
        cancel: CancellationToken,
    ) -> Result<Vec<EvalResult>> {
        let evaluators = configure_evaluators(&config.evaluators).change_context(Error::Config)?;

        let futures = config.evals.iter().map(|eval| {
            let evaluators = &evaluators;
            let inner = self.inner.clone();
            let eval_config = eval.clone();
            async move {
                let result = async move {
                    let evaluator = evaluators.get(&eval.evaluator).ok_or_else(|| {
                        Report::new(Error::Config)
                            .attach_printable(format!("Evaluator not found: {}", eval.evaluator))
                    })?;

                    let num_runs = eval.runs.unwrap_or(config.runs);

                    let run_futures = (0..num_runs).map(|i| {
                        info!("eval run started. query: {}, run: {}", eval.query, i + 1);

                        let mut inner = inner.clone();
                        let query = eval.query.clone();
                        let ideal = eval.ideal.clone();
                        async move {
                            let (response, trace) = inner
                                .query(&query, &eval.tool_filter)
                                .await
                                .change_context(Error::BrowserdClient)?;

                            let eval_result = evaluator
                                .evaluate(&query, &response, &ideal)
                                .await
                                .change_context(Error::Evaluator)?;

                            let latency_ms = trace.latency.map(|l| l as f64);
                            let (input_tokens, output_tokens) = extract_tokens_from_trace(&trace);
                            let total_tokens = match (input_tokens, output_tokens) {
                                (Some(input), Some(output)) => Some(input + output),
                                (Some(input), None) => Some(input),
                                (None, Some(output)) => Some(output),
                                (None, None) => None,
                            };

                            let num_inference_calls = inference_calls(&trace);

                            let mut nodes = vec![];
                            execution_plan_nodes(&trace, &mut nodes);
                            let concurrency_score = match nodes.len() {
                                0 => None,
                                _ => Some(nodes.iter().sum::<u64>() as f64 / nodes.len() as f64),
                            };

                            Ok::<_, error_stack::Report<Error>>(RunOutcome::Success(
                                SingleRunResult {
                                    response: response.to_string(),
                                    trace,
                                    passed: eval_result.passed,
                                    reason: eval_result.reason,
                                    eval_score: eval_result.eval_score,
                                    latency_ms,
                                    input_tokens,
                                    output_tokens,
                                    total_tokens,
                                    trace_explanation: None,
                                    num_inference_calls: Some(num_inference_calls),
                                    concurrency_score,
                                    error: None,
                                },
                            ))
                        }
                    });

                    let mut runs = vec![];
                    for run_future in run_futures {
                        match run_future.await {
                            Ok(outcome) => runs.push(outcome),
                            Err(e) => {
                                tracing::error!("Run failed for query '{}': {:?}", eval.label, e);
                                runs.push(RunOutcome::Failed {
                                    error_message: format!("{:?}", e),
                                    error_type: "runtime_error".to_string(),
                                });
                            }
                        }
                    }

                    let latencies: Vec<f64> = runs
                        .iter()
                        .filter_map(|r| match r {
                            RunOutcome::Success(r) => r.latency_ms,
                            RunOutcome::Failed { .. } => None,
                        })
                        .collect();
                    let eval_scores: Vec<f64> = runs
                        .iter()
                        .filter_map(|r| match r {
                            RunOutcome::Success(r) => Some(r.eval_score),
                            RunOutcome::Failed { .. } => None,
                        })
                        .collect();
                    let inference_calls: Vec<u64> = runs
                        .iter()
                        .filter_map(|r| match r {
                            RunOutcome::Success(r) => r.num_inference_calls,
                            RunOutcome::Failed { .. } => None,
                        })
                        .collect();
                    let concurrency_scores: Vec<f64> = runs
                        .iter()
                        .filter_map(|r| match r {
                            RunOutcome::Success(r) => r.concurrency_score,
                            RunOutcome::Failed { .. } => None,
                        })
                        .collect();

                    let passed_runs = runs
                        .iter()
                        .filter(|r| match r {
                            RunOutcome::Success(single_run) => single_run.passed,
                            RunOutcome::Failed { .. } => false,
                        })
                        .count();
                    let failed_runs = num_runs - passed_runs;
                    let success_rate = if num_runs > 0 {
                        (passed_runs as f64 / num_runs as f64) * 100.0
                    } else {
                        0.0
                    };

                    let input_tokens: Vec<u64> = runs
                        .iter()
                        .filter_map(|r| match r {
                            RunOutcome::Success(r) => r.input_tokens,
                            RunOutcome::Failed { .. } => None,
                        })
                        .collect();
                    let output_tokens: Vec<u64> = runs
                        .iter()
                        .filter_map(|r| match r {
                            RunOutcome::Success(r) => r.output_tokens,
                            RunOutcome::Failed { .. } => None,
                        })
                        .collect();

                    let token_stats = if !input_tokens.is_empty() || !output_tokens.is_empty() {
                        let total_input: u64 = input_tokens.iter().sum();
                        let total_output: u64 = output_tokens.iter().sum();
                        let total_all: u64 = total_input + total_output;

                        let mean_input_tokens = if !input_tokens.is_empty() {
                            total_input as f64 / input_tokens.len() as f64
                        } else {
                            0.0
                        };
                        let mean_output_tokens = if !output_tokens.is_empty() {
                            total_output as f64 / output_tokens.len() as f64
                        } else {
                            0.0
                        };
                        let mean_total_tokens = {
                            let total_tokens: Vec<u64> = runs
                                .iter()
                                .filter_map(|r| match r {
                                    RunOutcome::Success(r) => r.total_tokens,
                                    RunOutcome::Failed { .. } => None,
                                })
                                .collect();
                            if !total_tokens.is_empty() {
                                total_tokens.iter().sum::<u64>() as f64 / total_tokens.len() as f64
                            } else {
                                0.0
                            }
                        };

                        Some(TokenStats {
                            total_input_tokens: total_input,
                            total_output_tokens: total_output,
                            total_tokens: total_all,
                            mean_input_tokens,
                            mean_output_tokens,
                            mean_total_tokens,
                        })
                    } else {
                        None
                    };

                    let latency_stats = if !latencies.is_empty() {
                        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
                        let min = latencies.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                        let variance = latencies.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / latencies.len() as f64;
                        let std_dev = variance.sqrt();

                        Some(LatencyStats {
                            mean_ms: mean,
                            min_ms: min,
                            max_ms: max,
                            std_dev_ms: std_dev,
                        })
                    } else {
                        None
                    };

                    let eval_score_stats = if !eval_scores.is_empty() {
                        let mean = eval_scores.iter().sum::<f64>() / eval_scores.len() as f64;
                        let min = eval_scores.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max = eval_scores
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);

                        let variance = eval_scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / eval_scores.len() as f64;
                        let std_dev = variance.sqrt();

                        EvalScoreStats {
                            mean,
                            min,
                            max,
                            std_dev,
                        }
                    } else {
                        EvalScoreStats {
                            mean: 0.0,
                            min: 0.0,
                            max: 0.0,
                            std_dev: 0.0,
                        }
                    };

                    let inference_call_stats = if !inference_calls.is_empty() {
                        let mean = inference_calls.iter().sum::<u64>() as f64
                            / inference_calls.len() as f64;
                        let min = inference_calls.iter().cloned().fold(u64::MAX, u64::min);
                        let max = inference_calls.iter().cloned().fold(u64::MIN, u64::max);

                        let variance = inference_calls
                            .iter()
                            .map(|&x| (x as f64 - mean).powi(2))
                            .sum::<f64>()
                            / inference_calls.len() as f64;
                        let std_dev = variance.sqrt() as f64;

                        Some(InferenceCallStats {
                            mean,
                            min: min as f64,
                            max: max as f64,
                            std_dev,
                        })
                    } else {
                        None
                    };

                    let concurrency_score_stats = if !concurrency_scores.is_empty() {
                        let mean = concurrency_scores.iter().sum::<f64>()
                            / concurrency_scores.len() as f64;
                        let min = concurrency_scores
                            .iter()
                            .cloned()
                            .fold(f64::INFINITY, f64::min);
                        let max = concurrency_scores
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);

                        let variance = concurrency_scores
                            .iter()
                            .map(|&x| (x - mean).powi(2))
                            .sum::<f64>()
                            / concurrency_scores.len() as f64;
                        let std_dev = variance.sqrt();

                        Some(ConcurrencyScoreStats {
                            mean,
                            min,
                            max,
                            std_dev,
                        })
                    } else {
                        None
                    };

                    Ok::<_, error_stack::Report<Error>>(EvalResult {
                        label: eval.label.clone(),
                        query: eval.query.clone(),
                        ideal: eval.ideal.clone(),
                        evaluator: eval.evaluator.clone(),
                        runs,
                        aggregated_metrics: AggregatedMetrics {
                            total_runs: num_runs,
                            passed_runs,
                            failed_runs,
                            success_rate,
                            latency_stats,
                            eval_score_stats,
                            token_stats,
                            inference_call_stats,
                            concurrency_score_stats,
                        },
                        trace_explanation: None,
                        eval_error: None,
                    })
                }
                .await;

                (eval_config, result)
            }
        });

        let mut results = vec![];
        for future in futures {
            select! {
                res = future => {
                    match res {
                        (_eval_config, Ok(eval_result)) => {
                            results.push(eval_result);
                        },
                        (eval_config, Err(e)) => {
                            tracing::error!("Evaluation failed for '{}': {:?}", eval_config.query, e);
                            results.push(EvalResult {
                                label: eval_config.label,
                                query: eval_config.query,
                                ideal: eval_config.ideal,
                                evaluator: eval_config.evaluator,
                                runs: vec![],
                                aggregated_metrics: AggregatedMetrics {
                                    total_runs: 0,
                                    passed_runs: 0,
                                    failed_runs: 0,
                                    success_rate: 0.0,
                                    latency_stats: None,
                                    eval_score_stats: EvalScoreStats {
                                        mean: 0.0,
                                        min: 0.0,
                                        max: 0.0,
                                        std_dev: 0.0,
                                    },
                                    token_stats: None,
                                    inference_call_stats: None,
                                    concurrency_score_stats: None,
                                },
                                trace_explanation: None,
                                eval_error: Some(format!("{:?}", e)),
                            });
                        }
                    }
                }
                _ = cancel.cancelled() => {
                    return Ok(results);
                }
            }
        }

        Ok(results)
    }
}

fn configure_evaluators(
    configs: &Vec<evaluators::Config>,
) -> Result<HashMap<String, Box<dyn Evaluator + Send + Sync>>> {
    let mut evaluators: HashMap<String, Box<dyn Evaluator + Send + Sync>> = HashMap::new();

    for config in configs {
        match config {
            evaluators::Config::BasicMatch { id } => {
                evaluators.insert(id.to_string(), Box::new(evaluators::BasicMatchEvaluator));
            }
            evaluators::Config::LLM {
                id,
                base_prompt,
                inference_engine,
            } => match inference_engine {
                inference_engine::Config::Ollama { model, max_tokens } => {
                    evaluators.insert(
                        id.to_string(),
                        Box::new(evaluators::llm::LLMEvaluator::new(
                            inference_engine::ollama::Ollama::new(&model, *max_tokens),
                            &base_prompt,
                        )),
                    );
                }
                inference_engine::Config::Anthropic {
                    model,
                    max_tokens,
                    api_key,
                } => {
                    evaluators.insert(
                        id.to_string(),
                        Box::new(evaluators::llm::LLMEvaluator::new(
                            inference_engine::anthropic::Anthropic::new(
                                &model, max_tokens, &api_key,
                            )
                            .change_context(Error::Evaluator)?,
                            &base_prompt,
                        )),
                    );
                }
                inference_engine::Config::OpenAI {
                    model,
                    max_tokens,
                    api_key,
                    url,
                } => {
                    evaluators.insert(
                        id.to_string(),
                        Box::new(evaluators::llm::LLMEvaluator::new(
                            inference_engine::openai::OpenAI::new(
                                &model,
                                *max_tokens,
                                api_key,
                                url,
                            )
                            .change_context(Error::Evaluator)?,
                            &base_prompt,
                        )),
                    );
                }
            },
        }
    }

    Ok(evaluators)
}

fn inference_calls(trace: &Trace) -> u64 {
    let mut num_inference_calls = 0;

    if trace.label == "inference_call" {
        num_inference_calls += 1;
    }

    if let Some(inner_traces) = &trace.inner_traces {
        for inner_trace in inner_traces {
            num_inference_calls += inference_calls(inner_trace);
        }
    }

    num_inference_calls
}

fn execution_plan_nodes(trace: &Trace, nodes: &mut Vec<u64>) {
    if trace.label == "create_execution_plan" {
        if let Some(metadata) = &trace.metadata {
            if let Some(num_nodes) = _extract::<u64>(metadata, r"nodes:\s*(\d+)") {
                nodes.push(num_nodes);
            }
        }
    }

    if let Some(inner_traces) = &trace.inner_traces {
        for inner_trace in inner_traces {
            execution_plan_nodes(inner_trace, nodes);
        }
    }
}

fn _extract<T>(metadata: &str, pattern: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    regex::Regex::new(pattern)
        .ok()?
        .captures(metadata)?
        .get(1)?
        .as_str()
        .parse::<T>()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use browserd::trace::Trace;

    #[test]
    fn test_token_extraction_aggregates_all_calls() {
        let trace = Trace {
            label: "root".to_string(),
            metadata: Some("input_tokens: 100\noutput_tokens: 50".to_string()),
            inner_traces: Some(vec![
                Trace {
                    label: "tool_call".to_string(),
                    metadata: Some("input_tokens: 200\noutput_tokens: 150".to_string()),
                    inner_traces: None,
                    latency: None,
                },
                Trace {
                    label: "final_response".to_string(),
                    metadata: Some("input_tokens: 300\noutput_tokens: 250".to_string()),
                    inner_traces: None,
                    latency: None,
                },
            ]),
            latency: None,
        };

        let (input_tokens, output_tokens) = extract_tokens_from_trace(&trace);

        assert_eq!(input_tokens, Some(600));
        assert_eq!(output_tokens, Some(450));
    }

    #[test]
    fn test_token_extraction_handles_partial_data() {
        let trace = Trace {
            label: "root".to_string(),
            metadata: Some("input_tokens: 100".to_string()),
            inner_traces: Some(vec![Trace {
                label: "response".to_string(),
                metadata: Some("output_tokens: 200".to_string()),
                inner_traces: None,
                latency: None,
            }]),
            latency: None,
        };

        let (input_tokens, output_tokens) = extract_tokens_from_trace(&trace);

        assert_eq!(input_tokens, Some(100));
        assert_eq!(output_tokens, Some(200));
    }

    #[test]
    fn test_token_extraction_handles_missing_metadata() {
        let trace = Trace {
            label: "root".to_string(),
            metadata: Some("some other metadata".to_string()),
            inner_traces: None,
            latency: None,
        };

        let (input_tokens, output_tokens) = extract_tokens_from_trace(&trace);

        assert_eq!(input_tokens, None);
        assert_eq!(output_tokens, None);
    }

    #[test]
    fn test_token_stats_calculation_with_partial_data() {
        let runs = vec![
            RunOutcome::Success(SingleRunResult {
                response: "response1".to_string(),
                trace: Trace {
                    label: "run1".to_string(),
                    metadata: None,
                    inner_traces: None,
                    latency: None,
                },
                passed: true,
                reason: None,
                eval_score: 1.0,
                latency_ms: Some(100.0),
                input_tokens: Some(100),
                output_tokens: Some(50),
                total_tokens: Some(150),
                trace_explanation: None,
                num_inference_calls: Some(1),
                concurrency_score: Some(0.5),
                error: None,
            }),
            RunOutcome::Success(SingleRunResult {
                response: "response2".to_string(),
                trace: Trace {
                    label: "run2".to_string(),
                    metadata: None,
                    inner_traces: None,
                    latency: None,
                },
                passed: true,
                reason: None,
                eval_score: 1.0,
                latency_ms: Some(200.0),
                input_tokens: None,
                output_tokens: None,
                total_tokens: None,
                trace_explanation: None,
                num_inference_calls: Some(1),
                concurrency_score: Some(0.5),
                error: None,
            }),
            RunOutcome::Success(SingleRunResult {
                response: "response3".to_string(),
                trace: Trace {
                    label: "run3".to_string(),
                    metadata: None,
                    inner_traces: None,
                    latency: None,
                },
                passed: true,
                reason: None,
                eval_score: 1.0,
                latency_ms: Some(150.0),
                input_tokens: Some(200),
                output_tokens: Some(100),
                total_tokens: Some(300),
                trace_explanation: None,
                num_inference_calls: Some(1),
                concurrency_score: Some(0.5),
                error: None,
            }),
        ];

        let input_tokens: Vec<u64> = runs
            .iter()
            .filter_map(|r| match r {
                RunOutcome::Success(r) => r.input_tokens,
                RunOutcome::Failed { .. } => None,
            })
            .collect();
        let output_tokens: Vec<u64> = runs
            .iter()
            .filter_map(|r| match r {
                RunOutcome::Success(r) => r.output_tokens,
                RunOutcome::Failed { .. } => None,
            })
            .collect();

        let mean_input = input_tokens.iter().sum::<u64>() as f64 / input_tokens.len() as f64;
        let mean_output = output_tokens.iter().sum::<u64>() as f64 / output_tokens.len() as f64;

        assert_eq!(input_tokens.len(), 2);
        assert_eq!(output_tokens.len(), 2);

        assert_eq!(mean_input, 150.0);
        assert_eq!(mean_output, 75.0);
    }
}
