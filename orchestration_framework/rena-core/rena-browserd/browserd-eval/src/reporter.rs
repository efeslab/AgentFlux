use error_stack::ResultExt;
use std::fs::File;
use std::io::Write;
use thiserror::Error;
use tracing::info;

use crate::{EvalResult, RunOutcome};

#[derive(Error, Debug)]
pub enum Error {
    #[error("serialization error")]
    Serialization,
    #[error("io error")]
    Io,
}

pub type Result<T> = error_stack::Result<T, Error>;

pub trait Reporter {
    fn report(eval_report: &EvaluationReport, output_path: &str) -> Result<()>;
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct EvaluationReport {
    pub total_tests: usize,
    pub total_runs: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub success_rate: f64,
    pub average_latency_ms: f64,
    pub average_eval_score: f64,
    pub orchestrator_inference_engine: Option<String>,
    pub evaluator_inference_engine: Option<String>,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_tokens: u64,
    pub average_input_tokens: f64,
    pub average_output_tokens: f64,
    pub average_total_tokens: f64,
    pub average_concurrency_scores: f64,
    pub average_inference_calls: f64,
    pub results: Vec<EvalResult>,
}

impl From<Vec<EvalResult>> for EvaluationReport {
    fn from(results: Vec<EvalResult>) -> Self {
        let total_tests = results.len();
        let total_runs: usize = results
            .iter()
            .map(|r| r.aggregated_metrics.total_runs)
            .sum();
        let passed_tests = results
            .iter()
            .filter(|r| r.eval_error.is_none() && r.aggregated_metrics.success_rate >= 50.0)
            .count();
        let failed_tests = total_tests - passed_tests;

        let success_rate = match total_tests {
            0 => 0.0,
            _ => (passed_tests as f64 / total_tests as f64) * 100.0,
        };

        let all_latencies: Vec<f64> = results
            .iter()
            .flat_map(|r| r.runs.iter())
            .filter_map(|run| match run {
                RunOutcome::Success(single_run) => single_run.latency_ms,
                RunOutcome::Failed { .. } => None,
            })
            .collect();

        let average_latency_ms = match all_latencies.len() {
            0 => 0.0,
            _ => all_latencies.iter().sum::<f64>() / all_latencies.len() as f64,
        };

        let all_eval_scores: Vec<f64> = results
            .iter()
            .flat_map(|r| r.runs.iter())
            .filter_map(|run| match run {
                RunOutcome::Success(single_run) => Some(single_run.eval_score),
                RunOutcome::Failed { .. } => None,
            })
            .collect();

        let average_eval_score = match all_eval_scores.len() {
            0 => 0.0,
            _ => all_eval_scores.iter().sum::<f64>() / all_eval_scores.len() as f64,
        };

        let all_input_tokens: Vec<u64> = results
            .iter()
            .flat_map(|r| r.runs.iter())
            .filter_map(|run| match run {
                RunOutcome::Success(single_run) => single_run.input_tokens,
                RunOutcome::Failed { .. } => None,
            })
            .collect();

        let all_output_tokens: Vec<u64> = results
            .iter()
            .flat_map(|r| r.runs.iter())
            .filter_map(|run| match run {
                RunOutcome::Success(single_run) => single_run.output_tokens,
                RunOutcome::Failed { .. } => None,
            })
            .collect();

        let all_total_tokens: Vec<u64> = results
            .iter()
            .flat_map(|r| r.runs.iter())
            .filter_map(|run| match run {
                RunOutcome::Success(single_run) => single_run.total_tokens,
                RunOutcome::Failed { .. } => None,
            })
            .collect();

        let total_input_tokens: u64 = all_input_tokens.iter().sum();
        let total_output_tokens: u64 = all_output_tokens.iter().sum();
        let total_tokens: u64 = all_total_tokens.iter().sum();

        let average_input_tokens = match all_input_tokens.len() {
            0 => 0.0,
            _ => total_input_tokens as f64 / all_input_tokens.len() as f64,
        };

        let average_output_tokens = match all_output_tokens.len() {
            0 => 0.0,
            _ => total_output_tokens as f64 / all_output_tokens.len() as f64,
        };

        let average_total_tokens = match all_total_tokens.len() {
            0 => 0.0,
            _ => total_tokens as f64 / all_total_tokens.len() as f64,
        };

        let all_concurrency_scores: Vec<f64> = results
            .iter()
            .flat_map(|r| r.runs.iter())
            .filter_map(|run| match run {
                RunOutcome::Success(single_run) => single_run.concurrency_score,
                RunOutcome::Failed { .. } => None,
            })
            .collect();

        let average_concurrency_scores = match all_concurrency_scores.len() {
            0 => 0.0,
            _ => all_concurrency_scores.iter().sum::<f64>() / all_concurrency_scores.len() as f64,
        };

        let all_inference_calls: Vec<u64> = results
            .iter()
            .flat_map(|r| r.runs.iter())
            .filter_map(|run| match run {
                RunOutcome::Success(single_run) => single_run.num_inference_calls,
                RunOutcome::Failed { .. } => None,
            })
            .collect();

        let average_inference_calls = match all_inference_calls.len() {
            0 => 0.0,
            _ => all_inference_calls.iter().sum::<u64>() as f64 / all_inference_calls.len() as f64,
        };

        Self {
            total_tests,
            total_runs,
            passed_tests,
            failed_tests,
            success_rate,
            average_latency_ms,
            average_eval_score,
            orchestrator_inference_engine: None,
            evaluator_inference_engine: None,
            total_input_tokens,
            total_output_tokens,
            total_tokens,
            average_input_tokens,
            average_output_tokens,
            average_total_tokens,
            average_concurrency_scores,
            average_inference_calls,
            results: results.to_vec(),
        }
    }
}

impl std::fmt::Display for EvaluationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string_pretty(self).unwrap())
    }
}

pub struct JsonReporter;

impl Reporter for JsonReporter {
    fn report(eval_report: &EvaluationReport, output_path: &str) -> Result<()> {
        write_to_file(
            &serde_json::to_string_pretty(&eval_report).change_context(Error::Serialization)?,
            output_path,
        )?;

        Ok(())
    }
}

pub struct CsvReporter;

impl Reporter for CsvReporter {
    fn report(eval_report: &EvaluationReport, output_path: &str) -> Result<()> {
        let mut csv_content = String::new();

        csv_content.push_str("query,ideal,evaluator,total_runs,passed_runs,failed_runs,success_rate,mean_latency_ms,min_latency_ms,max_latency_ms,std_dev_latency_ms,mean_eval_score,min_eval_score,max_eval_score,std_dev_eval_score,eval_error\n");

        eval_report.results.iter().for_each(|result| {
            let metrics = &result.aggregated_metrics;
            csv_content.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",{},{},{},{:.2}",
                result.query.replace("\"", "\"\""),
                result.ideal.replace("\"", "\"\""),
                result.evaluator.replace("\"", "\"\""),
                metrics.total_runs,
                metrics.passed_runs,
                metrics.failed_runs,
                metrics.success_rate,
            ));

            if let Some(latency_stats) = &metrics.latency_stats {
                csv_content.push_str(&format!(
                    ",{:.2},{:.2},{:.2},{:.2}",
                    latency_stats.mean_ms,
                    latency_stats.min_ms,
                    latency_stats.max_ms,
                    latency_stats.std_dev_ms,
                ));
            } else {
                csv_content.push_str(",,,,");
            }

            csv_content.push_str(&format!(
                ",{:.2},{:.2},{:.2},{:.2}",
                metrics.eval_score_stats.mean,
                metrics.eval_score_stats.min,
                metrics.eval_score_stats.max,
                metrics.eval_score_stats.std_dev,
            ));

            if let Some(error) = &result.eval_error {
                csv_content.push_str(&format!(",\"{}\"", error.replace("\"", "\"\"")));
            } else {
                csv_content.push_str(",");
            }

            csv_content.push_str("\n");
        });

        write_to_file(&csv_content, output_path)?;

        Ok(())
    }
}

fn write_to_file(content: &str, output_path: &str) -> Result<()> {
    let mut file = File::create(output_path).change_context(Error::Io)?;
    file.write_all(content.as_bytes())
        .change_context(Error::Io)?;

    info!("dumped results. output_path: {}", output_path);

    Ok(())
}
