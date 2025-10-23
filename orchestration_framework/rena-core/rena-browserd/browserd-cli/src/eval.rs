use browserd::{
    agent_protocol::AgentClient, discovery::Discovery, orchestrator::Orchestrator, AppClient,
};
use browserd_eval::{
    self,
    config::Config,
    reporter::{CsvReporter, JsonReporter, Reporter},
};
use error_stack::ResultExt;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::common;

#[derive(Debug, clap::Args)]
pub struct Args {
    /// Path to the evaluation configuration file
    pub config: String,

    // Format of the reporter (json or csv)
    #[arg(short, long)]
    pub format: Option<String>,

    // Output path for the reporter
    #[arg(short, long)]
    pub output: Option<String>,
}

pub async fn run<T, D, O>(
    browserd_client: AppClient<T, D, O>,
    args: Args,
    browserd_config: browserd::config::Config,
    cancel: CancellationToken,
) -> common::Result<()>
where
    T: AgentClient + Clone,
    D: Discovery + Clone,
    O: Orchestrator + Clone,
{
    let config = common::init_config::<Config>(&args.config).change_context(common::Error::Eval)?;

    info!(
        "starting evaluation. config: {}, num_evals: {}",
        args.config,
        config.evals.len()
    );

    let mut results = browserd_eval::App::new(browserd_client)
        .run(&config, &browserd_config, cancel)
        .await
        .change_context(common::Error::Eval)?;

    info!("Starting post-evaluation trace explanation generation");

    let trace_engine_config = config.trace_explainer.as_ref().or_else(|| {
        config.evaluators.iter().find_map(|e| match e {
            browserd_eval::evaluators::Config::LLM {
                inference_engine, ..
            } => Some(inference_engine),
            _ => None,
        })
    });

    if let Some(engine_config) = trace_engine_config {
        match browserd_eval::trace_explainer::generate_trace_explanations(
            Some(engine_config),
            &mut results,
        )
        .await
        {
            Ok(_) => info!("Successfully generated trace explanations"),
            Err(e) => info!("Failed to generate trace explanations: {:?}", e),
        }
    } else {
        info!("No trace explainer or LLM evaluator configured, skipping trace explanations");
    }

    let orchestrator_engine = format_inference_engine(&browserd_config.inference_engine);
    let evaluator_engine = config.evaluators.iter().find_map(|e| match e {
        browserd_eval::evaluators::Config::LLM {
            inference_engine, ..
        } => Some(format_inference_engine(inference_engine)),
        _ => None,
    });

    let mut report: browserd_eval::reporter::EvaluationReport = results.into();
    report.orchestrator_inference_engine = Some(orchestrator_engine);
    report.evaluator_inference_engine = evaluator_engine;

    match args.format {
        Some(format) => {
            let output_path = args
                .output
                .ok_or(common::Error::Eval)
                .attach_printable("output_path not provided")?;

            match format.as_str() {
                "json" => JsonReporter::report(&report, &output_path)
                    .change_context(common::Error::Eval)?,
                "csv" => CsvReporter::report(&report, &output_path)
                    .change_context(common::Error::Eval)?,
                _ => Err(
                    error_stack::Report::new(common::Error::Eval).attach_printable(format!(
                        "invalid reporter format (use json, or csv). format: {}",
                        format,
                    )),
                )?,
            }
        }
        None => println!("{}", report),
    }

    Ok(())
}

fn format_inference_engine(config: &browserd::inference_engine::Config) -> String {
    match config {
        browserd::inference_engine::Config::Anthropic { model, .. } => {
            format!("anthropic-{}", model)
        }
        browserd::inference_engine::Config::Ollama { model, .. } => format!("ollama-{}", model),
        browserd::inference_engine::Config::OpenAI { model, .. } => format!("openai-{}", model),
    }
}
