use browserd::{
    agent_protocol::AgentClient, discovery::Discovery, orchestrator::Orchestrator, AppClient,
};
use error_stack::ResultExt;
use futures::future::try_join_all;
use tokio::select;
use tokio_util::sync::CancellationToken;

use crate::common::{Error, Result};

#[derive(clap::Args, Debug)]
pub struct Args {
    pub query_inputs: Vec<String>,

    #[arg(short, long)]
    pub tool_filter: Option<Vec<String>>,

    #[arg(short, long)]
    pub parallel: Option<u8>,
}

pub async fn run<T, D, O>(
    browserd_client: AppClient<T, D, O>,
    args: Args,
    cancel: CancellationToken,
) -> Result<()>
where
    T: AgentClient + Clone,
    D: Discovery + Clone,
    O: Orchestrator + Clone,
{
    let query_inputs = match args.parallel {
        Some(n) => vec![args.query_inputs[0].clone(); n as usize],
        None => args.query_inputs,
    };

    let futs = query_inputs.into_iter().map(|query_input| {
        let mut browserd_client = browserd_client.clone();
        let tool_filter = args.tool_filter.clone();
        async move {
            browserd_client
                .query(&query_input, &tool_filter)
                .await
                .map(|(res, trace)| (query_input, res, trace))
                .change_context(Error::Query)
        }
    });

    select! {
        res = try_join_all(futs) => {
            let _ = res
                .change_context(Error::Query)?
                .into_iter()
                .map(|(query_input, res, trace)| {
                    println!("Query: {}\nResponse: {}\nTrace: {}", query_input, res, trace);
                })
                .collect::<Vec<_>>();
        },
        _ = cancel.cancelled() => {}
    }

    Ok(())
}
