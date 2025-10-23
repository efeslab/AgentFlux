use browserd::{
    agent_protocol::AgentClient, discovery::Discovery, orchestrator::Orchestrator, AppClient,
};
use tokio_util::sync::CancellationToken;

use crate::common::Result;

pub async fn run<T, D, O>(
    _browserd_client: AppClient<T, D, O>,
    _cancel: CancellationToken,
) -> Result<()>
where
    T: AgentClient + Clone,
    D: Discovery + Clone,
    O: Orchestrator + Clone,
{
    todo!()
}
