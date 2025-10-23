use error_stack::ResultExt;
use std::future::Future;
use thiserror::Error;

use crate::app_manager::Tool;
use crate::executor::{ExecutionPlan, Node};
use crate::inference_engine::{
    HTTPClient, InferenceEngine, MessageContentBlock, MessageParam, MessageParamContent, Role,
};
use crate::trace::Trace;

#[derive(Error, Debug)]
pub enum Error {
    #[error("inference engine error")]
    InferenceEngine,
}

pub type Result<T> = error_stack::Result<T, Error>;

pub trait Orchestrator {
    fn orchestrate(
        &self,
        input: &mut Vec<MessageParam>,
        tools: Vec<Tool>,
    ) -> impl Future<Output = Result<(Option<ExecutionPlan>, Vec<Trace>)>> + Send;

    fn inference_engine(&self) -> &InferenceEngine<impl HTTPClient + Clone>;
}

#[derive(Clone)]
pub struct BasicOrchestrator<T>
where
    T: HTTPClient + Clone,
{
    pub inference_engine: InferenceEngine<T>,
}

impl<T> BasicOrchestrator<T>
where
    T: HTTPClient + Clone,
{
    pub fn new(inner: T) -> Self {
        Self {
            inference_engine: InferenceEngine::new(inner),
        }
    }
}

impl<T> Orchestrator for BasicOrchestrator<T>
where
    T: HTTPClient + Clone + Send + Sync,
{
    async fn orchestrate(
        &self,
        message_params: &mut Vec<MessageParam>,
        tools: Vec<Tool>,
    ) -> Result<(Option<ExecutionPlan>, Vec<Trace>)> {
        let result = self
            .inference_engine
            .generate(
                message_params.clone(),
                tools.into_iter().map(Into::into).collect(),
            )
            .await
            .change_context(Error::InferenceEngine);

        let (message, trace) = match result {
            Ok((message, trace)) => (message, trace),
            Err(e) => {
                message_params.push(MessageParam {
                    role: Role::Assistant,
                    content: MessageParamContent::Text(e.to_string()),
                });
                return Ok((None, vec![]));
            }
        };

        let mut nodes = vec![];

        for content in message.clone().content {
            match &content {
                MessageContentBlock::Text(text) => {
                    message_params.push(MessageParam {
                        role: Role::Assistant,
                        content: MessageParamContent::Text(text.text.clone()),
                    });
                }
                MessageContentBlock::ToolUse(tool_use_block) => {
                    nodes.push(Node {
                        id: 0,
                        tool_use_block: tool_use_block.clone(),
                        dependencies: None,
                    });
                }
            }
        }

        match nodes.len() {
            0 => Ok((None, vec![trace])),
            _ => Ok((Some(ExecutionPlan { nodes }), vec![trace])),
        }
    }

    fn inference_engine(&self) -> &InferenceEngine<impl HTTPClient + Clone> {
        &self.inference_engine
    }

}
