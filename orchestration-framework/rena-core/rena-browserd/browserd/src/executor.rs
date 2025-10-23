use thiserror::Error;

use crate::agent_protocol::AgentClient;
use crate::app_manager::AppManager;
use crate::inference_engine::{
    MessageParam, MessageParamContent, MessageParamContentBlock, Role, ToolResultBlockParam,
    ToolResultBlockParamContent, ToolResultBlockParamType, ToolUseBlock,
};
use crate::trace::Trace;

const CALL_TOOL_TIMEOUT: u64 = 60;

#[derive(Error, Debug)]
pub enum Error {
    #[error("agent client error")]
    AgentClient,
}

pub type Result<T> = error_stack::Result<T, Error>;

#[derive(Debug)]
pub struct Node {
    pub id: u64,
    pub tool_use_block: ToolUseBlock,
    pub dependencies: Option<Vec<u64>>,
}

#[derive(Debug)]
pub struct ExecutionPlan {
    pub nodes: Vec<Node>,
}

#[derive(Clone)]
pub struct Executor<T>
where
    T: AgentClient,
{
    pub app_manager: AppManager<T>,
}

impl<T> Executor<T>
where
    T: AgentClient,
{
    pub fn new(app_manager: AppManager<T>) -> Self {
        Self { app_manager }
    }

    pub async fn execute(
        &self,
        message_params: &mut Vec<MessageParam>,
        plan: ExecutionPlan,
    ) -> Result<Vec<Trace>> {
        let mut traces = vec![];

        // NOTE(sean): simply running them sequentially ignoring dependencies
        for node in plan.nodes {
            message_params.push(MessageParam {
                role: Role::Assistant,
                content: MessageParamContent::Blocks(vec![MessageParamContentBlock::ToolUse(
                    node.tool_use_block.clone(),
                )]),
            });

            let (mut trace, start) = Trace::start("tool call");
            let tool_result = match self
                .app_manager
                .call_tool(&node.tool_use_block, &CALL_TOOL_TIMEOUT)
                .await
            {
                Ok(tool_result) => tool_result,
                Err(e) => ToolResultBlockParam {
                    tool_use_id: node.tool_use_block.id,
                    type_field: ToolResultBlockParamType::ToolResult,
                    content: ToolResultBlockParamContent::Text(e.to_string()),
                    is_error: true,
                },
            };
            trace.end(
                start,
                Some(format!(
                    "tool_name: {}, tool_call_result: {}",
                    node.tool_use_block.name, tool_result
                )),
                None,
            );
            traces.push(trace);

            message_params.push(MessageParam {
                role: Role::User,
                content: MessageParamContent::Blocks(vec![MessageParamContentBlock::ToolResult(
                    tool_result.clone(),
                )]),
            });
        }

        Ok(traces)
    }
}
