use std::collections::HashMap;

use error_stack::ResultExt;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    Error, HTTPClient, Message, MessageContentBlock, MessageParam, MessageParamContent,
    MessageParamContentBlock, MessageType, Result, Role, TextBlock, TextBlockType, ToolParam,
    ToolResultBlockParam, ToolResultBlockParamContent, ToolUseBlock, ToolUseBlockType,
};

const OLLAMA_BASE_URL: &str = "http://localhost:11434";
const OLLAMA_CHAT_ENDPOINT: &str = "/api/chat";
const NANO_TO_MILLI: u64 = 1_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ToolParamType {
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaToolParamInner {
    name: String,
    description: String,
    parameters: HashMap<String, Value>,
}

impl From<ToolParam> for OllamaToolParamInner {
    fn from(tool_param: ToolParam) -> Self {
        Self {
            name: tool_param.name,
            description: tool_param.description,
            parameters: tool_param.input_schema,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaToolParam {
    #[serde(rename = "type")]
    type_filed: ToolParamType,
    function: OllamaToolParamInner,
}

impl From<ToolParam> for OllamaToolParam {
    fn from(tool_param: ToolParam) -> Self {
        Self {
            type_filed: ToolParamType::Function,
            function: tool_param.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolUseBlockInner {
    name: String,
    arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolUseBlock {
    function: OllamaToolUseBlockInner,
}

impl From<ToolUseBlock> for OllamaToolUseBlock {
    fn from(value: ToolUseBlock) -> Self {
        Self {
            function: OllamaToolUseBlockInner {
                name: value.name,
                arguments: value.input,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaMessage {
    role: Role,
    content: String,
    #[serde(default)]
    tool_calls: Vec<OllamaToolUseBlock>,
}

impl From<TextBlock> for String {
    fn from(text_block: TextBlock) -> Self {
        text_block.text
    }
}

impl From<ToolResultBlockParam> for String {
    fn from(tool_result_block: ToolResultBlockParam) -> Self {
        let mut result: Vec<String> = vec![];
        match tool_result_block.content {
            ToolResultBlockParamContent::Text(text) => result.push(text.to_string()),
            ToolResultBlockParamContent::TextBlocks(blocks) => {
                blocks
                    .iter()
                    .for_each(|block| result.push(block.text.to_string()));
            }
        };

        result.join("\n")
    }
}

impl From<MessageParam> for OllamaMessage {
    fn from(msg_param: MessageParam) -> Self {
        match msg_param.content {
            MessageParamContent::Blocks(blocks) => {
                let mut contents: Vec<String> = vec![];
                let mut tool_calls = vec![];

                blocks.iter().for_each(|block| match block {
                    MessageParamContentBlock::Text(text) => contents.push(text.clone().into()),
                    MessageParamContentBlock::ToolUse(tool_use) => {
                        tool_calls.push(tool_use.clone().into())
                    }
                    MessageParamContentBlock::ToolResult(tool_result) => {
                        contents.push(tool_result.clone().into())
                    }
                });
                OllamaMessage {
                    role: msg_param.role,
                    content: contents.join("\n"),
                    tool_calls,
                }
            }
            MessageParamContent::Text(text) => Self {
                role: msg_param.role,
                content: text,
                tool_calls: vec![],
            },
        }
    }
}

impl From<String> for MessageContentBlock {
    fn from(value: String) -> Self {
        MessageContentBlock::Text(TextBlock {
            text: value,
            type_field: TextBlockType::Text,
        })
    }
}

impl From<OllamaToolUseBlock> for MessageContentBlock {
    fn from(value: OllamaToolUseBlock) -> Self {
        MessageContentBlock::ToolUse(ToolUseBlock {
            id: uuid::Uuid::new_v4().to_string(),
            name: value.function.name,
            input: value.function.arguments,
            type_field: ToolUseBlockType::ToolUse,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OllamaResponse {
    model: String,
    message: OllamaMessage,
    prompt_eval_count: u64,
    eval_count: u64,
    load_duration: u64,
    prompt_eval_duration: u64,
    eval_duration: u64,
}

impl From<OllamaResponse> for Message {
    fn from(value: OllamaResponse) -> Self {
        let mut content = vec![];
        content.push(value.message.content.into());
        value
            .message
            .tool_calls
            .iter()
            .for_each(|tool_call| content.push(tool_call.clone().into()));

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: value.message.role,
            content,
            type_field: MessageType::Message,
            usage: (value.prompt_eval_count, value.eval_count).into(),
            latency: Some(
                (
                    value.load_duration / NANO_TO_MILLI,
                    value.prompt_eval_duration / NANO_TO_MILLI,
                    value.eval_duration / NANO_TO_MILLI,
                )
                    .into(),
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ollama {
    url: String,
    model: String,
    max_tokens: u64,
}

impl Ollama {
    pub fn new(model: &str, max_tokens: u64) -> Self {
        Self {
            url: format!("{}{}", OLLAMA_BASE_URL, OLLAMA_CHAT_ENDPOINT),
            model: model.to_string(),
            max_tokens,
        }
    }
}

impl HTTPClient for Ollama {
    fn url(&self) -> String {
        self.url.clone()
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        Ok(headers)
    }

    fn payload(
        &self,
        model: &str,
        max_tokens: u64,
        message_params: Vec<MessageParam>,
        tool_params: Vec<ToolParam>,
    ) -> Value {
        let tool_params: Vec<OllamaToolParam> = tool_params
            .iter()
            .map(|tool_param| tool_param.clone().into())
            .collect();

        let message_params: Vec<OllamaMessage> = message_params
            .iter()
            .map(|msg_param| msg_param.clone().into())
            .collect();

        let payload = serde_json::json!({
            "model": model.to_string(),
            "max_tokens": max_tokens,
            "messages": message_params,
            "stream": false,
            "tools": tool_params,
        });

        payload
    }

    fn resolve_res(&self, raw_res: &str) -> Result<Message> {
        let res: OllamaResponse = serde_json::from_str(raw_res)
            .change_context(Error::InferenceEngine)
            .attach_printable("failed to parse response")?;

        Ok(res.into())
    }

    fn get_model(&self) -> &str {
        &self.model
    }

    fn get_max_tokens(&self) -> u64 {
        self.max_tokens
    }
}
