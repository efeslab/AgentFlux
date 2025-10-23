use super::{Error, HTTPClient, Message, MessageParam, Result, ToolParam};
use error_stack::ResultExt;
use reqwest::header::HeaderMap;
use serde_json::Value;

use super::resolve_api_key;

const OPENAI_BASE_URL: &str = "https://api.openai.com";
const OPENAI_CHAT_ENDPOINT: &str = "/v1/chat/completions";
const OPENAI_API_KEY_NAME: &str = "OPENAI_API_KEY";

#[derive(Debug, Clone)]
pub struct OpenAI {
    url: String,
    api_key: String,
    model: String,
    max_tokens: u64,
}

impl OpenAI {
    pub fn new(
        model: &str,
        max_tokens: u64,
        api_key: &Option<String>,
        url: &Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            url: url
                .as_ref()
                .map(|url| format!("{}{}", url, OPENAI_CHAT_ENDPOINT))
                .unwrap_or_else(|| format!("{}{}", OPENAI_BASE_URL, OPENAI_CHAT_ENDPOINT)),
            model: model.to_string(),
            max_tokens,
            api_key: resolve_api_key(api_key, OPENAI_API_KEY_NAME)?,
        })
    }
}

impl HTTPClient for OpenAI {
    fn url(&self) -> String {
        self.url.clone()
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = reqwest::header::HeaderMap::new();

        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .change_context(Error::InferenceEngine)
                .attach_printable("failed to set Authorization header")?,
        );
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
        use super::{MessageParamContent, MessageParamContentBlock};

        let mut openai_messages = Vec::new();

        for message in message_params {
            match &message.content {
                MessageParamContent::Text(text) => {
                    openai_messages.push(serde_json::json!({
                        "role": message.role,
                        "content": text
                    }));
                }
                MessageParamContent::Blocks(blocks) => {
                    let mut text_content = String::new();
                    let mut tool_calls = Vec::new();

                    for block in blocks {
                        match block {
                            MessageParamContentBlock::Text(text_block) => {
                                if !text_content.is_empty() {
                                    text_content.push_str("\n");
                                }
                                text_content.push_str(&text_block.text);
                            }
                            MessageParamContentBlock::ToolUse(tool_use) => {
                                tool_calls.push(serde_json::json!({
                                    "id": tool_use.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_use.name,
                                        "arguments": serde_json::to_string(&tool_use.input).unwrap_or_default()
                                    }
                                }));
                            }
                            MessageParamContentBlock::ToolResult(tool_result) => {
                                let content = match &tool_result.content {
                                    super::ToolResultBlockParamContent::Text(text) => text.clone(),
                                    super::ToolResultBlockParamContent::TextBlocks(blocks) => {
                                        blocks
                                            .iter()
                                            .map(|b| b.text.clone())
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                    }
                                };

                                openai_messages.push(serde_json::json!({
                                    "role": "tool",
                                    "tool_call_id": tool_result.tool_use_id,
                                    "content": content
                                }));
                            }
                        }
                    }

                    if !tool_calls.is_empty() {
                        let mut msg = serde_json::json!({
                            "role": "assistant",
                            "tool_calls": tool_calls
                        });
                        if !text_content.is_empty() {
                            msg["content"] = serde_json::json!(text_content);
                        }
                        openai_messages.push(msg);
                    } else if !text_content.is_empty() {
                        openai_messages.push(serde_json::json!({
                            "role": message.role,
                            "content": text_content
                        }));
                    }
                }
            }
        }

        let mut payload = serde_json::json!({
            "model": model.to_string(),
            "messages": openai_messages,
        });

        // if max_tokens > 0 {
        //     if model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4") {
        //         payload["max_completion_tokens"] = serde_json::json!(max_tokens);
        //     } else {
        //         payload["max_tokens"] = serde_json::json!(max_tokens);
        //     }
        // }

        if !tool_params.is_empty() {
            let tools: Vec<Value> = tool_params
                .into_iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        }
                    })
                })
                .collect();
            payload["tools"] = serde_json::json!(tools);
        }

        payload
    }

    fn resolve_res(&self, raw_res: &str) -> Result<Message> {
        use super::{
            MessageContentBlock, MessageType, MessageUsage, Role, TextBlock, TextBlockType,
            ToolUseBlock, ToolUseBlockType,
        };
        println!("Raw OpenAI response: {}", raw_res);

        let openai_response: Value = serde_json::from_str(raw_res)
            .change_context(Error::InferenceEngine)
            .attach_printable("failed to parse OpenAI response")?;

        let choice = openai_response["choices"]
            .get(0)
            .ok_or_else(|| Error::InferenceEngine)
            .attach_printable("no choices in response")?;

        let message_obj = &choice["message"];

        let role_str = message_obj["role"]
            .as_str()
            .ok_or_else(|| Error::InferenceEngine)
            .attach_printable("missing role in response")?;
        let role = Role::try_from(role_str)?;

        let mut content_blocks = Vec::new();

        if let Some(text_content) = message_obj["content"].as_str() {
            content_blocks.push(MessageContentBlock::Text(TextBlock {
                text: text_content.to_string(),
                type_field: TextBlockType::Text,
            }));
        }

        if let Some(tool_calls) = message_obj["tool_calls"].as_array() {
            for tool_call in tool_calls {
                if let (Some(id), Some(name), Some(arguments)) = (
                    tool_call["id"].as_str(),
                    tool_call["function"]["name"].as_str(),
                    tool_call["function"]["arguments"].as_str(),
                ) {
                    let input: Value =
                        serde_json::from_str(arguments).unwrap_or_else(|_| serde_json::json!({}));

                    content_blocks.push(MessageContentBlock::ToolUse(ToolUseBlock {
                        id: id.to_string(),
                        name: name.to_string(),
                        input,
                        type_field: ToolUseBlockType::ToolUse,
                    }));
                }
            }
        }

        let usage_obj = &openai_response["usage"];
        let usage = MessageUsage {
            input_tokens: usage_obj["prompt_tokens"].as_u64().unwrap_or(0),
            output_tokens: usage_obj["completion_tokens"].as_u64().unwrap_or(0),
        };

        let message = Message {
            id: openai_response["id"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            type_field: MessageType::Message,
            role,
            content: content_blocks,
            usage,
            latency: None,
        };

        Ok(message)
    }

    fn get_model(&self) -> &str {
        &self.model
    }

    fn get_max_tokens(&self) -> u64 {
        self.max_tokens
    }
}
