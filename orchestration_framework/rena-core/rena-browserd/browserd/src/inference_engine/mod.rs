use error_stack::{Report, ResultExt};
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use thiserror::Error;

pub mod anthropic;
pub mod ollama;
pub mod openai;

use crate::app_manager;
use crate::container_comm::{self, proto::browserd};
use crate::trace::{instrument, Trace};

type Result<T> = error_stack::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("inference_engine failed")]
    InferenceEngine,
    #[error("invalid event")]
    InvalidEvent,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum Config {
    Anthropic {
        model: String,
        max_tokens: u64,
        api_key: Option<String>,
    },
    Ollama {
        model: String,
        max_tokens: u64,
    },
    OpenAI {
        model: String,
        max_tokens: u64,
        api_key: Option<String>,
        url: Option<String>,
    },
}

// shared
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

impl TryFrom<&str> for Role {
    type Error = Error;

    fn try_from(value: &str) -> std::result::Result<Self, Error> {
        match value {
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            _ => Err(Error::InvalidEvent),
        }
    }
}

// Text Block
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextBlockType {
    Text,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub text: String,
    #[serde(rename = "type")]
    pub type_field: TextBlockType,
}

impl From<&str> for TextBlock {
    fn from(value: &str) -> Self {
        Self {
            text: value.to_string(),
            type_field: TextBlockType::Text,
        }
    }
}

// ToolUse Block
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolUseBlockType {
    #[serde(rename = "tool_use")]
    ToolUse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseBlock {
    pub id: String,
    pub input: serde_json::Value,
    pub name: String,
    #[serde(rename = "type")]
    pub type_field: ToolUseBlockType,
}

// NOTE(sean): a param type  only can be fed to an inference_engine
// while a non-param can be used for both input and output of inference_engine
// ToolResult Block Param
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolResultBlockParamType {
    #[serde(rename = "tool_result")]
    ToolResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultBlockParamContent {
    Text(String),
    TextBlocks(Vec<TextBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultBlockParam {
    pub tool_use_id: String,
    #[serde(rename = "type")]
    pub type_field: ToolResultBlockParamType,
    pub content: ToolResultBlockParamContent,
    pub is_error: bool,
}

impl std::fmt::Display for ToolResultBlockParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ToolResultBlockParam {{ tool_use_id: {}, is_error: {}, content: {:?} }}",
            self.tool_use_id, self.is_error, self.content
        )
    }
}

impl From<container_comm::CallToolResponse> for ToolResultBlockParam {
    fn from(value: container_comm::CallToolResponse) -> Self {
        if let Some(error) = value.error {
            return Self {
                tool_use_id: value.id,
                type_field: ToolResultBlockParamType::ToolResult,
                content: ToolResultBlockParamContent::Text(error),
                is_error: true,
            };
        }

        Self {
            tool_use_id: value.id,
            type_field: ToolResultBlockParamType::ToolResult,
            content: ToolResultBlockParamContent::TextBlocks(
                value
                    .tool_results
                    .iter()
                    .map(|tool_result| TextBlock {
                        text: tool_result.clone(),
                        type_field: TextBlockType::Text,
                    })
                    .collect(),
            ),
            is_error: false,
        }
    }
}

// Message Param (inference_engine input)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageParamContentBlock {
    Text(TextBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlockParam),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageParamContent {
    Text(String),
    Blocks(Vec<MessageParamContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageParam {
    pub role: Role,
    pub content: MessageParamContent,
}

// Tool Param (inference_engine input)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam {
    pub input_schema: HashMap<String, serde_json::Value>,
    pub name: String,
    pub description: String,
}

impl From<app_manager::Tool> for ToolParam {
    fn from(value: app_manager::Tool) -> Self {
        Self {
            input_schema: value.input_schema,
            name: value.tool_name,
            description: value.description,
        }
    }
}

impl TryFrom<browserd::Tool> for ToolParam {
    type Error = Error;

    fn try_from(value: browserd::Tool) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            input_schema: serde_json::from_str(&value.input_schema)
                .map_err(|_| Error::InvalidEvent)?,
            name: value.name,
            description: value.description,
        })
    }
}

// Message (inference_engine output)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    Message,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContentBlock {
    Text(TextBlock),
    ToolUse(ToolUseBlock),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl From<(u64, u64)> for MessageUsage {
    fn from((input_tokens, output_tokens): (u64, u64)) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageLatency {
    pub load_model: u64,
    pub prefill: u64,
    pub decode: u64,
}

impl From<(u64, u64, u64)> for MessageLatency {
    fn from((load_model, prefill, decode): (u64, u64, u64)) -> Self {
        Self {
            load_model,
            prefill,
            decode,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: MessageType,
    pub role: Role,
    pub content: Vec<MessageContentBlock>,
    pub usage: MessageUsage,
    pub latency: Option<MessageLatency>,
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let latency = self
            .latency
            .as_ref()
            .map(|latency| {
                format!(
                    "load_model: {}, prefill: {}, decode: {}",
                    latency.load_model, latency.prefill, latency.decode
                )
            })
            .unwrap_or_else(|| "unknown".to_string());

        write!(
            f,
            "usage: input_tokens: {}, output_tokens: {}, latency: {}, content: {:?}",
            self.usage.input_tokens, self.usage.output_tokens, latency, self.content
        )
    }
}

pub trait HTTPClient {
    fn url(&self) -> String;
    fn headers(&self) -> Result<HeaderMap>;
    fn payload(
        &self,
        model: &str,
        max_tokens: u64,
        message_params: Vec<MessageParam>,
        tool_params: Vec<ToolParam>,
    ) -> serde_json::Value;
    fn resolve_res(&self, raw_res: &str) -> Result<Message>;
    fn get_model(&self) -> &str;
    fn get_max_tokens(&self) -> u64;
}

#[derive(Debug, Clone)]
pub struct InferenceEngine<T>
where
    T: HTTPClient,
{
    inner: T,
}

impl<T> InferenceEngine<T>
where
    T: HTTPClient,
{
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    async fn inner_generate(
        &self,
        message_params: Vec<MessageParam>,
        tool_params: Vec<ToolParam>,
    ) -> Result<Message> {
        let payload = self.inner.payload(
            self.inner.get_model(),
            self.inner.get_max_tokens(),
            message_params,
            tool_params,
        );

        // println!("payload: {:?}", payload);

        let response = reqwest::Client::new()
            .post(self.inner.url())
            .headers(self.inner.headers()?)
            .json(&payload)
            .send()
            .await
            .change_context(Error::InferenceEngine)
            .attach_printable("failed to send request")?;

        let status = response.status();
        if status != reqwest::StatusCode::OK {
            let error_body = response
                .text()
                .await
                .change_context(Error::InferenceEngine)
                .attach_printable("failed to read error body")?;
            
            return Err(
                Report::from(Error::InferenceEngine).attach_printable(format!(
                    "failed request: error: {}, error_body: {}",
                    status,
                    error_body
                )),
            );
        }

        let response_text = response
            .text()
            .await
            .change_context(Error::InferenceEngine)
            .attach_printable("failed to read raw res")?;
        
        let message = self.inner.resolve_res(&response_text)?;
        
        
        // println!("payload: {:?}", payload);
        // println!("response_text: {:?}", response_text);
        
        // // Check if the response contains tool calls
        // let has_tool_calls = message.content.iter().any(|content_block| {
        //     matches!(content_block, MessageContentBlock::ToolUse(_))
        // });
        
        // // Only log if there are no tool calls
        // if !has_tool_calls { 
        //     // Clone the payload to modify it for logging.
        //     let mut combined_payload = payload.clone();

        //     // Create the new assistant message from the response content.
        //     // If the response content has only one text block, flatten it to a simple string.
        //     let assistant_response_content = if message.content.len() == 1 {
        //         if let Some(MessageContentBlock::Text(text_block)) = message.content.first() {
        //             serde_json::Value::String(text_block.text.clone())
        //         } else {
        //             serde_json::to_value(&message.content).unwrap_or(serde_json::Value::Null)
        //         }
        //     } else {
        //         serde_json::to_value(&message.content).unwrap_or(serde_json::Value::Null)
        //     };

        //     let new_assistant_message = serde_json::json!({
        //         "role": message.role,
        //         "content": assistant_response_content
        //     });

        //     // By default, we will log the conversation.
        //     let mut should_log = true;

        //     // Check for the 'messages' array in the payload.
        //     if let Some(messages) = combined_payload.get_mut("messages").and_then(|v| v.as_array_mut()) {
        //         // If its length is 1 (the initial prompt), don't push the response and disable logging.
        //         if messages.len() <= 1 {
        //             should_log = false;
        //         } else {
        //             // For conversations with more than one message, add the assistant's response before logging.
        //             messages.push(new_assistant_message);
        //         }
        //     }
        //     // If 'messages' doesn't exist, should_log remains true, and we log the unmodified payload.

        //     if should_log {
        //         let combined_str = serde_json::to_string(&combined_payload)
        //             .unwrap_or_else(|_| format!("{:?}", combined_payload));
                
        //         if let Err(e) = log_to_file(&combined_str, &resolve_log_file_name()) {
        //             eprintln!("Failed to log conversation: {:?}", e);
        //         }
        //     }
        // }

        Ok(message)
    }

    pub async fn generate(
        &self,
        message_params: Vec<MessageParam>,
        tool_params: Vec<ToolParam>,
    ) -> Result<(Message, Trace)> {
        let (message, trace) = instrument(
            || self.inner_generate(message_params, tool_params),
            "inference_call",
            |message| Some(format!("{}", message)),
            |_| None,
        )
        .await?;

        Ok((message, trace))
    }


    pub fn print_inference_calls(
        &self,
        message_params: Vec<MessageParam>,
        tool_params: Vec<ToolParam>,
    ) -> Result<()> {
        let payload = self.inner.payload(
                    self.inner.get_model(),
                    self.inner.get_max_tokens(),
                    // self.inner.get_temperature(),
                    // self.inner.get_choices(),
                    message_params.clone(),
                    tool_params.clone(),
                );

        let combined_str = serde_json::to_string(&payload)
            .unwrap_or_else(|_| format!("{:?}", payload));

        if let Err(e) = log_to_file(&combined_str, &resolve_log_file_name()) {
            eprintln!("Failed to log inference calls: {}", e);
        }

        Ok(())
    }


}

fn resolve_api_key<'a>(api_key: &'a Option<String>, env_var: &str) -> Result<String> {
    match api_key {
        Some(api_key) => Ok(api_key.clone()),
        None => dotenv::var(env_var)
            .change_context(Error::InferenceEngine)
            .attach_printable(format!("{} not found in .env file", env_var)),
    }
}

fn resolve_log_file_name() -> String {
    match dotenv::var("RENA_LOG_NAME") {
        Ok(name) if !name.trim().is_empty() => name,
        _ => "inference_logs.json".to_string(),
    }
}

fn log_to_file(content: &str, log_file_path: &str) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file_path)
        .change_context(Error::InferenceEngine)
        .attach_printable("failed to open log file")?;

    // Write exactly one line per log entry. If `content` already ends with '\n',
    // don't add another; otherwise, append one.
    let write_result = if content.ends_with('\n') {
        write!(file, "{}", content)
    } else {
        writeln!(file, "{}", content)
    };

    write_result
        .change_context(Error::InferenceEngine)
        .attach_printable("failed to write log content")?;

    Ok(())
}
