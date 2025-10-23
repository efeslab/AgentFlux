use super::{Error, HTTPClient, Message, MessageParam, Result, ToolParam};
use error_stack::ResultExt;
use reqwest::header::HeaderMap;
use serde_json::Value;

use super::resolve_api_key;

const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_GENERATE_ENDPOINT: &str = "/v1/messages";
const ANTHROPIC_API_KEY_NAME: &str = "ANTHROPIC_API_KEY";

#[derive(Debug, Clone)]
pub struct Anthropic {
    url: String,
    api_key: String,
    model: String,
    max_tokens: u64,
}

impl Anthropic {
    pub fn new(model: &str, max_tokens: &u64, api_key: &Option<String>) -> Result<Self> {
        Ok(Self {
            url: format!("{}{}", ANTHROPIC_BASE_URL, ANTHROPIC_GENERATE_ENDPOINT),
            model: model.to_string(),
            max_tokens: max_tokens.clone(),
            api_key: resolve_api_key(api_key, ANTHROPIC_API_KEY_NAME)?,
        })
    }
}

impl HTTPClient for Anthropic {
    fn url(&self) -> String {
        self.url.clone()
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = reqwest::header::HeaderMap::new();

        headers.insert(
            reqwest::header::HeaderName::from_static("x-api-key"),
            reqwest::header::HeaderValue::from_str(&self.api_key)
                .change_context(Error::InferenceEngine)
                .attach_printable("failed to set x-api-key header")?,
        );
        headers.insert(
            reqwest::header::HeaderName::from_static("anthropic-version"),
            reqwest::header::HeaderValue::from_static("2023-06-01"),
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
        serde_json::json!({
          "model": model.to_string(),
          "max_tokens": max_tokens,
          "messages": message_params,
          "tools": tool_params,
        })
    }

    fn resolve_res(&self, raw_res: &str) -> Result<Message> {
        let message: Message = serde_json::from_str(raw_res)
            .change_context(Error::InferenceEngine)
            .attach_printable("failed to parse response")?;

        Ok(message)
    }

    fn get_model(&self) -> &str {
        &self.model
    }

    fn get_max_tokens(&self) -> u64 {
        self.max_tokens
    }
}
