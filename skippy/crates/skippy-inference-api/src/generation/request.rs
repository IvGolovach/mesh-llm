//! Backend generation input. HTTP deserialization belongs to each protocol
//! adapter; this type deliberately has no Serde wire representation.
//! Both token-limit aliases remain available so existing defaults and hooks
//! preserve their precedence and provenance during the boundary migration.

use super::content::{GenerationMessage, messages_to_plain_prompt};
use crate::{
    common::{
        AgentSessionIdentity, PromptCacheRetention, ReasoningConfig, ReasoningEffort, StopSequence,
        StreamOptions, agent_session_metadata, agent_session_source_metadata,
        set_agent_session_metadata,
    },
    errors::InferenceError,
};
use serde_json::Value;
use std::collections::BTreeMap;

/// Input owned by a generation backend, independent of frontend serialization.
///
/// Extension values remain opaque so adaptation preserves currently accepted
/// inputs, including fields interpreted later by runtime defaults or hooks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationRequest {
    pub model: String,

    pub messages: Vec<GenerationMessage>,

    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<u32>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<BTreeMap<String, Value>>,
    pub response_format: Option<Value>,
    /// Existing function definitions, including arbitrary JSON Schema and
    /// extension fields. Validation is intentionally unchanged.
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub user: Option<String>,
    pub stop: Option<StopSequence>,
    pub seed: Option<u64>,
    pub reasoning: Option<ReasoningConfig>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    pub stream_options: Option<StreamOptions>,

    pub extra: BTreeMap<String, Value>,
}

impl GenerationRequest {
    pub(crate) fn set_agent_session(&mut self, identity: Option<AgentSessionIdentity>) {
        set_agent_session_metadata(&mut self.extra, identity);
    }

    #[must_use]
    pub fn agent_session(&self) -> Option<&str> {
        agent_session_metadata(&self.extra)
    }

    #[must_use]
    pub fn agent_session_source(&self) -> Option<&str> {
        agent_session_source_metadata(&self.extra)
    }

    pub fn effective_max_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .map(StreamOptions::include_usage)
            .unwrap_or(false)
    }

    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.model.trim().is_empty() {
            return Err(InferenceError::invalid_request("model is required"));
        }
        if self.messages.is_empty() {
            return Err(InferenceError::invalid_request("messages is required"));
        }
        if matches!(self.max_tokens, Some(0)) || matches!(self.max_completion_tokens, Some(0)) {
            return Err(InferenceError::invalid_request(
                "max_tokens must be greater than zero",
            ));
        }
        if self.n.is_some_and(|n| n == 0) {
            return Err(InferenceError::invalid_request(
                "n must be greater than zero",
            ));
        }
        if self.n.is_some_and(|n| n > 1) {
            return Err(InferenceError::unsupported(
                "n > 1 is parsed but multiple choices are not yet implemented",
            ));
        }
        if self.top_logprobs.is_some() && !self.logprobs.unwrap_or(false) {
            return Err(InferenceError::invalid_request(
                "top_logprobs requires logprobs=true",
            ));
        }
        if let Some(tools) = self.tools.as_ref() {
            validate_tools_value(tools)
                .map_err(|message| InferenceError::invalid_request(message).with_param("tools"))?;
        }
        if self
            .response_format
            .as_ref()
            .is_some_and(invalid_response_format_value)
        {
            return Err(InferenceError::invalid_request(
                "response_format must be an object with a type field",
            ));
        }
        let prompt_is_empty = messages_to_plain_prompt(&self.messages).trim().is_empty();
        let media_can_supply_prompt = crate::hooks::first_chat_media(&self.messages).is_some();
        if prompt_is_empty && !media_can_supply_prompt {
            return Err(InferenceError::invalid_request(
                "messages produced an empty prompt",
            ));
        }
        Ok(())
    }
}

fn invalid_response_format_value(value: &Value) -> bool {
    !value
        .as_object()
        .and_then(|object| object.get("type"))
        .is_some_and(Value::is_string)
}

fn validate_tools_value(value: &Value) -> Result<(), String> {
    let Some(tools) = value.as_array() else {
        return Err("tools must be an array".to_string());
    };
    for (index, tool) in tools.iter().enumerate() {
        let Some(tool) = tool.as_object() else {
            return Err(format!("tools[{index}] must be an object"));
        };
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(format!("tools[{index}].type must be `function`"));
        }
        let Some(function) = tool.get("function").and_then(Value::as_object) else {
            return Err(format!("tools[{index}].function must be an object"));
        };
        if function
            .get("name")
            .and_then(Value::as_str)
            .is_none_or(|name| name.trim().is_empty())
        {
            return Err(format!(
                "tools[{index}].function.name must be a non-empty string"
            ));
        }
        if function
            .get("parameters")
            .is_some_and(|parameters| !parameters.is_object())
        {
            return Err(format!(
                "tools[{index}].function.parameters must be an object"
            ));
        }
    }
    Ok(())
}
