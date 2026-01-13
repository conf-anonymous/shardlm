//! Tokenizer wrapper for LLM models (Llama 3.x, Qwen 2.5)

use std::path::Path;

use crate::error::{ModelError, Result};

/// Chat format for different model families
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    /// Llama 3.x format with <|start_header_id|>, <|end_header_id|>, <|eot_id|>
    Llama3,
    /// ChatML format with <|im_start|>, <|im_end|>
    ChatML,
}

/// Tokenizer for LLM models
pub struct Tokenizer {
    /// Inner tokenizer
    inner: tokenizers::Tokenizer,
    /// BOS token ID
    pub bos_token_id: u32,
    /// EOS token ID
    pub eos_token_id: u32,
    /// End of turn token ID (model-specific)
    pub eot_token_id: u32,
    /// Padding token ID
    pub pad_token_id: Option<u32>,
    /// Chat format for this tokenizer
    pub chat_format: ChatFormat,
}

impl Tokenizer {
    /// Load tokenizer from model directory
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| ModelError::TokenizerError(e.to_string()))?;

        // Detect chat format by checking for ChatML tokens
        let has_chatml = inner.token_to_id("<|im_start|>").is_some();

        let (bos_token_id, eos_token_id, eot_token_id, chat_format) = if has_chatml {
            // Qwen 2.5 uses ChatML format
            // Token IDs from Qwen tokenizer
            let bos = inner.token_to_id("<|endoftext|>").unwrap_or(151643);
            let eos = inner.token_to_id("<|endoftext|>").unwrap_or(151643);
            let eot = inner.token_to_id("<|im_end|>").unwrap_or(151645);
            (bos, eos, eot, ChatFormat::ChatML)
        } else {
            // Llama 3.x format
            let bos = inner.token_to_id("<|begin_of_text|>").unwrap_or(128000);
            let eos = inner.token_to_id("<|end_of_text|>").unwrap_or(128001);
            let eot = inner.token_to_id("<|eot_id|>").unwrap_or(128009);
            (bos, eos, eot, ChatFormat::Llama3)
        };

        tracing::info!("Tokenizer loaded: format={:?}, bos={}, eos={}, eot={}",
            chat_format, bos_token_id, eos_token_id, eot_token_id);

        Ok(Self {
            inner,
            bos_token_id,
            eos_token_id,
            eot_token_id,
            pad_token_id: None,
            chat_format,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| ModelError::TokenizerError(e.to_string()))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Encode with chat template
    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let formatted = match self.chat_format {
            ChatFormat::Llama3 => self.format_llama3_chat(messages),
            ChatFormat::ChatML => self.format_chatml_chat(messages),
        };

        let encoding = self
            .inner
            .encode(formatted, true)
            .map_err(|e| ModelError::TokenizerError(e.to_string()))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Format messages using Llama 3.x chat template
    fn format_llama3_chat(&self, messages: &[ChatMessage]) -> String {
        // Llama 3.x chat format:
        // <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        // {system_message}<|eot_id|>
        // <|start_header_id|>user<|end_header_id|>
        // {user_message}<|eot_id|>
        // <|start_header_id|>assistant<|end_header_id|>

        let mut formatted = String::new();

        for msg in messages {
            formatted.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                msg.role, msg.content
            ));
        }

        // Add assistant header for generation
        formatted.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

        formatted
    }

    /// Format messages using ChatML template (Qwen 2.5)
    fn format_chatml_chat(&self, messages: &[ChatMessage]) -> String {
        // ChatML format (Qwen 2.5):
        // <|im_start|>system
        // {system_message}<|im_end|>
        // <|im_start|>user
        // {user_message}<|im_end|>
        // <|im_start|>assistant
        // {assistant_message}<|im_end|>
        // <|im_start|>assistant

        let mut formatted = String::new();

        for msg in messages {
            formatted.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }

        // Add assistant header for generation
        formatted.push_str("<|im_start|>assistant\n");

        formatted
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(tokens, skip_special_tokens)
            .map_err(|e| ModelError::TokenizerError(e.to_string()))
    }

    /// Decode a single token
    pub fn decode_token(&self, token: u32) -> Result<String> {
        self.decode(&[token], false)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Check if token is a special token
    pub fn is_special_token(&self, token_id: u32) -> bool {
        match self.chat_format {
            ChatFormat::Llama3 => token_id >= 128000,
            ChatFormat::ChatML => token_id >= 151643, // Qwen special tokens start here
        }
    }

    /// Check if token is EOS or end-of-turn
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id || token_id == self.eot_token_id
    }

    /// Get the chat format
    pub fn chat_format(&self) -> ChatFormat {
        self.chat_format
    }
}

/// Chat message
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    /// Message content
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message() {
        let msg = ChatMessage::user("Hello!");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello!");
    }
}
