//! Tokenizer wrapper for TinyLlama

use std::path::Path;
use tokenizers::Tokenizer as HFTokenizer;

use crate::error::{ModelError, Result};

/// Wrapper around HuggingFace tokenizer
pub struct Tokenizer {
    inner: HFTokenizer,
}

impl Tokenizer {
    /// Load tokenizer from a tokenizer.json file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HFTokenizer::from_file(path)
            .map_err(|e| ModelError::Safetensors(format!("Tokenizer error: {}", e)))?;
        Ok(Self { inner })
    }

    /// Load tokenizer from a directory containing tokenizer.json
    pub fn from_directory<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let path = dir.as_ref().join("tokenizer.json");
        Self::from_file(path)
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| ModelError::Safetensors(format!("Encoding error: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text with special tokens (like BOS)
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| ModelError::Safetensors(format!("Encoding error: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| ModelError::Safetensors(format!("Decoding error: {}", e)))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get BOS token ID
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner.token_to_id("<s>")
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner.token_to_id("</s>")
    }

    /// Get token for an ID
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests require the actual tokenizer file
    #[test]
    #[ignore]
    fn test_tokenizer_loading() {
        let tokenizer = Tokenizer::from_directory("../../tinyllama-weights").unwrap();
        assert_eq!(tokenizer.vocab_size(), 32000);
    }

    #[test]
    #[ignore]
    fn test_encode_decode() {
        let tokenizer = Tokenizer::from_directory("../../tinyllama-weights").unwrap();
        let text = "Hello, world!";
        let ids = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&ids).unwrap();
        assert!(decoded.contains("Hello"));
    }
}
