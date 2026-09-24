//! Thin wrapper over the Hugging Face `tokenizers`` crate.

use anyhow::{Result, bail};
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub add_bos_token: bool,
    #[serde(default)]
    pub chat_template: Option<String>,
}

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    settings: Settings,
}

impl Tokenizer {
    pub fn from_bytes(json: &[u8], settings: Settings) -> Result<Self> {
        if settings.add_bos_token && settings.bos_token_id.is_none() {
            bail!("add_bos_token is set but bos_token_id is null");
        }
        let inner = tokenizers::Tokenizer::from_bytes(json).map_err(anyhow::Error::from_boxed)?;
        Ok(Tokenizer { inner, settings })
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Encodes a raw prompt.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = self.encode_chat(text)?;
        if self.settings.add_bos_token {
            ids.insert(
                0,
                self.settings.bos_token_id.expect("checked in from_bytes"),
            );
        }
        Ok(ids)
    }

    /// Encodes a rendered chat prompt.
    pub fn encode_chat(&self, text: &str) -> Result<Vec<u32>> {
        // `false`: the file's own post-processor must not add BOS or EOS.
        // The manifest's add_bos_token is the only mechanism.
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(anyhow::Error::from_boxed)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes to text for display. Special tokens are dropped.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(anyhow::Error::from_boxed)
    }

    /// Incremental decoding for streaming output. A multi-byte character split
    /// across tokens is held back until it is complete.
    pub fn decode_stream(&self) -> DecodeStream<'_> {
        DecodeStream(self.inner.decode_stream(true))
    }

    /// Whether generation should stop at `id`.
    pub fn is_eos(&self, id: u32) -> bool {
        self.settings.eos_token_ids.contains(&id)
    }
}

pub struct DecodeStream<'a>(
    tokenizers::DecodeStream<
        'a,
        tokenizers::ModelWrapper,
        tokenizers::NormalizerWrapper,
        tokenizers::PreTokenizerWrapper,
        tokenizers::PostProcessorWrapper,
        tokenizers::DecoderWrapper,
    >,
);

impl DecodeStream<'_> {
    /// Feeds one id. Returns the text it completed, if any.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.0.step(id).map_err(anyhow::Error::from_boxed)
    }
}
