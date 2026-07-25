use crate::errors::ExperaiError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use tokenizers::Tokenizer;
use tracing::{info, warn, debug};

use crate::at_protocol::{extract_text_from_value, ATProtocolClient};

/// A single tokenized training sample with optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSample {
    pub text: String,
    pub tokens: Vec<u32>,
    /// Optional classification label (e.g., "at_protocol" for Bluesky posts).
    pub label: Option<String>,
    /// AT Protocol DID of the content author, when sourced from the social graph.
    pub did: Option<String>,
}

/// In-memory dataset holding tokenized samples and the tokenizer used to produce them.
pub struct Dataset {
    pub samples: Vec<DatasetSample>,
    pub tokenizer: Tokenizer,
    /// Maximum token length; longer sequences are truncated during collation.
    pub max_length: usize,
}

impl Dataset {
    /// Load a dataset from a JSONL file. Each line must contain a `"text"` field;
    /// optional `"label"` and `"did"` fields are captured if present.
    pub fn from_jsonl(path: &str, tokenizer: Tokenizer, max_length: usize) -> Result<Self, ExperaiError> {
        info!("Loading dataset from JSONL: {} (max_length={})", path, max_length);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            let parsed: HashMap<String, String> = serde_json::from_str(&line)?;
            let text = parsed
                .get("text")
                .ok_or_else(|| ExperaiError::Data(format!("Missing 'text' field in line {}", idx)))?;

            let encoding = tokenizer
                .encode(text.as_str(), true)
                .map_err(ExperaiError::Tokenization)?;
            let tokens = encoding.get_ids().to_vec();

            samples.push(DatasetSample {
                text: text.to_string(),
                tokens,
                label: parsed.get("label").cloned(),
                did: parsed.get("did").cloned(),
            });
        }

        info!("Loaded {} samples from {}", samples.len(), path);

        Ok(Self {
            samples,
            tokenizer,
            max_length,
        })
    }

    /// Fetch posts from an AT Protocol PDS and build a dataset.
    /// Resolves the handle to a DID, paginates through `app.bsky.feed.post` records,
    /// and tokenizes each post's text.
    pub async fn from_at_protocol(
        pds_url: &str,
        handle: &str,
        max_samples: usize,
        tokenizer: Tokenizer,
    ) -> Result<Self, ExperaiError> {
        let max_length = 512;
        let mut samples = Vec::new();

        let client = ATProtocolClient::new(pds_url);

        let did = client
            .resolve_handle(handle)
            .await
            .map_err(|e| ExperaiError::AtProtocol(format!("failed to resolve handle {handle}: {e}")))?;
        info!(
            "Resolved DID: {} for handle: {} on PDS: {}",
            did, handle, pds_url
        );

        let records = client
            .list_records_paginated(&did, "app.bsky.feed.post", max_samples)
            .await
            .map_err(|e| ExperaiError::AtProtocol(format!("failed to list records for {did}: {e}")))?;
        info!("Found {} records for DID: {}", records.len(), did);

        for (_uri, _cid, value) in records {
            if let Some(text) = extract_text_from_value(&value) {
                let did = value.get("did").and_then(|v| v.as_str()).map(|s| s.to_string());
                let tokens = tokenizer
                    .encode(text.as_str(), true)
                    .map_err(ExperaiError::Tokenization)?
                    .get_ids()
                    .to_vec();

                samples.push(DatasetSample {
                    text,
                    tokens,
                    label: Some("at_protocol".to_string()),
                    did,
                });
            }

            if samples.len() >= max_samples {
                warn!(
                    "Reached maximum sample limit ({}) from AT Protocol",
                    max_samples
                );
                break;
            }
        }

        Ok(Self {
            samples,
            tokenizer,
            max_length,
        })
    }

    /// Split the dataset into (train, validation) by the given ratio.
    /// `ratio` is the fraction of samples assigned to the training set.
    pub fn split(&self, ratio: f64) -> (Self, Self) {
        let split_idx = (self.samples.len() as f64 * ratio) as usize;
        let train_samples = self.samples[..split_idx].to_vec();
        let val_samples = self.samples[split_idx..].to_vec();
        debug!("Dataset split: {} train, {} val (ratio={})", train_samples.len(), val_samples.len(), ratio);

        (
            Self {
                samples: train_samples,
                tokenizer: self.tokenizer.clone(),
                max_length: self.max_length,
            },
            Self {
                samples: val_samples,
                tokenizer: self.tokenizer.clone(),
                max_length: self.max_length,
            },
        )
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Convenience wrapper that splits a dataset into train/val subsets.
pub fn split_train_val(dataset: &Dataset, ratio: f64) -> (Dataset, Dataset) {
    dataset.split(ratio)
}
