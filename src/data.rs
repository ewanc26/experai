use anyhow::{anyhow, Result};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use tokenizers::Tokenizer;
use tracing::{info, warn};

use crate::at_protocol::{extract_text_from_value, ATProtocolClient};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSample {
    pub text: String,
    pub tokens: Vec<u32>,
    pub label: Option<String>,
}

pub struct Dataset {
    pub samples: Vec<DatasetSample>,
    pub tokenizer: Tokenizer,
    pub max_length: usize,
}

impl Dataset {
    pub fn from_jsonl(path: &str, tokenizer: Tokenizer, max_length: usize) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            let parsed: HashMap<String, String> = serde_json::from_str(&line)?;
            let text = parsed
                .get("text")
                .ok_or_else(|| anyhow!("Missing 'text' field in line {}", idx))?;

            let tokens = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow!("Tokenization error: {}", e))?
                .get_ids()
                .iter()
                .map(|&id| id as u32)
                .collect::<Vec<_>>();

            if tokens.len() > max_length {
                warn!(
                    "Sample {} exceeds max length ({} > {})",
                    idx,
                    tokens.len(),
                    max_length
                );
            }

            samples.push(DatasetSample {
                text: text.to_string(),
                tokens,
                label: parsed.get("label").cloned(),
            });
        }

        Ok(Self {
            samples,
            tokenizer,
            max_length,
        })
    }

    pub fn from_csv(path: &str, tokenizer: Tokenizer, max_length: usize) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        let mut csv_reader = csv::Reader::from_reader(reader);
        for result in csv_reader.deserialize() {
            let record: HashMap<String, String> = result?;
            let text = record
                .get("text")
                .ok_or_else(|| anyhow!("Missing 'text' column"))?;

            let tokens = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow!("Tokenization error: {}", e))?
                .get_ids()
                .iter()
                .map(|&id| id as u32)
                .collect::<Vec<_>>();

            samples.push(DatasetSample {
                text: text.to_string(),
                tokens,
                label: record.get("label").cloned(),
            });
        }

        Ok(Self {
            samples,
            tokenizer,
            max_length,
        })
    }

    pub async fn from_at_protocol(
        pds_url: &str,
        handle: &str,
        max_samples: usize,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let max_length = 512;
        let mut samples = Vec::new();

        let client = ATProtocolClient::new(pds_url);

        let did = client.resolve_handle(handle).await?;
        info!(
            "Resolved DID: {} for handle: {} on PDS: {}",
            did, handle, pds_url
        );

        let records = client
            .list_records_paginated(&did, "app.bsky.feed.post", max_samples)
            .await?;
        info!("Found {} records for DID: {}", records.len(), did);

        for (_uri, _cid, value) in records {
            if let Some(text) = extract_text_from_value(&value) {
                let tokens = tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|e| anyhow!("Tokenization error: {}", e))?
                    .get_ids()
                    .iter()
                    .map(|&id| id as u32)
                    .collect::<Vec<_>>();

                samples.push(DatasetSample {
                    text,
                    tokens,
                    label: Some("at_protocol".to_string()),
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

    pub fn split(&self, ratio: f64) -> (Self, Self) {
        let split_idx = (self.samples.len() as f64 * ratio) as usize;
        let train_samples = self.samples[..split_idx].to_vec();
        let val_samples = self.samples[split_idx..].to_vec();

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

pub fn split_train_val(dataset: &Dataset, ratio: f64) -> (Dataset, Dataset) {
    dataset.split(ratio)
}

pub struct DataCollator {
    pub pad_token_id: u32,
    pub max_length: usize,
}

impl DataCollator {
    pub fn new(pad_token_id: u32, max_length: usize) -> Self {
        Self {
            pad_token_id,
            max_length,
        }
    }

    pub fn collate(&self, samples: &[DatasetSample]) -> Result<(Tensor, Tensor)> {
        let batch_size = samples.len();
        let max_len = samples
            .iter()
            .map(|s| s.tokens.len())
            .max()
            .unwrap_or(0)
            .min(self.max_length);

        let mut input_ids = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask = Vec::with_capacity(batch_size * max_len);

        for sample in samples {
            let tokens = &sample.tokens;
            let seq_len = tokens.len().min(max_len);

            for i in 0..max_len {
                if i < seq_len {
                    input_ids.push(tokens[i]);
                    attention_mask.push(1u32);
                } else {
                    input_ids.push(self.pad_token_id);
                    attention_mask.push(0u32);
                }
            }
        }

        let input_tensor =
            Tensor::new(input_ids, &candle_core::Device::Cpu)?.reshape((batch_size, max_len))?;
        let mask_tensor = Tensor::new(attention_mask, &candle_core::Device::Cpu)?
            .reshape((batch_size, max_len))?;

        Ok((input_tensor, mask_tensor))
    }
}

pub fn load_tokenizer(path: &str) -> Result<Tokenizer> {
    let tokenizer = Tokenizer::from_file(path)
        .map_err(|e| anyhow!("Failed to load tokenizer from {}: {}", path, e))?;
    Ok(tokenizer)
}

pub fn validate_dataset(dataset: &Dataset) -> Result<()> {
    let mut empty_count = 0;
    let mut duplicate_count = 0;
    let mut seen = std::collections::HashSet::new();

    for sample in &dataset.samples {
        if sample.tokens.is_empty() {
            empty_count += 1;
        }
        if !seen.insert(sample.text.clone()) {
            duplicate_count += 1;
        }
    }

    if empty_count > 0 {
        warn!("Found {} empty sequences in dataset", empty_count);
    }
    if duplicate_count > 0 {
        warn!("Found {} duplicate examples in dataset", duplicate_count);
    }

    Ok(())
}
