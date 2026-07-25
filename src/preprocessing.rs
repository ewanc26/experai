use crate::errors::ExperaiError;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use tokenizers::{Encoding, Tokenizer};
use tracing::{debug, event, info, Level};

// These patterns are compile-time constants, so compilation cannot fail at runtime;
// `expect` documents that rather than silently swallowing a programming error.
static RE_HTML: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<[^>]*>").expect("RE_HTML is a valid regex"));
static RE_URL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"https?://[^\s]+").expect("RE_URL is a valid regex"));
static RE_EMAIL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
        .expect("RE_EMAIL is a valid regex")
});
static RE_WHITESPACE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\s+").expect("RE_WHITESPACE is a valid regex"));

use crate::data::Dataset;

/// Configuration for the text preprocessing pipeline.
///
/// Controls which cleaning steps are applied, length filtering thresholds,
/// and deduplication behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    pub lowercase: bool,
    pub remove_extra_whitespace: bool,
    pub remove_html_tags: bool,
    pub remove_urls: bool,
    pub remove_emails: bool,
    pub min_length: usize,
    pub max_length: usize,
    pub dedupe: bool,
    pub clean: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_extra_whitespace: true,
            remove_html_tags: true,
            remove_urls: true,
            remove_emails: true,
            min_length: 10,
            max_length: 1024,
            dedupe: false,
            clean: false,
        }
    }
}

/// A preprocessing pipeline that applies text cleaning, tokenisation, length
/// filtering, and optional deduplication.
pub struct Preprocessor {
    pub config: PreprocessConfig,
    pub tokenizer: Tokenizer,
}

impl Preprocessor {
    /// Create a new preprocessor with the given tokenizer and optional config.
    pub fn new(tokenizer: Tokenizer, config: Option<PreprocessConfig>) -> Self {
        let config = config.unwrap_or_default();
        info!(
            "Creating preprocessor: clean={}, dedupe={}, min_len={}, max_len={}",
            config.clean, config.dedupe, config.min_length, config.max_length
        );
        Self { tokenizer, config }
    }

    /// Apply the regex-based cleaning pipeline to a single text string.
    ///
    /// Steps (each gated by its `PreprocessConfig` flag, applied in order):
    /// 1. Strip HTML tags → empty string
    /// 2. Replace URLs → `[URL]` placeholder
    /// 3. Replace email addresses → `[EMAIL]` placeholder
    /// 4. Collapse runs of whitespace to a single space and trim
    /// 5. Lowercase the entire string
    pub fn clean_text(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        if self.config.remove_html_tags {
            cleaned = RE_HTML.replace_all(&cleaned, "").to_string();
        }

        if self.config.remove_urls {
            cleaned = RE_URL.replace_all(&cleaned, "[URL]").to_string();
        }

        if self.config.remove_emails {
            cleaned = RE_EMAIL.replace_all(&cleaned, "[EMAIL]").to_string();
        }

        if self.config.remove_extra_whitespace {
            cleaned = RE_WHITESPACE.replace_all(&cleaned, " ").to_string();
            cleaned = cleaned.trim().to_string();
        }

        if self.config.lowercase {
            cleaned = cleaned.to_lowercase();
        }

        cleaned
    }

    /// Remove duplicate texts, preserving the first occurrence of each.
    pub fn dedupe_text(&self, texts: &[String]) -> Vec<String> {
        let mut seen = HashSet::new();
        texts
            .iter()
            .filter(|text| seen.insert((*text).clone()))
            .cloned()
            .collect()
    }

    /// Preprocess a JSONL file: clean, tokenize, filter by length, and
    /// optionally deduplicate each record's `"text"` field.
    ///
    /// Writes enriched JSONL to `output_path` with `"text"` (cleaned) and
    /// `"tokens"` (token IDs) fields added.
    pub fn preprocess_file(&self, input_path: &str, output_path: &str) -> Result<(), ExperaiError> {
        info!("Preprocessing file: {} -> {}", input_path, output_path);
        let input_file = File::open(input_path)?;
        let reader = BufReader::new(input_file);
        let mut output_file = File::create(output_path)?;

        let mut seen_texts = if self.config.dedupe {
            Some(HashSet::new())
        } else {
            None
        };

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            let mut record: serde_json::Value = serde_json::from_str(&line)?;

            if let Some(text_val) = record.get("text") {
                if let Some(text_str) = text_val.as_str() {
                    let cleaned = if self.config.clean {
                        self.clean_text(text_str)
                    } else {
                        text_str.to_string()
                    };

                    let tokens = self
                        .tokenizer
                        .encode(cleaned.as_str(), true)
                        .map_err(ExperaiError::Tokenization)?;

                    // Only keep samples with enough tokens
                    if tokens.get_ids().len() >= self.config.min_length {
                        // Deduplicate by cleaned text content
                        if let Some(seen) = &mut seen_texts {
                            if !seen.insert(cleaned.clone()) {
                                event!(
                                    Level::DEBUG,
                                    "Skipped duplicate sample {} (dedupe enabled)",
                                    idx
                                );
                                continue;
                            }
                        }

                        record["text"] = serde_json::Value::String(cleaned);
                        record["tokens"] = serde_json::Value::Array(
                            tokens
                                .get_ids()
                                .iter()
                                .map(|&id| serde_json::Value::Number(id.into()))
                                .collect(),
                        );
                        output_file
                            .write_all((serde_json::to_string(&record)? + "\n").as_bytes())?;
                    } else {
                        event!(
                            Level::DEBUG,
                            "Filtered sample {}: too short ({} < {})",
                            idx,
                            tokens.get_ids().len(),
                            self.config.min_length
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Save an in-memory [`Dataset`] to a JSONL file.
    ///
    /// Applies optional cleaning and deduplication per sample. Each line
    /// contains `"text"`, `"tokens"`, and `"did"` fields.
    pub fn save_dataset_to_jsonl(
        &self,
        dataset: &Dataset,
        output_path: &str,
    ) -> Result<(), ExperaiError> {
        info!("Saving {} samples to {}", dataset.len(), output_path);
        let mut output_file = File::create(output_path)?;
        let mut seen_texts = if self.config.dedupe {
            Some(HashSet::new())
        } else {
            None
        };

        for sample in &dataset.samples {
            let cleaned = if self.config.clean {
                self.clean_text(&sample.text)
            } else {
                sample.text.clone()
            };

            // Skip duplicate texts when deduplication is enabled
            if let Some(seen) = &mut seen_texts {
                if !seen.insert(cleaned.clone()) {
                    continue;
                }
            }

            let tokens = self
                .tokenizer
                .encode(cleaned.as_str(), true)
                .map_err(ExperaiError::Tokenization)?;

            let token_ids = tokens.get_ids().to_vec();

            let record = serde_json::json!({
                "text": cleaned,
                "tokens": token_ids,
                "did": sample.label,
            });

            output_file.write_all((serde_json::to_string(&record)? + "\n").as_bytes())?;
        }

        Ok(())
    }

    /// Tokenize a batch of text strings, applying the cleaning pipeline first.
    pub fn tokenize_batch(&self, texts: &[String]) -> Result<Vec<Encoding>, ExperaiError> {
        debug!("Tokenizing batch of {} texts", texts.len());
        let mut encodings = Vec::with_capacity(texts.len());

        for text in texts {
            let cleaned = self.clean_text(text);
            let encoding = self
                .tokenizer
                .encode(cleaned.as_str(), true)
                .map_err(ExperaiError::Tokenization)?;
            encodings.push(encoding);
        }

        Ok(encodings)
    }
}

/// Build a [`PreprocessConfig`] from individual flag values.
#[allow(clippy::too_many_arguments)]
pub fn build_preprocess_config(
    lowercase: bool,
    remove_extra_whitespace: bool,
    remove_html_tags: bool,
    remove_urls: bool,
    remove_emails: bool,
    min_length: usize,
    max_length: usize,
    dedupe: bool,
    clean: bool,
) -> PreprocessConfig {
    PreprocessConfig {
        lowercase,
        remove_extra_whitespace,
        remove_html_tags,
        remove_urls,
        remove_emails,
        min_length,
        max_length,
        dedupe,
        clean,
    }
}
