use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use tokenizers::{Encoding, Tokenizer};
use tracing::{event, Level};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    pub lowercase: bool,
    pub remove_extra_whitespace: bool,
    pub remove_html_tags: bool,
    pub remove_urls: bool,
    pub remove_emails: bool,
    pub min_length: usize,
    pub max_length: usize,
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
        }
    }
}

pub struct Preprocessor {
    pub config: PreprocessConfig,
    pub tokenizer: Tokenizer,
}

impl Preprocessor {
    pub fn new(tokenizer: Tokenizer, config: Option<PreprocessConfig>) -> Self {
        Self {
            tokenizer,
            config: config.unwrap_or_default(),
        }
    }

    pub fn clean_text(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        if self.config.lowercase {
            cleaned = cleaned.to_lowercase();
        }

        if self.config.remove_html_tags {
            let html_regex = Regex::new(r"<[^>]*>").unwrap();
            cleaned = html_regex.replace_all(&cleaned, "").to_string();
        }

        if self.config.remove_urls {
            let url_regex = Regex::new(r"https?://[^\s]+").unwrap();
            cleaned = url_regex.replace_all(&cleaned, "[URL]").to_string();
        }

        if self.config.remove_emails {
            let email_regex =
                Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap();
            cleaned = email_regex.replace_all(&cleaned, "[EMAIL]").to_string();
        }

        if self.config.remove_extra_whitespace {
            let ws_regex = Regex::new(r"\s+").unwrap();
            cleaned = ws_regex.replace_all(&cleaned, " ").to_string();
            cleaned = cleaned.trim().to_string();
        }

        cleaned
    }

    pub fn preprocess_file(&self, input_path: &str, output_path: &str) -> Result<()> {
        let input_file = File::open(input_path)?;
        let reader = BufReader::new(input_file);
        let mut output_file = File::create(output_path)?;

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            let mut record: serde_json::Value = serde_json::from_str(&line)?;

            if let Some(text_val) = record.get("text") {
                if let Some(text_str) = text_val.as_str() {
                    let cleaned = self.clean_text(text_str);
                    let tokens = self
                        .tokenizer
                        .encode(cleaned.as_str(), true)
                        .map_err(|e| anyhow::anyhow!("{}", e))?;

                    if tokens.get_ids().len() >= self.config.min_length {
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

    pub fn tokenize_batch(&self, texts: &[String]) -> Result<Vec<Encoding>> {
        let mut encodings = Vec::with_capacity(texts.len());

        for text in texts {
            let cleaned = self.clean_text(text);
            let encoding = self
                .tokenizer
                .encode(cleaned.as_str(), true)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            encodings.push(encoding);
        }

        Ok(encodings)
    }
}

pub fn build_preprocess_config(
    lowercase: bool,
    remove_extra_whitespace: bool,
    remove_html_tags: bool,
    remove_urls: bool,
    remove_emails: bool,
    min_length: usize,
    max_length: usize,
) -> PreprocessConfig {
    PreprocessConfig {
        lowercase,
        remove_extra_whitespace,
        remove_html_tags,
        remove_urls,
        remove_emails,
        min_length,
        max_length,
    }
}
