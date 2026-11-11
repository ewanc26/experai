//! Universal dataset loading: train on any data.
//!
//! Accepts JSONL, JSON arrays, CSV/TSV, plain text (txt/md/code), whole
//! directories (walked recursively), and stdin (`-`). The format is
//! auto-detected from the file extension and content, or forced with
//! [`LoadOptions::format`].

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use serde_json::Value;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use crate::errors::ExperaiError;

use super::dataset::{Dataset, DatasetSample};

/// Supported input formats for [`load_dataset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataFormat {
    /// Detect from extension/content (default).
    #[default]
    Auto,
    /// One JSON object per line; text taken from `text` (or fallback fields).
    Jsonl,
    /// A JSON array of strings/objects, or an object wrapping such an array.
    Json,
    /// CSV/TSV with a header row.
    Csv,
    /// Plain text, split into paragraph chunks (hard-split at `max_length` tokens).
    Text,
    /// A directory walked recursively; each file is loaded by its own format.
    Dir,
}

impl std::str::FromStr for DataFormat {
    type Err = ExperaiError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "jsonl" | "ndjson" => Ok(Self::Jsonl),
            "json" => Ok(Self::Json),
            "csv" | "tsv" => Ok(Self::Csv),
            "text" | "txt" => Ok(Self::Text),
            "dir" | "directory" => Ok(Self::Dir),
            other => Err(ExperaiError::Data(format!(
                "Unknown data format '{other}' (expected auto|jsonl|json|csv|text|dir)"
            ))),
        }
    }
}

/// Options controlling how raw data is turned into a [`Dataset`].
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub format: DataFormat,
    /// CSV column holding the text (default: `text`, then `content`, `body`,
    /// then the first column).
    pub text_column: Option<String>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            format: DataFormat::Auto,
            text_column: None,
        }
    }
}

/// Field names probed (in order) when extracting text from a JSON object
/// without an explicit `text_column`.
const TEXT_FIELD_CANDIDATES: [&str; 5] = ["text", "content", "body", "document", "value"];

/// Extensions treated as structured data in directory mode.
fn structured_format_for_extension(ext: &str) -> Option<DataFormat> {
    match ext.to_ascii_lowercase().as_str() {
        "jsonl" | "ndjson" => Some(DataFormat::Jsonl),
        "json" => Some(DataFormat::Json),
        "csv" | "tsv" => Some(DataFormat::Csv),
        _ => None,
    }
}

/// Detect the format of a path. `-` (stdin) is always [`DataFormat::Auto`] and
/// resolved later from the piped content.
pub fn detect_format(path: &str) -> DataFormat {
    if path == "-" {
        return DataFormat::Auto;
    }
    let p = Path::new(path);
    if p.is_dir() {
        return DataFormat::Dir;
    }
    match p.extension().and_then(|e| e.to_str()) {
        Some(ext) => structured_format_for_extension(ext).unwrap_or(DataFormat::Text),
        None => DataFormat::Text,
    }
}

/// Extract a text string from a JSON value: a bare string, or an object with a
/// `text`/`content`/`body`/... field (or the explicitly requested column).
fn extract_text(value: &Value, text_column: Option<&str>) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Object(map) => {
            let key = text_column.map(|k| vec![k.to_string()]).unwrap_or_else(|| {
                TEXT_FIELD_CANDIDATES
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            });
            for k in key {
                if let Some(Value::String(s)) = map.get(&k) {
                    return Some(s.clone());
                }
            }
            // Fall back to the first string-valued field.
            map.values().find_map(|v| v.as_str().map(|s| s.to_string()))
        }
        _ => None,
    }
}

/// Extract an optional `label` from a JSON object.
fn extract_label(value: &Value) -> Option<String> {
    value
        .as_object()
        .and_then(|m| m.get("label"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Tokenize a text into a [`DatasetSample`], truncating to `max_length` tokens.
fn make_sample(
    text: String,
    label: Option<String>,
    tokenizer: &Tokenizer,
    max_length: usize,
) -> Result<DatasetSample, ExperaiError> {
    let tokens = tokenizer
        .encode(text.as_str(), true)
        .map_err(ExperaiError::Tokenization)?
        .get_ids()
        .to_vec();
    let tokens = if tokens.len() > max_length {
        tokens[..max_length].to_vec()
    } else {
        tokens
    };
    Ok(DatasetSample {
        text,
        tokens,
        label,
        did: None,
    })
}

/// Load a JSONL stream (one JSON value per line) into samples.
fn load_jsonl_lines(
    lines: impl Iterator<Item = std::io::Result<String>>,
    tokenizer: &Tokenizer,
    max_length: usize,
    options: &LoadOptions,
    source: &str,
) -> Result<Vec<DatasetSample>, ExperaiError> {
    let mut samples = Vec::new();
    for (idx, line) in lines.enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(trimmed).map_err(|e| {
            ExperaiError::Data(format!("Invalid JSON in {source} line {}: {e}", idx + 1))
        })?;
        let Some(text) = extract_text(&value, options.text_column.as_deref()) else {
            return Err(ExperaiError::Data(format!(
                "No text field found in {source} line {}",
                idx + 1
            )));
        };
        samples.push(make_sample(
            text,
            extract_label(&value),
            tokenizer,
            max_length,
        )?);
    }
    Ok(samples)
}

/// Load a whole-file JSON document into samples.
fn load_json_value(
    value: &Value,
    tokenizer: &Tokenizer,
    max_length: usize,
    options: &LoadOptions,
    source: &str,
) -> Result<Vec<DatasetSample>, ExperaiError> {
    let items: Vec<&Value> = match value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(map) => {
            // Common wrapper keys: {"data": [...]}, {"texts": [...]}, ...
            for key in ["data", "texts", "samples", "records", "documents"] {
                if let Some(Value::Array(items)) = map.get(key) {
                    return load_json_value(
                        &Value::Array(items.clone()),
                        tokenizer,
                        max_length,
                        options,
                        source,
                    );
                }
            }
            vec![value]
        }
        _ => {
            return Err(ExperaiError::Data(format!(
                "JSON input {source} must be an array or object"
            )))
        }
    };

    let mut samples = Vec::new();
    for (idx, item) in items.iter().enumerate() {
        let Some(text) = extract_text(item, options.text_column.as_deref()) else {
            return Err(ExperaiError::Data(format!(
                "No text field found in {source} item {}",
                idx + 1
            )));
        };
        samples.push(make_sample(
            text,
            extract_label(item),
            tokenizer,
            max_length,
        )?);
    }
    Ok(samples)
}

/// A minimal RFC-4180-ish CSV parser: handles quoted fields, escaped quotes
/// (`""`), and embedded delimiters/newlines inside quotes.
fn parse_csv(content: &str, delimiter: char) -> Result<Vec<Vec<String>>, ExperaiError> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = content.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
        } else if c == '"' && field.is_empty() {
            in_quotes = true;
        } else if c == delimiter {
            row.push(std::mem::take(&mut field));
        } else if c == '\n' {
            row.push(std::mem::take(&mut field));
            if !(row.len() == 1 && row[0].is_empty()) {
                rows.push(std::mem::take(&mut row));
            }
        } else if c == '\r' {
            // Swallow; the following \n (if any) ends the row.
        } else {
            field.push(c);
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}

/// Load CSV/TSV content into samples, using the header row to find the text column.
fn load_csv_content(
    content: &str,
    delimiter: char,
    tokenizer: &Tokenizer,
    max_length: usize,
    options: &LoadOptions,
    source: &str,
) -> Result<Vec<DatasetSample>, ExperaiError> {
    let rows = parse_csv(content, delimiter)?;
    if rows.is_empty() {
        return Err(ExperaiError::Data(format!("CSV input {source} is empty")));
    }

    let header = &rows[0];
    let col_idx = if let Some(col) = &options.text_column {
        header.iter().position(|h| h == col).ok_or_else(|| {
            ExperaiError::Data(format!(
                "Column '{col}' not found in {source}; available columns: {header:?}"
            ))
        })?
    } else {
        header
            .iter()
            .position(|h| TEXT_FIELD_CANDIDATES.contains(&h.to_ascii_lowercase().as_str()))
            .unwrap_or(0)
    };

    let mut samples = Vec::new();
    for (idx, row) in rows.iter().enumerate().skip(1) {
        let Some(text) = row.get(col_idx) else {
            return Err(ExperaiError::Data(format!(
                "Row {} of {source} has no column {}",
                idx + 1,
                col_idx + 1
            )));
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        samples.push(make_sample(text.to_string(), None, tokenizer, max_length)?);
    }
    Ok(samples)
}

/// Load plain text into samples: split on blank lines (paragraphs), then
/// hard-split any chunk whose token count exceeds `max_length`.
fn load_text_content(
    content: &str,
    tokenizer: &Tokenizer,
    max_length: usize,
    source: &str,
) -> Result<Vec<DatasetSample>, ExperaiError> {
    let mut samples = Vec::new();
    for para in content.split("\n\n") {
        let text = para.trim();
        if text.is_empty() {
            continue;
        }
        let tokens = tokenizer
            .encode(text, true)
            .map_err(ExperaiError::Tokenization)?
            .get_ids()
            .to_vec();
        if tokens.len() <= max_length {
            samples.push(DatasetSample {
                text: text.to_string(),
                tokens,
                label: None,
                did: None,
            });
        } else {
            // Hard-split long documents into max_length-token windows.
            for chunk in tokens.chunks(max_length) {
                let decoded = tokenizer
                    .decode(chunk, true)
                    .map_err(ExperaiError::Tokenization)?;
                samples.push(DatasetSample {
                    text: decoded,
                    tokens: chunk.to_vec(),
                    label: None,
                    did: None,
                });
            }
            debug!(
                "Split a {}-token document from {source} into {} chunks",
                tokens.len(),
                tokens.len().div_ceil(max_length)
            );
        }
    }
    Ok(samples)
}

/// Recursively collect loadable files under a directory, in sorted order for
/// deterministic datasets. Structured extensions keep their format; anything
/// else is treated as text if it is valid UTF-8.
fn collect_dir_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), ExperaiError> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|e| e.path())
        .collect();
    entries.sort();
    for entry in entries {
        let meta = std::fs::metadata(&entry)?;
        if meta.is_dir() {
            collect_dir_files(&entry, out)?;
        } else if meta.is_file() {
            out.push(entry);
        }
    }
    Ok(())
}

/// Sniff the format of piped/inline content when the path is `-`.
fn sniff_content_format(content: &str) -> DataFormat {
    let trimmed = content.trim_start();
    if trimmed.starts_with('[') {
        return DataFormat::Json;
    }
    if trimmed.starts_with('{') {
        return DataFormat::Jsonl;
    }
    // Heuristic: a first line with consistent delimiters and more than one
    // column looks like a header row.
    let mut lines = content.lines();
    if let (Some(first), Some(second)) = (lines.next(), lines.next()) {
        for delim in [',', '\t'] {
            let first_count = first.matches(delim).count();
            let second_count = second.matches(delim).count();
            if first_count > 0 && first_count == second_count {
                return DataFormat::Csv;
            }
        }
    }
    DataFormat::Text
}

/// Load a dataset from any supported source.
///
/// `path` may be a file, a directory, or `-` for stdin. `format` in
/// `options` overrides auto-detection.
pub fn load_dataset(
    path: &str,
    tokenizer: Tokenizer,
    max_length: usize,
    options: &LoadOptions,
) -> Result<Dataset, ExperaiError> {
    let format = match options.format {
        DataFormat::Auto => detect_format(path),
        f => f,
    };
    info!(
        "Loading dataset from {} (format={:?}, max_length={})",
        path, format, max_length
    );

    let samples = match format {
        DataFormat::Auto => unreachable!("auto format resolved above"),
        DataFormat::Dir => {
            let mut files = Vec::new();
            collect_dir_files(Path::new(path), &mut files)?;
            if files.is_empty() {
                return Err(ExperaiError::Data(format!(
                    "Directory {path} contains no files"
                )));
            }
            let mut samples = Vec::new();
            for file in &files {
                let ext = file
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let file_format = structured_format_for_extension(&ext).unwrap_or(DataFormat::Text);
                let content = std::fs::read_to_string(file).map_err(|e| {
                    ExperaiError::Data(format!(
                        "Skipping non-UTF-8 or unreadable file {}: {e}",
                        file.display()
                    ))
                })?;
                let file_samples = load_content(
                    &content,
                    file_format,
                    file.to_str().unwrap_or(path),
                    &tokenizer,
                    max_length,
                    options,
                )?;
                debug!(
                    "Loaded {} samples from {} ({:?})",
                    file_samples.len(),
                    file.display(),
                    file_format
                );
                samples.extend(file_samples);
            }
            samples
        }
        DataFormat::Jsonl | DataFormat::Json | DataFormat::Csv | DataFormat::Text => {
            let content = if path == "-" {
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                buf
            } else {
                std::fs::read_to_string(path)
                    .map_err(|e| ExperaiError::Data(format!("Failed to read {path}: {e}")))?
            };
            let effective = if path == "-" {
                sniff_content_format(&content)
            } else {
                format
            };
            load_content(&content, effective, path, &tokenizer, max_length, options)?
        }
    };

    if samples.is_empty() {
        warn!("Dataset from {path} is empty");
    }
    info!("Loaded {} samples from {}", samples.len(), path);

    Ok(Dataset {
        samples,
        tokenizer,
        max_length,
    })
}

/// Dispatch in-memory content to the right format parser.
fn load_content(
    content: &str,
    format: DataFormat,
    source: &str,
    tokenizer: &Tokenizer,
    max_length: usize,
    options: &LoadOptions,
) -> Result<Vec<DatasetSample>, ExperaiError> {
    match format {
        DataFormat::Jsonl => {
            let cursor = std::io::Cursor::new(content.as_bytes().to_vec());
            load_jsonl_lines(
                BufReader::new(cursor).lines(),
                tokenizer,
                max_length,
                options,
                source,
            )
        }
        DataFormat::Json => {
            let value: Value = serde_json::from_str(content)
                .map_err(|e| ExperaiError::Data(format!("Invalid JSON in {source}: {e}")))?;
            load_json_value(&value, tokenizer, max_length, options, source)
        }
        DataFormat::Csv => {
            let delimiter = if source.ends_with(".tsv") { '\t' } else { ',' };
            load_csv_content(content, delimiter, tokenizer, max_length, options, source)
        }
        DataFormat::Text => load_text_content(content, tokenizer, max_length, source),
        DataFormat::Auto | DataFormat::Dir => unreachable!("resolved before dispatch"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Write;

    /// Build a small word-level-ish tokenizer: BPE with an empty vocab encodes
    /// unknown text as no tokens, so instead we use a WordLevel tokenizer over
    /// a tiny vocab for deterministic token counts.
    fn test_tokenizer() -> Tokenizer {
        use tokenizers::models::wordlevel::WordLevel;
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (i, w) in [
            "hello", "world", "the", "cat", "sat", "on", "mat", "a", "b", "c",
        ]
        .iter()
        .enumerate()
        {
            vocab.insert(w.to_string(), i as u32);
        }
        vocab.insert("[UNK]".to_string(), 100);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(tokenizers::pre_tokenizers::whitespace::Whitespace);
        tok
    }

    fn write_tmp(dir: &tempfile::TempDir, name: &str, content: &str) -> String {
        let p = dir.path().join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        p.to_string_lossy().to_string()
    }

    #[test]
    fn test_detect_format_by_extension() {
        assert_eq!(detect_format("data.jsonl"), DataFormat::Jsonl);
        assert_eq!(detect_format("data.ndjson"), DataFormat::Jsonl);
        assert_eq!(detect_format("data.json"), DataFormat::Json);
        assert_eq!(detect_format("data.csv"), DataFormat::Csv);
        assert_eq!(detect_format("data.tsv"), DataFormat::Csv);
        assert_eq!(detect_format("notes.txt"), DataFormat::Text);
        assert_eq!(detect_format("README.md"), DataFormat::Text);
        assert_eq!(detect_format("noext"), DataFormat::Text);
        assert_eq!(detect_format("-"), DataFormat::Auto);
    }

    #[test]
    fn test_from_str_format() {
        assert_eq!("jsonl".parse::<DataFormat>().unwrap(), DataFormat::Jsonl);
        assert_eq!("CSV".parse::<DataFormat>().unwrap(), DataFormat::Csv);
        assert!("bogus".parse::<DataFormat>().is_err());
    }

    #[test]
    fn test_load_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_tmp(
            &dir,
            "data.jsonl",
            "{\"text\": \"hello world\"}\n{\"text\": \"the cat sat\", \"label\": \"animals\"}\n\n",
        );
        let ds = load_dataset(&path, test_tokenizer(), 512, &LoadOptions::default()).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.samples[0].text, "hello world");
        assert_eq!(ds.samples[1].label.as_deref(), Some("animals"));
    }

    #[test]
    fn test_load_jsonl_missing_text_field_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_tmp(&dir, "bad.jsonl", "{\"foo\": 1}\n");
        let err = load_dataset(&path, test_tokenizer(), 512, &LoadOptions::default());
        assert!(err.is_err());
    }

    #[test]
    fn test_load_json_array_of_strings_and_objects() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_tmp(
            &dir,
            "data.json",
            "[\"hello world\", {\"content\": \"the cat sat\"}]",
        );
        let ds = load_dataset(&path, test_tokenizer(), 512, &LoadOptions::default()).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.samples[1].text, "the cat sat");
    }

    #[test]
    fn test_load_json_wrapped_array() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_tmp(&dir, "data.json", "{\"data\": [{\"text\": \"hello\"}]}");
        let ds = load_dataset(&path, test_tokenizer(), 512, &LoadOptions::default()).unwrap();
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.samples[0].text, "hello");
    }

    #[test]
    fn test_load_csv_default_text_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_tmp(
            &dir,
            "data.csv",
            "id,text,label\n1,hello world,x\n2,\"the cat, sat\",y\n",
        );
        let ds = load_dataset(&path, test_tokenizer(), 512, &LoadOptions::default()).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.samples[1].text, "the cat, sat");
    }

    #[test]
    fn test_load_csv_custom_column_and_tsv() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_tmp(&dir, "data.tsv", "name\tbody\nalice\thello world\n");
        let opts = LoadOptions {
            format: DataFormat::Auto,
            text_column: Some("body".to_string()),
        };
        let ds = load_dataset(&path, test_tokenizer(), 512, &opts).unwrap();
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.samples[0].text, "hello world");
    }

    #[test]
    fn test_load_csv_missing_column_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_tmp(&dir, "data.csv", "a,b\n1,2\n");
        let opts = LoadOptions {
            format: DataFormat::Auto,
            text_column: Some("nope".to_string()),
        };
        assert!(load_dataset(&path, test_tokenizer(), 512, &opts).is_err());
    }

    #[test]
    fn test_load_text_paragraphs_and_long_chunks() {
        let dir = tempfile::tempdir().unwrap();
        // max_length=2 forces the 4-token paragraph to split into 2 chunks.
        let path = write_tmp(&dir, "doc.txt", "hello world\n\nthe cat sat on\n\n\na b\n");
        let ds = load_dataset(&path, test_tokenizer(), 2, &LoadOptions::default()).unwrap();
        assert_eq!(ds.len(), 4); // "hello world" + 2 chunks of "the cat sat on" + "a b"
        assert!(ds.samples.iter().all(|s| s.tokens.len() <= 2));
    }

    #[test]
    fn test_load_directory_recursively() {
        let dir = tempfile::tempdir().unwrap();
        write_tmp(&dir, "a.jsonl", "{\"text\": \"hello world\"}\n");
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        let mut f = File::create(sub.join("b.txt")).unwrap();
        f.write_all(b"the cat sat\n").unwrap();
        let ds = load_dataset(
            dir.path().to_str().unwrap(),
            test_tokenizer(),
            512,
            &LoadOptions::default(),
        )
        .unwrap();
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_forced_format_overrides_extension() {
        let dir = tempfile::tempdir().unwrap();
        // A .txt file forced to be parsed as JSONL.
        let path = write_tmp(&dir, "data.txt", "{\"text\": \"hello world\"}\n");
        let opts = LoadOptions {
            format: DataFormat::Jsonl,
            text_column: None,
        };
        let ds = load_dataset(&path, test_tokenizer(), 512, &opts).unwrap();
        assert_eq!(ds.len(), 1);
    }

    #[test]
    fn test_sniff_content_format() {
        assert_eq!(
            sniff_content_format("[{\"text\": \"hi\"}]"),
            DataFormat::Json
        );
        assert_eq!(
            sniff_content_format("{\"text\": \"hi\"}\n{\"text\": \"yo\"}"),
            DataFormat::Jsonl
        );
        assert_eq!(sniff_content_format("a,b\n1,2\n"), DataFormat::Csv);
        assert_eq!(
            sniff_content_format("just some prose\nover lines\n"),
            DataFormat::Text
        );
    }
}
