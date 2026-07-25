use crate::errors::ExperaiError;
use tokenizers::Tokenizer;
use tracing::{info, warn};

use super::dataset::Dataset;

/// Load a HuggingFace `tokenizers` JSON file from `path`.
pub fn load_tokenizer(path: &str) -> Result<Tokenizer, ExperaiError> {
    info!("Loading tokenizer from {}", path);
    let tokenizer = Tokenizer::from_file(path).map_err(|e| {
        ExperaiError::ModelLoad(format!("Failed to load tokenizer from {}: {}", path, e))
    })?;
    info!(
        "Tokenizer loaded (vocab_size={})",
        tokenizer.get_vocab_size(true)
    );
    Ok(tokenizer)
}

/// Validate a dataset by checking for empty token sequences and duplicate examples.
/// Warnings are logged for any issues found; the function always returns `Ok`.
pub fn validate_dataset(dataset: &Dataset) -> Result<(), ExperaiError> {
    info!("Validating dataset: {} samples", dataset.len());
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

    if empty_count == 0 && duplicate_count == 0 {
        info!("Dataset validation passed: no issues found");
    }

    Ok(())
}
