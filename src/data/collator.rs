use crate::errors::ExperaiError;
use candle_core::Tensor;
use tracing::trace;

use super::dataset::DatasetSample;

/// Pads and batches tokenized samples into uniform-length tensors.
pub struct DataCollator {
    pub pad_token_id: u32,
    pub max_length: usize,
}

impl DataCollator {
    /// Create a collator with the given pad token ID and maximum sequence length.
    pub fn new(pad_token_id: u32, max_length: usize) -> Self {
        Self {
            pad_token_id,
            max_length,
        }
    }

    /// Collate samples into `(input_ids, attention_mask)` tensors of shape
    /// `(batch_size, max_len)`. Sequences shorter than `max_len` are right-padded;
    /// longer sequences are truncated to `max_length`.
    pub fn collate(&self, samples: &[DatasetSample]) -> Result<(Tensor, Tensor), ExperaiError> {
        let batch_size = samples.len();
        // Use the longest sequence in the batch, capped by max_length
        let max_len = samples
            .iter()
            .map(|s| s.tokens.len())
            .max()
            .unwrap_or(0)
            .min(self.max_length);

        trace!("Collating {} samples, max_len={}", batch_size, max_len);

        let mut input_ids = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask = Vec::with_capacity(batch_size * max_len);

        for sample in samples {
            let tokens = &sample.tokens;
            let seq_len = tokens.len().min(max_len);

            // Copy real tokens with mask=1
            for &token in tokens.iter().take(seq_len) {
                input_ids.push(token);
                attention_mask.push(1u32);
            }
            // Pad to max_len with mask=0
            for _ in seq_len..max_len {
                input_ids.push(self.pad_token_id);
                attention_mask.push(0u32);
            }
        }

        let input_tensor =
            Tensor::new(input_ids, &candle_core::Device::Cpu)?.reshape((batch_size, max_len))?;
        let mask_tensor = Tensor::new(attention_mask, &candle_core::Device::Cpu)?
            .reshape((batch_size, max_len))?;

        Ok((input_tensor, mask_tensor))
    }
}
