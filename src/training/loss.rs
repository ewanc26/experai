use anyhow::Result;
use candle_core::{Tensor, D};

/// Average cross-entropy loss over all tokens in the batch.
///
/// `logits` has shape `(batch, seq, vocab)` and `labels` has shape `(batch, seq)`
/// with each value in `[0, vocab)`.
pub fn compute_loss(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    let (batch, seq, _vocab) = logits.dims3()?;
    let vocab_size = logits.shape().dims().last().copied().unwrap_or(0);

    let log_probs = candle_nn::ops::log_softmax(logits, D::Minus1)?;

    let labels_flat = labels.reshape((batch * seq, 1))?;
    let log_probs_flat = log_probs.reshape((batch * seq, vocab_size))?;

    // Gather the log-probability assigned to the correct token at each position.
    let nll = log_probs_flat.gather(&labels_flat, 1)?;
    let loss = nll.mean_all()?;

    Ok(loss)
}

/// Convert a mean cross-entropy loss value to perplexity.
pub fn compute_perplexity(loss: f64) -> f64 {
    loss.exp()
}
