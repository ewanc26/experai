use anyhow::Result;
use candle_core::IndexOp;
use candle_nn::Module;
use crate::data::load_tokenizer;
use crate::model::ModelConfig;
use crate::utils;
use tracing::{info, warn, debug};

/// Run autoregressive text generation from a prompt using a trained model.
pub fn run(
    model: String,
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    tokenizer: String,
) -> Result<()> {
    info!("Starting generation: model={}, prompt={}", model, prompt);
    let tok = load_tokenizer(&tokenizer)?;

    let model_config = ModelConfig::default();
    let device = utils::device_from_env().to_candle()?;
    let mut var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);

    let model_arch = crate::model::build_model(&model_config, vb)?;

    let weights_path = format!("{}/best.safetensors", model);
    if std::path::Path::new(&weights_path).exists() {
        var_map.load(&weights_path)?;
        info!("Loaded weights from {}", weights_path);
    } else {
        warn!(
            "No weights found at {}, using random initialisation",
            weights_path
        );
    }

    let encoding = tok
        .encode(prompt.to_string(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
    let input_ids = encoding.get_ids().to_vec();
    info!("Prompt tokenised to {} tokens", input_ids.len());

    info!(
        "Generating: max_tokens={}, temp={}, top_k={}, top_p={}",
        max_tokens, temperature, top_k, top_p
    );
    let mut generated = input_ids.clone();
    for step in 0..max_tokens {
        let input_tensor =
            candle_core::Tensor::new(generated.clone(), &device)?.unsqueeze(0)?;

        let logits = model_arch.forward(&input_tensor)?;
        let seq_len = logits.dim(1)?;
        let next_logits = logits.i((0, seq_len - 1))?;

        let next_logits = (next_logits / temperature)?;

        let next_token = sample_top_k_top_p(&next_logits, top_k, top_p)?;

        generated.push(next_token as u32);

        if step % 10 == 0 {
            debug!("Generation step {}/{}: token_id={}", step + 1, max_tokens, next_token);
        }

        if next_token == model_config.eos_token_id {
            info!("Generated EOS token at step {}, stopping", step + 1);
            break;
        }
    }

    let output_text = tok
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
    info!("Generated {} tokens", generated.len() - input_ids.len());
    println!("{}", output_text);

    Ok(())
}

/// Sample a token from logits using top-k and/or top-p (nucleus) filtering.
pub fn sample_top_k_top_p(logits: &candle_core::Tensor, top_k: usize, top_p: f64) -> Result<usize> {
    let mut logits_vec = logits.to_vec1::<f32>()?;

    // Top-k: zero out all logits except the k highest, then softmax.
    if top_k > 0 && top_k < logits_vec.len() {
        let mut indices: Vec<usize> = (0..logits_vec.len()).collect();
        indices.sort_by(|&a, &b| {
            logits_vec[b]
                .partial_cmp(&logits_vec[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(top_k);
        let top_k_set: std::collections::HashSet<usize> = indices.iter().copied().collect();
        for (i, v) in logits_vec.iter_mut().enumerate() {
            if !top_k_set.contains(&i) {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Top-p (nucleus): keep the smallest set of tokens whose cumulative probability >= top_p.
    if top_p > 0.0 && top_p < 1.0 {
        // Numerically-stable softmax to compute probabilities.
        let max_val = logits_vec
            .iter()
            .filter(|v| v.is_finite())
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits_vec.iter().map(|v| (v - max_val).exp()).sum();
        let probs: Vec<f32> = logits_vec
            .iter()
            .map(|v| (v - max_val).exp() / exp_sum)
            .collect();

        // Sort descending and accumulate until we cross the threshold.
        let mut sorted_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        sorted_probs
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cum = 0.0_f32;
        let mut cutoff_idx = sorted_probs.len();
        for &(idx, prob) in &sorted_probs {
            cum += prob;
            if cum >= top_p as f32 {
                cutoff_idx = idx;
                break;
            }
        }

        // Mask tokens beyond the nucleus.
        let top_p_set: std::collections::HashSet<usize> = sorted_probs
            .iter()
            .take(cutoff_idx + 1)
            .map(|&(i, _)| i)
            .collect();
        for (i, v) in logits_vec.iter_mut().enumerate() {
            if !top_p_set.contains(&i) {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Final softmax over the filtered logits.
    let max_val = logits_vec
        .iter()
        .filter(|v| v.is_finite())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits_vec.iter().map(|v| (v - max_val).exp()).sum();
    let probs: Vec<f32> = logits_vec
        .iter()
        .map(|v| (v - max_val).exp() / exp_sum)
        .collect();

    // Categorical sampling: draw a uniform random value and walk the CDF.
    let mut rng = rand::thread_rng();
    let r: f32 = rand::Rng::gen_range(&mut rng, 0.0..1.0);
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return Ok(i);
        }
    }
    Ok(probs.len().saturating_sub(1))
}
