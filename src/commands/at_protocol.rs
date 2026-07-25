use anyhow::Result;
use crate::at_protocol::ATProtocolConfig;
use crate::data::{load_tokenizer, Dataset};
use crate::model::ModelConfig;
use crate::training::{Trainer, TrainingConfig};
use crate::utils;
use tracing::info;

/// Run the end-to-end pipeline: fetch posts from an AT Protocol PDS, save them, and train.
#[allow(clippy::too_many_arguments)]
pub fn run(
    pds_url: String,
    handle: String,
    max_samples: usize,
    output_dir: String,
    model_name: String,
    lr: f64,
    epochs: usize,
    batch_size: usize,
    grad_accum: usize,
    tokenizer: String,
) -> Result<()> {
    info!(
        "Starting AT Protocol dataset loading from PDS: {} handle: {} max_samples: {}",
        pds_url, handle, max_samples
    );

    let at_config = ATProtocolConfig {
        pds_url: pds_url.clone(),
        handle: handle.clone(),
        max_samples,
    };

    let tok = load_tokenizer(&tokenizer)?;

    let rt = tokio::runtime::Runtime::new()?;
    let dataset = rt.block_on(Dataset::from_at_protocol(
        &at_config.pds_url,
        &at_config.handle,
        at_config.max_samples,
        tok,
    ))?;
    info!("Loaded {} samples from AT Protocol", dataset.len());

    std::fs::create_dir_all(&output_dir)?;
    let data_path = format!("{}/at_protocol_data.jsonl", output_dir);
    let mut file = std::fs::File::create(&data_path)?;
    for sample in &dataset.samples {
        use std::io::Write;
        let json = serde_json::to_string(&serde_json::json!({
            "text": sample.text,
            "label": sample.label,
        }))?;
        writeln!(file, "{}", json)?;
    }
    info!("Saved {} samples to {}", dataset.len(), data_path);

    let mut final_at_config = TrainingConfig {
        model_name: model_name.clone(),
        data_path,
        output_dir: output_dir.clone(),
        epochs,
        learning_rate: lr,
        batch_size,
        gradient_accumulation_steps: grad_accum,
        max_seq_len: 512,
        precision: "bf16".to_string(),
        warmup_steps: 500,
        weight_decay: 0.01,
        max_grad_norm: 1.0,
        seed: 42,
        enable_load_monitoring: true,
        max_cpu_load: 0.80,
        load_check_interval: 10,
        max_swap_usage: 0.0,
        min_free_mem_mb: 2048,
    };

    let profile = utils::HardwareProfile::detect();
    let tuned = utils::AutoTuner::recommend(&profile);
    final_at_config.batch_size = tuned.batch_size;
    final_at_config.gradient_accumulation_steps = tuned.gradient_accumulation_steps;
    final_at_config.precision = tuned.precision;
    final_at_config.max_seq_len = tuned.max_seq_len;

    let model_config = ModelConfig::default();
    let mut trainer = Trainer::new(final_at_config, model_config)?;
    info!("Trainer initialised on {:?}", trainer.device);

    let losses = trainer.train(&dataset)?;
    info!("AT Protocol training complete. Losses: {:?}", losses);

    Ok(())
}
