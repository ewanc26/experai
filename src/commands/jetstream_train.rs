use anyhow::Result;
use crate::data::load_tokenizer;
use crate::jetstream::JetstreamConfig;
use crate::model::ModelConfig;
use crate::training::{Trainer, TrainingConfig};
use crate::utils;
use tracing::info;

/// Run the end-to-end pipeline: collect posts from Jetstream, save them, and train.
#[allow(clippy::too_many_arguments)]
pub fn run(
    jetstream_host: String,
    collections: Vec<String>,
    dids: Vec<String>,
    max_samples: usize,
    max_duration_secs: u64,
    output_dir: String,
    model_name: String,
    lr: f64,
    epochs: usize,
    batch_size: usize,
    grad_accum: usize,
    tokenizer: String,
    auto_tune: bool,
) -> Result<()> {
    info!(
        "Starting Jetstream training: host={}, collections={:?}, max_samples={}",
        jetstream_host, collections, max_samples
    );

    let tok = load_tokenizer(&tokenizer)?;

    let jetstream_config = JetstreamConfig {
        host: jetstream_host.clone(),
        collections: collections.clone(),
        dids: dids.clone(),
        max_samples,
        batch_size: 100,
        max_duration_secs,
        compression: false,
    };

    let rt = tokio::runtime::Runtime::new()?;
    let (dataset, stats) = rt.block_on(crate::jetstream::collect_from_jetstream(
        &jetstream_config,
        tok,
    ))?;

    info!(
        "Collected {} posts from Jetstream ({:.1}s, {:.1} posts/s)",
        stats.valid_posts, stats.duration_secs, stats.posts_per_second
    );

    std::fs::create_dir_all(&output_dir)?;
    let data_path = format!("{}/jetstream_data.jsonl", output_dir);
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

    let mut final_config = TrainingConfig {
        model_name: model_name.clone(),
        data_path: data_path.clone(),
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
        min_lr_ratio: 0.1,
    };

    if auto_tune {
        let profile = utils::HardwareProfile::detect();
        let tuned = utils::AutoTuner::recommend(&profile);
        info!("Auto-tune: {:?}", tuned);

        final_config.batch_size = tuned.batch_size;
        final_config.gradient_accumulation_steps = tuned.gradient_accumulation_steps;
        final_config.precision = tuned.precision;
        final_config.max_seq_len = tuned.max_seq_len;
        final_config.learning_rate = tuned.learning_rate;
        final_config.warmup_steps = tuned.warmup_steps;
        final_config.weight_decay = tuned.weight_decay;
        final_config.max_grad_norm = tuned.max_grad_norm;
    }

    let config_path = format!("{}/config.json", output_dir);
    std::fs::write(&config_path, serde_json::to_string_pretty(&final_config)?)?;
    info!("Saved training config to {}", config_path);

    let model_config = ModelConfig::default();
    let mut trainer = Trainer::new(final_config, model_config)?;
    info!("Trainer initialised on {:?}", trainer.device);

    let losses = trainer.train(&dataset)?;
    info!("Jetstream training complete. Losses: {:?}", losses);

    Ok(())
}
