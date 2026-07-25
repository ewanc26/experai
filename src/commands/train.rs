use anyhow::Result;
use crate::data::{load_tokenizer, validate_dataset, Dataset};
use crate::model::ModelConfig;
use crate::training::{Trainer, TrainingConfig};
use crate::utils;
use tracing::{info, warn};

/// Run the training workflow: load data, build model, and train.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model: String,
    data: String,
    lr: f64,
    epochs: usize,
    batch_size: usize,
    grad_accum: usize,
    output_dir: String,
    tokenizer: String,
    auto_tune: bool,
    resume: bool,
) -> Result<()> {
    info!(
        "Starting training: model={}, data={}, lr={}, epochs={}",
        model, data, lr, epochs
    );

    let mut final_config = TrainingConfig {
        model_name: model.clone(),
        data_path: data.clone(),
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

    std::fs::create_dir_all(&output_dir)?;

    let config_path = format!("{}/config.json", output_dir);
    std::fs::write(&config_path, serde_json::to_string_pretty(&final_config)?)?;
    info!("Saved training config to {}", config_path);

    let tok = load_tokenizer(&tokenizer)?;
    info!("Loaded tokenizer from {}", tokenizer);

    let dataset = Dataset::from_jsonl(&data, tok, final_config.max_seq_len)?;
    info!("Loaded {} samples from {}", dataset.len(), data);

    validate_dataset(&dataset)?;

    let model_config = ModelConfig::default();
    let mut trainer = Trainer::new(final_config.clone(), model_config)?;
    info!("Trainer initialised on {:?}", trainer.device);

    if resume {
        let ckpt_path = format!("{}/best", output_dir);
        if std::path::Path::new(&format!("{}.safetensors", ckpt_path)).exists() {
            trainer.load_checkpoint(&ckpt_path)?;
            info!("Resumed from checkpoint at step {}", trainer.global_step);
        } else {
            warn!("No checkpoint found at {}, starting fresh", ckpt_path);
        }
    }

    let losses = trainer.train(&dataset)?;
    info!("Training complete. Losses: {:?}", losses);

    Ok(())
}
