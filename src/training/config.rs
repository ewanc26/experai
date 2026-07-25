use serde::{Deserialize, Serialize};

/// Hyperparameters and runtime settings for a training run.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingConfig {
    /// Hugging Face model identifier or local path.
    pub model_name: String,
    /// Path to the training dataset (JSONL or similar).
    pub data_path: String,
    /// Directory for checkpoints and logs.
    pub output_dir: String,
    /// Number of full passes over the dataset.
    pub epochs: usize,
    /// Peak learning rate for the optimizer.
    pub learning_rate: f64,
    /// Number of samples per forward/backward step.
    pub batch_size: usize,
    /// Number of micro-batches accumulated before each optimizer step.
    pub gradient_accumulation_steps: usize,
    /// Maximum token sequence length; shorter sequences are padded.
    pub max_seq_len: usize,
    /// Numeric precision for training (`"bf16"`, `"fp16"`, or `"f32"`).
    pub precision: String,
    /// Linear warmup steps before reaching the peak learning rate.
    pub warmup_steps: usize,
    /// L2 regularization coefficient.
    pub weight_decay: f64,
    /// Gradient clipping threshold by global L2 norm.
    pub max_grad_norm: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// When `true`, dynamically scale batch sizes based on system load.
    pub enable_load_monitoring: bool,
    /// CPU load threshold (0.0–1.0) that triggers batch scaling.
    pub max_cpu_load: f32,
    /// Check system load every N batches when monitoring is enabled.
    pub load_check_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_name: "gpt2".to_string(),
            data_path: "data/train.jsonl".to_string(),
            output_dir: "output".to_string(),
            epochs: 3,
            learning_rate: 5e-5,
            batch_size: 4,
            gradient_accumulation_steps: 8,
            max_seq_len: 512,
            precision: "bf16".to_string(),
            warmup_steps: 500,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            seed: 42,
            enable_load_monitoring: true,
            max_cpu_load: 0.80,
            load_check_interval: 10,
        }
    }
}
