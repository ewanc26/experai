use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{AdamW, Module, Optimizer, VarBuilder, VarMap};
use std::path::Path;
use tracing::info;

use crate::data::{DataCollator, Dataset};
use crate::model::{build_model, ModelConfig, TransformerModel};
use crate::utils::{self, GradientAccumulator, SystemLoadConfig, SystemLoadMonitor};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TrainingConfig {
    pub model_name: String,
    pub data_path: String,
    pub output_dir: String,
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub max_seq_len: usize,
    pub precision: String,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
    pub seed: u64,
    pub enable_load_monitoring: bool,
    pub max_cpu_load: f32,
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

pub struct Trainer {
    pub config: TrainingConfig,
    pub device: Device,
    pub var_map: VarMap,
    pub optimizer: AdamW,
    pub model: TransformerModel,
    pub best_loss: f64,
    pub global_step: usize,
    pub load_monitor: Option<SystemLoadMonitor>,
    pub current_lr: f64,
}

impl Trainer {
    pub fn new(config: TrainingConfig, model_config: ModelConfig) -> Result<Self> {
        let device = utils::select_optimal_device().device.to_candle()?;
        let var_map = VarMap::new();

        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = build_model(&model_config, vb)?;

        let current_lr = config.learning_rate;
        let optimizer = AdamW::new(
            var_map.all_vars(),
            candle_nn::ParamsAdamW {
                lr: current_lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: config.weight_decay,
            },
        )?;

        let load_monitor = if config.enable_load_monitoring {
            let load_config = SystemLoadConfig {
                max_cpu_load: config.max_cpu_load,
                check_interval_batches: config.load_check_interval,
                reduction_factor: 0.5,
            };
            Some(SystemLoadMonitor::new(load_config))
        } else {
            None
        };

        Ok(Self {
            config,
            device,
            var_map,
            optimizer,
            model,
            best_loss: f64::INFINITY,
            global_step: 0,
            load_monitor,
            current_lr,
        })
    }

    pub fn update_learning_rate(&mut self, new_lr: f64) -> Result<()> {
        self.optimizer = AdamW::new(
            self.var_map.all_vars(),
            candle_nn::ParamsAdamW {
                lr: new_lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: self.config.weight_decay,
            },
        )?;
        self.current_lr = new_lr;
        Ok(())
    }

    pub fn validate(&self, dataset: &Dataset) -> Result<f64> {
        let collator = DataCollator::new(
            self.model.config.pad_token_id as u32,
            self.config.max_seq_len,
        );

        let mut total_loss = 0.0_f64;
        let mut count = 0_usize;

        for chunk in dataset.samples.chunks(self.config.batch_size) {
            if chunk.len() < 2 {
                continue;
            }

            let (input_ids, _attention_mask) = collator.collate(chunk)?;
            let input_ids = input_ids.to_device(&self.device)?;

            let logits = self.model.forward(&input_ids)?;
            let shift_logits = logits.i((.., 0..input_ids.shape().dims()[1].saturating_sub(1), ..))?;
            let shift_labels = input_ids.i((.., 1..))?;

            let loss = self.compute_loss(&shift_logits, &shift_labels)?;
            let loss_val = loss.to_scalar::<f32>()? as f64;
            total_loss += loss_val;
            count += 1;
        }

        let avg_loss = if count > 0 { total_loss / count as f64 } else { 0.0 };
        Ok(avg_loss)
    }

    pub fn train(&mut self, dataset: &Dataset) -> Result<Vec<f64>> {
        let collator = DataCollator::new(
            self.model.config.pad_token_id as u32,
            self.config.max_seq_len,
        );

        let mut losses = Vec::new();
        let total_steps = self.config.epochs * (dataset.len() / self.config.batch_size).max(1);
        let mut grad_accum = GradientAccumulator::new(self.config.gradient_accumulation_steps);

        if let Some(ref mut monitor) = self.load_monitor {
            info!(
                "System load monitoring enabled. Max CPU load: {:.0}%",
                monitor.max_load() * 100.0
            );
        }

        for epoch in 0..self.config.epochs {
            info!("Epoch {}/{}", epoch + 1, self.config.epochs);

            let mut epoch_loss = 0.0_f64;
            let mut batch_count = 0_usize;

            let base_batch_size = self.config.batch_size;
            let mut dynamic_batch_size = base_batch_size;

            let samples = &dataset.samples;
            for chunk in samples.chunks(base_batch_size) {
                if chunk.len() < 2 {
                    continue;
                }

                if let Some(ref mut monitor) = self.load_monitor {
                    if self.global_step.is_multiple_of(self.config.load_check_interval) {
                        let cpu_load = monitor.current_load();
                        let scale = monitor.recommended_batch_scale();
                        dynamic_batch_size = ((base_batch_size as f32 * scale) as usize).max(1);
                        if (dynamic_batch_size as f32 / base_batch_size as f32) < 0.9 {
                            info!(
                                "[load-adapt] CPU load: {:.0}% | batch scaled to {}/{}",
                                cpu_load * 100.0,
                                dynamic_batch_size,
                                base_batch_size
                            );
                        } else {
                            info!(
                                "[load-adapt] CPU load: {:.0}% | batch size: {} (ok)",
                                cpu_load * 100.0,
                                dynamic_batch_size
                            );
                        }
                    }
                }

                let effective_chunk = if chunk.len() > dynamic_batch_size {
                    &chunk[..dynamic_batch_size]
                } else {
                    chunk
                };

                let (input_ids, _attention_mask) = collator.collate(effective_chunk)?;
                let input_ids = input_ids.to_device(&self.device)?;

                let logits = self.model.forward(&input_ids)?;

                let shift_logits =
                    logits.i((.., 0..input_ids.shape().dims()[1].saturating_sub(1), ..))?;
                let shift_labels = input_ids.i((.., 1..))?;

                let loss = self.compute_loss(&shift_logits, &shift_labels)?;
                let loss_val = loss.to_scalar::<f32>()? as f64;
                epoch_loss += loss_val;
                batch_count += 1;
                self.global_step += 1;

                let scheduled_lr = self.learning_rate(self.global_step);
                if (scheduled_lr - self.current_lr).abs() > 1e-12 {
                    self.update_learning_rate(scheduled_lr)?;
                    info!("Step {} | Learning rate updated to {:.6}", self.global_step, scheduled_lr);
                }

                self.optimizer.backward_step(&loss)?;

                if grad_accum.step() {
                    info!(
                        "Step {} | Loss: {:.4} | gradient accumulation cycle complete",
                        self.global_step, loss_val
                    );
                }

                if self.global_step.is_multiple_of(10) {
                    info!("Step {} | Loss: {:.4} | LR: {:.6}", self.global_step, loss_val, self.current_lr);
                }

                if self.global_step >= total_steps {
                    break;
                }
            }

            let avg_loss = if batch_count > 0 {
                epoch_loss / batch_count as f64
            } else {
                0.0
            };
            info!("Epoch {} avg train loss: {:.4}", epoch + 1, avg_loss);

            let val_loss = self.validate(dataset)?;
            let val_ppl = compute_perplexity(val_loss);
            info!("Epoch {} avg val loss: {:.4} | perplexity: {:.4}", epoch + 1, val_loss, val_ppl);
            losses.push(avg_loss);

            if avg_loss < self.best_loss {
                self.best_loss = avg_loss;
                let ckpt_path = format!("{}/best", self.config.output_dir);
                self.save_checkpoint(&ckpt_path)?;
                info!("New best train loss: {:.4}, saved checkpoint", avg_loss);
            }
        }

        Ok(losses)
    }

    fn compute_loss(&self, logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
        let (batch, seq, _vocab) = logits.dims3()?;
        let vocab_size = logits.shape().dims().last().copied().unwrap_or(0);

        let log_probs = candle_nn::ops::log_softmax(logits, D::Minus1)?;

        let labels_flat = labels.reshape((batch * seq, 1))?;
        let log_probs_flat = log_probs.reshape((batch * seq, vocab_size))?;

        let nll = log_probs_flat.gather(&labels_flat, 1)?;
        let loss = nll.mean_all()?;

        Ok(loss)
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let metadata = serde_json::json!({
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": &self.config,
        });

        let metadata_path = path.with_extension("json");
        std::fs::write(metadata_path, serde_json::to_string_pretty(&metadata)?)?;

        // Save model weights
        let weights_path = path.with_extension("safetensors");
        self.var_map.save(weights_path.to_str().unwrap())?;

        info!("Saved checkpoint to {}", path.display());
        Ok(())
    }

    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        let path = Path::new(path);
        let metadata_path = path.with_extension("json");

        if metadata_path.exists() {
            let metadata_str = std::fs::read_to_string(metadata_path)?;
            let metadata: serde_json::Value = serde_json::from_str(&metadata_str)?;

            if let Some(step) = metadata.get("global_step").and_then(|v| v.as_u64()) {
                self.global_step = step as usize;
            }
            if let Some(loss) = metadata.get("best_loss").and_then(|v| v.as_f64()) {
                self.best_loss = loss;
            }

            info!("Loaded checkpoint metadata from step {}", self.global_step);
        }

        let weights_path = path.with_extension("safetensors");
        if weights_path.exists() {
            self.var_map.load(weights_path.to_str().unwrap())?;
            info!("Loaded checkpoint weights from {}", weights_path.display());
        }

        Ok(())
    }

    pub fn learning_rate(&self, step: usize) -> f64 {
        let warmup = self.config.warmup_steps;
        let lr = self.config.learning_rate;

        if step < warmup {
            lr * (step as f64 / warmup.max(1) as f64)
        } else {
            lr
        }
    }
}

pub fn compute_perplexity(loss: f64) -> f64 {
    loss.exp()
}
