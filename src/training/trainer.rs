use anyhow::Result;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, IndexOp};
use candle_nn::{AdamW, Module, Optimizer, VarBuilder, VarMap};
use std::path::Path;
use tracing::{debug, info, trace};

use crate::data::{DataCollator, Dataset};
use crate::model::{build_model, ModelConfig, TransformerModel};
use crate::utils::{self, GradientAccumulator, SystemLoadConfig, SystemLoadMonitor};

use super::config::TrainingConfig;
use super::loss::{compute_loss, compute_perplexity};

/// Borrow a path as UTF-8, erroring instead of panicking on non-UTF-8 paths.
///
/// `candle`'s `VarMap::save`/`load` take `&str` rather than `AsRef<Path>`, so the
/// conversion has to happen somewhere; doing it here keeps checkpointing from
/// panicking on an unusual `--output-dir`.
fn path_str(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| {
        anyhow::anyhow!(
            "checkpoint path {} is not valid UTF-8; use an ASCII --output-dir",
            path.display()
        )
    })
}

/// The temporary sibling path a file is staged at before being renamed into place.
fn with_tmp_suffix(path: &Path) -> std::path::PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Write `contents` to `path` atomically: stage to a sibling temp file, then rename.
///
/// The rename is atomic because the temp file lives in the same directory, so readers
/// either see the old file or the complete new one — never a partial write.
fn atomic_write(path: &Path, contents: &str) -> Result<()> {
    let tmp = with_tmp_suffix(path);
    std::fs::write(&tmp, contents)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
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
    pub total_steps: usize,
}

impl Trainer {
    pub fn new(config: TrainingConfig, model_config: ModelConfig) -> Result<Self> {
        info!(
            "Creating trainer: lr={}, batch={}, grad_accum={}, epochs={}, precision={}, max_seq={}",
            config.learning_rate, config.batch_size, config.gradient_accumulation_steps, config.epochs, config.precision, config.max_seq_len
        );
        let device = utils::select_optimal_device().device.to_candle()?;
        let var_map = VarMap::new();

        let dtype = Self::precision_dtype_for(&config);
        let vb = VarBuilder::from_varmap(&var_map, dtype, &device);
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
                swap_warn: config.max_swap_usage,
                swap_crit: (config.max_swap_usage + 0.02).min(1.0),
                min_free_mem_mb: config.min_free_mem_mb,
                ..SystemLoadConfig::default()
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
            total_steps: 0,
        })
    }

    fn precision_dtype_for(config: &TrainingConfig) -> DType {
        match config.precision.as_str() {
            "bf16" => DType::BF16,
            "fp16" => DType::F16,
            _ => DType::F32,
        }
    }

    pub fn precision_dtype(&self) -> DType {
        Self::precision_dtype_for(&self.config)
    }

    pub fn update_learning_rate(&mut self, new_lr: f64) -> Result<()> {
        debug!("Updating learning rate: {:.6} -> {:.6}", self.current_lr, new_lr);
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
        debug!("Running validation on {} samples", dataset.len());
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
            let logits = logits.to_dtype(DType::F32)?;
            let shift_logits = logits.i((.., 0..input_ids.shape().dims()[1].saturating_sub(1), ..))?;
            let shift_labels = input_ids.i((.., 1..))?;

            let loss = compute_loss(&shift_logits, &shift_labels)?;
            let loss_val = loss.to_scalar::<f32>()? as f64;
            total_loss += loss_val;
            count += 1;
        }

        let avg_loss = if count > 0 { total_loss / count as f64 } else { 0.0 };
        debug!("Validation complete: avg_loss={:.4} ({} batches)", avg_loss, count);
        Ok(avg_loss)
    }

    fn clip_and_step(
        optimizer: &mut AdamW,
        grads: &mut GradStore,
        vars: &[candle_core::Var],
        max_norm: f64,
    ) -> Result<f64> {
        let mut total_norm_sq = 0.0_f64;
        for var in vars {
            if let Some(g) = grads.get(var.as_tensor()) {
                let norm_sq = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                total_norm_sq += norm_sq;
            }
        }
        let total_norm = total_norm_sq.sqrt();

        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for var in vars {
                if let Some(g) = grads.get(var.as_tensor()).cloned() {
                    grads.insert(var.as_tensor(), g.affine(scale, 0.0)?);
                }
            }
        }

        optimizer.step(grads)?;
        Ok(total_norm)
    }

    pub fn train(&mut self, dataset: &Dataset) -> Result<Vec<f64>> {
        let collator = DataCollator::new(
            self.model.config.pad_token_id as u32,
            self.config.max_seq_len,
        );

        let mut losses = Vec::new();
        let total_steps = self.config.epochs * (dataset.len() / self.config.batch_size).max(1);
        self.total_steps = total_steps;
        let mut grad_accum = GradientAccumulator::new(self.config.gradient_accumulation_steps);
        let accum_steps = self.config.gradient_accumulation_steps;
        let vars = self.var_map.all_vars();
        let fp16_scale: f64 = if self.precision_dtype() == DType::F16 { 1024.0 } else { 1.0 };

        if let Some(ref monitor) = self.load_monitor {
            info!(
                "System load monitoring enabled. Check every {} steps. \
                 Swap avoidance: max_swap={:.0}% min_free={}MB",
                monitor.check_interval(),
                self.config.max_swap_usage * 100.0,
                self.config.min_free_mem_mb,
            );
        }

        for epoch in 0..self.config.epochs {
            info!("Epoch {}/{}", epoch + 1, self.config.epochs);

            let mut epoch_loss = 0.0_f64;
            let mut batch_count = 0_usize;

            let base_batch_size = self.config.batch_size;
            let mut dynamic_batch_size = base_batch_size;

            let mut accum_grads: Option<GradStore> = None;
            let mut accum_count = 0usize;
            let mut accum_loss_sum = 0.0f64;

            let samples = &dataset.samples;
            for chunk in samples.chunks(base_batch_size) {
                if chunk.len() < 2 {
                    continue;
                }

                if let Some(ref mut monitor) = self.load_monitor {
                    if self.global_step > 0 && self.global_step.is_multiple_of(monitor.check_interval()) {
                        let rec = monitor.poll();
                        dynamic_batch_size = ((base_batch_size as f32 * rec.batch_scale) as usize).max(1);

                        let target_lr = self.config.learning_rate * rec.lr_scale as f64;
                        if (target_lr - self.current_lr).abs() > 1e-12 {
                            self.update_learning_rate(target_lr)?;
                        }

                        info!(
                            "[load-adapt] {} | batch {}/{} | lr {:.6}",
                            rec.reason, dynamic_batch_size, base_batch_size, self.current_lr
                        );

                        if rec.should_pause {
                            info!(
                                "[load-adapt] PAUSING 2s — system memory pressure too high, \
                                 avoiding swap"
                            );
                            std::thread::sleep(std::time::Duration::from_secs(2));
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
                let logits = logits.to_dtype(DType::F32)?;

                let shift_logits =
                    logits.i((.., 0..logits.shape().dims()[1].saturating_sub(1), ..))?;
                let shift_labels = input_ids.i((.., 1..))?;

                let loss = compute_loss(&shift_logits, &shift_labels)?;
                let loss_val = loss.to_scalar::<f32>()? as f64;
                epoch_loss += loss_val;
                batch_count += 1;
                self.global_step += 1;
                accum_loss_sum += loss_val;

                let scheduled_lr = self.learning_rate(self.global_step);
                if (scheduled_lr - self.current_lr).abs() > 1e-12 {
                    self.update_learning_rate(scheduled_lr)?;
                    info!("Step {} | Learning rate updated to {:.6}", self.global_step, scheduled_lr);
                }

                let scaled_loss = loss.affine(1.0 / accum_steps as f64, 0.0)?.affine(fp16_scale, 0.0)?;
                let batch_grads = scaled_loss.backward()?;

                if let Some(ref mut acc) = accum_grads {
                    for var in &vars {
                        if let Some(g) = batch_grads.get(var.as_tensor()) {
                            if let Some(existing) = acc.get(var.as_tensor()).cloned() {
                                acc.insert(var.as_tensor(), (existing + g)?);
                            } else {
                                acc.insert(var.as_tensor(), g.clone());
                            }
                        }
                    }
                } else {
                    accum_grads = Some(batch_grads);
                }
                accum_count += 1;

                if grad_accum.step() {
                    if let Some(ref mut acc) = accum_grads {
                        if self.precision_dtype() == DType::F16 {
                            for var in &vars {
                                if let Some(g) = acc.get(var.as_tensor()).cloned() {
                                    acc.insert(var.as_tensor(), g.affine(1.0 / fp16_scale, 0.0)?);
                                }
                            }
                        }

                        let total_norm = Self::clip_and_step(
                            &mut self.optimizer,
                            acc,
                            &vars,
                            self.config.max_grad_norm,
                        )?;

                        info!(
                            "Step {} | Loss: {:.4} | grad_norm: {:.4} | gradient accumulation cycle complete",
                            self.global_step,
                            accum_loss_sum / accum_count as f64,
                            total_norm
                        );
                    }

                    accum_grads = None;
                    accum_count = 0;
                    accum_loss_sum = 0.0;
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

    /// Write a checkpoint (weights, metadata, and model config) to `path`.
    ///
    /// Every file is written to a sibling `.tmp` path and then renamed into place, so
    /// a crash mid-save leaves the previous checkpoint intact rather than a truncated
    /// one. Weights are committed before metadata, so the metadata never advertises a
    /// step whose weights are missing.
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let weights_path = path.with_extension("safetensors");
        let weights_tmp = with_tmp_suffix(&weights_path);
        self.var_map.save(path_str(&weights_tmp)?)?;
        std::fs::rename(&weights_tmp, &weights_path)?;

        let model_config_path = path
            .parent()
            .unwrap_or(Path::new("."))
            .join("model_config.json");
        atomic_write(
            &model_config_path,
            &serde_json::to_string_pretty(&self.model.config)?,
        )?;

        let metadata = serde_json::json!({
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": &self.config,
        });
        atomic_write(
            &path.with_extension("json"),
            &serde_json::to_string_pretty(&metadata)?,
        )?;

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
            self.var_map.load(path_str(&weights_path)?)?;
            info!("Loaded checkpoint weights from {}", weights_path.display());
        }

        Ok(())
    }

    pub fn learning_rate(&self, step: usize) -> f64 {
        let warmup = self.config.warmup_steps;
        let lr = self.config.learning_rate;
        let min_lr = lr * self.config.min_lr_ratio;

        let scheduled = if step < warmup {
            lr * (step as f64 / warmup.max(1) as f64)
        } else if self.total_steps <= warmup {
            lr
        } else {
            let decay_steps = (self.total_steps - warmup) as f64;
            let progress = ((step - warmup) as f64 / decay_steps).min(1.0);
            min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
        };
        trace!("LR schedule: step={}, warmup={}, lr={:.6}", step, warmup, scheduled);
        scheduled
    }
}
