use serde::{Deserialize, Serialize};
use sysinfo::System;
use tracing::info;

use super::device::{detect_gpus, ComputeDevice, GpuKind};

/// Snapshot of the system's compute resources used for auto-tuning recommendations.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu_cores: usize,
    pub total_ram_mb: u64,
    pub gpu_vram_mb: Option<u64>,
    pub gpu_kind: Option<GpuKind>,
    pub gpu_name: Option<String>,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            total_ram_mb: 512,
            gpu_vram_mb: None,
            gpu_kind: None,
            gpu_name: None,
        }
    }
}

impl HardwareProfile {
    /// Probe the system and build a [`HardwareProfile`].
    ///
    /// Two refresh passes with a short sleep ensure accurate CPU core counts.
    pub fn detect() -> Self {
        info!("Detecting hardware profile...");
        let mut system = System::new_all();
        system.refresh_all();
        // Second refresh after a brief pause stabilizes CPU core detection.
        std::thread::sleep(std::time::Duration::from_millis(200));
        system.refresh_all();

        let cpu_cores = system
            .physical_core_count()
            .unwrap_or(system.cpus().len())
            .max(1)
            * 2;
        let total_ram_mb = system.total_memory() / 1024;

        let gpus = detect_gpus();
        let (gpu_vram_mb, gpu_kind, gpu_name) = if let Some(best_gpu) = gpus.first() {
            (
                Some(best_gpu.vram_mb),
                Some(best_gpu.kind),
                Some(best_gpu.name.clone()),
            )
        } else {
            (None, None, None)
        };

        let profile = Self {
            cpu_cores,
            total_ram_mb,
            gpu_vram_mb,
            gpu_kind,
            gpu_name,
        };

        info!(
            "Hardware profile: cpu_cores={}, ram={}MB, gpu={:?}, gpu_vram={}MB, tier={}",
            profile.cpu_cores,
            profile.total_ram_mb,
            profile.gpu_kind,
            profile.gpu_vram_mb.unwrap_or(0),
            profile.memory_tier()
        );

        profile
    }

    /// Returns `true` if any GPU (CUDA or Metal) is detected.
    pub fn has_gpu(&self) -> bool {
        self.gpu_vram_mb.is_some()
    }

    /// Returns `true` if a CUDA GPU is detected.
    pub fn has_cuda(&self) -> bool {
        matches!(self.gpu_kind, Some(GpuKind::Cuda))
    }

    /// Returns `true` if a Metal GPU is detected.
    pub fn has_metal(&self) -> bool {
        matches!(self.gpu_kind, Some(GpuKind::Metal))
    }

    /// Classify system RAM into tiers: `"large"` (>=32 GB), `"medium"` (>=16 GB), `"small"` (>=8 GB), `"tiny"`.
    pub fn memory_tier(&self) -> &'static str {
        if self.total_ram_mb >= 32768 {
            "large"
        } else if self.total_ram_mb >= 16384 {
            "medium"
        } else if self.total_ram_mb >= 8192 {
            "small"
        } else {
            "tiny"
        }
    }

    /// Classify GPU VRAM into tiers: `"large"` (>=24 GB), `"medium"` (>=12 GB), `"small"` (>=6 GB), `"tiny"`, or `"cpu"`.
    pub fn gpu_tier(&self) -> &'static str {
        match self.gpu_vram_mb {
            Some(v) if v >= 24000 => "large",
            Some(v) if v >= 12000 => "medium",
            Some(v) if v >= 6000 => "small",
            Some(_) => "tiny",
            None => "cpu",
        }
    }

    /// Pick the best [`ComputeDevice`] based on detected hardware.
    pub fn get_recommended_device(&self) -> ComputeDevice {
        if self.has_cuda() {
            ComputeDevice::Cuda(0)
        } else if self.has_metal() {
            ComputeDevice::Metal
        } else {
            ComputeDevice::Cpu
        }
    }
}

/// Preset training hyperparameters tuned for a specific hardware tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneConfig {
    pub tier: &'static str,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub precision: String,
    pub max_seq_len: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
}

/// Stateless helper that maps a [`HardwareProfile`] to an [`AutoTuneConfig`].
pub struct AutoTuner;

impl AutoTuner {
    /// Return recommended training config for the given hardware profile.
    pub fn recommend(profile: &HardwareProfile) -> AutoTuneConfig {
        let gpu_vram = profile.gpu_vram_mb.unwrap_or(0);
        let config = if gpu_vram >= 12000 {
            Self::for_gpu("medium")
        } else if gpu_vram >= 6000 {
            Self::for_gpu("small")
        } else if profile.total_ram_mb >= 16384 {
            Self::for_cpu("medium")
        } else if profile.total_ram_mb >= 8192 {
            Self::for_cpu("small")
        } else {
            Self::for_cpu("tiny")
        };

        info!(
            "Auto-tune recommendation: tier={}, batch={}, grad_accum={}, precision={}, lr={}, max_seq={}",
            config.tier, config.batch_size, config.gradient_accumulation_steps, config.precision, config.learning_rate, config.max_seq_len
        );

        config
    }

    fn for_cpu(tier: &'static str) -> AutoTuneConfig {
        match tier {
            "large" => AutoTuneConfig {
                tier,
                batch_size: 64,
                gradient_accumulation_steps: 16,
                precision: "f32".to_string(),
                max_seq_len: 1024,
                learning_rate: 5e-5,
                warmup_steps: 1000,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
            },
            "medium" => AutoTuneConfig {
                tier,
                batch_size: 16,
                gradient_accumulation_steps: 16,
                precision: "f32".to_string(),
                max_seq_len: 512,
                learning_rate: 5e-5,
                warmup_steps: 500,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
            },
            _ => AutoTuneConfig {
                tier,
                batch_size: 4,
                gradient_accumulation_steps: 32,
                precision: "f32".to_string(),
                max_seq_len: 128,
                learning_rate: 1e-5,
                warmup_steps: 200,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
            },
        }
    }

    fn for_gpu(tier: &'static str) -> AutoTuneConfig {
        match tier {
            "large" => AutoTuneConfig {
                tier,
                batch_size: 64,
                gradient_accumulation_steps: 8,
                precision: "bf16".to_string(),
                max_seq_len: 1024,
                learning_rate: 5e-5,
                warmup_steps: 500,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
            },
            _ => AutoTuneConfig {
                tier,
                batch_size: 16,
                gradient_accumulation_steps: 8,
                precision: "f32".to_string(),
                max_seq_len: 512,
                learning_rate: 3e-5,
                warmup_steps: 300,
                weight_decay: 0.01,
                max_grad_norm: 1.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_profile_detect() {
        let profile = HardwareProfile::detect();
        assert!(profile.cpu_cores > 0);
        assert!(profile.total_ram_mb > 0);
        println!("Hardware profile: {:?}", profile);
    }
}
