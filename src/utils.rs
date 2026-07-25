use anyhow::Result;
use candle_core::Tensor;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use sysinfo::{CpuRefreshKind, RefreshKind, System};
use tracing::{event, Level};

/// Device selection for compute. Maps to candle_core::Device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDevice {
    Cpu,
    Metal,
    Cuda(usize),
}

impl ComputeDevice {
    pub fn as_str(&self) -> &'static str {
        match self {
            ComputeDevice::Cpu => "cpu",
            ComputeDevice::Metal => "metal",
            ComputeDevice::Cuda(_) => "cuda",
        }
    }

    pub fn to_candle(&self) -> Result<candle_core::Device> {
        match self {
            ComputeDevice::Cpu => Ok(candle_core::Device::Cpu),
            ComputeDevice::Metal => {
                #[cfg(feature = "metal")]
                {
                    Ok(candle_core::Device::new_metal(0)?)
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(anyhow::anyhow!(
                        "Metal support not compiled in. Rebuild with --features metal"
                    ))
                }
            }
            ComputeDevice::Cuda(idx) => {
                #[cfg(feature = "cuda")]
                {
                    Ok(candle_core::Device::new_cuda(*idx)?)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = idx;
                    Err(anyhow::anyhow!(
                        "CUDA support not compiled in. Rebuild with --features cuda"
                    ))
                }
            }
        }
    }
}

/// Device selection strategy for auto-detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSelectionStrategy {
    /// Auto-detect best available device (CUDA > Metal > CPU)
    Auto,
    /// Prefer CUDA if available, otherwise fall back
    PreferCuda,
    /// Prefer Metal if available, otherwise fall back
    PreferMetal,
    /// Force CPU
    ForceCpu,
}

/// Detailed GPU information for device ranking
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub kind: GpuKind,
    pub index: usize,
    pub vram_mb: u64,
    pub name: String,
    pub compute_capability: Option<(i32, i32)>,
}

/// Type of GPU available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKind {
    Cuda,
    Metal,
}

/// Selection result containing chosen device and reasoning
#[derive(Debug, Clone)]
pub struct DeviceSelection {
    pub device: ComputeDevice,
    pub strategy_used: DeviceSelectionStrategy,
    pub gpu_detected: bool,
    pub fallback_reason: Option<String>,
}

pub fn init_tracing(log_level: Option<String>, test_mode: bool) -> Result<()> {
    crate::logging::init_logger(log_level, test_mode)
}

pub fn set_global_seed(seed: u64) {
    let _ = rand::rngs::StdRng::seed_from_u64(seed);
}

pub fn device_from_env() -> ComputeDevice {
    match std::env::var("EXPERAI_DEVICE") {
        Ok(v) => match v.as_str() {
            x if x.starts_with("cuda:") => {
                let idx: usize = x[5..].parse().unwrap_or(0);
                ComputeDevice::Cuda(idx)
            }
            "cuda" => ComputeDevice::Cuda(0),
            "metal" => ComputeDevice::Metal,
            _ => ComputeDevice::Cpu,
        },
        Err(_) => ComputeDevice::Cpu,
    }
}

/// Detect all available GPU devices on the system.
/// Returns a list of GPUs sorted by preference (CUDA with higher VRAM first, then Metal).
pub fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus: Vec<GpuInfo> = Vec::new();

    // Detect CUDA devices
    #[cfg(feature = "cuda")]
    {
        let num_devices = candle_core::Device::cuda_device_count().unwrap_or(0);
        for idx in 0..num_devices {
            if let Ok(device) = candle_core::Device::new_cuda(idx) {
                if let Ok(info) = device.cuda_device_info() {
                    let vram_mb = info.total_memory as u64 / 1024 / 1024;
                    let name = info.name.clone();
                    let compute_capability = Some((info.major as i32, info.minor as i32));
                    
                    gpus.push(GpuInfo {
                        kind: GpuKind::Cuda,
                        index: idx,
                        vram_mb,
                        name,
                        compute_capability,
                    });
                }
            }
        }
    }

    // Detect Metal devices
    #[cfg(feature = "metal")]
    {
        // Metal on macOS typically has one GPU; try to create device
        if let Ok(_device) = candle_core::Device::new_metal(0) {
            // Metal doesn't expose VRAM info easily in candle; estimate or use placeholder
            // We'll use a reasonable default and note it's Metal
            gpus.push(GpuInfo {
                kind: GpuKind::Metal,
                index: 0,
                vram_mb: 8192, // Placeholder; actual unified memory
                name: "Apple Silicon GPU".to_string(),
                compute_capability: None,
            });
        }
    }

    // Sort: CUDA first (by VRAM descending), then Metal
    gpus.sort_by(|a, b| {
        match (a.kind, b.kind) {
            (GpuKind::Cuda, GpuKind::Metal) => std::cmp::Ordering::Less,
            (GpuKind::Metal, GpuKind::Cuda) => std::cmp::Ordering::Greater,
            (GpuKind::Cuda, GpuKind::Cuda) => b.vram_mb.cmp(&a.vram_mb),
            (GpuKind::Metal, GpuKind::Metal) => a.index.cmp(&b.index),
        }
    });

    gpus
}

/// Auto-select the optimal computation device based on available hardware.
/// 
/// Priority order:
/// 1. CUDA GPU with highest VRAM (if CUDA feature enabled)
/// 2. Metal GPU (if Metal feature enabled)
/// 3. CPU (fallback)
///
/// The selection can be influenced by environment variables:
/// - `EXPERAI_DEVICE`: Override with explicit device (cuda, cuda:N, metal, cpu)
/// - `EXPERAI_DEVICE_STRATEGY`: Selection strategy (auto, prefer_cuda, prefer_metal, force_cpu)
pub fn select_optimal_device() -> DeviceSelection {
    // Check for explicit override first
    if let Ok(env_device) = std::env::var("EXPERAI_DEVICE") {
        let device = match env_device.as_str() {
            x if x.starts_with("cuda:") => {
                let idx = x[5..].parse().unwrap_or(0);
                ComputeDevice::Cuda(idx)
            }
            "cuda" => ComputeDevice::Cuda(0),
            "metal" => ComputeDevice::Metal,
            _ => ComputeDevice::Cpu,
        };
        return DeviceSelection {
            device,
            strategy_used: DeviceSelectionStrategy::Auto,
            gpu_detected: matches!(device, ComputeDevice::Cuda(_) | ComputeDevice::Metal),
            fallback_reason: Some(format!("Explicit EXPERAI_DEVICE={}", env_device)),
        };
    }

    // Check for strategy override
    let strategy = std::env::var("EXPERAI_DEVICE_STRATEGY")
        .ok()
        .and_then(|s| match s.as_str() {
            "auto" => Some(DeviceSelectionStrategy::Auto),
            "prefer_cuda" => Some(DeviceSelectionStrategy::PreferCuda),
            "prefer_metal" => Some(DeviceSelectionStrategy::PreferMetal),
            "force_cpu" => Some(DeviceSelectionStrategy::ForceCpu),
            _ => None,
        })
        .unwrap_or(DeviceSelectionStrategy::Auto);

    let gpus = detect_gpus();
    
    let result = match strategy {
        DeviceSelectionStrategy::ForceCpu => DeviceSelection {
            device: ComputeDevice::Cpu,
            strategy_used: strategy,
            gpu_detected: false,
            fallback_reason: Some("Forced CPU via EXPERAI_DEVICE_STRATEGY=force_cpu".to_string()),
        },
        DeviceSelectionStrategy::PreferCuda => {
            if let Some(gpu) = gpus.iter().find(|g| g.kind == GpuKind::Cuda) {
                DeviceSelection {
                    device: ComputeDevice::Cuda(gpu.index),
                    strategy_used: strategy,
                    gpu_detected: true,
                    fallback_reason: None,
                }
            } else if cfg!(feature = "metal") && gpus.iter().any(|g| g.kind == GpuKind::Metal) {
                // Fallback to Metal if CUDA not available
                let _gpu = gpus.iter().find(|g| g.kind == GpuKind::Metal).unwrap();
                DeviceSelection {
                    device: ComputeDevice::Metal,
                    strategy_used: strategy,
                    gpu_detected: true,
                    fallback_reason: Some("CUDA not available, falling back to Metal".to_string()),
                }
            } else {
                DeviceSelection {
                    device: ComputeDevice::Cpu,
                    strategy_used: strategy,
                    gpu_detected: false,
                    fallback_reason: Some("No CUDA or Metal GPU available".to_string()),
                }
            }
        },
        DeviceSelectionStrategy::PreferMetal => {
            if let Some(_gpu) = gpus.iter().find(|g| g.kind == GpuKind::Metal) {
                DeviceSelection {
                    device: ComputeDevice::Metal,
                    strategy_used: strategy,
                    gpu_detected: true,
                    fallback_reason: None,
                }
            } else if cfg!(feature = "cuda") && gpus.iter().any(|g| g.kind == GpuKind::Cuda) {
                let gpu = gpus.iter().find(|g| g.kind == GpuKind::Cuda).unwrap();
                DeviceSelection {
                    device: ComputeDevice::Cuda(gpu.index),
                    strategy_used: strategy,
                    gpu_detected: true,
                    fallback_reason: Some("Metal not available, falling back to CUDA".to_string()),
                }
            } else {
                DeviceSelection {
                    device: ComputeDevice::Cpu,
                    strategy_used: strategy,
                    gpu_detected: false,
                    fallback_reason: Some("No Metal or CUDA GPU available".to_string()),
                }
            }
        },
        DeviceSelectionStrategy::Auto => {
            // Default priority: CUDA > Metal > CPU
            if let Some(gpu) = gpus.iter().find(|g| g.kind == GpuKind::Cuda) {
                DeviceSelection {
                    device: ComputeDevice::Cuda(gpu.index),
                    strategy_used: strategy,
                    gpu_detected: true,
                    fallback_reason: None,
                }
            } else if let Some(_gpu) = gpus.iter().find(|g| g.kind == GpuKind::Metal) {
                DeviceSelection {
                    device: ComputeDevice::Metal,
                    strategy_used: strategy,
                    gpu_detected: true,
                    fallback_reason: None,
                }
            } else {
                DeviceSelection {
                    device: ComputeDevice::Cpu,
                    strategy_used: strategy,
                    gpu_detected: false,
                    fallback_reason: Some("No GPU available, using CPU".to_string()),
                }
            }
        },
    };

    result
}

pub fn load_model_cfg(cfg_path: &str) -> Result<serde_json::Value> {
    let contents = std::fs::read_to_string(cfg_path)?;
    let json: serde_json::Value = serde_json::from_str(&contents)?;
    Ok(json)
}

pub fn save_model(cfg_path: &str, json: &serde_json::Value) -> Result<()> {
    let contents = serde_json::to_string_pretty(json)?;
    std::fs::write(cfg_path, contents)?;
    Ok(())
}

pub fn check_tensor_shapes(tensors: &[Tensor], expected_shapes: &[Vec<usize>]) -> Result<()> {
    for (tensor, expected_shape) in tensors.iter().zip(expected_shapes.iter()) {
        let shape = tensor.shape().dims().to_vec();
        if shape != *expected_shape {
            event!(
                Level::ERROR,
                "Shape mismatch: expected {:?}, got {:?}",
                expected_shape,
                shape
            );
            return Err(anyhow::anyhow!(
                "Shape mismatch: expected {:?}, got {:?}",
                expected_shape,
                shape
            ));
        }
    }
    Ok(())
}

pub struct GradientAccumulator {
    pub steps: usize,
    pub current: usize,
}

impl GradientAccumulator {
    pub fn new(steps: usize) -> Self {
        Self { steps, current: 0 }
    }

    /// Returns true when it's time to do an optimizer step.
    pub fn step(&mut self) -> bool {
        self.current = (self.current + 1) % self.steps;
        self.current == 0
    }

    pub fn reset(&mut self) {
        self.current = 0;
    }
}

pub fn check_value_in_range<T>(value: T, min: T, max: T) -> bool
where
    T: PartialOrd + Copy,
{
    value >= min && value <= max
}

#[derive(Debug, Clone)]
pub struct SystemLoadConfig {
    pub max_cpu_load: f32,
    pub check_interval_batches: usize,
    pub reduction_factor: f32,
}

impl Default for SystemLoadConfig {
    fn default() -> Self {
        Self {
            max_cpu_load: 0.80,
            check_interval_batches: 10,
            reduction_factor: 0.5,
        }
    }
}

pub struct SystemLoadMonitor {
    config: SystemLoadConfig,
    system: System,
}

impl SystemLoadMonitor {
    pub fn max_load(&self) -> f32 {
        self.config.max_cpu_load
    }
    pub fn new(config: SystemLoadConfig) -> Self {
        let mut system =
            System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything()));
        system.refresh_cpu();
        std::thread::sleep(std::time::Duration::from_millis(200));
        system.refresh_cpu();
        Self { config, system }
    }

    pub fn cpu_usage(&mut self) -> f32 {
        self.system.refresh_cpu();
        self.system.global_cpu_info().cpu_usage() / 100.0
    }

    pub fn is_overloaded(&mut self) -> bool {
        self.cpu_usage() > self.config.max_cpu_load
    }

    pub fn current_load(&mut self) -> f32 {
        self.cpu_usage()
    }

    pub fn recommended_batch_scale(&mut self) -> f32 {
        let load = self.current_load();
        if load > self.config.max_cpu_load {
            self.config.reduction_factor
        } else if load > self.config.max_cpu_load * 0.75 {
            0.75
        } else {
            1.0
        }
    }
}

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
    pub fn detect() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        std::thread::sleep(std::time::Duration::from_millis(200));
        system.refresh_all();

        let cpu_cores = system.physical_core_count().unwrap_or(system.cpus().len()).max(1) * 2;
        let total_ram_mb = system.total_memory() / 1024;

        // Detect best GPU
        let gpus = detect_gpus();
        let (gpu_vram_mb, gpu_kind, gpu_name) = if let Some(best_gpu) = gpus.first() {
            (Some(best_gpu.vram_mb), Some(best_gpu.kind), Some(best_gpu.name.clone()))
        } else {
            (None, None, None)
        };

        Self {
            cpu_cores,
            total_ram_mb,
            gpu_vram_mb,
            gpu_kind,
            gpu_name,
        }
    }

    pub fn has_gpu(&self) -> bool {
        self.gpu_vram_mb.is_some()
    }

    pub fn has_cuda(&self) -> bool {
        matches!(self.gpu_kind, Some(GpuKind::Cuda))
    }

    pub fn has_metal(&self) -> bool {
        matches!(self.gpu_kind, Some(GpuKind::Metal))
    }

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

    pub fn gpu_tier(&self) -> &'static str {
        match self.gpu_vram_mb {
            Some(v) if v >= 24000 => "large",
            Some(v) if v >= 12000 => "medium",
            Some(v) if v >= 6000 => "small",
            Some(_) => "tiny",
            None => "cpu",
        }
    }

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

pub struct AutoTuner;

impl AutoTuner {
    pub fn recommend(profile: &HardwareProfile) -> AutoTuneConfig {
        let gpu_vram = profile.gpu_vram_mb.unwrap_or(0);
        if gpu_vram >= 12000 {
            Self::for_gpu("medium")
        } else if gpu_vram >= 6000 {
            Self::for_gpu("small")
        } else if profile.total_ram_mb >= 16384 {
            Self::for_cpu("medium")
        } else if profile.total_ram_mb >= 8192 {
            Self::for_cpu("small")
        } else {
            Self::for_cpu("tiny")
        }
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
    fn test_device_selection_auto() {
        let selection = select_optimal_device();
        // Should return a valid device
        assert!(matches!(
            selection.device,
            ComputeDevice::Cpu | ComputeDevice::Metal | ComputeDevice::Cuda(_)
        ));
    }

    #[test]
    fn test_device_selection_strategies() {
        // Test force_cpu
        std::env::set_var("EXPERAI_DEVICE_STRATEGY", "force_cpu");
        let selection = select_optimal_device();
        assert_eq!(selection.device, ComputeDevice::Cpu);
        assert_eq!(selection.strategy_used, DeviceSelectionStrategy::ForceCpu);
        
        std::env::remove_var("EXPERAI_DEVICE_STRATEGY");
    }

    #[test]
    fn test_device_override() {
        std::env::set_var("EXPERAI_DEVICE", "metal");
        let selection = select_optimal_device();
        assert_eq!(selection.device, ComputeDevice::Metal);
        
        std::env::remove_var("EXPERAI_DEVICE");
    }

    #[test]
    fn test_hardware_profile_detect() {
        let profile = HardwareProfile::detect();
        assert!(profile.cpu_cores > 0);
        assert!(profile.total_ram_mb > 0);
        // GPU detection depends on hardware
        println!("Hardware profile: {:?}", profile);
    }

    #[test]
    fn test_gpu_detection() {
        let gpus = detect_gpus();
        println!("Detected GPUs: {:?}", gpus);
        // On systems with GPU, should detect at least one
    }
}
