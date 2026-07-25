use anyhow::Result;
use tracing::{info, debug};

/// Device selection for compute. Maps to candle_core::Device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDevice {
    Cpu,
    Metal,
    Cuda(usize),
}

impl ComputeDevice {
    /// Returns a string label for the device kind (e.g., `"cpu"`, `"metal"`, `"cuda"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            ComputeDevice::Cpu => "cpu",
            ComputeDevice::Metal => "metal",
            ComputeDevice::Cuda(_) => "cuda",
        }
    }

    /// Converts to a `candle_core::Device`, returning an error if the feature is not compiled in.
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

/// Parse the `EXPERAI_DEVICE` environment variable into a [`ComputeDevice`].
///
/// Supports `"cuda"`, `"cuda:<index>"`, `"metal"`, or anything else as CPU.
/// Returns `ComputeDevice::Cpu` if the variable is unset.
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

    #[cfg(feature = "metal")]
    {
        if let Ok(_device) = candle_core::Device::new_metal(0) {
            gpus.push(GpuInfo {
                kind: GpuKind::Metal,
                index: 0,
                vram_mb: 8192,
                name: "Apple Silicon GPU".to_string(),
                compute_capability: None,
            });
        }
    }

    // Sort: CUDA before Metal, then by descending VRAM for CUDA, ascending index for Metal.
    gpus.sort_by(|a, b| match (a.kind, b.kind) {
        (GpuKind::Cuda, GpuKind::Metal) => std::cmp::Ordering::Less,
        (GpuKind::Metal, GpuKind::Cuda) => std::cmp::Ordering::Greater,
        (GpuKind::Cuda, GpuKind::Cuda) => b.vram_mb.cmp(&a.vram_mb),
        (GpuKind::Metal, GpuKind::Metal) => a.index.cmp(&b.index),
    });

    gpus
}

/// Auto-select the optimal computation device based on available hardware.
///
/// Priority order:
/// 1. CUDA GPU with highest VRAM (if CUDA feature enabled)
/// 2. Metal GPU (if Metal feature enabled)
/// 3. CPU (fallback)
pub fn select_optimal_device() -> DeviceSelection {
    if let Ok(env_device) = std::env::var("EXPERAI_DEVICE") {
        info!("EXPERAI_DEVICE override set: {}", env_device);
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
    debug!("Detected {} GPU(s)", gpus.len());
    for gpu in &gpus {
        debug!("  GPU: {} ({} MB, {:?})", gpu.name, gpu.vram_mb, gpu.kind);
    }

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
        }
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
        }
        DeviceSelectionStrategy::Auto => {
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
        }
    };

    info!(
        "Device selection: {:?} (gpu_detected={}, reason={})",
        result.device,
        result.gpu_detected,
        result.fallback_reason.as_deref().unwrap_or("none")
    );

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_selection_auto() {
        let selection = select_optimal_device();
        assert!(matches!(
            selection.device,
            ComputeDevice::Cpu | ComputeDevice::Metal | ComputeDevice::Cuda(_)
        ));
    }

    #[test]
    fn test_device_override() {
        let prev = std::env::var("EXPERAI_DEVICE").ok();
        std::env::set_var("EXPERAI_DEVICE", "metal");
        let selection = select_optimal_device();
        assert_eq!(selection.device, ComputeDevice::Metal);
        if let Some(v) = prev {
            std::env::set_var("EXPERAI_DEVICE", v);
        } else {
            std::env::remove_var("EXPERAI_DEVICE");
        }
    }

    #[test]
    fn test_gpu_detection() {
        let gpus = detect_gpus();
        println!("Detected GPUs: {:?}", gpus);
    }
}
