use anyhow::Result;
use candle_core::Tensor;
use rand::SeedableRng;
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
                    Err(anyhow::anyhow!("Metal support not compiled in. Rebuild with --features metal"))
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
                    Err(anyhow::anyhow!("CUDA support not compiled in. Rebuild with --features cuda"))
                }
            }
        }
    }
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
            event!(Level::ERROR, "Shape mismatch: expected {:?}, got {:?}", expected_shape, shape);
            return Err(anyhow::anyhow!("Shape mismatch: expected {:?}, got {:?}", expected_shape, shape));
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
