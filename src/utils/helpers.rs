use anyhow::Result;
use candle_core::Tensor;
use rand::SeedableRng;
use tracing::{event, Level};

/// Initialise the global tracing subscriber.
///
/// When `test_mode` is `true`, output is written to `stderr` instead of stdout
/// so that test `println!` output stays clean.
pub fn init_tracing(log_level: Option<String>, test_mode: bool) -> Result<()> {
    crate::logging::init_logger(log_level, test_mode)
}

/// Seed the global random number generator for reproducibility.
pub fn set_global_seed(seed: u64) {
    let _ = rand::rngs::StdRng::seed_from_u64(seed);
}

/// Load a JSON model configuration file from disk.
pub fn load_model_cfg(cfg_path: &str) -> Result<serde_json::Value> {
    let contents = std::fs::read_to_string(cfg_path)?;
    let json: serde_json::Value = serde_json::from_str(&contents)?;
    Ok(json)
}

/// Persist a model configuration as pretty-printed JSON.
pub fn save_model(cfg_path: &str, json: &serde_json::Value) -> Result<()> {
    let contents = serde_json::to_string_pretty(json)?;
    std::fs::write(cfg_path, contents)?;
    Ok(())
}

/// Validate that each tensor's shape matches the corresponding expected shape.
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

/// Tracks micro-batch steps to decide when to trigger an optimizer update.
pub struct GradientAccumulator {
    pub steps: usize,
    pub current: usize,
}

impl GradientAccumulator {
    /// Create a new accumulator that fires every `steps` micro-batches.
    pub fn new(steps: usize) -> Self {
        Self { steps, current: 0 }
    }

    /// Returns true when it's time to do an optimizer step.
    pub fn step(&mut self) -> bool {
        self.current = (self.current + 1) % self.steps;
        self.current == 0
    }

    /// Reset the internal counter to zero.
    pub fn reset(&mut self) {
        self.current = 0;
    }
}

/// Returns `true` if `value` is within `[min, max]` inclusive.
pub fn check_value_in_range<T>(value: T, min: T, max: T) -> bool
where
    T: PartialOrd + Copy,
{
    value >= min && value <= max
}
