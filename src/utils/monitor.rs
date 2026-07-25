use sysinfo::{CpuRefreshKind, RefreshKind, System};
use tracing::debug;

/// Thresholds controlling when [`SystemLoadMonitor`] triggers batch-size reduction.
#[derive(Debug, Clone)]
pub struct SystemLoadConfig {
    /// CPU load fraction (0.0–1.0) above which the system is considered overloaded.
    pub max_cpu_load: f32,
    /// How often (in batches) to re-check CPU load.
    pub check_interval_batches: usize,
    /// Batch-size multiplier applied when overloaded (e.g., 0.5 halves the batch).
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

/// Monitors system CPU load and recommends batch-size scaling during training.
pub struct SystemLoadMonitor {
    config: SystemLoadConfig,
    system: System,
}

impl SystemLoadMonitor {
    /// Returns the configured maximum CPU load threshold.
    pub fn max_load(&self) -> f32 {
        self.config.max_cpu_load
    }

    /// Create a new monitor, taking an initial CPU measurement baseline.
    pub fn new(config: SystemLoadConfig) -> Self {
        debug!(
            "Initializing system load monitor: max_cpu_load={:.0}%, check_interval={}",
            config.max_cpu_load * 100.0,
            config.check_interval_batches
        );
        let mut system =
            System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything()));
        system.refresh_cpu();
        std::thread::sleep(std::time::Duration::from_millis(200));
        system.refresh_cpu();
        Self { config, system }
    }

    /// Refresh and return current global CPU usage as a fraction (0.0–1.0).
    pub fn cpu_usage(&mut self) -> f32 {
        self.system.refresh_cpu();
        self.system.global_cpu_info().cpu_usage() / 100.0
    }

    /// Returns `true` if current CPU usage exceeds the configured threshold.
    pub fn is_overloaded(&mut self) -> bool {
        self.cpu_usage() > self.config.max_cpu_load
    }

    /// Alias for [`cpu_usage`](Self::cpu_usage).
    pub fn current_load(&mut self) -> f32 {
        self.cpu_usage()
    }

    /// Suggest a batch-size multiplier based on current CPU load.
    ///
    /// - Overloaded → `reduction_factor` (default 0.5)
    /// - Near threshold (>75%) → 0.75
    /// - Nominal → 1.0
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
