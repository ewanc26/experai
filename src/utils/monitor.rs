use sysinfo::{CpuRefreshKind, RefreshKind, System};
use tracing::debug;

/// What the monitor recommends changing right now.
#[derive(Debug, Clone)]
pub struct LoadRecommendation {
    /// Multiplier for batch size (0.0–1.0). Multiply `base_batch_size` by this.
    pub batch_scale: f32,
    /// Suggested learning rate multiplier (0.0–1.0). Multiply `base_lr` by this.
    pub lr_scale: f32,
    /// Whether to drop precision to f32 from bf16 (e.g. under thermal pressure).
    pub force_f32: bool,
    /// Human-readable reason for the current recommendation.
    pub reason: String,
}

impl Default for LoadRecommendation {
    fn default() -> Self {
        Self {
            batch_scale: 1.0,
            lr_scale: 1.0,
            force_f32: false,
            reason: "nominal".into(),
        }
    }
}

/// Thresholds for different system pressure levels.
#[derive(Debug, Clone)]
pub struct SystemLoadConfig {
    /// CPU load fraction (0.0–1.0) at which we start scaling down.
    pub cpu_warn: f32,
    /// CPU load fraction that triggers heavy throttling.
    pub cpu_crit: f32,
    /// Memory usage fraction (0.0–1.0) at which we start scaling down.
    pub mem_warn: f32,
    /// Memory usage fraction that triggers heavy throttling.
    pub mem_crit: f32,
    /// How often (in optimizer steps) to re-check system load.
    pub check_interval: usize,
}

impl Default for SystemLoadConfig {
    fn default() -> Self {
        Self {
            cpu_warn: 0.50,
            cpu_crit: 0.80,
            mem_warn: 0.75,
            mem_crit: 0.90,
            check_interval: 5,
        }
    }
}

/// Continuously monitors CPU, memory, and (on macOS) GPU to recommend
/// dynamic training parameter adjustments.
pub struct SystemLoadMonitor {
    config: SystemLoadConfig,
    system: System,
    /// Rolling average of CPU load (exponential moving average).
    ema_cpu: f32,
    /// Rolling average of memory usage.
    ema_mem: f32,
    /// Number of samples taken.
    sample_count: usize,
}

impl SystemLoadMonitor {
    pub fn new(config: SystemLoadConfig) -> Self {
        let mut system =
            System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything()));
        system.refresh_cpu();
        system.refresh_memory();
        std::thread::sleep(std::time::Duration::from_millis(200));
        system.refresh_cpu();
        system.refresh_memory();

        let cpu = system.global_cpu_info().cpu_usage() / 100.0;
        let mem = 1.0 - (system.available_memory() as f32 / system.total_memory() as f32);

        debug!(
            "System monitor init: cpu={:.0}% mem={:.0}%",
            cpu * 100.0,
            mem * 100.0
        );

        Self {
            config,
            system,
            ema_cpu: cpu,
            ema_mem: mem,
            sample_count: 1,
        }
    }

    /// How often (in steps) to poll.
    pub fn check_interval(&self) -> usize {
        self.config.check_interval
    }

    /// Refresh all metrics and return a recommendation.
    pub fn poll(&mut self) -> LoadRecommendation {
        self.system.refresh_cpu();
        self.system.refresh_memory();
        self.sample_count += 1;

        // Raw readings.
        let raw_cpu = self.system.global_cpu_info().cpu_usage() / 100.0;
        let total_mem = self.system.total_memory() as f32;
        let avail_mem = self.system.available_memory() as f32;
        let raw_mem = if total_mem > 0.0 {
            1.0 - (avail_mem / total_mem)
        } else {
            0.0
        };

        // Exponential moving average (alpha=0.3) to smooth spikes.
        let alpha = 0.3;
        self.ema_cpu = alpha * raw_cpu + (1.0 - alpha) * self.ema_cpu;
        self.ema_mem = alpha * raw_mem + (1.0 - alpha) * self.ema_mem;

        let cpu = self.ema_cpu;
        let mem = self.ema_mem;
        let c = &self.config;

        // Tiered response based on combined pressure.
        if cpu > c.cpu_crit || mem > c.mem_crit {
            LoadRecommendation {
                batch_scale: 0.25,
                lr_scale: 0.5,
                force_f32: true,
                reason: format!(
                    "CRITICAL: cpu={:.0}% mem={:.0}% — heavy throttle",
                    cpu * 100.0,
                    mem * 100.0
                ),
            }
        } else if cpu > c.cpu_warn || mem > c.mem_warn {
            // Interpolate: the closer to crit, the more we throttle.
            let pressure = ((cpu - c.cpu_warn) / (c.cpu_crit - c.cpu_warn).max(0.01))
                .max((mem - c.mem_warn) / (c.mem_crit - c.mem_warn).max(0.01))
                .clamp(0.0, 1.0);
            let batch_scale = 1.0 - (0.5 * pressure); // 1.0 → 0.5
            let lr_scale = 1.0 - (0.3 * pressure); // 1.0 → 0.7
            LoadRecommendation {
                batch_scale,
                lr_scale,
                force_f32: false,
                reason: format!(
                    "WARN: cpu={:.0}% mem={:.0}% pressure={:.0}%",
                    cpu * 100.0,
                    mem * 100.0,
                    pressure * 100.0
                ),
            }
        } else {
            LoadRecommendation {
                batch_scale: 1.0,
                lr_scale: 1.0,
                force_f32: false,
                reason: format!("OK: cpu={:.0}% mem={:.0}%", cpu * 100.0, mem * 100.0),
            }
        }
    }

    /// Current CPU usage (EMA-smoothed), 0.0–1.0.
    pub fn cpu_usage(&self) -> f32 {
        self.ema_cpu
    }

    /// Current memory usage (EMA-smoothed), 0.0–1.0.
    pub fn memory_usage(&self) -> f32 {
        self.ema_mem
    }

    /// Simple overloaded check (CPU above critical threshold).
    pub fn is_overloaded(&self) -> bool {
        self.ema_cpu > self.config.cpu_crit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_poll() {
        let mut monitor = SystemLoadMonitor::new(SystemLoadConfig::default());
        let rec = monitor.poll();
        println!("Recommendation: {:?}", rec);
        assert!(rec.batch_scale > 0.0 && rec.batch_scale <= 1.0);
    }
}
