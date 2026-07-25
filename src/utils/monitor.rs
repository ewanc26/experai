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
    /// If `true`, the trainer should sleep briefly before the next step to let
    /// the system reclaim memory and avoid dipping into swap.
    pub should_pause: bool,
    /// Human-readable reason for the current recommendation.
    pub reason: String,
}

impl Default for LoadRecommendation {
    fn default() -> Self {
        Self {
            batch_scale: 1.0,
            lr_scale: 1.0,
            force_f32: false,
            should_pause: false,
            reason: "nominal".into(),
        }
    }
}

/// Thresholds for different system pressure levels.
///
/// Defaults are conservative: the goal is to **never** cause the OS to swap
/// and to **never** make the system stutter.  Training is treated as a
/// background citizen that yields to system responsiveness.
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
    /// Swap usage fraction (0.0–1.0) at which we start scaling down.
    /// Any swap usage is treated as a warning by default (0.0).
    pub swap_warn: f32,
    /// Swap usage fraction that triggers a pause + heavy throttle.
    pub swap_crit: f32,
    /// Minimum free memory (in MB) to maintain.  If available memory drops
    /// below this, training pauses until memory is reclaimed.
    pub min_free_mem_mb: u64,
    /// How often (in optimizer steps) to re-check system load.
    pub check_interval: usize,
}

impl Default for SystemLoadConfig {
    fn default() -> Self {
        Self {
            cpu_warn: 0.50,
            cpu_crit: 0.75,
            mem_warn: 0.60,
            mem_crit: 0.75,
            // Any swap usage is a warning — we should never swap.
            swap_warn: 0.0,
            swap_crit: 0.02,
            // Keep at least 2 GB free for the OS and other processes.
            min_free_mem_mb: 2048,
            check_interval: 5,
        }
    }
}

/// Continuously monitors CPU, memory, swap, and (on macOS) GPU to recommend
/// dynamic training parameter adjustments.
///
/// The monitor's philosophy is **conservative**: it is better to slow down
/// training than to let the system swap or stutter.  Swap usage is treated
/// as a hard signal to reduce memory pressure immediately.
pub struct SystemLoadMonitor {
    config: SystemLoadConfig,
    system: System,
    /// Rolling average of CPU load (exponential moving average).
    ema_cpu: f32,
    /// Rolling average of memory usage.
    ema_mem: f32,
    /// Rolling average of swap usage (fraction of total swap).
    ema_swap: f32,
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
        let total_mem = system.total_memory() as f32;
        let avail_mem = system.available_memory() as f32;
        let mem = if total_mem > 0.0 {
            1.0 - (avail_mem / total_mem)
        } else {
            0.0
        };
        let total_swap = system.total_swap() as f32;
        let used_swap = system.used_swap() as f32;
        let swap = if total_swap > 0.0 {
            used_swap / total_swap
        } else {
            0.0
        };

        debug!(
            "System monitor init: cpu={:.0}% mem={:.0}% swap={:.0}% (swap: {}/{} MB)",
            cpu * 100.0,
            mem * 100.0,
            swap * 100.0,
            used_swap / 1024.0 / 1024.0,
            total_swap / 1024.0 / 1024.0,
        );

        Self {
            config,
            system,
            ema_cpu: cpu,
            ema_mem: mem,
            ema_swap: swap,
            sample_count: 1,
        }
    }

    /// How often (in steps) to poll.
    pub fn check_interval(&self) -> usize {
        self.config.check_interval
    }

    /// Refresh all metrics and return a recommendation.
    ///
    /// The recommendation is the most severe of all individual checks
    /// (swap, free memory, CPU, memory usage).  This ensures that any
    /// single pressure source triggers an appropriate response.
    pub fn poll(&mut self) -> LoadRecommendation {
        self.system.refresh_cpu();
        self.system.refresh_memory();
        self.sample_count += 1;

        // Raw readings.
        let raw_cpu = self.system.global_cpu_info().cpu_usage() / 100.0;
        let total_mem = self.system.total_memory() as f32;
        let avail_mem = self.system.available_memory() as f32;
        let free_mem_mb = (avail_mem / 1024.0 / 1024.0) as u64;
        let raw_mem = if total_mem > 0.0 {
            1.0 - (avail_mem / total_mem)
        } else {
            0.0
        };

        let total_swap = self.system.total_swap() as f32;
        let used_swap = self.system.used_swap() as f32;
        let raw_swap = if total_swap > 0.0 {
            used_swap / total_swap
        } else {
            0.0
        };

        // Exponential moving average (alpha=0.3) to smooth spikes.
        let alpha = 0.3;
        self.ema_cpu = alpha * raw_cpu + (1.0 - alpha) * self.ema_cpu;
        self.ema_mem = alpha * raw_mem + (1.0 - alpha) * self.ema_mem;
        self.ema_swap = alpha * raw_swap + (1.0 - alpha) * self.ema_swap;

        let cpu = self.ema_cpu;
        let mem = self.ema_mem;
        let swap = self.ema_swap;
        let c = &self.config;

        // ── Priority 1: Swap / free-memory emergency ──────────────
        // If swap is being used at all, or free memory is below the minimum,
        // we need to pause and let the system recover.  This is the hard
        // brake — training must never cause the OS to swap.
        if swap > c.swap_crit || free_mem_mb < c.min_free_mem_mb {
            return LoadRecommendation {
                batch_scale: 0.1,
                lr_scale: 0.25,
                force_f32: true,
                should_pause: true,
                reason: format!(
                    "SWAP EMERGENCY: swap={:.1}% free={}MB — pausing to avoid swap",
                    swap * 100.0,
                    free_mem_mb,
                ),
            };
        }

        // ── Priority 2: Any swap usage or critical CPU/mem ─────────
        if swap > c.swap_warn || cpu > c.cpu_crit || mem > c.mem_crit {
            let reason = if swap > c.swap_warn {
                format!(
                    "SWAP WARN: swap={:.1}% — heavy throttle to reduce memory pressure",
                    swap * 100.0,
                )
            } else {
                format!(
                    "CRITICAL: cpu={:.0}% mem={:.0}% swap={:.1}% — heavy throttle",
                    cpu * 100.0,
                    mem * 100.0,
                    swap * 100.0,
                )
            };
            return LoadRecommendation {
                batch_scale: 0.25,
                lr_scale: 0.5,
                force_f32: true,
                should_pause: false,
                reason,
            };
        }

        // ── Priority 3: Warning-level CPU or memory ──────────────
        if cpu > c.cpu_warn || mem > c.mem_warn {
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
                should_pause: false,
                reason: format!(
                    "WARN: cpu={:.0}% mem={:.0}% swap={:.1}% pressure={:.0}%",
                    cpu * 100.0,
                    mem * 100.0,
                    swap * 100.0,
                    pressure * 100.0,
                ),
            }
        } else {
            LoadRecommendation {
                batch_scale: 1.0,
                lr_scale: 1.0,
                force_f32: false,
                should_pause: false,
                reason: format!(
                    "OK: cpu={:.0}% mem={:.0}% swap={:.1}% free={}MB",
                    cpu * 100.0,
                    mem * 100.0,
                    swap * 100.0,
                    free_mem_mb,
                ),
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

    /// Current swap usage (EMA-smoothed), 0.0–1.0.
    /// Returns 0.0 if the system has no swap configured.
    pub fn swap_usage(&self) -> f32 {
        self.ema_swap
    }

    /// Simple overloaded check (CPU above critical threshold).
    pub fn is_overloaded(&self) -> bool {
        self.ema_cpu > self.config.cpu_crit
    }

    /// Returns `true` if the system is currently swapping or about to.
    pub fn is_swapping(&self) -> bool {
        self.ema_swap > self.config.swap_warn
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

    #[test]
    fn test_swap_tracking() {
        let monitor = SystemLoadMonitor::new(SystemLoadConfig::default());
        let swap = monitor.swap_usage();
        println!("Swap usage: {:.2}%", swap * 100.0);
        assert!((0.0..=1.0).contains(&swap));
    }
}
