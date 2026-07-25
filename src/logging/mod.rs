/// Structured logging configuration for the application.
///
/// Provides a centralized logger backed by `tracing-subscriber` with support
/// for per-level filtering (`EnvFilter`) and two output formats:
/// - **pretty**: human-readable, indented output (used in test mode)
/// - **json**: machine-parseable structured logs (used in production)
use tracing_subscriber::prelude::*;
use tracing_subscriber::{fmt, EnvFilter};

/// Initialize the global `tracing` subscriber.
///
/// # Arguments
/// * `log_level` - Optional log level string (e.g. `"info"`, `"debug"`).
///   Defaults to `"info"` when `None`.
/// * `test_mode` - When `true`, uses the pretty-print formatter for
///   human-readable output; otherwise emits JSON logs.
pub fn init_logger(
    log_level: Option<String>,
    test_mode: bool,
) -> Result<(), crate::errors::ExperaiError> {
    let level = log_level.unwrap_or_else(|| "info".to_string());
    let env_filter = EnvFilter::try_new(&level).map_err(|e| {
        crate::errors::ExperaiError::Config(format!(
            "invalid log level {level:?}: {e} (expected one of trace|debug|info|warn|error)"
        ))
    })?;

    if test_mode {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().pretty())
            .try_init()
            .ok();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().json())
            .try_init()
            .ok();
    }

    Ok(())
}
