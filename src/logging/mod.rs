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
/// * `logs_to_stderr` - When `true`, logs go to stderr so stdout stays
///   machine-readable (used with the global `--json` flag).
pub fn init_logger(
    log_level: Option<String>,
    test_mode: bool,
    logs_to_stderr: bool,
) -> Result<(), crate::errors::ExperaiError> {
    let level = log_level.unwrap_or_else(|| "info".to_string());
    let env_filter = EnvFilter::try_new(&level).map_err(|e| {
        crate::errors::ExperaiError::Config(format!(
            "invalid log level {level:?}: {e} (expected one of trace|debug|info|warn|error)"
        ))
    })?;

    use std::io;
    let writer: tracing_subscriber::fmt::writer::BoxMakeWriter = if logs_to_stderr {
        tracing_subscriber::fmt::writer::BoxMakeWriter::new(Box::new(io::stderr))
    } else {
        tracing_subscriber::fmt::writer::BoxMakeWriter::new(Box::new(io::stdout))
    };

    if test_mode {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().pretty().with_writer(writer))
            .try_init()
            .ok();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().json().with_writer(writer))
            .try_init()
            .ok();
    }

    Ok(())
}
