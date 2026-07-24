use tracing_subscriber::prelude::*;
use tracing_subscriber::{fmt, EnvFilter};

pub fn init_logger(log_level: Option<String>, test_mode: bool) -> anyhow::Result<()> {
    let level = log_level.unwrap_or_else(|| "info".to_string());
    let env_filter = EnvFilter::try_new(&level)?;

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
