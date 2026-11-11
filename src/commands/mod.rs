/// CLI command handlers for all subcommands.
pub mod at_protocol;
pub mod fetch;
pub mod generate;
pub mod jetstream_train;
pub mod opencode;
pub mod package;
pub mod preprocess;
pub mod publish_weight;
pub mod train;

use anyhow::{Context, Result};

use crate::model::ModelConfig;

/// Resolve which [`ModelConfig`] to build: an explicit `--model-config` file
/// wins; otherwise a `model_config.json` inside `--model` (when `--model` is a
/// checkpoint directory) is used; otherwise the default architecture.
pub fn resolve_model_config(model: &str, model_config: Option<&str>) -> Result<ModelConfig> {
    if let Some(path) = model_config {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read model config {path}"))?;
        let cfg: ModelConfig = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse model config {path}"))?;
        return Ok(cfg);
    }
    let dir_cfg = format!("{model}/model_config.json");
    if std::path::Path::new(model).is_dir() && std::path::Path::new(&dir_cfg).exists() {
        let raw = std::fs::read_to_string(&dir_cfg)
            .with_context(|| format!("Failed to read {dir_cfg}"))?;
        return serde_json::from_str(&raw).with_context(|| format!("Failed to parse {dir_cfg}"));
    }
    Ok(ModelConfig::default())
}
