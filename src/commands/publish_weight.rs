use crate::at_protocol::{ATProtocolPublishConfig, ATProtocolPublisher, PublishResult};
use crate::model::ModelConfig;
use anyhow::{Context, Result};
use tracing::info;

/// Publish trained model weights to an AT Protocol repository under
/// the `click.croft.experai.weight` collection.
///
/// The checkpoint directory must contain a `*.safetensors` file
/// (e.g. `best.safetensors`) and a `model_config.json`.
#[allow(clippy::too_many_arguments)]
pub fn run(
    checkpoint: String,
    handle: String,
    password: String,
    pds_url: String,
    rkey: Option<String>,
    collection: Option<String>,
    chunk_size: usize,
    name: String,
    description: Option<String>,
) -> Result<()> {
    info!(
        "Publishing weights from checkpoint: {} to PDS: {} handle: {}",
        checkpoint, pds_url, handle
    );

    // Resolve the safetensors file inside the checkpoint directory.
    let weights_path = resolve_weights_path(&checkpoint)?;
    info!("Using weights file: {}", weights_path);

    // Load model config from the checkpoint directory.
    let model_config = load_model_config(&checkpoint).context(
        "Failed to load model config; ensure the checkpoint directory contains model_config.json",
    )?;
    info!(
        "Loaded model config: vocab={}, hidden={}, layers={}",
        model_config.vocab_size, model_config.hidden_size, model_config.num_layers
    );

    // Build the publish configuration.
    let config = ATProtocolPublishConfig {
        pds_url,
        handle: handle.clone(),
        password,
        rkey,
        collection: collection.unwrap_or_else(|| "click.croft.experai.weight".to_string()),
        chunk_size,
        name: name.clone(),
        description,
    };

    // Authenticate and publish.
    let rt = tokio::runtime::Runtime::new()
        .context("Failed to create async runtime for AT Protocol publishing")?;

    let publisher = rt.block_on(async { ATProtocolPublisher::new(config).await })?;

    let result: PublishResult = rt.block_on(async {
        publisher
            .publish_weights(&weights_path, &model_config)
            .await
    })?;

    info!(
        "Publish complete: uri={}, cid={}, size={} bytes, {} chunks",
        result.uri, result.cid, result.size, result.chunks
    );

    // Output a summary (always human-readable, as the CLI dispatches via main).
    println!("Published {name} weights to AT Protocol:");
    println!("  URI:  {}", result.uri);
    println!("  CID:  {}", result.cid);
    println!("  Size: {} bytes ({} chunks)", result.size, result.chunks);

    Ok(())
}

/// Find the `.safetensors` file inside a checkpoint directory.
/// If `checkpoint` is itself a `.safetensors` file, use it directly.
fn resolve_weights_path(checkpoint: &str) -> Result<String> {
    let path = std::path::Path::new(checkpoint);

    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            return Ok(checkpoint.to_string());
        }
        return Err(anyhow::anyhow!(
            "Expected a .safetensors file or a checkpoint directory, found: {}",
            checkpoint
        ));
    }

    if path.is_dir() {
        let candidate = path.join("best.safetensors");
        if candidate.exists() {
            return Ok(candidate.to_string_lossy().to_string());
        }

        // Fall back to any .safetensors file in the directory.
        let found = std::fs::read_dir(path)
            .map_err(|e| anyhow::anyhow!("Cannot read checkpoint directory {}: {}", checkpoint, e))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    Some(path.to_string_lossy().to_string())
                } else {
                    None
                }
            })
            .next();

        if let Some(found) = found {
            return Ok(found);
        }

        return Err(anyhow::anyhow!(
            "No .safetensors file found in checkpoint directory: {}",
            checkpoint
        ));
    }

    Err(anyhow::anyhow!(
        "Checkpoint path does not exist: {}",
        checkpoint
    ))
}

/// Load `model_config.json` from a checkpoint directory.
fn load_model_config(checkpoint: &str) -> Result<ModelConfig> {
    let config_path = std::path::Path::new(checkpoint).join("model_config.json");
    let data = std::fs::read_to_string(&config_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read model config from {}: {}",
            config_path.display(),
            e
        )
    })?;
    let config: ModelConfig = serde_json::from_str(&data)
        .map_err(|e| anyhow::anyhow!("Failed to parse model config: {}", e))?;
    Ok(config)
}
