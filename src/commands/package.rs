use anyhow::Result;
use tracing::info;

/// Export a trained checkpoint to GGUF format for use with LM Studio or other runtimes.
pub fn run(
    checkpoint: String,
    name: String,
    lmstudio_dir: Option<String>,
    dtype: String,
) -> Result<()> {
    info!(
        "Exporting checkpoint to GGUF: checkpoint={}, name={}, dtype={}",
        checkpoint, name, dtype
    );

    crate::export::export_to_gguf(&checkpoint, &name, lmstudio_dir.as_deref(), &dtype)?;

    Ok(())
}
