use std::collections::HashMap;
use std::path::{Path, PathBuf};

use gguf_rs::writer::{GGUFWriter, TensorInfo};
use safetensors::{Dtype, SafeTensors};
use tracing::{debug, info};

use crate::errors::ExperaiError;
use crate::model::ModelConfig;

/// GGML dtype constants used in the GGUF file format.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_BF16: u32 = 30;

/// Convert a [`safetensors::Dtype`] to the corresponding GGML dtype constant.
fn safetensors_dtype_to_ggml(dt: Dtype) -> Result<u32, ExperaiError> {
    match dt {
        Dtype::F32 => Ok(GGML_F32),
        Dtype::F16 => Ok(GGML_F16),
        Dtype::BF16 => Ok(GGML_BF16),
        _ => Err(ExperaiError::Export(format!(
            "Unsupported dtype for GGUF export: {:?}",
            dt
        ))),
    }
}

/// Map a Candle/safetensors tensor name to its GGUF GPT-2 equivalent.
///
/// The GGUF format uses different naming conventions:
/// - `embedding.weight` → `token_embd.weight`
/// - `norm.weight` → `output_norm.weight`
/// - `lm_head.weight` → `output.weight`
/// - Layer tensors `layers.{i}.*` → `blk.{i}.*` with sub-component renaming
///   (e.g. `self_attn.q_proj` → `attn_q`, `mlp.gate_proj` → `ffn_gate`).
fn map_tensor_name(candle_name: &str) -> Result<String, ExperaiError> {
    if candle_name == "embedding.weight" {
        return Ok("token_embd.weight".to_string());
    }
    if candle_name == "norm.weight" {
        return Ok("output_norm.weight".to_string());
    }
    if candle_name == "lm_head.weight" {
        return Ok("output.weight".to_string());
    }

    if let Some(rest) = candle_name.strip_prefix("layers.") {
        let dot_pos = rest
            .find('.')
            .ok_or_else(|| ExperaiError::Export("Invalid layer tensor name".to_string()))?;
        let layer_idx: usize = rest[..dot_pos]
            .parse()
            .map_err(|_| ExperaiError::Export("Invalid layer index".to_string()))?;
        let suffix = &rest[dot_pos + 1..];

        let gguf_suffix = match suffix {
            s if s.starts_with("self_attn.q_proj.") => {
                let rest = s
                    .strip_prefix("self_attn.q_proj.")
                    .ok_or_else(|| ExperaiError::Export("strip_prefix failed".to_string()))?;
                format!("attn_q.{}", rest)
            }
            s if s.starts_with("self_attn.k_proj.") => {
                let rest = s
                    .strip_prefix("self_attn.k_proj.")
                    .ok_or_else(|| ExperaiError::Export("strip_prefix failed".to_string()))?;
                format!("attn_k.{}", rest)
            }
            s if s.starts_with("self_attn.v_proj.") => {
                let rest = s
                    .strip_prefix("self_attn.v_proj.")
                    .ok_or_else(|| ExperaiError::Export("strip_prefix failed".to_string()))?;
                format!("attn_v.{}", rest)
            }
            s if s.starts_with("self_attn.o_proj.") => {
                let rest = s
                    .strip_prefix("self_attn.o_proj.")
                    .ok_or_else(|| ExperaiError::Export("strip_prefix failed".to_string()))?;
                format!("attn_output.{}", rest)
            }
            s if s.starts_with("input_layernorm.") => {
                let rest = s
                    .strip_prefix("input_layernorm.")
                    .ok_or_else(|| ExperaiError::Export("strip_prefix failed".to_string()))?;
                format!("ln_1.{}", rest)
            }
            s if s.starts_with("post_attention_layernorm.") => {
                let rest = s
                    .strip_prefix("post_attention_layernorm.")
                    .ok_or_else(|| ExperaiError::Export("strip_prefix failed".to_string()))?;
                format!("ln_2.{}", rest)
            }
            s if s.starts_with("mlp.gate_proj.") => {
                let rest = s
                    .strip_prefix("mlp.gate_proj.")
                    .ok_or_else(|| ExperaiError::Export("strip_prefix failed".to_string()))?;
                format!("ffn_gate.{}", rest)
            }
            s if s.starts_with("mlp.up_proj.") => {
                let rest = s
                    .strip_prefix("mlp.up_proj.")
                    .ok_or_else(|| ExperaiError::Export("strip_prefix failed".to_string()))?;
                format!("ffn_up.{}", rest)
            }
            s if s.starts_with("mlp.down_proj.") => {
                let rest = s
                    .strip_prefix("mlp.down_proj.")
                    .ok_or_else(|| ExperaiError::Export("strip_prefix failed".to_string()))?;
                format!("ffn_down.{}", rest)
            }
            _ => {
                return Err(ExperaiError::Export(format!(
                    "Unknown tensor suffix in layer: {}",
                    suffix
                )))
            }
        };

        return Ok(format!("blk.{}.{}", layer_idx, gguf_suffix));
    }

    Err(ExperaiError::Export(format!(
        "Unmapped tensor name: {}",
        candle_name
    )))
}

/// Load a [`ModelConfig`] from `model_config.json` in the given directory.
fn load_model_config(checkpoint_dir: &Path) -> Result<ModelConfig, ExperaiError> {
    let config_path = checkpoint_dir.join("model_config.json");
    let data = std::fs::read_to_string(&config_path).map_err(|e| {
        ExperaiError::Export(format!(
            "Failed to read model config from {}: {}",
            config_path.display(),
            e
        ))
    })?;
    let config: ModelConfig = serde_json::from_str(&data)
        .map_err(|e| ExperaiError::Export(format!("Failed to parse model config: {}", e)))?;
    Ok(config)
}

/// Export a safetensors checkpoint to GGUF format and install it for LM Studio.
///
/// # Arguments
/// * `checkpoint_path` - Path to the `.safetensors` checkpoint file.
/// * `model_name` - Name used for the output GGUF file and directory.
/// * `lmstudio_dir` - Optional override for the LM Studio models directory.
///   Defaults to `~/.lmstudio/models/custom`.
/// * `_dtype` - Target dtype hint (currently unused; original dtype is preserved).
///
/// # Workflow
/// 1. Load `model_config.json` from the checkpoint directory.
/// 2. Deserialize all tensors from the safetensors file.
/// 3. Map each Candle tensor name to its GGUF equivalent.
/// 4. Write GGUF metadata (architecture, hyperparameters) and tensor data.
/// 5. Copy `model_config.json` alongside the GGUF for reference.
pub fn export_to_gguf(
    checkpoint_path: &str,
    model_name: &str,
    lmstudio_dir: Option<&str>,
    _dtype: &str,
) -> Result<(), ExperaiError> {
    let checkpoint_path = Path::new(checkpoint_path);
    let checkpoint_dir = checkpoint_path
        .parent()
        .ok_or_else(|| ExperaiError::Export("Invalid checkpoint path".to_string()))?;

    // Locate the safetensors file (same stem as the checkpoint path)
    let safetensors_path = checkpoint_path.with_extension("safetensors");
    if !safetensors_path.exists() {
        return Err(ExperaiError::Export(format!(
            "Safetensors file not found: {}",
            safetensors_path.display()
        )));
    }

    let model_config = load_model_config(checkpoint_dir)?;
    info!(
        "Loaded model config: vocab={}, hidden={}, layers={}, heads={}",
        model_config.vocab_size,
        model_config.hidden_size,
        model_config.num_layers,
        model_config.num_heads
    );

    // Deserialize the safetensors archive
    let st_data = std::fs::read(&safetensors_path).map_err(|e| {
        ExperaiError::Export(format!(
            "Failed to read {}: {}",
            safetensors_path.display(),
            e
        ))
    })?;
    let st = SafeTensors::deserialize(&st_data)
        .map_err(|e| ExperaiError::Export(format!("Failed to deserialize safetensors: {}", e)))?;

    let names = st.names();
    info!("Found {} tensors in checkpoint", names.len());
    debug!("Tensor names: {:?}", names);

    // Build Candle → GGUF name mapping for every tensor
    let mut name_map: HashMap<String, String> = HashMap::new();
    for name in &names {
        let gguf_name = map_tensor_name(name)?;
        debug!("  {} -> {}", name, gguf_name);
        name_map.insert(name.to_string(), gguf_name);
    }

    // Determine output directory (custom path or default LM Studio location)
    let lmstudio_base = match lmstudio_dir {
        Some(dir) => PathBuf::from(dir),
        None => {
            let home = std::env::var("HOME")
                .map_err(|_| ExperaiError::Config("HOME not set".to_string()))?;
            PathBuf::from(home)
                .join(".lmstudio")
                .join("models")
                .join("custom")
        }
    };

    let output_dir = lmstudio_base.join(model_name);
    std::fs::create_dir_all(&output_dir).map_err(|e| {
        ExperaiError::Export(format!("Failed to create {}: {}", output_dir.display(), e))
    })?;

    let gguf_path = output_dir.join(format!("{}.gguf", model_name));
    info!("Writing GGUF to {}", gguf_path.display());

    // Initialize GGUF writer (version 3)
    let mut writer = GGUFWriter::new(&gguf_path, 3)
        .map_err(|e| ExperaiError::Export(format!("Failed to create GGUF writer: {}", e)))?;

    // Write architecture and hyperparameter metadata
    writer.add_metadata("general.architecture", "gpt2");
    writer.add_metadata("general.name", model_name);

    // GPT-2 specific metadata
    writer.add_metadata_u32("gpt2.block_count", model_config.num_layers as u32);
    writer.add_metadata_u32("gpt2.attention.head_count", model_config.num_heads as u32);
    writer.add_metadata_u32("gpt2.embedding_length", model_config.hidden_size as u32);
    writer.add_metadata_u32(
        "gpt2.feed_forward_length",
        model_config.intermediate_size as u32,
    );
    writer.add_metadata_u32("gpt2.context_length", model_config.max_seq_len as u32);
    writer.add_metadata_u32("gpt2.vocab_size", model_config.vocab_size as u32);
    writer.add_metadata_f32(
        "gpt2.layer_norm_epsilon",
        model_config.layer_norm_eps as f32,
    );

    // Sort tensors by name for deterministic output order
    let mut ordered_names: Vec<(&String, &String)> = name_map.iter().collect();
    ordered_names.sort_by_key(|(candle, _)| candle.to_string());

    // Collect tensor data: (gguf_name, raw_bytes, shape, dtype)
    let mut tensor_data_map: Vec<(String, Vec<u8>, Vec<u64>, u32)> = Vec::new();

    for (candle_name, gguf_name) in &ordered_names {
        let tensor = st.tensor(candle_name).map_err(|e| {
            ExperaiError::Export(format!("Failed to read tensor {}: {}", candle_name, e))
        })?;

        let dtype = safetensors_dtype_to_ggml(tensor.dtype())?;
        let shape: Vec<u64> = tensor.shape().iter().map(|&s| s as u64).collect();
        let data = tensor.data().to_vec();

        debug!(
            "  Registering: {} shape={:?} dtype={}",
            gguf_name, shape, dtype
        );

        // Register tensor metadata (must happen before writing the header)
        writer.add_tensor(TensorInfo {
            name: gguf_name.to_string(),
            shape: shape.clone(),
            dtype,
        });

        tensor_data_map.push((gguf_name.to_string(), data, shape, dtype));
    }

    // Write the GGUF header and tensor info section
    writer
        .write()
        .map_err(|e| ExperaiError::Export(format!("Failed to write GGUF header: {}", e)))?;

    for (i, (_name, data, _shape, _dtype)) in tensor_data_map.iter().enumerate() {
        writer
            .write_tensor_data(i, data)
            .map_err(|e| ExperaiError::Export(format!("Failed to write tensor data: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| ExperaiError::Export(format!("Failed to finalize GGUF: {}", e)))?;

    // Copy model_config.json alongside the GGUF for runtime reference
    let config_dest = output_dir.join("model_config.json");
    std::fs::copy(checkpoint_dir.join("model_config.json"), &config_dest).map_err(|e| {
        ExperaiError::Export(format!(
            "Failed to copy model_config.json to {}: {}",
            config_dest.display(),
            e
        ))
    })?;

    let gguf_size = std::fs::metadata(&gguf_path)?.len();
    info!(
        "GGUF export complete: {} ({:.1} MB)",
        gguf_path.display(),
        gguf_size as f64 / 1_048_576.0
    );

    Ok(())
}
