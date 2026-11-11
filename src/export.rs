use std::collections::HashMap;
use std::path::{Path, PathBuf};

use gguf_rs::writer::{GGUFWriter, MetadataValue, TensorInfo};
use safetensors::{Dtype, SafeTensors};
use tracing::{debug, info};

use crate::errors::ExperaiError;
use crate::model::ModelConfig;

/// llama.cpp `token_type` values used in `tokenizer.ggml.token_type`.
const TOKEN_TYPE_NORMAL: u32 = 0;
const TOKEN_TYPE_UNKNOWN: u32 = 1;
const TOKEN_TYPE_CONTROL: u32 = 2;
const TOKEN_TYPE_BYTE: u32 = 5;

/// Tokenizer metadata embedded into a GGUF file so llama.cpp-based runtimes
/// (Ollama, LM Studio, OpenCode via a local server) can tokenize prompts.
#[derive(Debug, Clone)]
pub struct TokenizerMetadata {
    /// One entry per token id (index = id, padded with empty strings).
    pub tokens: Vec<String>,
    /// Parallel to `tokens`: llama.cpp token types.
    pub token_types: Vec<u32>,
    /// BPE merges as `"a b"` strings.
    pub merges: Vec<String>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub unk_token_id: u32,
    pub chat_template: String,
}

/// Default chat template written into the GGUF. Uses the bos/eos tokens so a
/// base model can be driven as a chat model by OpenCode/Ollama.
const DEFAULT_CHAT_TEMPLATE: &str = "{%- if messages[0]['role'] == 'system' -%}\n{{- bos_token + messages[0]['content'] + eos_token }}\n{%- set messages = messages[1:] -%}\n{%- endif -%}\n{%- for message in messages -%}\n{{- bos_token + message['role'] | capitalize + ': ' + message['content'] + eos_token }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n{{- bos_token + 'Assistant: ' }}\n{%- endif -%}";

/// Token contents recognised as bos/eos/pad/unk special tokens.
const BOS_NAMES: [&str; 2] = ["<s>", "<|startoftext|>"];
const EOS_NAMES: [&str; 2] = ["</s>", "<|endoftext|>"];
const PAD_NAMES: [&str; 2] = ["<pad>", "<|pad|>"];
const UNK_NAMES: [&str; 3] = ["<unk>", "[UNK]", "<|unk|>"];

/// In GPT-2's byte-level BPE, byte-fallback tokens are single chars with a
/// code point >= 256 (e.g. `Ġ` for byte 32).
fn is_byte_fallback_token(token: &str) -> bool {
    let mut chars = token.chars();
    matches!(chars.next(), Some(c) if (c as u32) >= 256) && chars.next().is_none()
}

/// Build [`TokenizerMetadata`] from a parsed HuggingFace `tokenizers` JSON
/// document, falling back to `model_config` special-token ids.
pub fn tokenizer_metadata_from_json(
    tokenizer_json: &serde_json::Value,
    model_config: &ModelConfig,
) -> Result<TokenizerMetadata, ExperaiError> {
    let vocab_size = model_config.vocab_size;
    let mut tokens: Vec<String> = vec![String::new(); vocab_size];
    let mut token_types: Vec<u32> = vec![TOKEN_TYPE_NORMAL; vocab_size];

    // Base vocab from model.vocab (token -> id).
    if let Some(vocab) = tokenizer_json
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
    {
        for (token, id) in vocab {
            if let Some(id) = id.as_u64() {
                if (id as usize) < vocab_size {
                    tokens[id as usize] = token.clone();
                    if is_byte_fallback_token(token) {
                        token_types[id as usize] = TOKEN_TYPE_BYTE;
                    }
                }
            }
        }
    }

    // Added (special) tokens override base vocab entries and become control
    // tokens; the unk token becomes type unknown.
    let mut bos = None;
    let mut eos = None;
    let mut pad = None;
    let mut unk = None;
    if let Some(added) = tokenizer_json
        .get("added_tokens")
        .and_then(|t| t.as_array())
    {
        for entry in added {
            let (Some(id), Some(content)) = (
                entry.get("id").and_then(|v| v.as_u64()),
                entry.get("content").and_then(|v| v.as_str()),
            ) else {
                continue;
            };
            if (id as usize) >= vocab_size {
                continue;
            }
            tokens[id as usize] = content.to_string();
            let special = entry
                .get("special")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if special {
                token_types[id as usize] = TOKEN_TYPE_CONTROL;
            }
            if BOS_NAMES.contains(&content) {
                bos = Some(id as u32);
            } else if EOS_NAMES.contains(&content) {
                eos = Some(id as u32);
            } else if PAD_NAMES.contains(&content) {
                pad = Some(id as u32);
            } else if UNK_NAMES.contains(&content) {
                unk = Some(id as u32);
                token_types[id as usize] = TOKEN_TYPE_UNKNOWN;
            }
        }
    }

    // BPE merges: entries are either "a b" strings or ["a", "b"] pairs.
    let mut merges = Vec::new();
    if let Some(m) = tokenizer_json
        .get("model")
        .and_then(|m| m.get("merges"))
        .and_then(|v| v.as_array())
    {
        for merge in m {
            match merge {
                serde_json::Value::String(s) => merges.push(s.clone()),
                serde_json::Value::Array(pair) if pair.len() == 2 => {
                    if let (Some(a), Some(b)) = (pair[0].as_str(), pair[1].as_str()) {
                        merges.push(format!("{a} {b}"));
                    }
                }
                _ => {}
            }
        }
    }

    Ok(TokenizerMetadata {
        tokens,
        token_types,
        merges,
        bos_token_id: bos.unwrap_or(model_config.bos_token_id as u32),
        eos_token_id: eos.unwrap_or(model_config.eos_token_id as u32),
        pad_token_id: pad.unwrap_or(model_config.pad_token_id as u32),
        unk_token_id: unk.unwrap_or(model_config.pad_token_id as u32),
        chat_template: DEFAULT_CHAT_TEMPLATE.to_string(),
    })
}

/// Write tokenizer metadata keys into a GGUF writer.
fn write_tokenizer_metadata(
    writer: &mut GGUFWriter,
    meta: &TokenizerMetadata,
) -> Result<(), ExperaiError> {
    writer.add_metadata_array(
        "tokenizer.ggml.tokens",
        meta.tokens
            .iter()
            .map(|t| MetadataValue::String(t.clone()))
            .collect(),
    );
    writer.add_metadata_array(
        "tokenizer.ggml.token_type",
        meta.token_types
            .iter()
            .map(|t| MetadataValue::Uint32(*t))
            .collect(),
    );
    if !meta.merges.is_empty() {
        writer.add_metadata_array(
            "tokenizer.ggml.merges",
            meta.merges
                .iter()
                .map(|m| MetadataValue::String(m.clone()))
                .collect(),
        );
    }
    // Key names verified against llama.cpp's gguf-py `Keys.Tokenizer`.
    writer.add_metadata("tokenizer.ggml.model", "bpe");
    writer.add_metadata("tokenizer.ggml.pre", "default");
    writer.add_metadata_u32("tokenizer.ggml.bos_token_id", meta.bos_token_id);
    writer.add_metadata_u32("tokenizer.ggml.eos_token_id", meta.eos_token_id);
    writer.add_metadata_u32("tokenizer.ggml.padding_token_id", meta.pad_token_id);
    writer.add_metadata_u32("tokenizer.ggml.unknown_token_id", meta.unk_token_id);
    writer.add_metadata_bool("tokenizer.ggml.add_bos_token", false);
    writer.add_metadata_bool("tokenizer.ggml.add_eos_token", false);
    writer.add_metadata("tokenizer.chat_template", &meta.chat_template);
    Ok(())
}

/// GGML dtype constants used in the GGUF file format.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_BF16: u32 = 30;

/// RoPE base, matching [`crate::model`]'s `ROPE_THETA` (GPT-NeoX/Llama convention).
const ROPE_THETA_F32: f32 = 10000.0;

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

/// Map a Candle/safetensors tensor name to its GGUF `llama`-arch equivalent
/// (names verified against llama.cpp's gguf-py tensor name map).
///
/// - `embedding.weight` → `token_embd.weight`
/// - `norm.weight` → `output_norm.weight`
/// - `lm_head.weight` → `output.weight`
/// - Layer tensors `layers.{i}.*` → `blk.{i}.*` with sub-component renaming
///   (e.g. `self_attn.q_proj` → `attn_q`, `input_layernorm` → `attn_norm`,
///   `post_attention_layernorm` → `ffn_norm`, `mlp.gate_proj` → `ffn_gate`).
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
                format!("attn_norm.{}", rest)
            }
            s if s.starts_with("post_attention_layernorm.") => {
                let rest = s
                    .strip_prefix("post_attention_layernorm.")
                    .ok_or_else(|| ExperaiError::Export("strip_prefix failed".to_string()))?;
                format!("ffn_norm.{}", rest)
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
pub fn load_model_config(checkpoint_dir: &Path) -> Result<ModelConfig, ExperaiError> {
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
    export_to_gguf_to(checkpoint_path, model_name, &output_dir, _dtype, None)?;
    Ok(())
}

/// Export a safetensors checkpoint to GGUF in an explicit output directory,
/// optionally embedding tokenizer metadata so the file is self-contained for
/// llama.cpp-based runtimes (Ollama, LM Studio, OpenCode local providers).
///
/// Returns the path of the written GGUF file.
pub fn export_to_gguf_to(
    checkpoint_path: &str,
    model_name: &str,
    output_dir: &Path,
    _dtype: &str,
    tokenizer_meta: Option<&TokenizerMetadata>,
) -> Result<PathBuf, ExperaiError> {
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

    std::fs::create_dir_all(output_dir).map_err(|e| {
        ExperaiError::Export(format!("Failed to create {}: {}", output_dir.display(), e))
    })?;

    let gguf_path = output_dir.join(format!("{}.gguf", model_name));
    info!("Writing GGUF to {}", gguf_path.display());

    // Initialize GGUF writer (version 3)
    let mut writer = GGUFWriter::new(&gguf_path, 3)
        .map_err(|e| ExperaiError::Export(format!("Failed to create GGUF writer: {}", e)))?;

    // Write architecture and hyperparameter metadata.
    //
    // The model is RoPE + RMSNorm + SwiGLU with no biases and no learned
    // positional embeddings, which is exactly llama.cpp's `llama` architecture
    // (key/tensor names verified against llama.cpp's gguf-py constants). Using
    // `gpt2` here would make Ollama/llama-server misinterpret the weights.
    writer.add_metadata("general.architecture", "llama");
    writer.add_metadata("general.name", model_name);
    // general.file_type: 0 = all F32, 1 = mostly F16 (llama.cpp convention).
    writer.add_metadata_u32("general.file_type", 0);

    let head_dim = model_config.hidden_size / model_config.num_heads;
    writer.add_metadata_u32("llama.block_count", model_config.num_layers as u32);
    writer.add_metadata_u32("llama.attention.head_count", model_config.num_heads as u32);
    writer.add_metadata_u32(
        "llama.attention.head_count_kv",
        model_config.num_heads as u32,
    );
    writer.add_metadata_u32("llama.embedding_length", model_config.hidden_size as u32);
    writer.add_metadata_u32(
        "llama.feed_forward_length",
        model_config.intermediate_size as u32,
    );
    writer.add_metadata_u32("llama.context_length", model_config.max_seq_len as u32);
    writer.add_metadata_u32("llama.vocab_size", model_config.vocab_size as u32);
    writer.add_metadata_f32(
        "llama.attention.layer_norm_rms_epsilon",
        model_config.layer_norm_eps as f32,
    );
    writer.add_metadata_f32("llama.rope.freq_base", ROPE_THETA_F32);
    writer.add_metadata_u32("llama.rope.dimension_count", head_dim as u32);

    // Embed tokenizer metadata (vocab, merges, special ids, chat template) so
    // the GGUF is directly consumable by Ollama / OpenCode local providers.
    if let Some(meta) = tokenizer_meta {
        write_tokenizer_metadata(&mut writer, meta)?;
        info!(
            "Embedded tokenizer metadata: {} tokens, {} merges",
            meta.tokens.len(),
            meta.merges.len()
        );
    }

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

    Ok(gguf_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_names_map_to_llama_arch() {
        assert_eq!(
            map_tensor_name("embedding.weight").unwrap(),
            "token_embd.weight"
        );
        assert_eq!(
            map_tensor_name("norm.weight").unwrap(),
            "output_norm.weight"
        );
        assert_eq!(map_tensor_name("lm_head.weight").unwrap(), "output.weight");
        assert_eq!(
            map_tensor_name("layers.3.self_attn.q_proj.weight").unwrap(),
            "blk.3.attn_q.weight"
        );
        assert_eq!(
            map_tensor_name("layers.0.self_attn.o_proj.weight").unwrap(),
            "blk.0.attn_output.weight"
        );
        assert_eq!(
            map_tensor_name("layers.1.input_layernorm.weight").unwrap(),
            "blk.1.attn_norm.weight"
        );
        assert_eq!(
            map_tensor_name("layers.1.post_attention_layernorm.weight").unwrap(),
            "blk.1.ffn_norm.weight"
        );
        assert_eq!(
            map_tensor_name("layers.2.mlp.gate_proj.weight").unwrap(),
            "blk.2.ffn_gate.weight"
        );
        assert_eq!(
            map_tensor_name("layers.2.mlp.up_proj.weight").unwrap(),
            "blk.2.ffn_up.weight"
        );
        assert_eq!(
            map_tensor_name("layers.2.mlp.down_proj.weight").unwrap(),
            "blk.2.ffn_down.weight"
        );
        assert!(map_tensor_name("bogus.weight").is_err());
    }

    #[test]
    fn tokenizer_metadata_from_hf_json() {
        let cfg = ModelConfig::default();
        let tok_json: serde_json::Value = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {"hello": 10, "Ġ": 11},
                "merges": ["hello world", ["a", "b"]],
            },
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": true},
                {"id": 1, "content": "<s>", "special": true},
                {"id": 2, "content": "</s>", "special": true},
                {"id": 3, "content": "<unk>", "special": true},
            ],
        });
        let meta = tokenizer_metadata_from_json(&tok_json, &cfg).unwrap();
        assert_eq!(meta.tokens.len(), cfg.vocab_size);
        assert_eq!(meta.tokens[10], "hello");
        assert_eq!(meta.tokens[11], "Ġ");
        assert_eq!(meta.token_types[0], TOKEN_TYPE_CONTROL);
        assert_eq!(meta.token_types[3], TOKEN_TYPE_UNKNOWN);
        // "Ġ" (U+0120 >= 256, single char) is a byte-fallback token.
        assert_eq!(meta.token_types[11], TOKEN_TYPE_BYTE);
        assert_eq!(
            meta.merges,
            vec!["hello world".to_string(), "a b".to_string()]
        );
        assert_eq!(meta.bos_token_id, 1);
        assert_eq!(meta.eos_token_id, 2);
        assert_eq!(meta.pad_token_id, 0);
        assert_eq!(meta.unk_token_id, 3);
    }

    #[test]
    fn tokenizer_metadata_falls_back_to_model_config_ids() {
        let cfg = ModelConfig::default();
        let tok_json: serde_json::Value =
            serde_json::json!({"model": {"type": "BPE", "vocab": {}, "merges": []}});
        let meta = tokenizer_metadata_from_json(&tok_json, &cfg).unwrap();
        assert_eq!(meta.bos_token_id, cfg.bos_token_id as u32);
        assert_eq!(meta.eos_token_id, cfg.eos_token_id as u32);
        assert_eq!(meta.pad_token_id, cfg.pad_token_id as u32);
    }
}
