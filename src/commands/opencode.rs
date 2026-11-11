//! Package a trained checkpoint so it can be used as a model in OpenCode.
//!
//! Workflow:
//! 1. Export the checkpoint to a self-contained GGUF (weights + tokenizer
//!    metadata + chat template), the format llama.cpp-based runtimes load.
//! 2. Write an Ollama `Modelfile` so `ollama create <name> -f Modelfile`
//!    serves the model on an OpenAI-compatible endpoint (`/v1`).
//! 3. Emit (and optionally merge into `opencode.json`) the OpenCode provider
//!    config so the model shows up as `<provider>/<name>` in `/models`.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{json, Value};
use tracing::info;

use crate::export::{export_to_gguf_to, load_model_config, tokenizer_metadata_from_json};

/// npm package OpenCode uses to talk to any OpenAI-compatible endpoint.
const OPENCODE_OPENAI_COMPATIBLE_PACKAGE: &str = "@opencode/ai/providers/openai-compatible";

/// Sanitize a name into a valid Ollama/OpenCode model id
/// (`[a-zA-Z0-9_-]` with `.`/`:` for tags, lowercase by convention).
pub fn sanitize_model_name(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
            out.push(c.to_ascii_lowercase());
        } else if c == ':' {
            out.push(':');
        } else {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches(['-', '_', '.']);
    if trimmed.is_empty() {
        "experai-model".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Build the OpenCode `providers` + `model` config snippet for a local model.
///
/// Uses the current (v2) `providers` schema: an OpenAI-compatible package with
/// a `settings.baseURL` pointing at the local server, plus an explicit model
/// entry so the model is selectable as `<provider>/<name>`.
pub fn build_opencode_config(
    provider: &str,
    name: &str,
    base_url: &str,
    context_len: usize,
) -> Value {
    let output_limit = (context_len / 2).clamp(1024, 8192);
    json!({
        "$schema": "https://opencode.ai/config.json",
        "providers": {
            provider: {
                "name": format!("experai local ({name})"),
                "package": OPENCODE_OPENAI_COMPATIBLE_PACKAGE,
                "settings": {
                    "baseURL": base_url,
                },
                "models": {
                    name: {
                        "name": name,
                        "modelID": name,
                        "type": "chat",
                        "limit": {
                            "context": context_len,
                            "output": output_limit,
                        },
                        "capabilities": {
                            "tools": false,
                            "temperature": true,
                            "reasoning": { "supported": false },
                        },
                    }
                }
            }
        },
        "model": format!("{provider}/{name}"),
    })
}

/// Merge the provider/model snippet into an existing opencode config file,
/// preserving all other keys. Creates the file (with `$schema`) when missing.
fn merge_into_config(
    config_path: &Path,
    provider: &str,
    name: &str,
    base_url: &str,
    context_len: usize,
) -> Result<()> {
    let snippet = build_opencode_config(provider, name, base_url, context_len);
    let mut config = if config_path.exists() {
        let raw = std::fs::read_to_string(config_path)?;
        if raw.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&raw)?
        }
    } else {
        json!({ "$schema": "https://opencode.ai/config.json" })
    };

    if let Some(parent) = config_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let obj = config.as_object_mut().expect("config root is an object");
    let providers = obj.entry("providers").or_insert_with(|| json!({}));
    let providers_obj = providers.as_object_mut().expect("providers is an object");
    let provider_snippet = &snippet["providers"][provider];
    providers_obj.insert(provider.to_string(), provider_snippet.clone());
    obj.insert("model".to_string(), snippet["model"].clone());

    let pretty = serde_json::to_string_pretty(&config)?;
    std::fs::write(config_path, pretty + "\n")?;
    Ok(())
}

/// Write an Ollama Modelfile referencing the exported GGUF.
fn write_modelfile(
    output_dir: &Path,
    gguf_name: &str,
    context_len: usize,
    stop_token: Option<&str>,
) -> Result<PathBuf> {
    let mut mf = format!(
        "# Modelfile for the experai-trained model '{gguf_name}'.\n\
         # Build the Ollama model with:  ollama create {gguf_name} -f Modelfile\n\
         FROM ./{gguf_name}.gguf\n\n\
         PARAMETER num_ctx {context_len}\n\
         PARAMETER temperature 0.2\n\
         PARAMETER top_p 0.95\n\
         PARAMETER top_k 40\n"
    );
    if let Some(stop) = stop_token {
        mf.push_str(&format!("PARAMETER stop \"{stop}\"\n"));
    }
    let path = output_dir.join("Modelfile");
    std::fs::write(&path, mf)?;
    Ok(path)
}

/// Locate the checkpoint stem (`<dir>/best` preferred) and its model config.
fn resolve_checkpoint(checkpoint: &str) -> Result<(PathBuf, PathBuf)> {
    let dir = Path::new(checkpoint);
    if !dir.is_dir() {
        return Err(anyhow::anyhow!(
            "Checkpoint directory not found: {checkpoint}"
        ));
    }
    if !dir.join("model_config.json").exists() {
        return Err(anyhow::anyhow!(
            "model_config.json not found in {checkpoint} (run training first)"
        ));
    }
    let best = dir.join("best.safetensors");
    if best.exists() {
        return Ok((dir.join("best"), dir.to_path_buf()));
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    candidates.sort();
    let Some(first) = candidates.first() else {
        return Err(anyhow::anyhow!(
            "No .safetensors checkpoint found in {checkpoint}"
        ));
    };
    Ok((first.with_extension(""), dir.to_path_buf()))
}

/// Run the OpenCode packaging workflow.
#[allow(clippy::too_many_arguments)]
pub fn run(
    checkpoint: String,
    name: String,
    output_dir: Option<String>,
    tokenizer: String,
    provider: String,
    base_url: String,
    config_path: Option<String>,
    json: bool,
) -> Result<()> {
    let model_name = sanitize_model_name(&name);
    info!(
        "Packaging checkpoint {} for OpenCode as '{}' (provider={}, base_url={})",
        checkpoint, model_name, provider, base_url
    );

    let (ckpt_stem, ckpt_dir) = resolve_checkpoint(&checkpoint)?;
    let output_dir = match output_dir {
        Some(d) => PathBuf::from(d),
        None => ckpt_dir.join("opencode"),
    };
    std::fs::create_dir_all(&output_dir)?;

    // 1. Export GGUF with embedded tokenizer metadata.
    let model_config = load_model_config(&ckpt_dir)?;
    let tokenizer_json: Value = {
        let raw = std::fs::read_to_string(&tokenizer)
            .map_err(|e| anyhow::anyhow!("Failed to read tokenizer {tokenizer}: {e}"))?;
        serde_json::from_str(&raw)?
    };
    let tok_meta = tokenizer_metadata_from_json(&tokenizer_json, &model_config)?;
    let gguf_path = export_to_gguf_to(
        ckpt_stem.to_str().expect("non-UTF8 checkpoint path"),
        &model_name,
        &output_dir,
        "f16",
        Some(&tok_meta),
    )?;
    info!("GGUF written to {}", gguf_path.display());

    // 2. Write the Ollama Modelfile.
    let eos_token = tok_meta
        .tokens
        .get(tok_meta.eos_token_id as usize)
        .cloned()
        .filter(|t| !t.is_empty());
    let modelfile_path = write_modelfile(
        &output_dir,
        &model_name,
        model_config.max_seq_len,
        eos_token.as_deref(),
    )?;
    info!("Modelfile written to {}", modelfile_path.display());

    // 3. Build (and optionally merge) the OpenCode provider config.
    let snippet =
        build_opencode_config(&provider, &model_name, &base_url, model_config.max_seq_len);
    let snippet_path = output_dir.join("opencode.snippet.json");
    std::fs::write(
        &snippet_path,
        serde_json::to_string_pretty(&snippet)? + "\n",
    )?;

    let mut merged_config: Option<String> = None;
    if let Some(cp) = &config_path {
        merge_into_config(
            Path::new(cp),
            &provider,
            &model_name,
            &base_url,
            model_config.max_seq_len,
        )?;
        merged_config = Some(cp.clone());
        info!("Merged provider config into {}", cp);
    }

    let model_id = format!("{provider}/{model_name}");
    let next_steps = vec![
        format!("ollama create {model_name} -f {}", modelfile_path.display()),
        "ollama serve  # if not already running (OpenCode probes http://127.0.0.1:11434)"
            .to_string(),
        format!("In OpenCode, select the model `{model_id}` (see /models)"),
    ];

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "gguf": gguf_path,
                "modelfile": modelfile_path,
                "snippet": snippet_path,
                "config_path": merged_config,
                "provider": provider,
                "model": model_name,
                "model_id": model_id,
                "base_url": base_url,
                "context_len": model_config.max_seq_len,
                "config_snippet": snippet,
                "next_steps": next_steps,
            }))?
        );
    } else {
        println!("Packaged '{}' for OpenCode:", model_name);
        println!("  GGUF:      {}", gguf_path.display());
        println!("  Modelfile: {}", modelfile_path.display());
        println!(
            "  Config:    {} (merge into opencode.json with --config-path)",
            snippet_path.display()
        );
        if let Some(cp) = &merged_config {
            println!("  Merged into: {cp}");
        }
        println!("  Model id:  {model_id}  (endpoint {base_url})");
        println!("\nNext steps:");
        for step in &next_steps {
            println!("  {step}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_names() {
        assert_eq!(sanitize_model_name("My Model!"), "my-model");
        assert_eq!(sanitize_model_name("coderv2.1"), "coderv2.1");
        assert_eq!(sanitize_model_name("---"), "experai-model");
        assert_eq!(sanitize_model_name("alice:best"), "alice:best");
    }

    #[test]
    fn opencode_config_shape() {
        let cfg = build_opencode_config("experai", "my-model", "http://localhost:11434/v1", 2048);
        assert_eq!(cfg["model"], "experai/my-model");
        let prov = &cfg["providers"]["experai"];
        assert_eq!(prov["package"], "@opencode/ai/providers/openai-compatible");
        assert_eq!(prov["settings"]["baseURL"], "http://localhost:11434/v1");
        let model = &prov["models"]["my-model"];
        assert_eq!(model["modelID"], "my-model");
        assert_eq!(model["type"], "chat");
        assert_eq!(model["limit"]["context"], 2048);
        assert!(model["capabilities"]["temperature"].as_bool().unwrap());
        assert_eq!(model["capabilities"]["reasoning"]["supported"], false);
    }

    #[test]
    fn merge_preserves_existing_keys() {
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("opencode.json");
        std::fs::write(&cfg_path, r#"{"$schema":"https://opencode.ai/config.json","mcp":{"experai":{"type":"local","command":["node","mcp/experai-server.mjs"]}}}"#).unwrap();
        merge_into_config(
            &cfg_path,
            "experai",
            "my-model",
            "http://localhost:11434/v1",
            1024,
        )
        .unwrap();
        let merged: Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        assert!(
            merged["mcp"]["experai"].is_object(),
            "existing mcp config preserved"
        );
        assert_eq!(merged["model"], "experai/my-model");
        assert!(merged["providers"]["experai"]["models"]["my-model"].is_object());
    }

    #[test]
    fn merge_creates_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("opencode.json");
        merge_into_config(&cfg_path, "local", "m", "http://localhost:11434/v1", 512).unwrap();
        let merged: Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        assert_eq!(merged["$schema"], "https://opencode.ai/config.json");
        assert_eq!(merged["model"], "local/m");
    }
}
