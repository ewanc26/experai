//! Fetch tokenizer/config files for a HuggingFace model id.
//!
//! `experai fetch --model gpt2` downloads `tokenizer.json` (and optionally
//! `config.json`) from `https://huggingface.co/<model>/resolve/<revision>/<file>`
//! so training/packaging work out of the box without manual `curl`.

use anyhow::{Context, Result};
use tracing::info;

const HF_BASE: &str = "https://huggingface.co";

/// What to download from the HuggingFace repo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FetchWhat {
    #[default]
    Tokenizer,
    Config,
    All,
}

impl std::str::FromStr for FetchWhat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "tokenizer" => Ok(Self::Tokenizer),
            "config" => Ok(Self::Config),
            "all" => Ok(Self::All),
            other => Err(anyhow::anyhow!(
                "Unknown --what '{other}' (expected tokenizer|config|all)"
            )),
        }
    }
}

/// One downloaded file.
pub struct FetchedFile {
    pub file_name: String,
    pub path: String,
    pub url: String,
    pub bytes: usize,
}

/// Download a single file from a HuggingFace repo, following redirects.
async fn download(
    client: &reqwest::Client,
    model: &str,
    revision: &str,
    file: &str,
) -> Result<(String, Vec<u8>)> {
    let url = format!("{HF_BASE}/{model}/resolve/{revision}/{file}");
    info!("GET {url}");
    let resp = client
        .get(&url)
        .header(reqwest::header::USER_AGENT, "experai/0.3")
        .send()
        .await
        .with_context(|| format!("request failed for {url}"))?;
    let status = resp.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(anyhow::anyhow!(
            "{file} not found in HuggingFace repo '{model}' (revision {revision}) — check the model id"
        ));
    }
    if !status.is_success() {
        return Err(anyhow::anyhow!("HuggingFace returned {status} for {url}"));
    }
    let bytes = resp.bytes().await?.to_vec();
    Ok((url, bytes))
}

/// Run the fetch workflow.
pub fn run(
    model: String,
    revision: String,
    what: String,
    output_dir: String,
    json: bool,
) -> Result<()> {
    let what: FetchWhat = what.parse()?;
    let files: Vec<&str> = match what {
        FetchWhat::Tokenizer => vec!["tokenizer.json"],
        FetchWhat::Config => vec!["config.json"],
        FetchWhat::All => vec!["tokenizer.json", "config.json"],
    };

    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create output dir {output_dir}"))?;

    let rt = tokio::runtime::Runtime::new()?;
    let fetched = rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;
        let mut out = Vec::new();
        for file in files {
            let (url, bytes) = download(&client, &model, &revision, file).await?;
            let path = format!("{output_dir}/{file}");
            std::fs::write(&path, &bytes).with_context(|| format!("Failed to write {path}"))?;
            info!("Saved {} ({} bytes) to {}", file, bytes.len(), path);
            out.push(FetchedFile {
                file_name: file.to_string(),
                path,
                url,
                bytes: bytes.len(),
            });
        }
        Ok::<Vec<FetchedFile>, anyhow::Error>(out)
    })?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": model,
                "revision": revision,
                "files": fetched.iter().map(|f| serde_json::json!({
                    "file": f.file_name,
                    "path": f.path,
                    "url": f.url,
                    "bytes": f.bytes,
                })).collect::<Vec<_>>(),
            }))?
        );
    } else {
        for f in &fetched {
            println!(
                "Saved {} ({} bytes)\n  from {}\n  to   {}",
                f.file_name, f.bytes, f.url, f.path
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_what() {
        assert_eq!(
            "tokenizer".parse::<FetchWhat>().unwrap(),
            FetchWhat::Tokenizer
        );
        assert_eq!("ALL".parse::<FetchWhat>().unwrap(), FetchWhat::All);
        assert!("weights".parse::<FetchWhat>().is_err());
    }
}
