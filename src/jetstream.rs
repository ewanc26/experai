use anyhow::{Result, anyhow};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;
use tracing::{info, error, debug, warn, trace};

use crate::data::{Dataset, DatasetSample};

/// Jetstream configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JetstreamConfig {
    /// Jetstream host (e.g., "jetstream2.us-east.bsky.network")
    pub host: String,
    /// Collections to subscribe to (e.g., ["app.bsky.feed.post"])
    pub collections: Vec<String>,
    /// DIDs to filter (empty = all users)
    pub dids: Vec<String>,
    /// Maximum samples to collect before stopping
    pub max_samples: usize,
    /// Batch size for accumulating samples
    pub batch_size: usize,
    /// Maximum time to collect (in seconds)
    pub max_duration_secs: u64,
    /// Enable compression
    pub compression: bool,
}

impl Default for JetstreamConfig {
    fn default() -> Self {
        Self {
            host: "jetstream2.us-east.bsky.network".to_string(),
            collections: vec!["app.bsky.feed.post".to_string()],
            dids: Vec::new(),
            max_samples: 10000,
            batch_size: 100,
            max_duration_secs: 3600, // 1 hour
            compression: false,
        }
    }
}

/// A single event from Jetstream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JetstreamPost {
    pub did: String,
    pub text: String,
    pub created_at: Option<String>,
    pub rkey: Option<String>,
}

/// Statistics from a Jetstream collection run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JetstreamStats {
    pub total_events: usize,
    pub valid_posts: usize,
    pub filtered_posts: usize,
    pub duration_secs: f64,
    pub posts_per_second: f64,
}

/// Collect posts from Jetstream into a Dataset
pub async fn collect_from_jetstream(
    config: &JetstreamConfig,
    tokenizer: tokenizers::Tokenizer,
) -> Result<(Dataset, JetstreamStats)> {
    let start = Instant::now();
    let collected = Arc::new(AtomicUsize::new(0));
    let filtered = Arc::new(AtomicUsize::new(0));
    let cancelled = Arc::new(AtomicBool::new(false));

    let (tx, mut rx) = mpsc::channel::<JetstreamPost>(config.batch_size * 2);

    // Build the WebSocket URL with query parameters
    let url = build_jetstream_url(config)?;
    info!("Connecting to Jetstream: {}", url);

    // Set up cancellation token
    let cancel_token = CancellationToken::new();
    let cancel_clone = cancel_token.clone();

    // Spawn the WebSocket consumer
    let config_for_consumer = config.clone();
    let collected_clone = collected.clone();
    let cancelled_clone = cancelled.clone();

    tokio::spawn(async move {
        match run_websocket_consumer(&config_for_consumer, tx.clone(), cancel_clone.clone()).await {
            Ok(()) => {}
            Err(e) => {
                error!("WebSocket consumer error: {}", e);
            }
        }
        let _ = tx.closed().await;
    });

    // Collect posts into a Vec until we hit limits or cancellation
    let mut posts = Vec::with_capacity(config.max_samples);
    let timeout = Duration::from_secs(config.max_duration_secs);
    let dids_filter = config.dids.clone();

    let mut tick_count: u64 = 0;
    loop {
        if posts.len() >= config.max_samples {
            info!("Reached max samples limit: {}", config.max_samples);
            break;
        }

        if start.elapsed() >= timeout {
            info!("Reached max duration limit: {}s", config.max_duration_secs);
            break;
        }

        if cancelled_clone.load(Ordering::Relaxed) {
            info!("Received cancellation signal");
            break;
        }

        tokio::select! {
            Some(post) = rx.recv() => {
                let sample = DatasetSample {
                    text: post.text.clone(),
                    tokens: Vec::new(), // Will be tokenized later
                    label: Some(format!("jetstream:{}", post.did)),
                    did: Some(post.did.clone()),
                };

                if should_include_post(&post, &dids_filter) {
                    posts.push(sample);
                    collected_clone.fetch_add(1, Ordering::Relaxed);
                } else {
                    filtered.fetch_add(1, Ordering::Relaxed);
                }
            }
            _ = tokio::time::sleep(Duration::from_secs(5)) => {
                tick_count += 1;
                info!("Tick #{}: {:.0}s elapsed, {} posts collected", tick_count, start.elapsed().as_secs_f64(), posts.len());
            }
        }
    }

    // Signal completion and wait for consumer to finish
    cancel_token.cancel();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Tokenize all collected samples
    for sample in &mut posts {
        let tokens = tokenizer
            .encode(sample.text.as_str(), true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?
            .get_ids()
            .to_vec();
        sample.tokens = tokens;
    }

    let duration = start.elapsed().as_secs_f64();
    let stats = JetstreamStats {
        total_events: collected.load(Ordering::Relaxed) + filtered.load(Ordering::Relaxed),
        valid_posts: posts.len(),
        filtered_posts: filtered.load(Ordering::Relaxed),
        duration_secs: duration,
        posts_per_second: if duration > 0.0 { posts.len() as f64 / duration } else { 0.0 },
    };

    info!(
        "Jetstream collection complete: {} posts in {:.1}s ({:.1} posts/s)",
        stats.valid_posts, stats.duration_secs, stats.posts_per_second
    );

    Ok((
        Dataset {
            samples: posts,
            tokenizer,
            max_length: 512, // Default max length
        },
        stats,
    ))
}

/// Build the WebSocket URL with query parameters
fn build_jetstream_url(config: &JetstreamConfig) -> Result<String> {
    let mut url = format!("wss://{}/subscribe", config.host);

    let mut params = Vec::new();

    if !config.collections.is_empty() {
        for collection in &config.collections {
            params.push(format!("wantedCollections={}", collection));
        }
    }

    if !config.dids.is_empty() {
        for did in &config.dids {
            params.push(format!("wantedDids={}", did));
        }
    }

    if config.compression {
        params.push("compression=zstd".to_string());
    }

    if !params.is_empty() {
        url.push('?');
        url.push_str(&params.join("&"));
    }

    debug!("Built Jetstream URL: {}", url);
    Ok(url)
}

/// Check if a post should be included based on DID filtering
fn should_include_post(post: &JetstreamPost, dids: &[String]) -> bool {
    if dids.is_empty() {
        return true;
    }
    dids.iter().any(|d| d == &post.did)
}

/// Run the WebSocket consumer (real implementation using tokio-tungstenite)
async fn run_websocket_consumer(
    config: &JetstreamConfig,
    tx: mpsc::Sender<JetstreamPost>,
    cancel_token: CancellationToken,
) -> Result<()> {
    let url = build_jetstream_url(config)?;
    info!("Connecting to Jetstream: {}", url);

    loop {
        if cancel_token.is_cancelled() {
            info!("Consumer cancelled before connecting");
            break;
        }

        match connect_and_stream(&url, &tx, &cancel_token, &config.collections, &config.dids).await {
            Ok(()) => {
                info!("WebSocket stream ended gracefully");
                break;
            }
            Err(e) => {
                warn!("Jetstream connection error: {}", e);
                if cancel_token.is_cancelled() {
                    break;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }

    Ok(())
}

async fn connect_and_stream(
    url: &str,
    tx: &mpsc::Sender<JetstreamPost>,
    cancel_token: &CancellationToken,
    wanted_collections: &[String],
    dids_filter: &[String],
) -> Result<()> {
    let (ws_stream, _) = tokio_tungstenite::connect_async(url).await?;
    info!("WebSocket connected to Jetstream");

    let (mut write, mut read) = ws_stream.split();
    let mut msg_count: u64 = 0;
    let mut parsed_count: u64 = 0;
    let mut sent_count: u64 = 0;

    loop {
        tokio::select! {
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        msg_count += 1;
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                            if let Some(post) = parse_jetstream_message(&json, wanted_collections) {
                                parsed_count += 1;
                                if should_include_post(&post, dids_filter) {
                                    if tx.send(post).await.is_err() {
                                        break;
                                    }
                                    sent_count += 1;
                                }
                            }
                        }
                        if msg_count.is_multiple_of(500) {
                            debug!("Stats: {} received, {} parsed, {} sent", msg_count, parsed_count, sent_count);
                        }
                    }
                    Some(Ok(Message::Binary(_))) => {
                        msg_count += 1;
                    }
                    Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) | Some(Ok(Message::Frame(_))) => {}
                    Some(Ok(Message::Close(_))) => {
                        info!("WebSocket closed by server: {} received, {} parsed, {} sent", msg_count, parsed_count, sent_count);
                        break;
                    }
                    Some(Err(e)) => {
                        return Err(anyhow!("WebSocket error: {}", e));
                    }
                    None => {
                        info!("WebSocket stream ended: {} received, {} parsed, {} sent", msg_count, parsed_count, sent_count);
                        break;
                    }
                }
            }
            _ = cancel_token.cancelled() => {
                info!("Consumer cancelled: {} received, {} parsed, {} sent", msg_count, parsed_count, sent_count);
                let _ = write.close().await;
                break;
            }
        }
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
struct JetstreamCommit {
    #[serde(rename = "collection")]
    collection: Option<String>,
    #[serde(rename = "rkey")]
    rkey: Option<String>,
    #[serde(rename = "record")]
    record: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct JetstreamMessage {
    #[serde(rename = "did")]
    did: Option<String>,
    #[serde(rename = "kind")]
    kind: Option<String>,
    #[serde(rename = "commit")]
    commit: Option<JetstreamCommit>,
    #[serde(rename = "time")]
    _time: Option<String>,
}

fn parse_jetstream_message(
    value: &serde_json::Value,
    wanted_collections: &[String],
) -> Option<JetstreamPost> {
    let msg: JetstreamMessage = match serde_json::from_value(value.clone()) {
        Ok(m) => m,
        Err(_) => {
            trace!("Failed to deserialize Jetstream message");
            return None;
        }
    };

    if msg.kind.as_deref() != Some("commit") {
        trace!("Skipping non-commit message: kind={:?}", msg.kind);
        return None;
    }

    let did = msg.did?;
    let commit = msg.commit?;

    if let Some(col) = &commit.collection {
        if !wanted_collections.is_empty() && !wanted_collections.contains(col) {
            trace!("Skipping collection '{}' (not in wanted list)", col);
            return None;
        }
    }

    let record = commit.record?;

    let text = record.get("text")?.as_str()?.trim().to_string();
    if text.is_empty() {
        trace!("Skipping empty post from {}", did);
        return None;
    }

    if commit.collection.as_deref() == Some("app.bsky.feed.post") {
        let langs = record.get("langs").and_then(|v| v.as_array());
        let is_english = langs.is_some_and(|arr| {
            arr.iter().any(|l| l.as_str() == Some("en"))
        });
        if !is_english {
            trace!("Skipping non-English post from {}", did);
            return None;
        }
    }

    trace!("Accepted post from {} ({} chars)", did, text.len());

    let created_at = record
        .get("createdAt")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Some(JetstreamPost {
        did,
        text,
        created_at,
        rkey: commit.rkey,
    })
}

/// Create a Dataset from saved Jetstream posts (JSONL file)
pub fn load_jetstream_dataset(
    path: &str,
    tokenizer: tokenizers::Tokenizer,
    max_length: usize,
) -> Result<Dataset> {
    info!("Loading Jetstream dataset from {} (max_length={})", path, max_length);
    Dataset::from_jsonl(path, tokenizer, max_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jetstream_config_defaults() {
        let config = JetstreamConfig::default();
        assert_eq!(config.host, "jetstream2.us-east.bsky.network");
        assert!(config.collections.contains(&"app.bsky.feed.post".to_string()));
        assert!(config.dids.is_empty());
        assert_eq!(config.max_samples, 10000);
    }

    #[test]
    fn test_build_jetstream_url_no_filters() {
        let config = JetstreamConfig::default();
        let url = build_jetstream_url(&config).unwrap();
        assert!(url.starts_with("wss://"));
        assert!(url.contains("/subscribe?"));
    }

    #[test]
    fn test_build_jetstream_url_with_collections() {
        let config = JetstreamConfig {
            collections: vec!["app.bsky.feed.like".to_string()],
            ..Default::default()
        };
        let url = build_jetstream_url(&config).unwrap();
        assert!(url.contains("wantedCollections=app.bsky.feed.like"));
    }

    #[test]
    fn test_build_jetstream_url_with_dids() {
        let config = JetstreamConfig {
            dids: vec!["did:plc:abc123".to_string()],
            ..Default::default()
        };
        let url = build_jetstream_url(&config).unwrap();
        assert!(url.contains("wantedDids=did:plc:abc123"));
    }

    #[test]
    fn test_should_include_post_no_filter() {
        let post = JetstreamPost {
            did: "did:plc:abc".to_string(),
            text: "Hello".to_string(),
            created_at: None,
            rkey: None,
        };
        assert!(should_include_post(&post, &[]));
    }

    #[test]
    fn test_should_include_post_with_filter() {
        let post = JetstreamPost {
            did: "did:plc:abc".to_string(),
            text: "Hello".to_string(),
            created_at: None,
            rkey: None,
        };
        let dids = vec!["did:plc:abc".to_string(), "did:plc:def".to_string()];
        assert!(should_include_post(&post, &dids));

        let post2 = JetstreamPost {
            did: "did:plc:xyz".to_string(),
            text: "Hello".to_string(),
            created_at: None,
            rkey: None,
        };
        assert!(!should_include_post(&post2, &dids));
    }

    #[test]
    fn test_jetstream_stats_default() {
        let stats = JetstreamStats {
            total_events: 100,
            valid_posts: 95,
            filtered_posts: 5,
            duration_secs: 10.0,
            posts_per_second: 9.5,
        };
        assert_eq!(stats.valid_posts, 95);
        assert!((stats.posts_per_second - 9.5).abs() < 0.01);
    }
}
