use anyhow::{anyhow, Result};
use atrium_api::client::AtpServiceClient;
use atrium_api::com::atproto::identity::resolve_handle::ParametersData as ResolveParams;
use atrium_api::com::atproto::repo::list_records::ParametersData as ListRecordsParams;
use atrium_api::types::string::Handle;
use atrium_xrpc::types::AuthorizationToken;
use atrium_xrpc::{HttpClient, XrpcClient};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Configuration for fetching posts via the AT Protocol (used by Bluesky/PDS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATProtocolConfig {
    pub pds_url: String,
    pub handle: String,
    pub max_samples: usize,
}

impl Default for ATProtocolConfig {
    fn default() -> Self {
        Self {
            pds_url: "https://bsky.social".to_string(),
            handle: "public".to_string(),
            max_samples: 1000,
        }
    }
}

/// A single post record fetched via the AT Protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATProtocolSample {
    pub uri: String,
    pub cid: String,
    pub text: String,
    pub created_at: String,
    pub author: String,
}

/// AT Protocol client for resolving handles and listing repository records.
pub struct ATProtocolClient {
    client: AtpServiceClient<atrium_xrpc_client::reqwest::ReqwestClient>,
}

impl ATProtocolClient {
    /// Create a new client pointed at the given PDS URL.
    pub fn new(pds_url: &str) -> Self {
        debug!("Creating AT Protocol client for PDS: {}", pds_url);
        let http_client = atrium_xrpc_client::reqwest::ReqwestClient::new(pds_url);
        let client = AtpServiceClient::new(http_client);
        Self { client }
    }

    /// Resolve a Bluesky handle (e.g. `"alice.bsky.social"`) to a DID.
    pub async fn resolve_handle(&self, handle: &str) -> Result<String> {
        info!("Resolving handle: {}", handle);
        let response = self
            .client
            .service
            .com
            .atproto
            .identity
            .resolve_handle(
                ResolveParams {
                    handle: Handle::new(handle.to_string())
                        .map_err(|e| anyhow!("Invalid handle '{}': {}", handle, e))?,
                }
                .into(),
            )
            .await
            .map_err(|e| anyhow!("Failed to resolve handle '{}': {}", handle, e))?;

        let did = response.did.to_string();
        info!("Resolved handle '{}' -> DID: {}", handle, did);
        Ok(did)
    }

    /// List records from a repository collection in a single page.
    ///
    /// Returns a list of `(uri, cid, value)` tuples for each record.
    pub async fn list_records(
        &self,
        repo: &str,
        collection: &str,
        limit: Option<u8>,
        cursor: Option<String>,
    ) -> Result<Vec<(String, String, serde_json::Value)>> {
        debug!(
            "Listing records: repo={}, collection={}, limit={:?}",
            repo, collection, limit
        );
        let params = ListRecordsParams {
            collection: collection
                .parse()
                .map_err(|e| anyhow!("Invalid collection NSID: {}", e))?,
            repo: repo
                .parse()
                .map_err(|e| anyhow!("Invalid repo identifier: {}", e))?,
            limit: limit
                .map(|l| l.try_into().map_err(|_| anyhow!("Invalid limit: {}", l)))
                .transpose()?,
            cursor,
            reverse: None,
        };

        let response = self
            .client
            .service
            .com
            .atproto
            .repo
            .list_records(params.into())
            .await
            .map_err(|e| anyhow!("Failed to list records: {}", e))?;

        let mut records = Vec::new();
        for record in response.data.records {
            let value = serde_json::to_value(&record.data.value).unwrap_or(serde_json::Value::Null);
            let cid = serde_json::to_value(&record.data.cid)
                .unwrap_or(serde_json::Value::Null)
                .as_str()
                .unwrap_or("")
                .to_string();
            records.push((record.data.uri, cid, value));
        }

        debug!(
            "Listed {} records from collection '{}'",
            records.len(),
            collection
        );
        Ok(records)
    }

    /// List records from a repository collection with automatic pagination.
    ///
    /// Fetches records in batches of up to 100, following the cursor until
    /// `max_records` is reached or no more pages remain.
    pub async fn list_records_paginated(
        &self,
        repo: &str,
        collection: &str,
        max_records: usize,
    ) -> Result<Vec<(String, String, serde_json::Value)>> {
        let mut all_records = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            // Cap the batch size to remaining records needed
            let batch_size = (max_records - all_records.len()).min(100) as u8;
            if batch_size == 0 {
                break;
            }

            debug!(
                "Fetching records (batch of {}, total so far: {})",
                batch_size,
                all_records.len()
            );

            let params = ListRecordsParams {
                collection: collection
                    .parse()
                    .map_err(|e| anyhow!("Invalid collection NSID: {}", e))?,
                repo: repo
                    .parse()
                    .map_err(|e| anyhow!("Invalid repo identifier: {}", e))?,
                limit: Some(
                    batch_size
                        .try_into()
                        .map_err(|_| anyhow!("Invalid limit"))?,
                ),
                cursor: cursor.clone(),
                reverse: None,
            };

            let response = self
                .client
                .service
                .com
                .atproto
                .repo
                .list_records(params.into())
                .await
                .map_err(|e| anyhow!("Failed to list records: {}", e))?;

            let batch_len = response.data.records.len();
            for record in response.data.records {
                let value =
                    serde_json::to_value(&record.data.value).unwrap_or(serde_json::Value::Null);
                let cid = serde_json::to_value(&record.data.cid)
                    .unwrap_or(serde_json::Value::Null)
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                all_records.push((record.data.uri, cid, value));
            }

            // Fewer records than requested means we've reached the end
            if batch_len < batch_size as usize {
                break;
            }

            // Advance to the next page using the returned cursor
            cursor = response.data.cursor;
            if cursor.is_none() {
                break;
            }
        }

        info!(
            "Paginated list complete: {} total records from collection '{}'",
            all_records.len(),
            collection
        );
        Ok(all_records)
    }
}

/// An in-memory collection of AT Protocol post samples.
pub struct ATProtocolDataset {
    pub config: ATProtocolConfig,
    pub samples: Vec<ATProtocolSample>,
}

impl ATProtocolDataset {
    /// Create an empty dataset for the given configuration.
    pub fn new(config: ATProtocolConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
        }
    }

    /// Fetch posts from the AT Protocol and populate `self.samples`.
    ///
    /// Resolves the configured handle to a DID, then paginates through
    /// the `app.bsky.feed.post` collection, extracting text and metadata
    /// from each record.
    pub async fn load_from_at_protocol(&mut self) -> Result<()> {
        let client = ATProtocolClient::new(&self.config.pds_url);

        let did = client.resolve_handle(&self.config.handle).await?;
        info!("Resolved DID: {} for handle: {}", did, self.config.handle);

        let records = client
            .list_records_paginated(&did, "app.bsky.feed.post", self.config.max_samples)
            .await?;
        info!("Found {} records for DID: {}", records.len(), did);

        for (uri, cid, value) in records.into_iter().take(self.config.max_samples) {
            if let Some(text) = extract_text_from_value(&value) {
                let created_at = value
                    .get("createdAt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                self.samples.push(ATProtocolSample {
                    uri,
                    cid,
                    text,
                    created_at,
                    author: did.clone(),
                });
            }
        }

        info!("Loaded {} samples from AT Protocol", self.samples.len());
        Ok(())
    }

    /// Return a slice of all loaded samples.
    pub fn get_samples(&self) -> &[ATProtocolSample] {
        &self.samples
    }

    /// Return the number of loaded samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Return `true` if no samples have been loaded.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Extract and trim the `"text"` field from a record's JSON value.
///
/// Returns `None` if the field is missing, not a string, or empty after trimming.
pub fn extract_text_from_value(value: &serde_json::Value) -> Option<String> {
    if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
        let cleaned = text.trim();
        if !cleaned.is_empty() {
            return Some(cleaned.to_string());
        }
    }
    None
}

/// Convenience function: resolve a handle, fetch posts, and return samples.
pub async fn load_at_protocol_dataset(config: ATProtocolConfig) -> Result<Vec<ATProtocolSample>> {
    let mut dataset = ATProtocolDataset::new(config);
    dataset.load_from_at_protocol().await?;
    Ok(dataset.samples)
}

// ---------------------------------------------------------------------------
// Authenticated publishing of model weights
// ---------------------------------------------------------------------------

/// Default blob chunk size (1 MB). AT Protocol PDS servers enforce a per-blob
/// size limit of roughly 1 MB, so large checkpoint files must be split.
pub const DEFAULT_BLOB_CHUNK_SIZE: usize = 1_000_000;

/// Default collection NSID for published weight records.
pub const DEFAULT_WEIGHT_COLLECTION: &str = "click.croft.experai.weight";

/// Configuration for publishing trained model weights to an AT Protocol repo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATProtocolPublishConfig {
    /// Personal Data Server URL (e.g. `https://bsky.social`).
    pub pds_url: String,
    /// Bluesky handle or DID of the account that will own the record.
    pub handle: String,
    /// Password or app password for the account.
    pub password: String,
    /// Record key (rkey) for the published record. Auto-generated when `None`.
    pub rkey: Option<String>,
    /// NSID of the collection to publish under (default `click.croft.experai.weight`).
    pub collection: String,
    /// Maximum bytes per blob chunk during upload.
    pub chunk_size: usize,
    /// Human-readable name for the weight record.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
}

impl Default for ATProtocolPublishConfig {
    fn default() -> Self {
        Self {
            pds_url: "https://bsky.social".to_string(),
            handle: String::new(),
            password: String::new(),
            rkey: None,
            collection: DEFAULT_WEIGHT_COLLECTION.to_string(),
            chunk_size: DEFAULT_BLOB_CHUNK_SIZE,
            name: "experai-model".to_string(),
            description: None,
        }
    }
}

/// Result returned after a successful weight publication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishResult {
    /// AT Protocol URI of the created record (e.g. `at://did:plc:xxx/click.croft.experai.weight/rkey`).
    pub uri: String,
    /// CID of the created record.
    pub cid: String,
    /// Total bytes published.
    pub size: usize,
    /// Number of blob chunks uploaded.
    pub chunks: usize,
}

/// Wrapper around [`atrium_xrpc_client::reqwest::ReqwestClient`] that injects
/// a bearer access-JWT into every XRPC request, enabling authenticated writes
/// such as `com.atproto.repo.uploadBlob` and `com.atproto.repo.createRecord`.
pub struct AuthXrpcClient {
    inner: atrium_xrpc_client::reqwest::ReqwestClient,
    access_jwt: String,
}

impl AuthXrpcClient {
    pub fn new(base_uri: &str, access_jwt: String) -> Self {
        Self {
            inner: atrium_xrpc_client::reqwest::ReqwestClient::new(base_uri),
            access_jwt,
        }
    }
}

impl HttpClient for AuthXrpcClient {
    async fn send_http(
        &self,
        request: atrium_xrpc::http::Request<Vec<u8>>,
    ) -> Result<
        atrium_xrpc::http::Response<Vec<u8>>,
        Box<dyn std::error::Error + Send + Sync + 'static>,
    > {
        self.inner.send_http(request).await
    }
}

impl XrpcClient for AuthXrpcClient {
    fn base_uri(&self) -> String {
        self.inner.base_uri()
    }
    async fn authorization_token(&self, _is_refresh: bool) -> Option<AuthorizationToken> {
        Some(AuthorizationToken::Bearer(self.access_jwt.clone()))
    }
}

/// Publisher that authenticates to a PDS and writes weight records.
pub struct ATProtocolPublisher {
    config: ATProtocolPublishConfig,
    client: AtpServiceClient<AuthXrpcClient>,
}

impl ATProtocolPublisher {
    /// Create a new publisher and authenticate a session against the PDS.
    ///
    /// Uses `com.atproto.server.createSession` with the configured handle and
    /// password (or app password) to obtain an access JWT, then wraps an
    /// authenticated [`AuthXrpcClient`].
    pub async fn new(config: ATProtocolPublishConfig) -> Result<Self> {
        let pds_url = config.pds_url.clone();
        let handle = config.handle.clone();
        let password = config.password.clone();

        let http_client = atrium_xrpc_client::reqwest::ReqwestClient::new(&pds_url);
        let unauth_client = AtpServiceClient::new(http_client);

        info!("Creating AT Protocol session for handle: {}", handle);
        let session = unauth_client
            .service
            .com
            .atproto
            .server
            .create_session(
                atrium_api::com::atproto::server::create_session::InputData {
                    identifier: handle,
                    password,
                    allow_takendown: None,
                    auth_factor_token: None,
                }
                .into(),
            )
            .await
            .map_err(|e| anyhow!("Failed to create AT Protocol session: {}", e))?;

        info!(
            "Authenticated as DID: {} handle: {}",
            session.did.as_str(),
            session.handle.as_str()
        );

        let auth_client = AuthXrpcClient::new(&pds_url, session.data.access_jwt);
        let client = AtpServiceClient::new(auth_client);

        Ok(Self { config, client })
    }

    /// Publish a serialized weights file to the PDS.
    ///
    /// The file is split into chunks of [`ATProtocolPublishConfig::chunk_size`]
    /// bytes (default 1 MB) and each chunk is uploaded via
    /// `com.atproto.repo.uploadBlob`. A record referencing all chunks is then
    /// written to the configured collection via `com.atproto.repo.createRecord`.
    pub async fn publish_weights(
        &self,
        weights_path: &str,
        model_config: &crate::model::ModelConfig,
    ) -> Result<PublishResult> {
        use atrium_api::com::atproto::repo::create_record;
        use atrium_api::types::TryIntoUnknown;

        info!(
            "Publishing weights from {} (collection: {})",
            weights_path, self.config.collection
        );

        let data = std::fs::read(weights_path)
            .map_err(|e| anyhow!("Failed to read weights file {}: {}", weights_path, e))?;
        let file_size = data.len();
        info!("Read {} bytes from {}", file_size, weights_path);

        // Chunk and upload each piece, collecting BlobRef results.
        let mut chunk_refs: Vec<serde_json::Value> = Vec::new();
        let mut offset = 0usize;
        let mut chunk_index = 0usize;
        while offset < data.len() {
            let end = (offset + self.config.chunk_size).min(data.len());
            let chunk = data[offset..end].to_vec();

            debug!(
                "Uploading chunk {} (bytes {}..{}, {} bytes)",
                chunk_index,
                offset,
                end,
                chunk.len()
            );

            let blob_output = self
                .client
                .service
                .com
                .atproto
                .repo
                .upload_blob(chunk)
                .await
                .map_err(|e| anyhow!("Failed to upload blob chunk {}: {}", chunk_index, e))?;

            let blob_ref = blob_output.data.blob;
            let json_value = serde_json::to_value(&blob_ref).map_err(|e| {
                anyhow!(
                    "Failed to serialize BlobRef for chunk {}: {}",
                    chunk_index,
                    e
                )
            })?;
            chunk_refs.push(json_value);

            offset = end;
            chunk_index += 1;
        }

        info!("Uploaded {} blob chunks", chunk_refs.len());

        // Serialize model config as a JSON string to avoid float-rejection by IPLD.
        let model_config_json = serde_json::to_string(model_config)?;

        let now = chrono::Utc::now().to_rfc3339();

        let mut record = serde_json::json!({
            "$type": self.config.collection,
            "name": self.config.name,
            "description": self.config.description.clone().unwrap_or_default(),
            "fileSize": file_size,
            "chunkSize": self.config.chunk_size,
            "chunks": chunk_refs,
            "modelConfig": model_config_json,
            "createdAt": now,
        });

        if let Some(ref rkey) = self.config.rkey {
            record["rkey"] = serde_json::Value::String(rkey.clone());
        }

        let record_value: serde_json::Value = record;
        let record_unknown = record_value
            .try_into_unknown()
            .map_err(|e| anyhow!("Failed to convert record to Unknown: {}", e))?;

        let rkey = self
            .config
            .rkey
            .clone()
            .unwrap_or_else(|| format!("weight-{}", chrono::Utc::now().timestamp()));

        let input =
            create_record::InputData {
                collection: self.config.collection.parse().map_err(|e| {
                    anyhow!(
                        "Invalid collection NSID '{}': {}",
                        self.config.collection,
                        e
                    )
                })?,
                record: record_unknown,
                repo: self.config.handle.clone().parse().map_err(|e| {
                    anyhow!("Invalid repo identifier '{}': {}", self.config.handle, e)
                })?,
                rkey: Some(
                    atrium_api::types::string::RecordKey::new(rkey.clone())
                        .map_err(|e| anyhow!("Invalid record key '{}': {}", rkey, e))?,
                ),
                swap_commit: None,
                validate: Some(false),
            }
            .into();

        let output = self
            .client
            .service
            .com
            .atproto
            .repo
            .create_record(input)
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to create record in collection '{}': {}",
                    self.config.collection,
                    e
                )
            })?;

        info!(
            "Published weights: uri={}, cid={}, size={} bytes, {} chunks",
            output.data.uri,
            output.data.cid.as_ref(),
            file_size,
            chunk_refs.len()
        );

        Ok(PublishResult {
            uri: output.data.uri,
            cid: output.data.cid.as_ref().to_string(),
            size: file_size,
            chunks: chunk_refs.len(),
        })
    }

    /// Resolve a handle to its DID (useful for validation before publishing).
    pub async fn resolve_did(handle: &str, pds_url: &str) -> Result<String> {
        let client = ATProtocolClient::new(pds_url);
        client.resolve_handle(handle).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_config_defaults() {
        let config = ATProtocolPublishConfig::default();
        assert_eq!(config.pds_url, "https://bsky.social");
        assert_eq!(config.collection, DEFAULT_WEIGHT_COLLECTION);
        assert_eq!(config.chunk_size, DEFAULT_BLOB_CHUNK_SIZE);
        assert!(config.rkey.is_none());
        assert_eq!(config.name, "experai-model");
    }

    #[test]
    fn publish_config_serializes() {
        let config = ATProtocolPublishConfig {
            pds_url: "https://example.com".to_string(),
            handle: "user.bsky.social".to_string(),
            password: "secret".to_string(),
            rkey: Some("my-rkey".to_string()),
            collection: "click.croft.experai.weight".to_string(),
            chunk_size: 500_000,
            name: "test-model".to_string(),
            description: Some("A test model".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ATProtocolPublishConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.handle, "user.bsky.social");
        assert_eq!(parsed.rkey, Some("my-rkey".to_string()));
        assert_eq!(parsed.chunk_size, 500_000);
    }

    #[test]
    fn default_weight_collection_nsid() {
        assert_eq!(DEFAULT_WEIGHT_COLLECTION, "click.croft.experai.weight");
    }

    #[test]
    fn default_chunk_size_is_one_mb() {
        assert_eq!(DEFAULT_BLOB_CHUNK_SIZE, 1_000_000);
    }
}
