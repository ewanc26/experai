use anyhow::{anyhow, Result};
use atrium_api::client::AtpServiceClient;
use atrium_api::com::atproto::identity::resolve_handle::ParametersData as ResolveParams;
use atrium_api::com::atproto::repo::list_records::ParametersData as ListRecordsParams;
use atrium_api::types::string::Handle;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATProtocolSample {
    pub uri: String,
    pub cid: String,
    pub text: String,
    pub created_at: String,
    pub author: String,
}

pub struct ATProtocolClient {
    client: AtpServiceClient<atrium_xrpc_client::reqwest::ReqwestClient>,
}

impl ATProtocolClient {
    pub fn new(pds_url: &str) -> Self {
        debug!("Creating AT Protocol client for PDS: {}", pds_url);
        let http_client = atrium_xrpc_client::reqwest::ReqwestClient::new(pds_url);
        let client = AtpServiceClient::new(http_client);
        Self { client }
    }

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

    pub async fn list_records(
        &self,
        repo: &str,
        collection: &str,
        limit: Option<u8>,
        cursor: Option<String>,
    ) -> Result<Vec<(String, String, serde_json::Value)>> {
        debug!("Listing records: repo={}, collection={}, limit={:?}", repo, collection, limit);
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

        debug!("Listed {} records from collection '{}'", records.len(), collection);
        Ok(records)
    }

    pub async fn list_records_paginated(
        &self,
        repo: &str,
        collection: &str,
        max_records: usize,
    ) -> Result<Vec<(String, String, serde_json::Value)>> {
        let mut all_records = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
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

            if batch_len < batch_size as usize {
                break;
            }

            cursor = response.data.cursor;
            if cursor.is_none() {
                break;
            }
        }

        info!("Paginated list complete: {} total records from collection '{}'", all_records.len(), collection);
        Ok(all_records)
    }
}

pub struct ATProtocolDataset {
    pub config: ATProtocolConfig,
    pub samples: Vec<ATProtocolSample>,
}

impl ATProtocolDataset {
    pub fn new(config: ATProtocolConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
        }
    }

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

    pub fn get_samples(&self) -> &[ATProtocolSample] {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

pub fn extract_text_from_value(value: &serde_json::Value) -> Option<String> {
    if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
        let cleaned = text.trim();
        if !cleaned.is_empty() {
            return Some(cleaned.to_string());
        }
    }
    None
}

pub async fn load_at_protocol_dataset(config: ATProtocolConfig) -> Result<Vec<ATProtocolSample>> {
    let mut dataset = ATProtocolDataset::new(config);
    dataset.load_from_at_protocol().await?;
    Ok(dataset.samples)
}
