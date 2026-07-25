use anyhow::Result;
use crate::data::load_tokenizer;
use crate::jetstream::JetstreamConfig;
use crate::preprocessing::Preprocessor;
use tracing::info;

/// Run the preprocessing pipeline, loading data from a file or live from AT Protocol Jetstream.
#[allow(clippy::too_many_arguments)]
pub fn run(
    input: Option<String>,
    output: String,
    tokenizer: String,
    max_length: usize,
    atproto_handle: Option<String>,
    collection: String,
    dedupe: bool,
    clean: bool,
    dids: Vec<String>,
) -> Result<()> {
    info!(
        "Starting preprocessing: output={}, tokenizer={}, max_length={}, dedupe={}, clean={}",
        output, tokenizer, max_length, dedupe, clean
    );

    let tok = load_tokenizer(&tokenizer)?;

    if let Some(handle) = atproto_handle {
        info!(
            "Fetching live Jetstream data from handle={}, collection={}, dids={:?}",
            handle, collection, dids
        );

        let jetstream_config = JetstreamConfig {
            host: "jetstream2.us-east.bsky.network".to_string(),
            collections: vec![collection.clone()],
            dids: dids.clone(),
            max_samples: 10000,
            batch_size: 100,
            max_duration_secs: 3600,
            compression: true,
        };

        let rt = tokio::runtime::Runtime::new()?;
        let (dataset, stats) = rt.block_on(crate::jetstream::collect_from_jetstream(
            &jetstream_config,
            tok.clone(),
        ))?;

        info!(
            "Collected {} posts from Jetstream ({:.1}s, {:.1} posts/s)",
            stats.valid_posts, stats.duration_secs, stats.posts_per_second
        );

        let mut preprocessor = Preprocessor::new(tok, None);
        preprocessor.config.dedupe = dedupe;
        preprocessor.config.clean = clean;
        preprocessor.config.max_length = max_length;

        preprocessor.save_dataset_to_jsonl(&dataset, &output)?;
        info!(
            "Preprocessing complete: {} samples saved to {}",
            dataset.len(),
            output
        );
    } else {
        let input_path = input
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--input is required when not using --atproto-handle"))?;

        let mut preprocessor = Preprocessor::new(tok, None);
        preprocessor.config.dedupe = dedupe;
        preprocessor.config.clean = clean;
        preprocessor.config.max_length = max_length;

        preprocessor.preprocess_file(input_path, &output)?;
        info!("Preprocessing complete: {} -> {}", input_path, output);
    }

    Ok(())
}
