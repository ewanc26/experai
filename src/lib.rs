/// AT Protocol client for fetching posts from a PDS.
pub mod at_protocol;
/// CLI command handlers for all subcommands.
pub mod commands;
/// Dataset loading, tokenization, and batching.
pub mod data;
/// Checkpoint export to GGUF format.
pub mod export;
/// Live streaming from the AT Protocol Jetstream.
pub mod jetstream;
/// Structured logging setup with tracing.
pub mod logging;
/// Transformer model architecture definitions.
pub mod model;
/// Text cleaning and tokenization pipeline.
pub mod preprocessing;
/// Training loop, loss, optimizers, and checkpointing.
pub mod training;
/// Shared utilities: device detection, metrics, seeds.
pub mod utils;
