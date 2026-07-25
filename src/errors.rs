use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExperaiError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Tokenization error: {0}")]
    Tokenization(#[from] tokenizers::Error),

    #[error("Model loading failed: {0}")]
    ModelLoad(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Data error: {0}")]
    Data(String),

    #[error("Export error: {0}")]
    Export(String),

    #[error("AT Protocol error: {0}")]
    AtProtocol(String),

    #[error("WebSocket error: {0}")]
    WebSocket(String),
}
