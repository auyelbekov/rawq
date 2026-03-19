use thiserror::Error;

/// Typed error for embedding operations.
/// Enables callers to match on specific failure modes instead of opaque anyhow strings.
#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("failed to load tokenizer: {0}")]
    TokenizerLoad(String),

    #[error("failed to load ONNX model: {0}")]
    ModelLoad(String),

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("no embedding model installed; run: rawq model download")]
    NoModel,

    #[error("model not found: {name}")]
    ModelNotFound { name: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
