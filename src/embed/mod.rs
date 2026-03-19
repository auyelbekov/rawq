pub mod config;
pub mod error;
pub mod gpu;
pub mod model;

pub use error::EmbedError;
pub use gpu::GpuInfo;
pub use model::{cosine_similarity, Embedder, TokenCounter};

/// Embedding dimension for snowflake-arctic-embed-s (default model).
pub(crate) const EMBED_DIM: usize = 384;
/// Maximum token sequence length.
pub(crate) const MAX_SEQ_LEN: usize = 512;
/// HuggingFace model repository.
pub(crate) const HF_REPO: &str = "Snowflake/snowflake-arctic-embed-s";
/// ONNX model filename within the repo.
pub(crate) const ONNX_MODEL_FILE: &str = "onnx/model_quantized.onnx";
/// Query prefix for asymmetric retrieval (snowflake-arctic-embed convention).
pub(crate) const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";
