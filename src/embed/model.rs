use std::path::Path;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use ndarray::{Array1, Array2, ArrayView3};
use ort::memory::Allocator;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::embed::config::{self, ModelConfig};
use crate::embed::{EMBED_DIM, MAX_SEQ_LEN};

/// Custom progress bar for HF Hub downloads matching rawq's style.
pub(crate) struct DownloadProgress {
    pb: ProgressBar,
}

impl DownloadProgress {
    pub(crate) fn new() -> Self {
        Self {
            pb: ProgressBar::with_draw_target(Some(0), ProgressDrawTarget::stderr()),
        }
    }
}

impl hf_hub::api::Progress for DownloadProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.pb.set_length(size as u64);
        self.pb.set_style(
            ProgressStyle::with_template(
                "  {msg:>20} [{bar:30}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        // Show just the filename, not the full path
        let short = filename
            .rsplit('/')
            .next()
            .unwrap_or(filename);
        self.pb.set_message(short.to_string());
    }

    fn update(&mut self, size: usize) {
        self.pb.inc(size as u64);
    }

    fn finish(&mut self) {
        self.pb.finish_and_clear();
    }
}

/// Fetch a file from HF Hub for a specific repo: use cache if available, otherwise download.
pub(crate) fn hf_get_or_download_repo(
    repo: &hf_hub::api::sync::ApiRepo,
    repo_id: &str,
    filename: &str,
) -> Result<std::path::PathBuf> {
    let cache = hf_hub::Cache::default();
    let cached = cache
        .repo(hf_hub::Repo::model(repo_id.to_string()))
        .get(filename);
    if let Some(path) = cached {
        return Ok(path);
    }
    repo.download_with_progress(filename, DownloadProgress::new())
        .map_err(|e| anyhow::anyhow!("download {filename}: {e}"))
}

/// Find an ONNX model file in a directory using common naming patterns.
fn find_onnx_in_dir(dir: &Path) -> Result<std::path::PathBuf> {
    if dir.join("model_quantized.onnx").exists() {
        Ok(dir.join("model_quantized.onnx"))
    } else if dir.join("model.onnx").exists() {
        Ok(dir.join("model.onnx"))
    } else {
        let nested = dir.join("onnx").join("model_quantized.onnx");
        if nested.exists() {
            Ok(nested)
        } else {
            let nested_regular = dir.join("onnx").join("model.onnx");
            if nested_regular.exists() {
                Ok(nested_regular)
            } else {
                anyhow::bail!("no ONNX model found in {}", dir.display())
            }
        }
    }
}

/// Embedding pipeline: tokenize -> ONNX inference -> mean pool -> L2 normalize.
pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
    name: String,
    query_prefix: Option<String>,
    embed_dim: usize,
    max_seq_len: usize,
    uses_gpu: bool,
    /// Detected GPU VRAM in bytes. 0 means unknown or CPU-only.
    gpu_vram: u64,
}

impl Embedder {
    /// Load from explicit model and tokenizer file paths.
    /// Probes model.json from the model file's parent directory if available.
    pub fn from_paths(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let mut emb = Self::new_inner(model_path, tokenizer_path)?;
        // Try to load model.json from the model file's parent directory
        if let Some(dir) = model_path.parent() {
            if let Ok(cfg) = crate::embed::config::load_model_config(dir) {
                emb.apply_config(&cfg);
            }
        }
        Ok(emb)
    }

    /// Low-level constructor from explicit file paths.
    fn new_inner(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        #[allow(unused_mut)]
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
            .with_intra_threads(cpus)
            .map_err(|e| anyhow::anyhow!("set threads: {e}"))?
            // Disable memory pattern optimization — it pre-plans memory reuse
            // assuming fixed tensor shapes, but our batch size is dynamic (the
            // last batch is smaller).  GPU EPs manage their own device memory
            // and the planner conflicts with that, causing hangs on repeated runs.
            .with_memory_pattern(false)
            .map_err(|e| anyhow::anyhow!("disable mem pattern: {e}"))?;

        // GPU acceleration (compile-time feature flags).
        // Auto-detect best GPU and its VRAM; env vars override.
        // ort silently falls back to CPU if the provider isn't available.
        #[allow(unused_mut, unused_assignments)]
        let mut gpu_vram: u64 = 0;

        #[cfg(feature = "directml")]
        {
            let (device_id, vram) = if let Some(dev) = std::env::var("RAWQ_DML_DEVICE")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
            {
                // Manual override — query VRAM for that specific device
                let vram = crate::embed::gpu::detect_gpu_vram(dev as u32).unwrap_or(0);
                (dev, vram)
            } else {
                // Auto-detect: pick the adapter with the most dedicated VRAM
                crate::embed::gpu::detect_best_gpu()
                    .map(|info| (info.device_id, info.vram_bytes))
                    .unwrap_or((0, 0))
            };
            gpu_vram = vram;
            builder = builder
                .with_execution_providers([
                    ort::execution_providers::DirectMLExecutionProvider::default()
                        .with_device_id(device_id)
                        .build(),
                ])
                .map_err(|e| anyhow::anyhow!("DirectML EP: {e}"))?;
        }
        #[cfg(feature = "cuda")]
        {
            let (device_id, vram) = if let Some(dev) = std::env::var("RAWQ_CUDA_DEVICE")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
            {
                let vram = crate::embed::gpu::detect_gpu_vram(dev as u32).unwrap_or(0);
                (dev, vram)
            } else {
                crate::embed::gpu::detect_best_gpu()
                    .map(|info| (info.device_id, info.vram_bytes))
                    .unwrap_or((0, 0))
            };
            gpu_vram = vram;
            builder = builder
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default()
                        .with_device_id(device_id)
                        .build(),
                ])
                .map_err(|e| anyhow::anyhow!("CUDA EP: {e}"))?;
        }
        #[cfg(feature = "coreml")]
        {
            gpu_vram = crate::embed::gpu::detect_best_gpu()
                .map(|info| info.vram_bytes)
                .unwrap_or(0);
            builder = builder
                .with_execution_providers([ort::execution_providers::CoreMLExecutionProvider::default().build()])
                .map_err(|e| anyhow::anyhow!("CoreML EP: {e}"))?;
        }

        // true when a GPU EP was registered; ort may silently fall back to CPU
        // if the EP is unavailable. Use RAWQ_NO_GPU=1 to force CPU.
        let gpu = cfg!(any(feature = "directml", feature = "cuda", feature = "coreml"))
            && std::env::var("RAWQ_NO_GPU").is_err();

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("load model: {e}"))?;

        Ok(Self {
            session,
            tokenizer,
            name: "custom".to_string(),
            query_prefix: None,
            embed_dim: EMBED_DIM,
            max_seq_len: MAX_SEQ_LEN,
            uses_gpu: gpu,
            gpu_vram: if gpu { gpu_vram } else { 0 },
        })
    }

    /// Load from a directory containing an ONNX model and tokenizer.json.
    /// Reads `model.json` for metadata if present; falls back to defaults.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let tokenizer_path = dir.join("tokenizer.json");
        anyhow::ensure!(tokenizer_path.exists(), "tokenizer.json not found in {}", dir.display());

        // Try to load model.json for metadata
        let config = config::load_model_config(dir).ok();

        let model_path = if let Some(ref cfg) = config {
            let from_config = dir.join(&cfg.onnx_file);
            if from_config.exists() {
                from_config
            } else {
                find_onnx_in_dir(dir)?
            }
        } else {
            find_onnx_in_dir(dir)?
        };

        let mut embedder = Self::new_inner(&model_path, &tokenizer_path)?;

        // Override metadata from config
        if let Some(cfg) = config {
            embedder.apply_config(&cfg);
        } else {
            embedder.name = dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "custom".to_string());
        }

        Ok(embedder)
    }

    /// Download default model from HuggingFace Hub and load it.
    pub fn from_hf_hub() -> Result<Self> {
        Self::from_managed(None)
    }

    /// Load a model by short name or path.
    pub fn from_managed(name: Option<&str>) -> Result<Self> {
        let (dir, cfg) = config::resolve_model(name)?;
        anyhow::ensure!(
            dir.join("tokenizer.json").exists(),
            "model '{}' is incomplete (missing tokenizer.json)\n\
            Re-download with: rawq model download {}",
            cfg.name,
            cfg.hf_repo.as_deref().unwrap_or(&cfg.name)
        );
        Self::from_dir(&dir)
    }

    /// Apply metadata from a ModelConfig.
    fn apply_config(&mut self, cfg: &ModelConfig) {
        self.name = cfg.name.clone();
        self.query_prefix = cfg.query_prefix.clone();
        self.embed_dim = cfg.embed_dim;
        self.max_seq_len = cfg.max_seq_len;
    }

    /// Model name (e.g. "snowflake-arctic-embed-s").
    pub fn model_name(&self) -> &str {
        &self.name
    }

    /// Embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Maximum token sequence length for this model.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Whether a GPU execution provider was registered at session creation.
    /// Note: ort may silently fall back to CPU if the provider is unavailable.
    /// Use `RAWQ_NO_GPU=1` to force CPU mode.
    pub fn uses_gpu(&self) -> bool {
        self.uses_gpu
    }

    /// Detected GPU VRAM in bytes. Returns 0 if CPU-only or detection failed.
    pub fn gpu_vram(&self) -> u64 {
        self.gpu_vram
    }

    /// Embed a batch of texts. Returns `[batch_size, embed_dim]`.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Array2<f32>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let batch_size = encodings.len();
        // Always pad to max_seq_len so every batch produces identical tensor
        // shapes.  GPU execution providers (DirectML, CUDA) cache compiled
        // execution plans by shape — a shape change triggers a recompile.
        // Attention mask already zeros out padding, so results are unchanged.
        let seq_len = self.max_seq_len;

        // Use ort-managed tensors (CreateTensorAsOrtValue) instead of wrapping
        // external ndarray memory (CreateTensorWithDataAsOrtValue).  GPU EPs
        // may hold internal references to input memory between runs; wrapping
        // caller-owned memory that is freed and reallocated at different
        // addresses between batches can cause GPU hangs or stale references.
        let allocator = Allocator::default();
        let mut input_ids_tensor = Tensor::<i64>::new(&allocator, [batch_size, seq_len])
            .map_err(|e| anyhow::anyhow!("input_ids tensor: {e}"))?;
        let mut attention_mask_tensor = Tensor::<i64>::new(&allocator, [batch_size, seq_len])
            .map_err(|e| anyhow::anyhow!("attention_mask tensor: {e}"))?;

        // Fill the ort-managed buffers with tokenized data
        {
            let (_, ids_buf) = input_ids_tensor.extract_tensor_mut();
            let (_, mask_buf) = attention_mask_tensor.extract_tensor_mut();
            for (i, enc) in encodings.iter().enumerate() {
                let ids = enc.get_ids();
                let mask = enc.get_attention_mask();
                let len = ids.len().min(seq_len);
                let row = i * seq_len;
                for j in 0..len {
                    ids_buf[row + j] = ids[j] as i64;
                    mask_buf[row + j] = mask[j] as i64;
                }
            }
        }

        // Keep a copy of the mask for mean pooling (the tensor is consumed by run)
        let attention_mask = {
            let (_, mask_data) = attention_mask_tensor.extract_tensor();
            Array2::<i64>::from_shape_vec((batch_size, seq_len), mask_data.to_vec())
                .map_err(|e| anyhow::anyhow!("mask copy: {e}"))?
        };

        // Determine output index before running inference (outputs borrows session mutably,
        // preventing access to session.outputs() afterwards).
        let output_idx = self
            .session
            .outputs()
            .iter()
            .enumerate()
            .find(|(_, o)| o.name() == "last_hidden_state" || o.name() == "token_embeddings")
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Check if model expects token_type_ids
        let has_token_type_ids = self
            .session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let outputs = if has_token_type_ids {
            let token_type_ids_tensor = Tensor::<i64>::new(&allocator, [batch_size, seq_len])
                .map_err(|e| anyhow::anyhow!("token_type_ids tensor: {e}"))?;
            self.session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                    "token_type_ids" => token_type_ids_tensor,
                ])
                .map_err(|e| anyhow::anyhow!("inference: {e}"))?
        } else {
            self.session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                ])
                .map_err(|e| anyhow::anyhow!("inference: {e}"))?
        };

        let (_shape, data) = outputs[output_idx]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extract tensor[{output_idx}]: {e}"))?;

        // Output shape is [batch_size, seq_len, hidden_dim]
        let dim = if batch_size > 0 && seq_len > 0 {
            data.len() / (batch_size * seq_len)
        } else {
            0
        };
        if dim != self.embed_dim {
            anyhow::bail!(
                "embed_dim mismatch: model output has dim={dim} but expected {}. \
                 Index was built with a different model or the ONNX file is corrupt.",
                self.embed_dim
            );
        }
        let hidden_states = ArrayView3::from_shape((batch_size, seq_len, dim), data)
            .map_err(|e| anyhow::anyhow!("shape error: {e}"))?;

        Ok(mean_pool_and_normalize(&hidden_states, &attention_mask))
    }

    /// Embed a single query (with retrieval prefix if configured).
    pub fn embed_query(&mut self, query: &str) -> Result<Array1<f32>> {
        let text = match &self.query_prefix {
            Some(prefix) => format!("{prefix}{query}"),
            None => query.to_string(),
        };
        let result = self.embed(&[&text])?;
        Ok(result.row(0).to_owned())
    }

    /// Embed a single document (no prefix).
    pub fn embed_document(&mut self, doc: &str) -> Result<Array1<f32>> {
        let result = self.embed(&[doc])?;
        Ok(result.row(0).to_owned())
    }

    /// Count tokens in a text string using the loaded tokenizer.
    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|enc| enc.get_ids().len())
            .unwrap_or(0)
    }
}

/// Lightweight token counter — loads only the tokenizer (no ONNX model).
pub struct TokenCounter {
    tokenizer: Tokenizer,
}

impl TokenCounter {
    /// Load tokenizer from a directory containing tokenizer.json.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        Ok(Self { tokenizer })
    }

    /// Load tokenizer from HuggingFace Hub cache.
    pub fn from_hf_hub() -> Result<Self> {
        Self::from_managed(None)
    }

    /// Load tokenizer by model name or path.
    pub fn from_managed(name: Option<&str>) -> Result<Self> {
        let (dir, _cfg) = config::resolve_model(name)?;
        Self::from_dir(&dir)
    }

    /// Count tokens in a text string.
    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|enc| enc.get_ids().len())
            .unwrap_or(0)
    }
}

/// Mean-pool hidden states using attention mask, then L2-normalize.
fn mean_pool_and_normalize(
    hidden_states: &ArrayView3<f32>, // [batch, seq_len, dim]
    attention_mask: &Array2<i64>,    // [batch, seq_len]
) -> Array2<f32> {
    let batch_size = hidden_states.shape()[0];
    let seq_len = hidden_states.shape()[1];
    let dim = hidden_states.shape()[2];
    let mut result = Array2::<f32>::zeros((batch_size, dim));

    for i in 0..batch_size {
        let mut sum = Array1::<f32>::zeros(dim);
        let mut count = 0.0_f32;

        for j in 0..seq_len {
            let mask_val = attention_mask[[i, j]] as f32;
            if mask_val > 0.0 {
                for k in 0..dim {
                    sum[k] += hidden_states[[i, j, k]] * mask_val;
                }
                count += mask_val;
            }
        }

        if count > 0.0 {
            sum /= count;
        }

        // L2 normalize
        let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm.is_finite() && norm > 0.0 {
            sum /= norm;
        }

        result.row_mut(i).assign(&sum);
    }

    result
}

/// Cosine similarity between two L2-normalized vectors (= dot product).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
