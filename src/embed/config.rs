use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::embed::{EMBED_DIM, HF_REPO, MAX_SEQ_LEN, ONNX_MODEL_FILE, QUERY_PREFIX};

/// Metadata for an embedding model, persisted as `model.json` in each model directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_repo: Option<String>,
    pub embed_dim: usize,
    pub max_seq_len: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_prefix: Option<String>,
    pub onnx_file: String,
}

/// Root directory for managed models: `~/.cache/rawq/models/`.
pub fn models_dir() -> Result<PathBuf> {
    let cache = dirs::cache_dir().context("could not determine cache directory")?;
    Ok(cache.join("rawq").join("models"))
}

/// Directory for a specific managed model: `~/.cache/rawq/models/<name>/`.
pub fn model_dir(name: &str) -> Result<PathBuf> {
    Ok(models_dir()?.join(name))
}

/// Load `model.json` from a directory.
pub fn load_model_config(dir: &Path) -> Result<ModelConfig> {
    let path = dir.join("model.json");
    let data = fs::read_to_string(&path)
        .with_context(|| format!("read model.json from {}", dir.display()))?;
    serde_json::from_str(&data).context("parse model.json")
}

/// Write `model.json` to a directory.
pub fn save_model_config(dir: &Path, config: &ModelConfig) -> Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join("model.json");
    let data = serde_json::to_string_pretty(config)?;
    fs::write(&path, data).context("write model.json")?;
    Ok(())
}

// --- User config (default model, stored in config dir) ---

#[derive(Debug, Default, Serialize, Deserialize)]
struct UserConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    default_model: Option<String>,
}

fn config_path() -> Result<PathBuf> {
    let config = dirs::config_dir().context("could not determine config directory")?;
    Ok(config.join("rawq").join("config.json"))
}

fn load_user_config() -> UserConfig {
    let Ok(path) = config_path() else {
        return UserConfig::default();
    };
    if !path.exists() {
        return UserConfig::default();
    }
    fs::read_to_string(&path)
        .ok()
        .and_then(|data| serde_json::from_str(&data).ok())
        .unwrap_or_default()
}

fn save_user_config(config: &UserConfig) -> Result<()> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let data = serde_json::to_string_pretty(config)?;
    fs::write(&path, data)?;
    Ok(())
}

/// Get the configured default model name.
pub fn get_default_model() -> Option<String> {
    load_user_config().default_model
}

/// Set the default model. Verifies the model is installed.
pub fn set_default_model(name: &str) -> Result<()> {
    let dir = model_dir(name)?;
    anyhow::ensure!(
        dir.join("model.json").exists(),
        "model '{}' is not installed",
        name
    );
    let mut config = load_user_config();
    config.default_model = Some(name.to_string());
    save_user_config(&config)?;
    Ok(())
}

/// Clear the default model setting.
pub fn clear_default_model() -> Result<()> {
    let mut config = load_user_config();
    config.default_model = None;
    save_user_config(&config)?;
    Ok(())
}

// --- Model catalog (single source of truth) ---

struct CatalogEntry {
    name: &'static str,
    repo: &'static str,
    dim: usize,
    seq_len: usize,
    query_prefix: Option<&'static str>,
    onnx_file: &'static str,
    description: &'static str,
    /// Show in `rawq model list --recommended`
    recommended: bool,
}

const MODEL_CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        name: "snowflake-arctic-embed-s",
        repo: "Snowflake/snowflake-arctic-embed-s",
        dim: 384,
        seq_len: 512,
        query_prefix: Some("Represent this sentence for searching relevant passages: "),
        onnx_file: "onnx/model_quantized.onnx",
        description: "small, fast",
        recommended: true,
    },
    CatalogEntry {
        name: "bge-small-en-v1.5",
        repo: "BAAI/bge-small-en-v1.5",
        dim: 384,
        seq_len: 512,
        query_prefix: Some("Represent this sentence for searching relevant passages: "),
        onnx_file: "onnx/model.onnx",
        description: "small, general",
        recommended: true,
    },
    CatalogEntry {
        name: "bge-base-en-v1.5",
        repo: "BAAI/bge-base-en-v1.5",
        dim: 768,
        seq_len: 512,
        query_prefix: Some("Represent this sentence for searching relevant passages: "),
        onnx_file: "onnx/model.onnx",
        description: "medium, general",
        recommended: false,
    },
    CatalogEntry {
        name: "snowflake-arctic-embed-m-v1.5",
        repo: "Snowflake/snowflake-arctic-embed-m-v1.5",
        dim: 768,
        seq_len: 512,
        query_prefix: Some("Represent this sentence for searching relevant passages: "),
        onnx_file: "onnx/model_int8.onnx",
        description: "medium, better quality",
        recommended: true,
    },
    CatalogEntry {
        name: "jina-embeddings-v2-base-code",
        repo: "jinaai/jina-embeddings-v2-base-code",
        dim: 768,
        seq_len: 8192,
        query_prefix: None,
        onnx_file: "onnx/model.onnx",
        description: "code-optimized",
        recommended: true,
    },
    CatalogEntry {
        name: "all-MiniLM-L6-v2",
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        dim: 384,
        seq_len: 256,
        query_prefix: None,
        onnx_file: "onnx/model.onnx",
        description: "tiny, fast",
        recommended: true,
    },
    CatalogEntry {
        name: "nomic-embed-text-v1.5",
        repo: "nomic-ai/nomic-embed-text-v1.5",
        dim: 768,
        seq_len: 8192,
        query_prefix: Some("search_query: "),
        onnx_file: "onnx/model.onnx",
        description: "long context",
        recommended: false,
    },
];

pub struct RecommendedModel {
    pub repo_id: &'static str,
    pub name: &'static str,
    pub embed_dim: usize,
    pub max_seq_len: usize,
    pub description: &'static str,
}

pub fn recommended_models() -> Vec<RecommendedModel> {
    MODEL_CATALOG
        .iter()
        .filter(|e| e.recommended)
        .map(|e| RecommendedModel {
            repo_id: e.repo,
            name: e.name,
            embed_dim: e.dim,
            max_seq_len: e.seq_len,
            description: e.description,
        })
        .collect()
}

fn no_model_error() -> String {
    let mut msg = String::from("no embedding model installed\n\nRecommended models:\n");
    for m in recommended_models() {
        msg.push_str(&format!(
            "  {:<48} {:>3} dim, {:>5} seq  ({})\n",
            m.repo_id, m.embed_dim, m.max_seq_len, m.description
        ));
    }
    msg.push_str("\nInstall with: rawq model download <repo>");
    msg
}

/// Default config for the built-in snowflake-arctic-embed-s model.
pub fn default_config() -> ModelConfig {
    ModelConfig {
        name: "snowflake-arctic-embed-s".to_string(),
        hf_repo: Some(HF_REPO.to_string()),
        embed_dim: EMBED_DIM,
        max_seq_len: MAX_SEQ_LEN,
        query_prefix: Some(QUERY_PREFIX.to_string()),
        onnx_file: ONNX_MODEL_FILE.to_string(),
    }
}

/// Resolve a model name or filesystem path to a (directory, config) pair.
///
/// Precedence: explicit name > config default > single installed model > error.
/// Does NOT auto-download models.
pub fn resolve_model(name_or_path: Option<&str>) -> Result<(PathBuf, ModelConfig)> {
    match name_or_path {
        None => {
            // 1. Config default
            if let Some(default_name) = get_default_model() {
                let dir = model_dir(&default_name)?;
                if dir.join("model.json").exists() {
                    if let Ok(config) = load_model_config(&dir) {
                        return Ok((dir, config));
                    }
                } else {
                    eprintln!(
                        "warning: default model '{}' is missing, clearing stale config",
                        default_name
                    );
                    let _ = clear_default_model();
                }
            }
            // 2. Single installed model
            let models = list_installed_models()?;
            match models.len() {
                0 => {
                    if std::env::var("RAWQ_OFFLINE").is_ok() {
                        anyhow::bail!("{}", no_model_error());
                    }
                    eprintln!("No embedding model installed. Downloading default (snowflake-arctic-embed-s, ~33 MB)...");
                    download_model(crate::embed::HF_REPO, None)
                }
                1 => Ok(models.into_iter().next().unwrap()),
                _ => anyhow::bail!(
                    "multiple models installed but no default set\n\n\
                    Set a default with: rawq model default <name>\n\
                    Or specify with: --model <name>"
                ),
            }
        }
        Some(name) => {
            // Short name (no path separators or drive letters)
            if !name.contains('/') && !name.contains('\\') && !name.contains(':') {
                let dir = model_dir(name)?;
                if let Ok(config) = load_model_config(&dir) {
                    return Ok((dir, config));
                }
                // Not installed — provide helpful error
                if let Some(repo_id) = known_repo_for_name(name) {
                    anyhow::bail!(
                        "model '{name}' is not installed\n\n\
                        Install with: rawq model download {repo_id}"
                    );
                }
                anyhow::bail!(
                    "model '{name}' not found\n\n\
                    Install a model with: rawq model download <repo>"
                );
            }
            // Filesystem path
            let dir = PathBuf::from(name);
            if !dir.exists() {
                anyhow::bail!("model path does not exist: {name}");
            }
            let config = load_model_config(&dir).unwrap_or_else(|_| ModelConfig {
                name: dir
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "custom".to_string()),
                hf_repo: None,
                ..default_config()
            });
            Ok((dir, config))
        }
    }
}

/// List all installed models in the managed directory.
pub fn list_installed_models() -> Result<Vec<(PathBuf, ModelConfig)>> {
    let dir = models_dir()?;
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut models = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let path = entry.path();
            if let Ok(config) = load_model_config(&path) {
                models.push((path, config));
            }
        }
    }
    Ok(models)
}

/// Total bytes in a directory (recursive).
pub fn disk_size(dir: &Path) -> Result<u64> {
    let mut total = 0u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    Ok(total)
}

/// Known model configurations for popular embedding models.
pub fn known_model_config(repo_id: &str) -> Option<ModelConfig> {
    MODEL_CATALOG.iter().find(|e| e.repo == repo_id).map(|e| ModelConfig {
        name: e.name.to_string(),
        hf_repo: Some(e.repo.to_string()),
        embed_dim: e.dim,
        max_seq_len: e.seq_len,
        query_prefix: e.query_prefix.map(String::from),
        onnx_file: e.onnx_file.to_string(),
    })
}

/// Reverse lookup: map a short model name to its HuggingFace repo ID.
pub fn known_repo_for_name(name: &str) -> Option<&'static str> {
    MODEL_CATALOG.iter().find(|e| e.name == name).map(|e| e.repo)
}

/// Copy a file if the destination doesn't already exist.
/// Handles concurrent access by using a temp file and catching race errors.
fn copy_if_missing(src: &Path, dest: &Path) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    let tmp = dest.with_extension(format!("tmp.{}", std::process::id()));
    if let Err(e) = fs::copy(src, &tmp) {
        // Another process may have completed the copy
        if dest.exists() {
            let _ = fs::remove_file(&tmp);
            return Ok(());
        }
        let _ = fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("copy to {}", dest.display()));
    }
    // Try to rename temp to final; if it fails, another process beat us
    if let Err(_e) = fs::rename(&tmp, dest) {
        let _ = fs::remove_file(&tmp);
        if dest.exists() {
            return Ok(());
        }
        // Fallback: try direct copy (rename may fail cross-device)
        fs::copy(src, dest)
            .with_context(|| format!("copy to {}", dest.display()))?;
    }
    Ok(())
}

/// Download a model from HuggingFace Hub into the managed directory.
pub fn download_model(repo_id: &str, name: Option<&str>) -> Result<(PathBuf, ModelConfig)> {
    let model_name = name.map(String::from).unwrap_or_else(|| {
        repo_id
            .rsplit('/')
            .next()
            .unwrap_or(repo_id)
            .to_string()
    });

    let dest = model_dir(&model_name)?;

    // If already fully downloaded, return existing config
    if dest.join("model.json").exists() && dest.join("tokenizer.json").exists() {
        let config = load_model_config(&dest)?;
        return Ok((dest, config));
    }

    fs::create_dir_all(&dest)?;

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(false)
        .build()
        .map_err(|e| anyhow::anyhow!("HF Hub API: {e}"))?;
    let repo = api.model(repo_id.to_string());

    // Try ONNX files in priority order
    let onnx_candidates = [
        "onnx/model_quantized.onnx",
        "onnx/model_int8.onnx",
        "onnx/model.onnx",
        "model.onnx",
        "model_quantized.onnx",
    ];

    let mut onnx_file = None;
    for candidate in &onnx_candidates {
        match crate::embed::model::hf_get_or_download_repo(&repo, repo_id, candidate) {
            Ok(src) => {
                let dest_name = candidate.rsplit('/').next().unwrap_or(candidate);
                let dest_path = dest.join(dest_name);
                copy_if_missing(&src, &dest_path)
                    .with_context(|| format!("copy {candidate} to managed dir"))?;
                onnx_file = Some(dest_name.to_string());
                break;
            }
            Err(_) => continue,
        }
    }

    let onnx_file = onnx_file.ok_or_else(|| {
        anyhow::anyhow!(
            "no ONNX model found in {repo_id} (tried: {:?})",
            onnx_candidates
        )
    })?;

    // Download tokenizer
    let tok_src = crate::embed::model::hf_get_or_download_repo(&repo, repo_id, "tokenizer.json")?;
    copy_if_missing(&tok_src, &dest.join("tokenizer.json"))
        .context("copy tokenizer.json to managed dir")?;

    // Build config — check known catalog first, then probe dimensions
    let mut config = known_model_config(repo_id).unwrap_or_else(|| ModelConfig {
        name: model_name.clone(),
        hf_repo: Some(repo_id.to_string()),
        embed_dim: 0,
        max_seq_len: 512,
        query_prefix: None,
        onnx_file: onnx_file.clone(),
    });
    config.name = model_name;
    config.onnx_file = onnx_file;

    // Probe embed_dim if unknown
    if config.embed_dim == 0 {
        eprintln!("Probing embedding dimensions...");
        let mut embedder = crate::embed::model::Embedder::from_dir(&dest)?;
        let result = embedder.embed(&["test"])?;
        config.embed_dim = result.shape()[1];
    }

    save_model_config(&dest, &config)?;

    // Auto-set as default if no default is configured
    if get_default_model().is_none() {
        let _ = set_default_model(&config.name);
    }

    Ok((dest, config))
}

/// Remove a managed model by name. Returns bytes freed.
pub fn remove_model(name: &str) -> Result<u64> {
    let dir = model_dir(name)?;
    if !dir.exists() {
        anyhow::bail!("model '{}' not found in managed directory", name);
    }
    let bytes = disk_size(&dir)?;
    fs::remove_dir_all(&dir)
        .with_context(|| format!("remove model directory {}", dir.display()))?;
    Ok(bytes)
}

/// Remove all managed models. Returns (count, bytes_freed).
pub fn remove_all_models() -> Result<(usize, u64)> {
    let dir = models_dir()?;
    if !dir.exists() {
        return Ok((0, 0));
    }
    let mut count = 0usize;
    let mut total_bytes = 0u64;
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            total_bytes += disk_size(&entry.path()).unwrap_or(0);
            fs::remove_dir_all(entry.path())?;
            count += 1;
        }
    }
    Ok((count, total_bytes))
}
