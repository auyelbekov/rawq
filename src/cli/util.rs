use std::path::Path;

use anyhow::{Context, Result};
use crate::daemon::client::DaemonClient;

use crate::cli::daemon_spawn;

pub const DAEMON_READY_MSG: &str =
    "Daemon ready \u{2014} subsequent operations will be faster (auto-stops after 30m of inactivity).";

/// Resolve the effective model: --model flag > RAWQ_MODEL env > None (config default).
pub fn effective_model(cli_model: &Option<String>) -> Option<String> {
    cli_model
        .clone()
        .or_else(|| std::env::var("RAWQ_MODEL").ok())
}

/// Resolve the model name string for daemon/embed operations.
/// Looks up the config to get the canonical name, falls back to the raw string.
pub fn resolve_model_name(model: &Option<String>) -> String {
    match crate::embed::config::resolve_model(model.as_deref()) {
        Ok((_dir, cfg)) => cfg.name,
        Err(_) => model.clone().unwrap_or_default(),
    }
}

/// Resolve search mode from CLI flags.
pub fn resolve_search_mode(exact: bool, semantic: bool) -> crate::search::SearchMode {
    if exact {
        crate::search::SearchMode::Lexical
    } else if semantic {
        crate::search::SearchMode::Semantic
    } else {
        crate::search::SearchMode::Hybrid
    }
}

/// Ensure the daemon is running for a given model. Fire-and-forget.
///
/// Called after any command that loaded a model locally (index build, watch, etc.)
/// so the next operation (typically search) gets a hot daemon. No-op if daemon
/// is already running or if daemon is disabled.
pub fn ensure_daemon_running(model_name: &str, no_daemon: bool) {
    if no_daemon || std::env::var("RAWQ_NO_DAEMON").is_ok() {
        return;
    }
    if model_name.is_empty() {
        return;
    }
    // Block until daemon is ready so subsequent operations can use it immediately.
    match DaemonClient::connect_or_start(model_name, daemon_spawn::spawn_daemon) {
        Ok((_client, freshly_started)) => {
            if freshly_started {
                eprintln!("{DAEMON_READY_MSG}");
            }
        }
        Err(e) => {
            eprintln!("warning: could not start daemon: {e:#}");
        }
    }
}

/// Max files changed before we skip daemon IPC and use local pipeline.
/// Daemon embed_batch does per-item IPC calls — fast for small incremental
/// reindexes but painfully slow for large builds (each chunk = one round-trip).
const DAEMON_FILE_LIMIT: usize = 50;

/// Build index, then ensure daemon is running afterward.
///
/// If the daemon is already running, uses it for embedding via IPC (fast for small
/// incremental reindexes — avoids cold ONNX model load). Falls back to local
/// embedding for fresh builds or when daemon is unavailable.
pub fn build_index_and_ensure_daemon(
    path: &Path,
    force: bool,
    opts: &crate::index::IndexOptions,
    no_daemon: bool,
) -> Result<crate::index::IndexStats> {
    let model_name = resolve_model_name(&opts.model);

    // Only use daemon for small incremental reindexes — per-item IPC is too slow
    // for large builds. Skip daemon if force-reindex or many files changed.
    let use_daemon = !force
        && !no_daemon
        && std::env::var("RAWQ_NO_DAEMON").is_err()
        && is_small_incremental(path, &opts.exclude_patterns);

    if use_daemon {
        if let Some(client) = DaemonClient::try_connect(&model_name) {
            let stats = crate::index::build_index_with_embed_fn(path, force, opts, &|texts| {
                client.embed_batch(texts)
            })?;
            return Ok(stats);
        }
    }

    // No daemon, large build, or force — load model locally, then start daemon afterward
    let stats = crate::index::build_index_with_options(path, force, opts)?;
    ensure_daemon_running(&model_name, no_daemon);
    Ok(stats)
}

/// Quick check: is the diff small enough for daemon IPC?
fn is_small_incremental(path: &Path, exclude_patterns: &[String]) -> bool {
    let canon = match std::fs::canonicalize(path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let index_dir = match crate::index::index_dir_for(&canon) {
        Ok(d) => d,
        Err(_) => return false, // no index dir → fresh build
    };
    let manifest = match crate::index::Manifest::load(&index_dir) {
        Ok(Some(m)) => m,
        _ => return false, // no manifest → fresh build
    };
    let walk_opts = crate::index::walker::WalkOptions {
        exclude_patterns: exclude_patterns.to_vec(),
        ..Default::default()
    };
    match manifest.diff_with_options(&canon, &walk_opts) {
        Ok(diff) => {
            let total_changed = diff.added.len() + diff.changed.len() + diff.removed.len();
            total_changed <= DAEMON_FILE_LIMIT
        }
        Err(_) => false,
    }
}

/// Check if an existing index is stale (files changed since last build).
///
/// Returns `Some((added, changed, removed))` if stale, `None` if fresh or no index.
/// Lightweight: uses git status (fast path) or mtime+hash walk (fallback).
pub fn check_index_freshness(
    root: &Path,
    exclude: &[String],
) -> Result<Option<(usize, usize, usize)>> {
    let canon = std::fs::canonicalize(root).context("canonicalize root")?;
    let index_dir = crate::index::index_dir_for(&canon)?;

    let manifest = match crate::index::Manifest::load(&index_dir)? {
        Some(m) => m,
        None => return Ok(None), // no manifest — auto-index handles this
    };

    let walk_opts = crate::index::walker::WalkOptions {
        exclude_patterns: exclude.to_vec(),
        ..Default::default()
    };
    let diff = manifest.diff_with_options(&canon, &walk_opts)?;

    if diff.is_empty() {
        Ok(None)
    } else {
        Ok(Some((diff.added.len(), diff.changed.len(), diff.removed.len())))
    }
}

/// Resolve model source and embed a query via daemon.
///
/// Centralizes the daemon resolution + query embedding pattern used by search and diff.
/// Returns `(pre_embedded_query, model_name_override, freshly_started)`.
pub fn resolve_and_embed_query(
    model: &Option<String>,
    query: &str,
    no_daemon: bool,
) -> (Option<Vec<f32>>, Option<String>, bool) {
    use crate::search::{ModelSource, embed_query_via_source, resolve_model_source};

    let model_name = resolve_model_name(model);
    let source = resolve_model_source(
        &model_name,
        no_daemon,
        Some(|name: &str| daemon_spawn::spawn_daemon(name)),
    );

    let freshly_started = matches!(&source, ModelSource::Daemon(_, true));

    if let Some((vec, name)) = embed_query_via_source(&source, query) {
        if !freshly_started {
            eprintln!("Using daemon ({name}).");
        }
        (Some(vec), Some(name), freshly_started)
    } else {
        (None, None, freshly_started)
    }
}

pub fn count_dir_contents(dir: &std::path::Path) -> Result<(usize, u64)> {
    let mut files = 0usize;
    let mut bytes = 0u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                files += 1;
                bytes += meta.len();
            }
        }
    }
    Ok((files, bytes))
}
