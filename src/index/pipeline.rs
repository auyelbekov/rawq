use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use serde::Serialize;
use crate::embed::Embedder;

use crate::index::chunk::Chunk;
use crate::index::manifest::{compute_file_hash, index_dir_for, FileRecord, Manifest, COMMIT_MARKER};
use crate::index::store::{self, IndexStore};
use crate::index::chunker;

/// Compute batch size for embedding. If override is given, use it;
/// otherwise estimate from model dimensions and available memory.
/// GPU budget is derived from detected VRAM (75% to leave headroom).
/// `RAWQ_VRAM_BUDGET` env var overrides when set explicitly.
///
/// The per-item cost accounts for both input tensors and transformer
/// attention/activation memory, which scales with seq_len² and dominates
/// actual GPU usage.
fn batch_size_for(embedder: &mut Embedder, override_size: Option<usize>) -> usize {
    if let Some(bs) = override_size {
        return bs.clamp(1, 128);
    }
    if !embedder.uses_gpu() {
        // CPU: measure real throughput instead of guessing a memory budget.
        return crate::embed::calibrate::calibrated_batch_size(embedder);
    }
    // GPU: compute from detected VRAM (a real, measurable hardware limit).
    let dim = embedder.embed_dim();
    let seq_len = embedder.max_seq_len();
    let activation_factor = (seq_len / 6).max(4);
    let per_item = match dim
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_mul(activation_factor))
    {
        Some(v) if v > 0 => v,
        _ => return 1,
    };
    let mem_limit: usize = std::env::var("RAWQ_VRAM_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            let vram = embedder.gpu_vram();
            if vram > 0 {
                ((vram / 4) * 3) as usize
            } else {
                2 * 1024 * 1024 * 1024
            }
        });
    let computed = mem_limit / per_item;
    computed.max(1)
}

pub struct IndexedChunk {
    pub id: u64,
    pub chunk: Chunk,
}

#[derive(Serialize)]
pub struct IndexStats {
    pub total_files: usize,
    pub total_chunks: usize,
    pub added_files: usize,
    pub changed_files: usize,
    pub removed_files: usize,
}

/// Callback type for remote embedding (e.g. daemon IPC).
/// Receives text slices, returns `(vectors, model_name)`.
pub type EmbedFn<'a> = &'a dyn Fn(&[&str]) -> Result<(Vec<Vec<f32>>, String)>;

/// Options for the indexing pipeline.
#[derive(Default)]
pub struct IndexOptions {
    /// Model name or path (uses default model if None).
    pub model: Option<String>,
    /// Override batch size for embedding (auto-computed if None).
    pub batch_size: Option<usize>,
    /// Glob patterns to exclude from indexing.
    pub exclude_patterns: Vec<String>,
}

/// Full indexing pipeline: walk → diff → chunk → embed → store → persist.
pub fn build_index(root: &Path) -> Result<IndexStats> {
    build_index_with_options(root, false, &IndexOptions::default())
}

/// Full re-index: ignores existing manifest and rebuilds from scratch.
pub fn build_index_force(root: &Path) -> Result<IndexStats> {
    build_index_with_options(root, true, &IndexOptions::default())
}

/// Build index with custom options.
pub fn build_index_with_options(
    root: &Path,
    force: bool,
    opts: &IndexOptions,
) -> Result<IndexStats> {
    build_index_inner(root, force, opts, None, None)
}

/// Build index using a pre-loaded embedder (avoids model reload).
/// Used by the watch loop to re-index with its hot ONNX session.
pub fn build_index_with_embedder(
    root: &Path,
    force: bool,
    opts: &IndexOptions,
    embedder: &mut Embedder,
) -> Result<IndexStats> {
    build_index_inner(root, force, opts, Some(embedder), None)
}

/// Build index using a remote embedding callback (e.g. daemon IPC).
/// Avoids cold ONNX model load for small incremental reindexes.
/// The callback receives text slices and returns `(vectors, model_name)`.
pub fn build_index_with_embed_fn(
    root: &Path,
    force: bool,
    opts: &IndexOptions,
    embed_fn: EmbedFn<'_>,
) -> Result<IndexStats> {
    build_index_inner(root, force, opts, None, Some(embed_fn))
}

fn make_progress_bar(len: u64, prefix: &str, template: &str) -> ProgressBar {
    let pb = ProgressBar::with_draw_target(Some(len), ProgressDrawTarget::stderr());
    pb.set_prefix(prefix.to_string());
    pb.set_style(
        ProgressStyle::with_template(template)
            .unwrap()
            .progress_chars("##-"),
    );
    pb
}

// ─── Phase helpers ────────────────────────────────────────────────────────────

/// Phase 2: Walk and chunk files listed in `files_to_process` in parallel using rayon.
fn diff_and_chunk(root: &Path, files_to_process: &[&String]) -> Result<Vec<Chunk>> {
    let new_chunks: Vec<Chunk> = files_to_process
        .par_iter()
        .flat_map(|rel| {
            let abs = root.join(rel);
            let source = match std::fs::read_to_string(&abs) {
                Ok(s) => s.replace('\r', ""),
                Err(e) => {
                    eprintln!("warning: skipping {rel}: {e}");
                    return Vec::new();
                }
            };
            let lang = match crate::index::lang::detect_language(Path::new(rel)) {
                Some(l) => l,
                None => {
                    return Vec::new();
                }
            };
            match chunker::chunk_file(rel, &source, lang) {
                Ok(chunks) => chunks,
                Err(e) => {
                    eprintln!("warning: failed to chunk {rel}: {e}");
                    Vec::new()
                }
            }
        })
        .collect();
    Ok(new_chunks)
}

/// Phase 3: Compute embeddings for new chunks, reusing existing vectors where content hash matches.
/// Returns `(embeddings, model_name)`.
fn embed_chunks(
    new_chunks: &[Chunk],
    existing_records: &HashMap<u64, store::ChunkRecord>,
    old_manifest_model: &str,
    opts: &IndexOptions,
    store: &mut IndexStore,
    external_embedder: Option<&mut Embedder>,
    embed_fn: Option<EmbedFn<'_>>,
) -> Result<(Vec<Vec<f32>>, String)> {
    // Build content_hash -> chunk_id map from existing records
    let mut hash_to_id: HashMap<String, u64> = HashMap::new();
    for record in existing_records.values() {
        if !record.content_hash.is_empty() {
            hash_to_id.insert(record.content_hash.clone(), record.id);
        }
    }

    // Check if model has changed — if so, re-embed everything
    let model_changed = if !old_manifest_model.is_empty() {
        match crate::embed::config::resolve_model(opts.model.as_deref()) {
            Ok((_, cfg)) => cfg.name != old_manifest_model,
            Err(_) => false,
        }
    } else {
        false
    };

    // Classify chunks: reuse existing embeddings where possible
    let mut reuse_vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(new_chunks.len());
    let mut need_embed_indices: Vec<usize> = Vec::new();

    if model_changed {
        eprintln!("  Model changed, re-embedding all chunks");
        for i in 0..new_chunks.len() {
            reuse_vectors.push(None);
            need_embed_indices.push(i);
        }
    } else {
        for (i, chunk) in new_chunks.iter().enumerate() {
            let hash = store::compute_chunk_hash(&chunk.file, chunk.lines[0], &chunk.content);
            if let Some(&old_id) = hash_to_id.get(&hash) {
                if let Some(vec) = store.get_vector(old_id) {
                    reuse_vectors.push(Some(vec));
                    continue;
                }
            }
            reuse_vectors.push(None);
            need_embed_indices.push(i);
        }
    }

    let skip_count = new_chunks.len() - need_embed_indices.len();
    if skip_count > 0 {
        eprintln!(
            "  Incremental: {} chunks unchanged, {} to embed",
            skip_count,
            need_embed_indices.len()
        );
    }

    let model_name: String;
    if need_embed_indices.is_empty() {
        model_name = old_manifest_model.to_string();
        eprintln!("  No embedding needed, all chunks unchanged");
    } else {
        // Prepend context (file, language, scope) to each chunk before embedding.
        // The stored content stays clean — only the embedding input gets the prefix.
        let texts_to_embed: Vec<String> = need_embed_indices
            .iter()
            .map(|&i| {
                let c = &new_chunks[i];
                format!(
                    "// File: {}\n// Language: {}\n// Scope: {}\n\n{}",
                    c.file, c.language, c.scope, c.content
                )
            })
            .collect();
        let text_refs: Vec<&str> = texts_to_embed.iter().map(|s| s.as_str()).collect();

        // Priority: embed_fn (daemon IPC) > external_embedder (hot session) > local load
        if let Some(ef) = embed_fn {
            eprintln!("  Embedding {} chunks via daemon...", text_refs.len());
            let (vecs, name) = ef(&text_refs)?;
            model_name = name;
            for (j, &i) in need_embed_indices.iter().enumerate() {
                reuse_vectors[i] = Some(vecs[j].clone());
            }
        } else {
            let mut owned_embedder;
            let embedder: &mut Embedder = if let Some(ext) = external_embedder {
                eprintln!("  Using hot embedder: {}", ext.model_name());
                ext
            } else {
                eprintln!("  Loading embedding model...");
                let model_start = Instant::now();
                owned_embedder = Embedder::from_managed(opts.model.as_deref())?;
                eprintln!(
                    "  Loaded {} ({} dim, {} seq) in {}ms",
                    owned_embedder.model_name(),
                    owned_embedder.embed_dim(),
                    owned_embedder.max_seq_len(),
                    model_start.elapsed().as_millis()
                );
                &mut owned_embedder
            };
            model_name = embedder.model_name().to_string();

            let new_embeddings =
                embed_texts_with_progress(embedder, &text_refs, text_refs.len(), opts.batch_size)?;

            for (j, &i) in need_embed_indices.iter().enumerate() {
                reuse_vectors[i] = Some(new_embeddings[j].clone());
            }
        }
    }

    let embeddings: Vec<Vec<f32>> = reuse_vectors
        .into_iter()
        .map(|v| v.expect("all chunks should have embeddings"))
        .collect();

    Ok((embeddings, model_name))
}

struct PersistArgs<'a> {
    root: &'a Path,
    index_dir: &'a Path,
    new_chunks: &'a [Chunk],
    embeddings: Vec<Vec<f32>>,
    model_name: String,
    old_manifest: &'a Manifest,
    diff: &'a crate::index::manifest::ManifestDiff,
    ids_to_remove: &'a HashSet<u64>,
    store: IndexStore,
    files_to_process: &'a [&'a String],
    existing_records: &'a HashMap<u64, store::ChunkRecord>,
    start_id: u64,
}

/// Phase 4: Insert chunks into store, write chunks.jsonl, build and save manifest, write commit marker.
/// Returns `(total_files, total_chunks)`.
fn persist_index(args: PersistArgs<'_>) -> Result<(usize, usize)> {
    let PersistArgs {
        root, index_dir, new_chunks, embeddings, model_name,
        old_manifest, diff, ids_to_remove, store: mut store_inner,
        files_to_process, existing_records, start_id,
    } = args;
    let mut next_id = start_id;
    let mut indexed: Vec<(u64, &Chunk)> = Vec::new();
    let mut file_chunk_ids: HashMap<String, Vec<u64>> = HashMap::new();

    for (i, chunk) in new_chunks.iter().enumerate() {
        let id = next_id;
        next_id += 1;
        store_inner.insert(id, chunk, embeddings[i].clone())?;
        indexed.push((id, chunk));
        file_chunk_ids
            .entry(chunk.file.clone())
            .or_default()
            .push(id);
    }

    store::save_chunks_jsonl(index_dir, existing_records, ids_to_remove, &indexed)?;
    store_inner.persist()?;

    // Build new manifest
    let mut new_manifest = Manifest::new();
    new_manifest.model = model_name;
    new_manifest.next_chunk_id = next_id;

    // Keep unchanged files from old manifest
    for rel in &diff.unchanged {
        if let Some(record) = old_manifest.files.get(rel) {
            new_manifest.files.insert(rel.clone(), record.clone());
        }
    }

    // Keep mtime_only files from old manifest (update mtime)
    for rel in &diff.mtime_only {
        let abs = root.join(rel);
        if let Some(mut record) = old_manifest.files.get(rel).cloned() {
            if let Ok(meta) = std::fs::metadata(&abs) {
                record.mtime_secs = meta
                    .modified()
                    .unwrap_or(SystemTime::UNIX_EPOCH)
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                record.size = meta.len();
            }
            new_manifest.files.insert(rel.clone(), record);
        }
    }

    // Add new/changed files with content hash
    for rel in files_to_process {
        let abs = root.join(rel);
        let meta = match std::fs::metadata(&abs) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("warning: skipping {rel}: {e}");
                continue;
            }
        };
        let mtime = meta
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let size = meta.len();
        let chunk_ids = file_chunk_ids.remove(*rel).unwrap_or_default();
        let content_hash = compute_file_hash(&abs).ok();

        new_manifest.files.insert(
            (*rel).clone(),
            FileRecord {
                mtime_secs: mtime,
                size,
                chunk_ids,
                content_hash,
            },
        );
    }

    new_manifest.save(index_dir)?;

    // Write commit marker as the last step — marks this index as complete
    std::fs::write(index_dir.join(COMMIT_MARKER), b"ok")
        .context("write commit marker")?;

    let total_files = new_manifest.files.len();
    let total_chunks = new_manifest.total_chunks();

    Ok((total_files, total_chunks))
}

// ─── Main pipeline ─────────────────────────────────────────────────────────────

fn build_index_inner(
    root: &Path,
    force: bool,
    opts: &IndexOptions,
    external_embedder: Option<&mut Embedder>,
    embed_fn: Option<EmbedFn<'_>>,
) -> Result<IndexStats> {
    let pipeline_start = Instant::now();
    let root = std::fs::canonicalize(root).context("canonicalize root")?;
    let index_dir = index_dir_for(&root)?;

    // Phase 1: Acquire lock, optionally clear index, load manifest, compute diff.
    // The early-return for "nothing to do" keeps Phase 1 inline — extracting it would
    // require returning the lock (to keep it alive) plus the early-return path, which
    // adds more complexity than it saves.
    let _lock = crate::index::lock::IndexLock::write(&index_dir)?;

    // Verify index directory is safe (not world-writable on Unix)
    crate::index::manifest::verify_index_dir_safety(&index_dir)?;

    eprintln!("Index directory: {}", index_dir.display());

    // Remove commit marker before writing — incomplete builds won't be trusted
    let _ = std::fs::remove_file(index_dir.join(COMMIT_MARKER));

    if force {
        // Clear existing index contents for a clean rebuild
        if index_dir.exists() {
            for entry in std::fs::read_dir(&index_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.file_name() == Some(std::ffi::OsStr::new("lock")) {
                    continue;
                }
                if path.is_dir() {
                    std::fs::remove_dir_all(&path)?;
                } else {
                    std::fs::remove_file(&path)?;
                }
            }
            eprintln!("Cleared existing index for re-index");
        }
    }

    let walk_opts = crate::index::walker::WalkOptions {
        exclude_patterns: opts.exclude_patterns.clone(),
        ..Default::default()
    };

    let old_manifest = if force {
        Manifest::new()
    } else {
        Manifest::load(&index_dir)?.unwrap_or_else(Manifest::new)
    };
    let diff = old_manifest.diff_with_options(&root, &walk_opts)?;

    let stats_added = diff.added.len();
    let stats_changed = diff.changed.len();
    let stats_removed = diff.removed.len();

    if !diff.mtime_only.is_empty() {
        eprintln!(
            "Diff: {} added, {} changed, {} removed, {} unchanged, {} mtime-only",
            stats_added,
            stats_changed,
            stats_removed,
            diff.unchanged.len(),
            diff.mtime_only.len()
        );
    } else {
        eprintln!(
            "Diff: {} added, {} changed, {} removed, {} unchanged",
            stats_added,
            stats_changed,
            stats_removed,
            diff.unchanged.len()
        );
    }

    // Collect chunk IDs to remove (from removed + changed files)
    let mut ids_to_remove = HashSet::new();
    for rel in diff.removed.iter().chain(diff.changed.iter()) {
        if let Some(record) = old_manifest.files.get(rel) {
            for &id in &record.chunk_ids {
                ids_to_remove.insert(id);
            }
        }
    }

    // If nothing to do (only mtime-only updates at most), short-circuit
    if diff.is_empty() {
        // Update mtime for mtime_only files in the manifest
        if !diff.mtime_only.is_empty() {
            let mut updated = old_manifest.clone();
            updated.schema_version = crate::index::manifest::CURRENT_SCHEMA_VERSION;
            for rel in &diff.mtime_only {
                let abs = root.join(rel);
                if let Ok(meta) = std::fs::metadata(&abs) {
                    let mtime = meta
                        .modified()
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    if let Some(record) = updated.files.get_mut(rel) {
                        record.mtime_secs = mtime;
                        record.size = meta.len();
                    }
                }
            }
            updated.save(&index_dir)?;
            std::fs::write(index_dir.join(COMMIT_MARKER), b"ok")
                .context("write commit marker")?;
        } else {
            // No changes at all — ensure commit marker exists for old indexes
            if !index_dir.join(COMMIT_MARKER).exists() {
                std::fs::write(index_dir.join(COMMIT_MARKER), b"ok")
                    .context("write commit marker")?;
            }
        }

        let total_files = old_manifest.files.len();
        let total_chunks = old_manifest.total_chunks();
        eprintln!("Index is up to date.");
        return Ok(IndexStats {
            total_files,
            total_chunks,
            added_files: 0,
            changed_files: 0,
            removed_files: 0,
        });
    }

    let files_to_process: Vec<&String> = diff.added.iter().chain(diff.changed.iter()).collect();

    // Phase 2: Parallel chunking
    let new_chunks = diff_and_chunk(&root, &files_to_process)?;

    // Load existing chunk records for incremental embedding dedup
    let existing_records = store::load_chunks_jsonl(&index_dir)?;

    // Open store and remove stale chunks before embedding (to free reuse vector slots)
    let mut store = IndexStore::open_or_create(&index_dir)?;
    if !ids_to_remove.is_empty() {
        eprintln!("Removing {} stale chunks", ids_to_remove.len());
        store.remove(&ids_to_remove);
    }

    // Prune orphaned vectors not referenced by manifest or chunks.jsonl.
    // These can accumulate from crashed builds that wrote vectors.bin
    // but failed before writing manifest/chunks.jsonl.
    let valid_ids: HashSet<u64> = old_manifest
        .files
        .values()
        .flat_map(|r| &r.chunk_ids)
        .copied()
        .chain(existing_records.keys().copied())
        .collect();
    let pruned = store.prune_orphaned_vectors(&valid_ids);
    if pruned > 0 {
        eprintln!("Pruned {pruned} orphaned vectors");
    }

    // Phase 3: Embed (hash dedup + embedder selection + batch embed)
    let (embeddings, model_name) = embed_chunks(
        &new_chunks,
        &existing_records,
        &old_manifest.model,
        opts,
        &mut store,
        external_embedder,
        embed_fn,
    )?;

    // Phase 4: Store insert + chunks.jsonl + manifest save + commit marker
    let (total_files, total_chunks) = persist_index(PersistArgs {
        root: &root,
        index_dir: &index_dir,
        new_chunks: &new_chunks,
        embeddings,
        model_name,
        old_manifest: &old_manifest,
        diff: &diff,
        ids_to_remove: &ids_to_remove,
        store,
        files_to_process: &files_to_process,
        existing_records: &existing_records,
        start_id: old_manifest.next_chunk_id,
    })?;

    let elapsed = pipeline_start.elapsed().as_secs_f64();
    eprintln!(
        "  Indexed {} chunks from {} files in {:.1}s",
        total_chunks, total_files, elapsed
    );

    Ok(IndexStats {
        total_files,
        total_chunks,
        added_files: stats_added,
        changed_files: stats_changed,
        removed_files: stats_removed,
    })
}

/// Embed text slices in batches with progress bar.
/// On GPU OOM or inference failure, automatically quarters batch size down to 1.
/// The progress bar only appears after the first successful batch to avoid
/// a visual flash during GPU warmup (first inference can take seconds).
fn embed_texts_with_progress(
    embedder: &mut Embedder,
    texts: &[&str],
    total: usize,
    batch_size_override: Option<usize>,
) -> Result<Vec<Vec<f32>>> {
    let mut batch_size = batch_size_for(embedder, batch_size_override);
    if embedder.uses_gpu() {
        eprintln!(
            "  Batch size: {} (vram={:.1} GB)",
            batch_size,
            embedder.gpu_vram() as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    let mut all_embeddings = Vec::with_capacity(texts.len());
    let mut offset = 0;
    let mut pb: Option<ProgressBar> = None;

    while offset < texts.len() {
        let end = (offset + batch_size).min(texts.len());
        let batch = &texts[offset..end];

        match embedder.embed(batch) {
            Ok(result) => {
                for row_idx in 0..batch.len() {
                    all_embeddings.push(result.row(row_idx).to_vec());
                }

                // Show progress bar only after first successful batch
                let bar = pb.get_or_insert_with(|| {
                    make_progress_bar(
                        total as u64,
                        "Embedding",
                        "  {prefix:>10} [{bar:40}] {pos}/{len} chunks ({eta})",
                    )
                });
                bar.inc(batch.len() as u64);
                offset = end;
            }
            Err(e) => {
                let msg = format!("{e:#}");
                let is_oom = msg.contains("out of memory")
                    || msg.contains("OOM")
                    || msg.contains("alloc")
                    || msg.contains("memory allocation")
                    || msg.contains("Not enough memory")
                    || msg.contains("8007000E");

                // GPU device crash — no point retrying, the device is gone
                let is_device_lost = msg.contains("887A0005")
                    || msg.contains("887A0006")
                    || msg.contains("device instance has been suspended")
                    || msg.contains("GPU will not respond");

                if is_device_lost {
                    return Err(e).context(
                        "GPU device lost — try reducing batch size with --batch-size \
                         or use RAWQ_NO_GPU=1 for CPU mode"
                    );
                } else if is_oom && batch_size > 1 {
                    // Quarter batch size (not halve) to minimize consecutive GPU
                    // failures — repeated OOM errors can crash the DX12 device.
                    let new_size = (batch_size / 4).max(1);
                    eprintln!(
                        "warning: OOM at batch size {batch_size}, retrying with {new_size}"
                    );
                    if new_size <= 2 {
                        eprintln!(
                            "warning: batch size critically low — GPU overhead \
                             will dominate. Consider RAWQ_NO_GPU=1 for CPU mode"
                        );
                    }
                    batch_size = new_size;
                } else if !is_oom && batch_size > 1 {
                    eprintln!(
                        "warning: batch failed ({e:#}), retrying with batch_size=1"
                    );
                    batch_size = 1;
                } else {
                    return Err(e).context("embed batch (batch_size=1)");
                }
            }
        }
    }

    if let Some(bar) = pb {
        bar.finish_and_clear();
    }

    Ok(all_embeddings)
}
