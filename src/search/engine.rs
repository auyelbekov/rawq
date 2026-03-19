use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use crate::embed::{Embedder, TokenCounter};
use crate::index::{IndexReader, StoredChunk};

#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Semantic,
    Lexical,
    Hybrid,
}

#[non_exhaustive]
pub struct SearchOptions {
    pub mode: SearchMode,
    pub top_n: usize,
    pub threshold: f32,
    pub context_lines: usize,
    pub full_file: bool,
    pub lang_filter: Option<String>,
    /// Model name or path (uses default model if None).
    pub model: Option<String>,
    /// Maximum token budget for all results combined.
    pub token_budget: Option<usize>,
    /// RRF smoothing constant k (default: 60).
    pub rrf_k: usize,
    /// Semantic weight in RRF merge (0.0–1.0). None = auto-detect from query type.
    pub semantic_weight: Option<f64>,
    /// Glob patterns to exclude from indexing/searching.
    pub exclude_patterns: Vec<String>,
    /// Pre-computed query embedding from daemon. Skips local embedder loading.
    pub pre_embedded_query: Option<Vec<f32>>,
    /// Model name override for output when using daemon-provided embedding.
    pub model_name_override: Option<String>,
    /// Re-rank top results with a keyword overlap heuristic after initial scoring.
    /// Boosts results that contain more query terms. Off by default (adds slight latency).
    pub rerank: bool,
    /// Suppress the "Model: ..." status line printed to stderr after each search.
    /// Set to true in contexts like benchmark where the line pollutes output.
    pub suppress_status_line: bool,
    /// Weight multiplier for text/markdown chunks (0.0–1.0). Default: 0.5.
    /// Set to 1.0 to treat text chunks equally with code chunks.
    pub text_weight: Option<f32>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            mode: SearchMode::Hybrid,
            top_n: 10,
            threshold: 0.3,
            context_lines: 3,
            full_file: false,
            lang_filter: None,
            model: None,
            token_budget: None,
            rrf_k: 60,
            semantic_weight: None,
            exclude_patterns: Vec::new(),
            pre_embedded_query: None,
            model_name_override: None,
            rerank: false,
            suppress_status_line: false,
            text_weight: None,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub file: String,
    pub lines: [usize; 2],
    /// Start line for displayed content (differs from `lines[0]` when using --context or --full-file)
    pub display_start_line: usize,
    pub language: String,
    pub scope: String,
    /// Confidence score: raw cosine similarity (0.0–1.0) for semantic/hybrid,
    /// normalized BM25 (0.0–1.0) for lexical-only.
    pub confidence: f64,
    pub content: String,
    /// Lines above the chunk for surrounding context.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub context_before: String,
    /// Lines below the chunk for surrounding context.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub context_after: String,
    pub token_count: usize,
    /// Line numbers (1-based) within the result that match query terms.
    pub matched_lines: Vec<usize>,
}

/// Current search output schema version. Increment on breaking field changes.
pub const SEARCH_OUTPUT_VERSION: u32 = 1;

#[derive(Debug, Serialize)]
pub struct SearchOutput {
    pub schema_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub results: Vec<SearchResult>,
    pub query_ms: u64,
    pub total_tokens: usize,
}

/// Lazily load the embedder only when needed (semantic/hybrid modes).
fn load_embedder<'a>(
    cached: &'a mut Option<Embedder>,
    model: Option<&str>,
    suppress_status: bool,
) -> Result<&'a mut Embedder> {
    if cached.is_none() {
        let emb = Embedder::from_managed(model)?;
        if !suppress_status {
            eprintln!("Model: {}", emb.model_name());
        }
        *cached = Some(emb);
    }
    Ok(cached.as_mut().unwrap())
}

/// Build a compiled GlobSet from exclude patterns. Returns None if empty or all invalid.
fn build_exclude_set(patterns: &[String]) -> Option<globset::GlobSet> {
    if patterns.is_empty() {
        return None;
    }
    use globset::{Glob, GlobSetBuilder};
    let mut builder = GlobSetBuilder::new();
    for pat in patterns {
        if let Ok(g) = Glob::new(pat) {
            builder.add(g);
        }
    }
    builder.build().ok()
}

/// Returns true if `file` matches the pre-built exclude GlobSet.
fn matches_exclude(file: &str, set: Option<&globset::GlobSet>) -> bool {
    let Some(set) = set else { return false; };
    let filename = std::path::Path::new(file)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    set.is_match(file) || set.is_match(filename)
}

/// Top-level search: embed query, search, fuse, enrich.
/// Auto-indexing is the caller's responsibility (rawq-cli handles it before calling here).
///
/// `rawq-core` is the search library; it does NOT trigger side effects like indexing.
pub fn search(root: &Path, query: &str, opts: &SearchOptions) -> Result<SearchOutput> {
    search_inner(None, root, query, opts)
}

/// Low-level search entry point that accepts a pre-opened `IndexReader`.
/// Use this when running multiple searches (e.g., benchmark across 3 modes) to avoid
/// reopening the index and re-reading vectors.bin / chunks.jsonl for each search.
pub fn search_with_reader(
    reader: &IndexReader,
    root: &Path,
    query: &str,
    opts: &SearchOptions,
) -> Result<SearchOutput> {
    search_inner(Some(reader), root, query, opts)
}

/// Shared search implementation used by both `search()` and `search_with_reader()`.
///
/// When `reader_opt` is `None`, acquires a lock, opens the reader, and loads the
/// manifest for model resolution. When `Some`, uses the provided reader directly.
fn search_inner(
    reader_opt: Option<&IndexReader>,
    root: &Path,
    query: &str,
    opts: &SearchOptions,
) -> Result<SearchOutput> {
    let start = Instant::now();

    if query.trim().is_empty() {
        return Ok(SearchOutput {
            schema_version: SEARCH_OUTPUT_VERSION,
            model: None,
            results: vec![],
            query_ms: 0,
            total_tokens: 0,
        });
    }

    let root = std::fs::canonicalize(root).context("canonicalize root")?;

    // Acquire resources based on whether a reader was provided.
    // _lock and owned_reader must live until the function returns.
    let _lock;
    let owned_reader;
    let owned_index_dir;
    let reader: &IndexReader = if let Some(r) = reader_opt {
        _lock = None::<crate::index::lock::IndexLock>;
        owned_index_dir = None;
        r
    } else {
        let index_dir = crate::index::index_dir_for(&root)?;
        _lock = Some(crate::index::lock::IndexLock::read(&index_dir)?);
        owned_reader = IndexReader::open(&index_dir).context("open index")?;
        owned_index_dir = Some(index_dir);
        &owned_reader
    };

    // Model resolution: full manifest-based when we own the reader, simple otherwise
    let effective_model: Option<String> = if let Some(ref index_dir) = owned_index_dir {
        let manifest = crate::index::Manifest::load(index_dir)?;
        if opts.model.is_some() {
            // User explicitly specified a model — warn if it doesn't match the index
            if !matches!(opts.mode, SearchMode::Lexical) {
                if let Some(ref m) = manifest {
                    let (_, resolved_config) =
                        crate::embed::config::resolve_model(opts.model.as_deref())?;
                    if m.model != resolved_config.name {
                        eprintln!(
                            "warning: index was built with model '{}' but searching with '{}'",
                            m.model, resolved_config.name
                        );
                    }
                }
            }
            opts.model.clone()
        } else {
            // No --model flag: auto-detect from index manifest
            manifest.map(|m| m.model).filter(|m| !m.is_empty())
        }
    } else {
        opts.model.clone()
    };

    // Lazy embedder — only loaded for semantic/hybrid modes when no pre-computed vector
    let mut embedder: Option<Embedder> = None;
    let mut query_vec_opt: Option<Vec<f32>> = None;

    let suppress = opts.suppress_status_line;
    let get_query_vec = |embedder: &mut Option<Embedder>,
                         model: Option<&str>,
                         pre: &Option<Vec<f32>>|
     -> Result<Vec<f32>> {
        if let Some(vec) = pre {
            return Ok(vec.clone());
        }
        let emb = load_embedder(embedder, model, suppress)?;
        Ok(emb.embed_query(query)?.to_vec())
    };

    // Retrieve candidates based on mode
    let scored = match opts.mode {
        SearchMode::Semantic => {
            let query_vec =
                get_query_vec(&mut embedder, effective_model.as_deref(), &opts.pre_embedded_query)?;
            let results = reader.search_vector(&query_vec, opts.top_n * 2);
            query_vec_opt = Some(query_vec);
            apply_language_weights(&results, reader, opts.text_weight.unwrap_or(DEFAULT_TEXT_WEIGHT))
        }
        SearchMode::Lexical => {
            reader.search_bm25(query, opts.top_n * 2)?
        }
        SearchMode::Hybrid => {
            let query_vec =
                get_query_vec(&mut embedder, effective_model.as_deref(), &opts.pre_embedded_query)?;
            // SAFETY: both closures are read-only on the IndexReader — no shared
            // mutable state. Adding side effects to either branch would be a data race.
            let (semantic, lex_result) = rayon::join(
                || reader.search_vector(&query_vec, opts.top_n * 2),
                || reader.search_bm25(query, opts.top_n * 2),
            );
            let lexical = lex_result?;
            let weight = effective_semantic_weight(query, opts.semantic_weight);
            let merged = rrf_merge(&semantic, &lexical, opts.rrf_k, weight);
            query_vec_opt = Some(query_vec);
            apply_language_weights(&merged, reader, opts.text_weight.unwrap_or(DEFAULT_TEXT_WEIGHT))
        }
    };

    // Deduplicate overlapping chunks within the same file
    let scored = deduplicate(&scored, reader);

    // Load token counter (lightweight — tokenizer only, no ONNX model)
    let counter = match TokenCounter::from_managed(effective_model.as_deref()) {
        Ok(c) => Some(c),
        Err(e) => {
            if !opts.suppress_status_line {
                eprintln!("warning: token counter unavailable ({e:#}), token_count will be 0");
            }
            None
        }
    };

    // Compute max BM25 score for lexical normalization
    let max_bm25 = if matches!(opts.mode, SearchMode::Lexical) {
        scored.first().map(|(_, s)| *s).unwrap_or(1.0).max(f32::EPSILON)
    } else {
        1.0
    };

    // Language filter + truncate + token budget
    let exclude_set = build_exclude_set(&opts.exclude_patterns);
    let mut results = Vec::new();
    let mut total_tokens: usize = 0;
    for (id, _score) in &scored {
        if results.len() >= opts.top_n {
            break;
        }
        if let Some(chunk) = reader.get_chunk(*id) {
            if matches_exclude(&chunk.file, exclude_set.as_ref()) {
                continue;
            }
            if let Some(ref lang) = opts.lang_filter {
                if !chunk.language.to_string().eq_ignore_ascii_case(lang) {
                    continue;
                }
            }

            // Compute confidence BEFORE enrichment to skip below-threshold
            // results without expensive file I/O (RWQ-216)
            let confidence = match opts.mode {
                SearchMode::Lexical => (*_score / max_bm25) as f64,
                _ => {
                    if let Some(ref qv) = query_vec_opt {
                        if let Some(cv) = reader.get_vector(*id) {
                            crate::embed::cosine_similarity(qv, &cv) as f64
                        } else {
                            *_score as f64
                        }
                    } else {
                        *_score as f64
                    }
                }
            };

            if confidence < opts.threshold as f64 {
                continue;
            }

            let enriched =
                enrich_content(chunk, &root, opts.context_lines, opts.full_file);

            let matched_lines = compute_matched_lines(&enriched.content, query, enriched.display_start_line);

            let token_count = counter
                .as_ref()
                .map(|c| c.count_tokens(&enriched.content))
                .unwrap_or(0);

            if let Some(budget) = opts.token_budget {
                if total_tokens + token_count > budget && !results.is_empty() {
                    break;
                }
            }

            total_tokens += token_count;
            results.push(SearchResult {
                file: chunk.file.clone(),
                lines: chunk.lines,
                display_start_line: enriched.display_start_line,
                language: chunk.language.to_string(),
                scope: chunk.scope.clone(),
                confidence,
                content: enriched.content,
                context_before: enriched.context_before,
                context_after: enriched.context_after,
                token_count,
                matched_lines,
            });
        }
    }

    if opts.rerank {
        rerank_by_keyword_overlap(&mut results, query);
    } else {
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal));
    }

    let model_name = opts
        .model_name_override
        .clone()
        .or_else(|| embedder.as_ref().map(|e| e.model_name().to_string()))
        .or(effective_model);

    let query_ms = start.elapsed().as_millis() as u64;
    Ok(SearchOutput {
        schema_version: SEARCH_OUTPUT_VERSION,
        model: model_name,
        results,
        query_ms,
        total_tokens,
    })
}

/// BM25 search over in-memory chunks using a RAM tantivy index.
/// Includes fuzzy fallback (Levenshtein distance 1) when exact search returns 0 results.
/// Returns (chunk_index, bm25_score) pairs sorted descending.
fn search_bm25_ram(
    chunks: &[crate::index::Chunk],
    query: &str,
    top_n: usize,
) -> Result<Vec<(usize, f32)>> {
    use tantivy::collector::TopDocs;
    use tantivy::query::QueryParser;
    use tantivy::schema::{Schema, TextFieldIndexing, TextOptions, Value, STORED};
    use tantivy::{doc, Index};

    let mut schema_builder = Schema::builder();
    let f_idx = schema_builder.add_u64_field("idx", STORED | tantivy::schema::INDEXED);
    let code_indexing = TextFieldIndexing::default()
        .set_tokenizer(crate::index::CODE_TOKENIZER_NAME)
        .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions);
    let code_opts = TextOptions::default()
        .set_indexing_options(code_indexing)
        .set_stored();
    let f_content = schema_builder.add_text_field("content", code_opts.clone());
    let f_scope = schema_builder.add_text_field("scope", code_opts);
    let schema = schema_builder.build();

    let index = Index::create_in_ram(schema);
    crate::index::register_code_tokenizer(&index);
    let mut writer = index.writer(15_000_000).context("tantivy ram writer")?;
    for (i, chunk) in chunks.iter().enumerate() {
        writer.add_document(doc!(
            f_idx => i as u64,
            f_content => chunk.content.clone(),
            f_scope => chunk.scope.clone(),
        ))?;
    }
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let mut parser = QueryParser::for_index(&index, vec![f_content, f_scope]);
    parser.set_field_boost(f_scope, 3.0);
    let tantivy_query = parser.parse_query(query).context("parse BM25 query")?;
    let top_docs = searcher
        .search(&tantivy_query, &TopDocs::with_limit(top_n))
        .context("tantivy search")?;

    let mut results: Vec<(usize, f32)> = top_docs
        .into_iter()
        .filter_map(|(score, addr)| {
            let doc = searcher.doc::<tantivy::TantivyDocument>(addr).ok()?;
            let idx = doc.get_first(f_idx).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            Some((idx, score))
        })
        .collect();

    // Fuzzy fallback if exact BM25 returned nothing
    if results.is_empty() && !query.trim().is_empty() {
        let fuzzy_query = crate::index::build_fuzzy_tantivy_query(query, f_content, f_scope);
        if let Some(fq) = fuzzy_query {
            if let Ok(fuzzy_docs) = searcher.search(&fq, &TopDocs::with_limit(top_n)) {
                for (score, addr) in fuzzy_docs {
                    if let Ok(doc) = searcher.doc::<tantivy::TantivyDocument>(addr) {
                        if let Some(val) = doc.get_first(f_idx) {
                            if let Some(id) = val.as_u64() {
                                results.push((id as usize, score));
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Search in-memory content (for stdin/pipe mode). Chunks the content, embeds, and searches.
pub fn search_content(
    content: &str,
    lang_hint: &str,
    query: &str,
    opts: &SearchOptions,
) -> Result<SearchOutput> {
    use crate::embed::cosine_similarity;

    let start = Instant::now();

    // Warn on very large stdin input (RWQ-246)
    const STDIN_WARN_BYTES: usize = 10 * 1024 * 1024; // 10 MB
    if content.len() > STDIN_WARN_BYTES {
        eprintln!(
            "warning: stdin input is large ({:.1} MB), search may be slow",
            content.len() as f64 / (1024.0 * 1024.0)
        );
    }

    // Detect language from hint (delegates to Language::FromStr for consistency)
    let language = lang_hint
        .parse::<crate::index::Language>()
        .unwrap_or(crate::index::Language::Other("text".to_string()));

    let chunks = crate::index::chunk_content("<stdin>", content, language)
        .context("chunk stdin content")?;

    if chunks.is_empty() || query.is_empty() {
        return Ok(SearchOutput {
            schema_version: SEARCH_OUTPUT_VERSION,
            model: None,
            results: Vec::new(),
            query_ms: start.elapsed().as_millis() as u64,
            total_tokens: 0,
        });
    }

    let mut embedder: Option<Embedder> = None;

    // Pre-compute embeddings for semantic/hybrid modes
    let mut query_vec_opt: Option<Vec<f32>> = None;
    let mut chunk_embeddings: Option<ndarray::Array2<f32>> = None;

    if matches!(opts.mode, SearchMode::Semantic | SearchMode::Hybrid) {
        let emb = load_embedder(&mut embedder, opts.model.as_deref(), opts.suppress_status_line)?;
        let qv = emb.embed_query(query)?.to_vec();
        let texts: Vec<String> = chunks.iter().map(|c| {
            format!("// File: {}\n// Language: {}\n// Scope: {}\n\n{}",
                c.file, c.language, c.scope, c.content)
        }).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embs = emb.embed(&text_refs).context("embed chunks")?;
        query_vec_opt = Some(qv);
        chunk_embeddings = Some(embs);
    }

    // Build scored results based on mode
    let mut scored: Vec<(usize, f32)> = match opts.mode {
        SearchMode::Semantic => {
            let query_vec = query_vec_opt.as_ref().unwrap();
            let embeddings = chunk_embeddings.as_ref().unwrap();
            let mut results: Vec<(usize, f32)> = (0..chunks.len())
                .map(|i| {
                    let row = embeddings.row(i).to_vec();
                    (i, cosine_similarity(query_vec, &row))
                })
                .collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results
        }
        SearchMode::Lexical => {
            search_bm25_ram(&chunks, query, opts.top_n * 2)?
        }
        SearchMode::Hybrid => {
            let query_vec = query_vec_opt.as_ref().unwrap();
            let embeddings = chunk_embeddings.as_ref().unwrap();
            let mut semantic: Vec<(usize, f32)> = (0..chunks.len())
                .map(|i| {
                    let row = embeddings.row(i).to_vec();
                    (i, cosine_similarity(query_vec, &row))
                })
                .collect();
            semantic.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            semantic.truncate(opts.top_n * 2);

            let lexical = search_bm25_ram(&chunks, query, opts.top_n * 2)?;

            // RRF merge using index-based IDs
            let sem_ids: Vec<(u64, f32)> = semantic.iter().map(|(i, s)| (*i as u64, *s)).collect();
            let lex_ids: Vec<(u64, f32)> = lexical.iter().map(|(i, s)| (*i as u64, *s)).collect();
            let weight = effective_semantic_weight(query, opts.semantic_weight);
            let merged = rrf_merge(&sem_ids, &lex_ids, opts.rrf_k, weight);
            merged.into_iter().map(|(id, s)| (id as usize, s)).collect()
        }
    };

    // Sort by score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate overlapping chunks
    scored = deduplicate_stdin(&scored, &chunks);

    // Compute max BM25 for lexical normalization
    let max_bm25 = if matches!(opts.mode, SearchMode::Lexical) {
        scored.first().map(|(_, s)| *s).unwrap_or(1.0).max(f32::EPSILON)
    } else {
        1.0
    };

    // Load token counter (stdin mode: failure is expected, no warning needed)
    let counter = TokenCounter::from_managed(opts.model.as_deref()).ok();

    // Build results with proper confidence scores
    let mut results = Vec::new();
    let mut total_tokens: usize = 0;
    for (idx, _score) in &scored {
        if results.len() >= opts.top_n {
            break;
        }
        if *idx >= chunks.len() {
            continue;
        }
        let chunk = &chunks[*idx];
        let lang_str = chunk.language.to_string();

        if let Some(ref lang) = opts.lang_filter {
            if !lang_str.eq_ignore_ascii_case(lang) {
                continue;
            }
        }

        let token_count = counter
            .as_ref()
            .map(|c| c.count_tokens(&chunk.content))
            .unwrap_or(0);

        if let Some(budget) = opts.token_budget {
            if total_tokens + token_count > budget && !results.is_empty() {
                break;
            }
        }

        // Compute confidence by mode
        let confidence = match opts.mode {
            SearchMode::Lexical => {
                (*_score / max_bm25) as f64
            }
            SearchMode::Semantic => {
                // Already cosine similarity
                *_score as f64
            }
            SearchMode::Hybrid => {
                // Compute actual cosine similarity for hybrid results
                if let (Some(ref qv), Some(ref embs)) = (&query_vec_opt, &chunk_embeddings) {
                    let row = embs.row(*idx).to_vec();
                    cosine_similarity(qv, &row) as f64
                } else {
                    *_score as f64
                }
            }
        };

        if confidence < opts.threshold as f64 {
            continue;
        }

        let matched_lines = compute_matched_lines(&chunk.content, query, chunk.lines[0]);
        total_tokens += token_count;
        results.push(SearchResult {
            file: chunk.file.clone(),
            lines: chunk.lines,
            display_start_line: chunk.lines[0],
            language: lang_str,
            scope: chunk.scope.clone(),
            confidence,
            content: chunk.content.clone(),
            context_before: String::new(),
            context_after: String::new(),
            token_count,
            matched_lines,
        });
    }

    let model_name = embedder.as_ref().map(|e| e.model_name().to_string());

    let query_ms = start.elapsed().as_millis() as u64;
    Ok(SearchOutput {
        schema_version: SEARCH_OUTPUT_VERSION,
        model: model_name,
        results,
        query_ms,
        total_tokens,
    })
}

/// Deduplicate overlapping search results within the same file.
/// If chunk B's line range is fully contained within higher-scored chunk A, drop B.
///
/// Complexity: O(n * m) where m is max kept results per file. For typical result
/// sets (n < 100, m < 10) this is effectively O(n). An interval tree would give
/// O(n log n) but the constant factor is worse for small n.
fn deduplicate(scored: &[(u64, f32)], reader: &IndexReader) -> Vec<(u64, f32)> {
    // Group by file, keeping original order (already sorted by score desc)
    let mut kept: Vec<(u64, f32)> = Vec::new();
    // Track kept ranges per file: file -> Vec<[start, end]>
    let mut kept_ranges: HashMap<String, Vec<[usize; 2]>> = HashMap::new();

    for &(id, score) in scored {
        let chunk = match reader.get_chunk(id) {
            Some(c) => c,
            None => {
                eprintln!("warning: chunk {id} not found in index, skipping");
                continue;
            }
        };

        let dominated = kept_ranges
            .get(&chunk.file)
            .is_some_and(|ranges| {
                ranges.iter().any(|r| r[0] <= chunk.lines[0] && chunk.lines[1] <= r[1])
            });

        if !dominated {
            kept_ranges
                .entry(chunk.file.clone())
                .or_default()
                .push(chunk.lines);
            kept.push((id, score));
        }
    }

    kept
}

/// Deduplicate overlapping chunks for stdin/content search (uses index-based IDs).
fn deduplicate_stdin(scored: &[(usize, f32)], chunks: &[crate::index::Chunk]) -> Vec<(usize, f32)> {
    let mut kept: Vec<(usize, f32)> = Vec::new();
    let mut kept_ranges: HashMap<String, Vec<[usize; 2]>> = HashMap::new();

    for &(idx, score) in scored {
        if idx >= chunks.len() {
            continue;
        }
        let chunk = &chunks[idx];
        let dominated = kept_ranges
            .get(&chunk.file)
            .is_some_and(|ranges| {
                ranges.iter().any(|r| r[0] <= chunk.lines[0] && chunk.lines[1] <= r[1])
            });

        if !dominated {
            kept_ranges
                .entry(chunk.file.clone())
                .or_default()
                .push(chunk.lines);
            kept.push((idx, score));
        }
    }

    kept
}

/// Re-rank results using a lightweight keyword overlap heuristic (RWQ-36).
/// Computes what fraction of query terms appear in each result's content,
/// then blends that fraction into the existing confidence score.
/// `alpha` controls how much keyword overlap affects the final score.
pub fn rerank_by_keyword_overlap(results: &mut [SearchResult], query: &str) {

    let terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() > 2)
        .map(|t| t.to_ascii_lowercase()) // RWQ-210: ascii for code-dominated content
        .collect();

    // RWQ-217: skip rerank for single-term queries — binary overlap (0 or 1)
    // is too coarse and destroys the existing confidence ordering
    if terms.len() <= 1 {
        return;
    }

    for result in results.iter_mut() {
        let content_lower = result.content.to_ascii_lowercase();
        let matched = terms
            .iter()
            .filter(|t| content_lower.contains(t.as_str()))
            .count();
        let overlap = matched as f64 / terms.len() as f64;
        result.confidence = result.confidence * (1.0 - RERANK_ALPHA) + overlap * RERANK_ALPHA;
    }

    results.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Default weight multiplier for text chunks (markdown, plain text).
/// Code chunks get 1.0, text chunks get this value.
const DEFAULT_TEXT_WEIGHT: f32 = 0.5;

/// Apply language-based weights to scored results.
/// Text chunks (markdown, plain text) are down-weighted relative to code chunks.
fn apply_language_weights(scored: &[(u64, f32)], reader: &IndexReader, text_weight: f32) -> Vec<(u64, f32)> {
    let mut weighted: Vec<(u64, f32)> = scored
        .iter()
        .map(|&(id, score)| {
            let weight = reader
                .get_chunk(id)
                .map(|c| if c.language == crate::index::Language::Other("text".to_string()) { text_weight } else { 1.0 })
                .unwrap_or(1.0);
            (id, score * weight)
        })
        .collect();
    weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    weighted
}

/// Query type classification for adaptive RRF weighting.
#[derive(Debug, Clone, Copy)]
enum QueryType {
    /// Single token or code identifier (e.g. "search", "DatabaseClient", "fn")
    Identifier,
    /// 2-3 short tokens without natural language markers (e.g. "parse query")
    ShortPhrase,
    /// Multi-word query with natural language patterns (e.g. "how does chunking work")
    NaturalLanguage,
}

/// Classify a query to determine optimal RRF weighting.
fn classify_query(query: &str) -> QueryType {
    let tokens: Vec<&str> = query.split_whitespace().collect();
    if tokens.len() <= 1 {
        return QueryType::Identifier;
    }

    // Code keyword detection: queries starting with language keywords like
    // "fn main", "class MyService", "def parse" should favor BM25 (Identifier)
    const CODE_KEYWORDS: &[&str] = &[
        "fn", "func", "function", "def", "class", "struct", "enum", "impl",
        "trait", "interface", "module", "use", "import", "pub", "async",
        "const", "static", "type", "var", "let", "val",
    ];
    if let Some(first) = tokens.first() {
        if CODE_KEYWORDS.contains(&first.to_lowercase().as_str()) {
            return QueryType::Identifier;
        }
    }

    let nl_indicators = [
        "how", "what", "which", "where", "when", "why", "does", "that", "this",
        "the", "for", "with", "from", "is", "are", "can", "should", "would",
    ];
    let has_nl = tokens
        .iter()
        .any(|t| nl_indicators.contains(&t.to_lowercase().as_str()));
    if has_nl || tokens.len() > 3 {
        return QueryType::NaturalLanguage;
    }
    QueryType::ShortPhrase
}

/// Semantic weights for adaptive RRF by query type.
/// Identifier queries favor BM25 (0.3 semantic); NL queries favor semantic (0.7).
const SEMANTIC_WEIGHT_IDENTIFIER: f64 = 0.3;
const SEMANTIC_WEIGHT_SHORT_PHRASE: f64 = 0.45;
const SEMANTIC_WEIGHT_NATURAL_LANG: f64 = 0.7;

/// Minimum RRF score as fraction of 1/(k+1) to filter noise results.
const RRF_MIN_SCORE_FRACTION: f64 = 0.3;

/// Keyword overlap blending factor for reranking.
const RERANK_ALPHA: f64 = 0.25;

/// Compute effective semantic weight, using adaptive classification when no override is set.
fn effective_semantic_weight(query: &str, override_weight: Option<f64>) -> f64 {
    if let Some(w) = override_weight {
        return w;
    }
    match classify_query(query) {
        QueryType::Identifier => SEMANTIC_WEIGHT_IDENTIFIER,
        QueryType::ShortPhrase => SEMANTIC_WEIGHT_SHORT_PHRASE,
        QueryType::NaturalLanguage => SEMANTIC_WEIGHT_NATURAL_LANG,
    }
}

/// Weighted Reciprocal Rank Fusion.
/// score = sem_weight * 1/(k + rank_s) + (1 - sem_weight) * 1/(k + rank_l).
fn rrf_merge(
    semantic: &[(u64, f32)],
    lexical: &[(u64, f32)],
    k: usize,
    semantic_weight: f64,
) -> Vec<(u64, f32)> {
    let lexical_weight = 1.0 - semantic_weight;
    let mut scores: HashMap<u64, f64> = HashMap::with_capacity(semantic.len() + lexical.len());

    for (rank, (id, _)) in semantic.iter().enumerate() {
        *scores.entry(*id).or_default() +=
            semantic_weight / (k as f64 + rank as f64 + 1.0);
    }
    for (rank, (id, _)) in lexical.iter().enumerate() {
        *scores.entry(*id).or_default() +=
            lexical_weight / (k as f64 + rank as f64 + 1.0);
    }

    let min_rrf = RRF_MIN_SCORE_FRACTION / (k as f64 + 1.0);
    let mut merged: Vec<(u64, f32)> = scores
        .into_iter()
        .map(|(id, score)| (id, score as f32))
        .filter(|(_, score)| *score >= min_rrf as f32)
        .collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

/// Compute which lines in the content match query terms (case-insensitive).
/// Expands camelCase/PascalCase terms to match tantivy's code tokenizer.
/// Returns 1-based line numbers relative to the file.
fn compute_matched_lines(content: &str, query: &str, start_line: usize) -> Vec<usize> {
    let raw_terms: Vec<String> = query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .collect();
    if raw_terms.is_empty() {
        return Vec::new();
    }
    // Expand with camelCase sub-tokens to match tantivy's code tokenizer.
    // Filter noise: drop single-char tokens from CamelCase splits (e.g.
    // "HTTPSServer" → ["H","T","T","P","S","Server"] → keep only ["Server"]).
    let mut terms: Vec<String> = Vec::new();
    for term in &raw_terms {
        terms.push(term.clone());
        let parts = crate::index::split_camel_case(term);
        if parts.len() > 1 {
            terms.extend(
                parts.into_iter()
                    .filter(|p| p.len() > 1)
                    .map(|p| p.to_lowercase()),
            );
        }
    }
    terms.sort_unstable();
    terms.dedup();

    let mut matched = Vec::new();
    for (offset, line) in content.lines().enumerate() {
        let lower = line.to_lowercase();
        if terms.iter().any(|t| lower.contains(t.as_str())) {
            matched.push(start_line + offset);
        }
    }
    matched
}

/// Enriched content with separate context.
struct EnrichedContent {
    content: String,
    context_before: String,
    context_after: String,
    display_start_line: usize,
}

/// Expand chunk content with surrounding context lines or full file.
fn enrich_content(
    chunk: &StoredChunk,
    root: &Path,
    context_lines: usize,
    full_file: bool,
) -> EnrichedContent {
    if !full_file && context_lines == 0 {
        return EnrichedContent {
            content: chunk.content.clone(),
            context_before: String::new(),
            context_after: String::new(),
            display_start_line: chunk.lines[0],
        };
    }

    let file_path = root.join(&chunk.file);
    let file_path = match file_path.canonicalize() {
        Ok(p) if p.starts_with(root) => p,
        Ok(_) => {
            eprintln!("warning: {} resolves outside index root, using stored content", chunk.file);
            return EnrichedContent {
                content: chunk.content.clone(),
                context_before: String::new(),
                context_after: String::new(),
                display_start_line: chunk.lines[0],
            };
        }
        Err(_) => return EnrichedContent {
            content: chunk.content.clone(),
            context_before: String::new(),
            context_after: String::new(),
            display_start_line: chunk.lines[0],
        },
    };
    let source = match std::fs::read_to_string(&file_path) {
        Ok(s) => s.replace('\r', ""),
        Err(_) => return EnrichedContent {
            content: chunk.content.clone(),
            context_before: String::new(),
            context_after: String::new(),
            display_start_line: chunk.lines[0],
        },
    };

    if full_file {
        return EnrichedContent {
            content: source,
            context_before: String::new(),
            context_after: String::new(),
            display_start_line: 1,
        };
    }

    let lines: Vec<&str> = source.lines().collect();

    // Context before (chunk.lines is 1-based)
    let ctx_before_start = chunk.lines[0].saturating_sub(1 + context_lines);
    let ctx_before_end = chunk.lines[0].saturating_sub(1);
    let context_before = if ctx_before_start < ctx_before_end {
        lines[ctx_before_start..ctx_before_end].join("\n")
    } else {
        String::new()
    };

    // Chunk content from source (fresher than stored)
    let chunk_start = chunk.lines[0].saturating_sub(1);
    let chunk_end = chunk.lines[1].min(lines.len());
    let content = lines[chunk_start..chunk_end].join("\n");

    // Context after
    let ctx_after_start = chunk.lines[1].min(lines.len());
    let ctx_after_end = (chunk.lines[1] + context_lines).min(lines.len());
    let context_after = if ctx_after_start < ctx_after_end {
        lines[ctx_after_start..ctx_after_end].join("\n")
    } else {
        String::new()
    };

    let display_start = ctx_before_start + 1;
    EnrichedContent {
        content,
        context_before,
        context_after,
        display_start_line: display_start,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_merge_basic() {
        let semantic = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let lexical = vec![(2, 5.0), (3, 4.0), (4, 3.0)];

        let merged = rrf_merge(&semantic, &lexical, 60, 0.6);

        // Chunk 2 should be top: it's rank 2 in semantic + rank 1 in lexical
        assert_eq!(merged[0].0, 2);
    }

    #[test]
    fn rrf_merge_dedup() {
        let semantic = vec![(1, 0.9)];
        let lexical = vec![(1, 5.0)];

        let merged = rrf_merge(&semantic, &lexical, 60, 0.5);

        // Should have exactly 1 entry (deduped), with combined score
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].0, 1);
        // Score = 0.5/(60+1) + 0.5/(60+1) = 1/61
        let expected = 1.0 / 61.0;
        assert!((merged[0].1 - expected as f32).abs() < 0.001);
    }

    #[test]
    fn rrf_merge_weight_affects_ranking() {
        // Chunk 1 only in semantic, chunk 2 only in lexical
        let semantic = vec![(1, 0.9)];
        let lexical = vec![(2, 5.0)];

        // High semantic weight -> chunk 1 wins
        let high_sem = rrf_merge(&semantic, &lexical, 60, 0.8);
        assert_eq!(high_sem[0].0, 1);

        // Low semantic weight -> chunk 2 wins
        let low_sem = rrf_merge(&semantic, &lexical, 60, 0.2);
        assert_eq!(low_sem[0].0, 2);
    }

    #[test]
    fn context_expansion_boundaries() {
        // Verify context expansion uses correct 1-based line numbers.
        // chunk.lines is [start, end] 1-based. context_lines expands symmetrically.
        let chunk = StoredChunk {
            id: 1,
            file: "test.rs".to_string(),
            lines: [3, 5], // 1-based
            language: crate::index::Language::Rust,
            scope: "fn foo".to_string(),
            content: "line3\nline4\nline5".to_string(),
            kind: "function_item".to_string(),
        };

        // With 0 context, should return original content
        let enriched = enrich_content(&chunk, std::path::Path::new("/nonexistent"), 0, false);
        assert_eq!(enriched.display_start_line, 3);
        assert_eq!(enriched.content, "line3\nline4\nline5");
        assert!(enriched.context_before.is_empty());
        assert!(enriched.context_after.is_empty());
    }
}
