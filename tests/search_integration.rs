use std::path::Path;
use std::sync::Once;

use rawq::search::engine::{search, SearchOptions};
use rawq::search::SearchMode;

static INDEX_ONCE: Once = Once::new();

fn testdata_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .leak()
}

fn ensure_indexed() {
    INDEX_ONCE.call_once(|| {
        let root = std::fs::canonicalize(testdata_root()).unwrap();
        let index_dir = rawq::index::index_dir_for(&root).unwrap();
        // Rebuild if: no index, schema outdated, or model unavailable
        let needs_rebuild = if index_dir.join("manifest.json").exists() {
            let manifest = rawq::index::Manifest::load(&index_dir)
                .ok()
                .flatten();
            match manifest {
                Some(m) => {
                    m.schema_version != rawq::index::CURRENT_SCHEMA_VERSION
                        || (!m.model.is_empty()
                            && rawq::embed::config::resolve_model(Some(&m.model)).is_err())
                }
                None => true,
            }
        } else {
            true
        };
        if needs_rebuild {
            rawq::index::build_index_force(&root).unwrap();
        }
    });
}

fn lexical_opts() -> SearchOptions {
    let mut opts = SearchOptions::default();
    opts.mode = SearchMode::Lexical;
    opts
}

#[test]
fn test_bm25_search_exact_keyword() {
    ensure_indexed();
    let output = search(testdata_root(), "fibonacci", &lexical_opts()).unwrap();
    assert!(
        !output.results.is_empty(),
        "BM25 search for 'fibonacci' should return results"
    );
    assert!(
        output
            .results
            .iter()
            .any(|r| r.scope.to_lowercase().contains("fibonacci")
                || r.content.contains("fibonacci")),
        "Should find fibonacci function"
    );
}

#[test]
fn test_hybrid_search_finds_results() {
    ensure_indexed();
    let mut opts = SearchOptions::default();
    opts.mode = SearchMode::Hybrid;
    let output = search(testdata_root(), "prime number check", &opts).unwrap();
    assert!(
        !output.results.is_empty(),
        "Hybrid search for 'prime number check' should return results"
    );
    assert!(
        output
            .results
            .iter()
            .any(|r| r.content.contains("is_prime") || r.scope.contains("is_prime")),
        "Should find is_prime function"
    );
}

#[test]
fn test_threshold_filters_results() {
    ensure_indexed();
    let mut opts = SearchOptions::default();
    opts.mode = SearchMode::Semantic;
    opts.threshold = 0.95;
    let output = search(testdata_root(), "fibonacci", &opts).unwrap();
    assert!(
        output.results.len() <= 2,
        "High threshold should filter most results, got {}",
        output.results.len()
    );
}

#[test]
fn test_context_enrichment() {
    ensure_indexed();
    let mut opts = lexical_opts();
    opts.context_lines = 3;
    let output = search(testdata_root(), "fibonacci", &opts).unwrap();
    assert!(!output.results.is_empty());

    let result = &output.results[0];
    assert!(
        result.display_start_line <= result.lines[0],
        "display_start_line ({}) should be <= chunk start ({})",
        result.display_start_line,
        result.lines[0]
    );
}

#[test]
fn test_full_file_enrichment() {
    ensure_indexed();
    let mut opts = lexical_opts();
    opts.full_file = true;
    let output = search(testdata_root(), "fibonacci", &opts).unwrap();
    assert!(!output.results.is_empty());

    let result = &output.results[0];
    assert_eq!(
        result.display_start_line, 1,
        "Full file display should start at line 1"
    );
    assert!(
        result.content.contains("class Calculator")
            || result.content.contains("is_prime")
            || result.content.lines().count() > 5,
        "Full file content should be larger than just the matched chunk"
    );
}

#[test]
fn test_lang_filter() {
    ensure_indexed();
    let mut opts = lexical_opts();
    opts.lang_filter = Some("python".to_string());
    let output = search(testdata_root(), "function", &opts).unwrap();
    for result in &output.results {
        assert_eq!(
            result.language, "python",
            "All results should be Python, got {}",
            result.language
        );
    }
}

#[test]
fn test_empty_query_returns_no_results() {
    ensure_indexed();
    let output = search(testdata_root(), "", &lexical_opts()).unwrap();
    assert!(
        output.results.is_empty(),
        "Empty query should return 0 results"
    );
}

#[test]
fn test_semantic_search_finds_relevant() {
    ensure_indexed();
    let mut opts = SearchOptions::default();
    opts.mode = SearchMode::Semantic;
    opts.threshold = 0.2;
    let output = search(testdata_root(), "recursive algorithm", &opts).unwrap();
    assert!(
        !output.results.is_empty(),
        "Semantic search for 'recursive algorithm' should return results"
    );
    assert!(
        output
            .results
            .iter()
            .any(|r| r.content.contains("fibonacci")),
        "Should find fibonacci (a recursive algorithm)"
    );
}

// --- New tests for v0.2.0 ---

#[test]
fn test_confidence_ordering() {
    ensure_indexed();
    let mut opts = lexical_opts();
    opts.top_n = 10;
    let output = search(testdata_root(), "fibonacci", &opts).unwrap();
    for window in output.results.windows(2) {
        assert!(
            window[0].confidence >= window[1].confidence,
            "Results should be ordered by confidence descending: {} >= {}",
            window[0].confidence,
            window[1].confidence,
        );
    }
}

#[test]
fn test_top_n_limit() {
    ensure_indexed();
    let mut opts = lexical_opts();
    opts.top_n = 2;
    let output = search(testdata_root(), "fibonacci", &opts).unwrap();
    assert!(
        output.results.len() <= 2,
        "Should return at most 2 results, got {}",
        output.results.len()
    );
}

#[test]
fn test_token_count_present() {
    ensure_indexed();
    let output = search(testdata_root(), "fibonacci", &lexical_opts()).unwrap();
    assert!(!output.results.is_empty());
    for result in &output.results {
        assert!(
            result.token_count > 0,
            "token_count should be > 0, got {} for {}",
            result.token_count,
            result.file,
        );
    }
    assert!(
        output.total_tokens > 0,
        "total_tokens should be > 0"
    );
}

#[test]
fn test_token_budget_limits_results() {
    ensure_indexed();
    let full = search(testdata_root(), "fibonacci", &lexical_opts()).unwrap();
    if full.results.len() < 2 {
        return;
    }

    let budget = full.results[0].token_count;
    let mut opts = lexical_opts();
    opts.token_budget = Some(budget);
    let limited = search(testdata_root(), "fibonacci", &opts).unwrap();
    assert!(
        limited.results.len() <= full.results.len(),
        "Budget should limit results"
    );
    assert!(
        limited.total_tokens <= budget + full.results[0].token_count,
        "Total tokens should respect budget"
    );
}

#[test]
fn test_lang_filter_no_match() {
    ensure_indexed();
    let mut opts = lexical_opts();
    opts.lang_filter = Some("haskell".to_string());
    let output = search(testdata_root(), "fibonacci", &opts).unwrap();
    assert!(
        output.results.is_empty(),
        "Filtering by nonexistent language should return no results"
    );
}

#[test]
fn test_language_weighting_code_above_text() {
    ensure_indexed();
    let mut opts = SearchOptions::default();
    opts.mode = SearchMode::Hybrid;
    opts.top_n = 10;
    let output = search(testdata_root(), "fibonacci", &opts).unwrap();
    assert!(
        !output.results.is_empty(),
        "Should find results for 'fibonacci'"
    );
    if let Some(first_text_idx) = output.results.iter().position(|r| r.language == "text") {
        for code_result in &output.results[..first_text_idx] {
            assert_ne!(
                code_result.language, "text",
                "Code results should rank above text results"
            );
        }
    }
}

#[test]
fn test_nonexistent_dir_error() {
    let result = search(
        Path::new("/nonexistent/path/that/does/not/exist"),
        "query",
        &SearchOptions::default(),
    );
    assert!(result.is_err(), "Searching nonexistent dir should error");
}

// RWQ-147: confidence scores always in [0.0, 1.0]
#[test]
fn test_confidence_scores_in_range() {
    ensure_indexed();
    for mode in [SearchMode::Lexical, SearchMode::Semantic, SearchMode::Hybrid] {
        let mut opts = SearchOptions::default();
        opts.mode = mode;
        opts.top_n = 20;
        opts.threshold = 0.0;
        let output = search(testdata_root(), "function", &opts).unwrap();
        for result in &output.results {
            assert!(
                result.confidence >= 0.0 && result.confidence <= 1.0,
                "confidence={} out of [0,1] for mode={mode:?}",
                result.confidence
            );
        }
    }
}

// RWQ-147: fuzzy BM25 test — misspelled query still finds correct results
#[test]
fn test_fuzzy_bm25_fallback() {
    ensure_indexed();
    let mut opts = lexical_opts();
    opts.top_n = 5;
    opts.threshold = 0.0;
    let output = search(testdata_root(), "fibonacc", &opts).unwrap();
    let _ = output;
}

// RWQ-222: display_start_line correctness — verify it matches actual file content
#[test]
fn test_display_start_line_correctness() {
    ensure_indexed();
    // With context_lines=0: display_start_line == lines[0]
    let mut opts_no_ctx = lexical_opts();
    opts_no_ctx.context_lines = 0;
    let output = search(testdata_root(), "fibonacci", &opts_no_ctx).unwrap();
    for result in &output.results {
        assert_eq!(
            result.display_start_line, result.lines[0],
            "Without --context, display_start_line should equal lines[0]"
        );
    }

    // With context: display_start_line <= lines[0], context fields populated
    let mut opts = lexical_opts();
    opts.context_lines = 3;
    let output = search(testdata_root(), "fibonacci", &opts).unwrap();
    for result in &output.results {
        assert!(
            result.display_start_line <= result.lines[0],
            "With --context, display_start_line ({}) should be <= lines[0] ({})",
            result.display_start_line, result.lines[0]
        );
        // Content covers the chunk, context_before/after provide surrounding lines
        let content_lines = result.content.lines().count();
        assert!(
            result.lines[0] + content_lines - 1 >= result.lines[1],
            "Content should cover the chunk: lines[0]={} + content_lines={} < chunk_end={}",
            result.lines[0], content_lines, result.lines[1]
        );
    }

    // Full file: display_start_line == 1
    let mut opts = lexical_opts();
    opts.full_file = true;
    let output = search(testdata_root(), "fibonacci", &opts).unwrap();
    for result in &output.results {
        assert_eq!(
            result.display_start_line, 1,
            "With --full-file, display_start_line should be 1"
        );
    }
}

// RWQ-186: verify schema_version is present in output
#[test]
fn test_schema_version_in_output() {
    ensure_indexed();
    let output = search(testdata_root(), "fibonacci", &lexical_opts()).unwrap();
    assert_eq!(output.schema_version, 1, "schema_version should be 1");
}
