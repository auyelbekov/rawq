use anyhow::{Context, Result};
use crate::search::SearchOptions;

use crate::cli::args::SearchArgs;
use crate::cli::output::{print_human, print_json, print_ndjson};
use crate::cli::util::{
    build_index_and_ensure_daemon, check_index_freshness, effective_model,
    resolve_and_embed_query, resolve_search_mode, DAEMON_READY_MSG,
};

/// Build SearchOptions from common SearchArgs fields.
#[allow(clippy::field_reassign_with_default)]
fn base_search_opts(args: &SearchArgs, effective: &Option<String>) -> SearchOptions {
    let mut opts = SearchOptions::default();
    opts.top_n = args.top_n;
    opts.threshold = args.threshold;
    opts.lang_filter = args.lang.clone();
    opts.model = effective.clone();
    opts.token_budget = args.token_budget;
    opts.rrf_k = args.rrf_k;
    opts.semantic_weight = args.rrf_weight;
    opts.rerank = args.rerank;
    opts.text_weight = args.text_weight;
    opts
}

/// Print search output and summary, return exit code.
fn print_and_summarize(output: &crate::search::SearchOutput, stream: bool, json: bool) -> i32 {
    if stream {
        print_ndjson(output);
    } else if json {
        print_json(output);
    } else {
        print_human(output);
    }

    if let Some(ref model) = output.model {
        eprintln!(
            "Model: {} | {} results in {}ms",
            model,
            output.results.len(),
            output.query_ms
        );
    } else {
        eprintln!(
            "{} results in {}ms",
            output.results.len(),
            output.query_ms
        );
    }

    if output.results.is_empty() { 1 } else { 0 }
}

pub fn cmd_search(args: SearchArgs) -> Result<i32> {
    if args.query.trim().is_empty() {
        eprintln!("error: query cannot be empty");
        return Ok(2);
    }

    // Stdin pipe mode
    if args.stdin || args.path == std::path::Path::new("-") {
        return cmd_search_stdin(&args);
    }

    let effective = effective_model(&args.model);

    if args.reindex {
        eprintln!("Forcing re-index before search...");
        let opts = crate::index::IndexOptions {
            model: effective.clone(),
            batch_size: None,
            exclude_patterns: args.exclude.clone(),
        };
        build_index_and_ensure_daemon(&args.path, true, &opts, args.no_daemon)?;
    } else {
        let canon = std::fs::canonicalize(&args.path).context("canonicalize path")?;
        let index_dir = crate::index::index_dir_for(&canon)?;
        if !index_dir.join("manifest.json").exists() {
            // No index at all — full build
            eprintln!("No index found, building...");
            let opts = crate::index::IndexOptions {
                model: effective.clone(),
                exclude_patterns: args.exclude.clone(),
                ..Default::default()
            };
            build_index_and_ensure_daemon(&canon, false, &opts, args.no_daemon)?;
        } else if !args.no_reindex {
            // Index exists — check freshness and auto-reindex if stale (RWQ-257)
            if let Some((added, changed, removed)) =
                check_index_freshness(&args.path, &args.exclude)?
            {
                eprintln!(
                    "Updating index (+{added} ~{changed} -{removed} files)..."
                );
                let opts = crate::index::IndexOptions {
                    model: effective.clone(),
                    exclude_patterns: args.exclude.clone(),
                    ..Default::default()
                };
                build_index_and_ensure_daemon(&canon, false, &opts, args.no_daemon)?;
            }
        }
    }

    let mode = resolve_search_mode(args.exact, args.semantic);

    // Try daemon for fast query embedding (semantic/hybrid modes only).
    // If index was just built, daemon is already running from ensure_daemon_running.
    // If index existed, daemon may be started here for the first time.
    let (pre_embedded_query, model_name_override, freshly_started) =
        if !matches!(mode, crate::search::SearchMode::Lexical) {
            resolve_and_embed_query(&effective, &args.query, args.no_daemon)
        } else {
            (None, None, false)
        };

    if freshly_started {
        eprintln!("{DAEMON_READY_MSG}");
    }

    let mut opts = base_search_opts(&args, &effective);
    opts.mode = mode;
    opts.context_lines = args.context_lines;
    opts.full_file = args.full_file;
    opts.exclude_patterns = args.exclude.clone();
    opts.pre_embedded_query = pre_embedded_query;
    opts.model_name_override = model_name_override;

    let output = crate::search::engine::search(&args.path, &args.query, &opts)?;

    Ok(print_and_summarize(&output, args.stream, args.json))
}

fn cmd_search_stdin(args: &SearchArgs) -> Result<i32> {
    use std::io::Read;

    if args.context_lines > 0 {
        eprintln!("warning: --context is ignored with --stdin");
    }
    if args.full_file {
        eprintln!("warning: --full-file is ignored with --stdin");
    }

    let mut content = String::new();
    std::io::stdin().read_to_string(&mut content)?;

    let effective = effective_model(&args.model);
    let mode = resolve_search_mode(args.exact, args.semantic);

    let mut opts = base_search_opts(args, &effective);
    opts.mode = mode;

    let output = crate::search::search_content(&content, &args.lang_hint, &args.query, &opts)?;

    Ok(print_and_summarize(&output, args.stream, args.json))
}
