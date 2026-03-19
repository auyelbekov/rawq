use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::{Context, Result};
use crate::search::{SearchMode, SearchOptions};

use crate::cli::args::parse_unit_f32;
use crate::cli::output::{print_human, print_json, print_ndjson};
use crate::cli::util::{effective_model, resolve_and_embed_query, resolve_search_mode, DAEMON_READY_MSG};

#[derive(clap::Parser, Debug)]
pub struct DiffArgs {
    /// Search query
    pub query: String,

    /// Path to search (defaults to current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Use exact/lexical BM25 search only
    #[arg(short = 'e', long = "exact", group = "search_mode")]
    pub exact: bool,

    /// Use semantic-only search
    #[arg(short = 's', long = "semantic", group = "search_mode")]
    pub semantic: bool,

    /// Output as JSON
    #[arg(long = "json", group = "output_format")]
    pub json: bool,

    /// Output as streaming NDJSON
    #[arg(long, group = "output_format")]
    pub stream: bool,

    /// Number of results to return
    #[arg(short = 'n', long = "top", default_value = "10")]
    pub top_n: usize,

    /// Number of context lines around each result
    #[arg(short = 'C', long = "context", default_value = "3")]
    pub context_lines: usize,

    /// Minimum confidence threshold (0.0-1.0)
    #[arg(long = "threshold", default_value = "0.3", value_parser = parse_unit_f32)]
    pub threshold: f32,

    /// Filter results by language (e.g. rust, python, typescript)
    #[arg(long = "lang")]
    pub lang: Option<String>,

    /// Model name or directory path
    #[arg(long)]
    pub model: Option<String>,

    /// Maximum total tokens across all results
    #[arg(long = "token-budget")]
    pub token_budget: Option<usize>,

    /// Include staged changes in addition to unstaged (runs both git diff and git diff --cached)
    #[arg(long)]
    pub staged: bool,

    /// Compare against a specific commit or branch instead of working tree
    #[arg(long)]
    pub base: Option<String>,

    /// Skip daemon and embed locally
    #[arg(long = "no-daemon")]
    pub no_daemon: bool,
}

#[allow(clippy::field_reassign_with_default)]
pub fn cmd_diff(args: DiffArgs) -> Result<i32> {
    if args.query.trim().is_empty() {
        eprintln!("error: query cannot be empty");
        return Ok(2);
    }

    let path = std::fs::canonicalize(&args.path).context("canonicalize path")?;

    // Step 1: collect changed line ranges from git
    let diff_ranges = collect_diff_ranges(&path, args.staged, args.base.as_deref())?;

    if diff_ranges.is_empty() {
        eprintln!("No changes found in git diff.");
        return Ok(1);
    }

    // Step 2: open index and collect chunk IDs that overlap with the diff
    let index_dir = crate::index::index_dir_for(&path)?;
    let _lock = crate::index::lock::IndexLock::read(&index_dir)?;
    let reader = crate::index::IndexReader::open(&index_dir).context("open index")?;

    let diff_chunk_ids: HashSet<u64> = reader
        .all_chunks()
        .filter(|chunk| overlaps_diff(&chunk.file, chunk.lines, &diff_ranges))
        .map(|chunk| chunk.id)
        .collect();

    if diff_chunk_ids.is_empty() {
        eprintln!("No indexed chunks overlap with the current diff.");
        eprintln!("(Run `rawq index build` to index the changed files first.)");
        return Ok(1);
    }

    eprintln!("{} chunk(s) in diff scope.", diff_chunk_ids.len());

    // Step 3: search using the open reader, then filter to diff chunks
    let mode = resolve_search_mode(args.exact, args.semantic);

    let effective = effective_model(&args.model);

    // Try daemon for fast query embedding (semantic/hybrid modes only)
    let (pre_embedded_query, model_name_override, freshly_started) =
        if !matches!(mode, SearchMode::Lexical) {
            resolve_and_embed_query(&effective, &args.query, args.no_daemon)
        } else {
            (None, None, false)
        };

    if freshly_started {
        eprintln!("{DAEMON_READY_MSG}");
    }

    let search_top_n = (diff_chunk_ids.len() + args.top_n).max(50);
    let mut opts = SearchOptions::default();
    opts.mode = mode;
    opts.top_n = search_top_n;
    opts.threshold = args.threshold;
    opts.context_lines = args.context_lines;
    opts.lang_filter = args.lang.clone();
    opts.model = effective;
    opts.token_budget = args.token_budget;
    opts.pre_embedded_query = pre_embedded_query;
    opts.model_name_override = model_name_override;

    let mut output = crate::search::search_with_reader(&reader, &path, &args.query, &opts)?;

    // Filter to only results that overlap with the diff
    output.results.retain(|r| {
        // Match by file + line overlap. r.lines are [start, end] 1-based.
        let result_file = crate::index::normalize_path(&r.file);
        diff_ranges.iter().any(|(diff_file, ranges)| {
            let df = crate::index::normalize_path(diff_file);
            // The stored path may be relative or include the root prefix
            let files_match = result_file == df
                || result_file.ends_with(&format!("/{df}"))
                || df.ends_with(&format!("/{result_file}"));
            if !files_match {
                return false;
            }
            ranges.iter().any(|&[rstart, rend]| {
                // Overlap: chunk [r.lines[0], r.lines[1]] vs hunk [rstart, rend]
                r.lines[0] <= rend && r.lines[1] >= rstart
            })
        })
    });
    output.results.truncate(args.top_n);
    output.total_tokens = output.results.iter().map(|r| r.token_count).sum();

    if output.results.is_empty() {
        eprintln!("No results match the query within the diff scope.");
        return Ok(1);
    }

    if args.stream {
        print_ndjson(&output);
    } else if args.json {
        print_json(&output);
    } else {
        print_human(&output);
    }

    eprintln!(
        "{} result(s) in diff scope in {}ms",
        output.results.len(),
        output.query_ms
    );

    Ok(0)
}

/// Run `git diff -U0` (and optionally `--cached`) and parse hunk line ranges.
/// Returns a map of relative file path → list of [start, end] new-file line ranges.
fn collect_diff_ranges(
    repo_root: &std::path::Path,
    include_staged: bool,
    base: Option<&str>,
) -> Result<HashMap<String, Vec<[usize; 2]>>> {
    let mut result: HashMap<String, Vec<[usize; 2]>> = HashMap::new();

    // Build the git diff command(s) to run
    let mut commands: Vec<Vec<String>> = Vec::new();

    if let Some(base_ref) = base {
        // Compare HEAD against a specific base
        commands.push(vec!["diff".into(), "-U0".into(), base_ref.to_string()]);
    } else {
        // Working tree (unstaged) changes
        commands.push(vec!["diff".into(), "-U0".into()]);
        if include_staged {
            commands.push(vec!["diff".into(), "--cached".into(), "-U0".into()]);
        }
    }

    for git_args in commands {
        let output = std::process::Command::new("git")
            .args(&git_args)
            .current_dir(repo_root)
            .output()
            .with_context(|| format!("failed to run `git {}`", git_args.join(" ")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git diff failed: {}", stderr.trim());
        }

        let text = String::from_utf8_lossy(&output.stdout);
        parse_diff_output(&text, &mut result);
    }

    Ok(result)
}

/// Parse the output of `git diff -U0` and accumulate file → hunk ranges.
fn parse_diff_output(diff_text: &str, result: &mut HashMap<String, Vec<[usize; 2]>>) {
    let mut current_file: Option<String> = None;

    for line in diff_text.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            // `+++ b/path/to/file.rs` — extract the path after "b/"
            let path = rest.to_string();
            current_file = Some(path);
        } else if line.starts_with("@@ ") {
            if let (Some(ref file), Some(range)) = (&current_file, parse_hunk_range(line)) {
                result.entry(file.clone()).or_default().push(range);
            }
        }
    }
}

/// Parse `@@ -a,b +c,d @@` and return the new-file line range [c, c+d-1].
/// Returns None for pure-deletion hunks (count == 0).
fn parse_hunk_range(line: &str) -> Option<[usize; 2]> {
    // Format: `@@ -old_start[,old_count] +new_start[,new_count] @@`
    let plus_part = line
        .split_whitespace()
        .find(|s| s.starts_with('+') && !s.starts_with("+++"))?;

    let rest = plus_part.get(1..)?; // strip '+', returns None if empty
    if rest.is_empty() {
        return None;
    }
    let (start_str, count_str) = if let Some(comma) = rest.find(',') {
        (&rest[..comma], &rest[comma + 1..])
    } else {
        (rest, "1")
    };

    let start: usize = start_str.parse().ok()?;
    let count: usize = count_str.parse().ok()?;
    if count == 0 {
        // Pure deletion — no new lines added
        return None;
    }
    Some([start, start + count - 1])
}

/// Check if a chunk (file, lines) overlaps with any changed hunk in the diff.
fn overlaps_diff(
    chunk_file: &str,
    chunk_lines: [usize; 2],
    diff_ranges: &HashMap<String, Vec<[usize; 2]>>,
) -> bool {
    let norm_chunk = crate::index::normalize_path(chunk_file);

    for (diff_file, ranges) in diff_ranges {
        let norm_diff = crate::index::normalize_path(diff_file);

        // Files match if paths are equal or one is a suffix of the other
        let files_match = norm_chunk == norm_diff
            || norm_chunk.ends_with(&format!("/{norm_diff}"))
            || norm_diff.ends_with(&format!("/{norm_chunk}"));

        if files_match {
            for &[rstart, rend] in ranges {
                if chunk_lines[0] <= rend && chunk_lines[1] >= rstart {
                    return true;
                }
            }
        }
    }
    false
}
