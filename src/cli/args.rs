use std::path::PathBuf;

use clap::{Parser, Subcommand};
use clap_complete::Shell;

pub fn parse_unit_f32(s: &str) -> std::result::Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!("{v} is not in 0.0..=1.0"));
    }
    Ok(v)
}

pub fn parse_unit_f64(s: &str) -> std::result::Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!("{v} is not in 0.0..=1.0"));
    }
    Ok(v)
}

#[derive(Parser)]
#[command(name = "rawq", version, about = "Context retrieval engine for AI agents")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    // --- Core workflow ---

    /// Search the index (default when no subcommand given)
    Search(SearchArgs),

    /// Manage the search index (build, status, remove)
    Index {
        #[command(subcommand)]
        command: IndexCommand,
    },

    /// Search only within the current git diff
    Diff(crate::cli::commands::diff::DiffArgs),

    /// Show codebase structure (symbols and definitions)
    Map(MapArgs),

    /// Watch a directory and auto-re-index on changes
    Watch(WatchArgs),

    // --- Infrastructure ---

    /// Manage embedding models
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },

    /// Manage the embedding daemon
    Daemon {
        #[command(subcommand)]
        command: DaemonCommand,
    },

    // --- Debug / dev ---

    /// Generate an embedding vector for text
    Embed {
        /// Text to embed
        text: String,

        /// Treat input as a query (adds retrieval prefix)
        #[arg(long)]
        query: bool,

        /// Model name, directory path, or ONNX file path (downloads default if not set)
        #[arg(long)]
        model: Option<String>,

        /// Path to tokenizer.json (only needed with explicit ONNX file path)
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// Skip daemon, always load model locally
        #[arg(long = "no-daemon")]
        no_daemon: bool,
    },

    /// Walk a directory and chunk files, printing NDJSON to stdout
    Chunk {
        /// Path to walk (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Subcommand)]
pub enum IndexCommand {
    /// Build or update the search index for a directory
    Build {
        /// Path to index (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Force full re-index (ignore existing manifest)
        #[arg(long)]
        reindex: bool,

        /// Model name or directory path
        #[arg(long)]
        model: Option<String>,

        /// Override embedding batch size (auto-computed if not set)
        #[arg(long = "batch-size")]
        batch_size: Option<usize>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Exclude files matching glob patterns (can be repeated)
        #[arg(short = 'x', long = "exclude")]
        exclude: Vec<String>,
    },

    /// Show index status for a directory
    Status {
        /// Path to check (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Remove the search index for a directory
    Remove {
        /// Path to remove index for (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Remove all cached indexes
        #[arg(long)]
        all: bool,
    },
}

#[derive(Subcommand)]
pub enum ModelCommand {
    /// Download a model from HuggingFace Hub
    Download {
        /// HuggingFace repo ID (e.g. Snowflake/snowflake-arctic-embed-s)
        repo_id: String,

        /// Custom name for the model (defaults to repo name)
        #[arg(long)]
        name: Option<String>,
    },

    /// List installed models
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Remove installed models
    Remove {
        /// Model name to remove
        name: Option<String>,

        /// Remove all installed models
        #[arg(long)]
        all: bool,
    },

    /// Set or show the default model
    Default {
        /// Model name to set as default (shows current default if omitted)
        name: Option<String>,
    },
}

#[derive(Parser, Debug)]
pub struct SearchArgs {
    /// Search query
    pub query: String,

    /// Path to search (defaults to current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Use exact/lexical BM25 search only
    #[arg(short = 'e', long = "exact", group = "search_mode")]
    pub exact: bool,

    /// Use hybrid search (semantic + BM25 fused with RRF) [default]
    #[arg(short = 'H', long = "hybrid", group = "search_mode")]
    pub hybrid: bool,

    /// Use semantic-only search
    #[arg(short = 's', long = "semantic", group = "search_mode")]
    pub semantic: bool,

    /// Output as JSON
    #[arg(long = "json", group = "output_format")]
    pub json: bool,

    /// Output as streaming NDJSON (one result per line, metadata last)
    #[arg(long, group = "output_format")]
    pub stream: bool,

    /// Number of results to return
    #[arg(short = 'n', long = "top", default_value = "10")]
    pub top_n: usize,

    /// Number of context lines around each result
    #[arg(short = 'C', long = "context", default_value = "3")]
    pub context_lines: usize,

    /// Show full file content for each result
    #[arg(long = "full-file")]
    pub full_file: bool,

    /// Minimum confidence threshold (0.0-1.0)
    #[arg(long = "threshold", default_value = "0.3", value_parser = parse_unit_f32)]
    pub threshold: f32,

    /// Filter results by language (e.g. rust, python, typescript)
    #[arg(long = "lang")]
    pub lang: Option<String>,

    /// Force re-index before searching
    #[arg(long)]
    pub reindex: bool,

    /// Model name or directory path
    #[arg(long)]
    pub model: Option<String>,

    /// Maximum total tokens across all results
    #[arg(long = "token-budget")]
    pub token_budget: Option<usize>,

    /// Read content from stdin instead of indexed files
    #[arg(long)]
    pub stdin: bool,

    /// Language hint for stdin content (e.g. rust, python, typescript)
    #[arg(long = "lang-hint", default_value = "text")]
    pub lang_hint: String,

    /// RRF smoothing constant k (default: 60, higher = more uniform ranking)
    #[arg(long = "rrf-k", default_value = "60")]
    pub rrf_k: usize,

    /// Semantic weight in hybrid RRF (0.0-1.0, auto-detected if omitted)
    #[arg(long = "rrf-weight", value_parser = parse_unit_f64)]
    pub rrf_weight: Option<f64>,

    /// Exclude files matching glob patterns (can be repeated)
    #[arg(short = 'x', long = "exclude")]
    pub exclude: Vec<String>,

    /// Re-rank top results with a keyword overlap heuristic after initial scoring
    #[arg(long)]
    pub rerank: bool,

    /// Weight multiplier for text/markdown results (0.0-1.0, default 0.5).
    /// Set to 1.0 to treat text equally with code.
    #[arg(long = "text-weight")]
    pub text_weight: Option<f32>,

    /// Skip daemon and embed locally
    #[arg(long = "no-daemon")]
    pub no_daemon: bool,

    /// Skip automatic index freshness check before searching
    #[arg(long = "no-reindex")]
    pub no_reindex: bool,
}

#[derive(Parser, Debug)]
pub struct MapArgs {
    /// Path to map (defaults to current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Filter by language (e.g. rust, python, typescript)
    #[arg(long = "lang")]
    pub lang: Option<String>,

    /// Max nesting depth for symbols (default: 2 = definitions + one level of children)
    #[arg(long, default_value = "2")]
    pub depth: usize,

    /// Exclude files matching glob patterns (can be repeated)
    #[arg(short = 'x', long = "exclude")]
    pub exclude: Vec<String>,
}

#[derive(Parser, Debug)]
pub struct WatchArgs {
    /// Path to watch (defaults to current directory)
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Poll interval in seconds (minimum 1)
    #[arg(long = "interval", default_value = "2")]
    pub interval: u64,

    /// Model name or directory path
    #[arg(long)]
    pub model: Option<String>,

    /// Exclude files matching glob patterns (can be repeated)
    #[arg(short = 'x', long = "exclude")]
    pub exclude: Vec<String>,

    /// Skip daemon and embed locally
    #[arg(long = "no-daemon")]
    pub no_daemon: bool,
}

#[derive(Subcommand)]
pub enum DaemonCommand {
    /// Start the embedding daemon
    Start {
        /// Model name or directory path
        #[arg(long)]
        model: Option<String>,

        /// Idle timeout in minutes before auto-shutdown
        #[arg(long, default_value = "30")]
        idle_timeout: u64,

        /// Run in background (detached)
        #[arg(long)]
        background: bool,
    },

    /// Stop the embedding daemon
    Stop {
        /// Model name (to identify which daemon to stop)
        #[arg(long)]
        model: Option<String>,
    },

    /// Show embedding daemon status
    Status {
        /// Model name (to identify which daemon to query)
        #[arg(long)]
        model: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}
