# rawq

Context retrieval engine for AI agents.

Semantic + lexical search over codebases. Single Rust binary. Fully offline. Built for AI agents.

## Why

AI agents waste tokens reading irrelevant files. rawq returns only the relevant code — with file paths, line ranges, scope names, and confidence scores. Searching a 10k-file codebase yields 5-10 relevant chunks instead of 50+ full files.

## Install

**Quick install** (prebuilt binary, auto-adds to PATH):

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/auyelbekov/rawq/main/scripts/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/auyelbekov/rawq/main/scripts/install.ps1 | iex"
```

Or download manually from [GitHub Releases](https://github.com/auyelbekov/rawq/releases).

**Cargo** (requires Rust toolchain):

```bash
cargo install rawq
```

**GPU acceleration** — prebuilt binaries include GPU support. For cargo installs, enable with feature flags:

```bash
cargo install rawq --features directml   # Windows (DirectML)
cargo install rawq --features cuda       # Linux (CUDA)
cargo install rawq --features coreml     # macOS (CoreML)
```

**Build from source:**

```bash
git clone https://github.com/auyelbekov/rawq.git
cd rawq
cargo build --release --features directml   # or cuda / coreml
```

## Quick start

```bash
# Search a codebase (auto-downloads snowflake-arctic-embed-s + indexes on first run)
rawq "database connection retry" ./src

# Structured JSON output
rawq search "database connection retry" ./src --json

# Lexical BM25 only
rawq search -e "reconnect" ./src

# Semantic only
rawq search -s "how does retry logic work" ./src
```

## Search output

```
src/db/connection.py:23-41  [91%]  DatabaseClient.reconnect
   23 | def reconnect(self, max_retries=3):
   24 |     """Attempt to re-establish database connection"""
   25 |     for attempt in range(max_retries):
```

With `--json`:

```json
{
  "schema_version": 1,
  "model": "snowflake-arctic-embed-s",
  "results": [
    {
      "file": "src/db/connection.py",
      "lines": [23, 41],
      "display_start_line": 23,
      "language": "python",
      "scope": "DatabaseClient.reconnect",
      "confidence": 0.91,
      "content": "def reconnect(self, max_retries=3): ...",
      "token_count": 45,
      "matched_lines": [23]
    }
  ],
  "query_ms": 8,
  "total_tokens": 45
}
```

## Features

- **Hybrid search** — RRF-fused semantic (ONNX embeddings) + lexical (tantivy BM25) with adaptive query weighting
- **16 languages** — tree-sitter AST chunking for Rust, Python, TypeScript, JavaScript, Go, Java, C, C++, C#, Ruby, PHP, Swift, Bash, Lua, Scala, Dart
- **Universal fallback** — any text file automatically indexed with its real extension as the language label (`.sql`, `.yaml`, `.proto`, `.tf`, etc.)
- **Incremental indexing** — SHA-256 per chunk, git-aware change detection, sub-second re-index
- **Fully offline** — ONNX Runtime inference, no network calls after initial model download
- **Agent-friendly** — `--json`, `--stream` (NDJSON), `--token-budget`, exit codes (0=found, 1=none, 2=error)
- **GPU acceleration** — auto-detects best GPU and computes batch sizes from actual VRAM. DirectML, CUDA, CoreML with automatic CPU fallback
- **Daemon mode** — holds ONNX model hot in background, auto-starts on first search, auto-exits after 30min idle
- **Diff-scoped search** — `rawq diff "query"` searches only within the current git diff
- **Re-ranking** — `--rerank` applies keyword overlap heuristic for two-pass result ordering
- **Codebase map** — `rawq map` shows AST-based structure with real hierarchy (impl > methods)
- **Terminal UX** — syntax highlighting via `bat`, paged output, context lines around matches

## Commands

```bash
rawq "query" [path]                     # Search (default)
rawq search "query" [path]              # Search with options
rawq search "query" [path] --json       # JSON output
rawq search "query" [path] --stream     # NDJSON streaming
rawq search "query" [path] --rerank     # Two-pass re-ranking
rawq search "query" [path] --context 5  # 5 context lines
rawq search "query" [path] --full-file  # Full file content
rawq index build [path]                 # Build index explicitly
rawq index build --reindex [path]       # Force full re-index
rawq index status [path]                # Show index stats
rawq index remove [path]               # Remove index
rawq diff "query" [path]               # Search within git diff
rawq map [path]                         # Show codebase structure
rawq watch [path]                       # Auto-re-index on changes
rawq model download [name]             # Download a model
rawq model list                         # List available models
rawq embed "text"                       # Generate embedding vector
rawq daemon status                      # Check daemon status
rawq daemon stop                        # Stop daemon
```

## Models

rawq auto-downloads the default model on first use. Available models:

| Model | Dimensions | Sequence Length | Notes |
|-------|-----------|----------------|-------|
| snowflake-arctic-embed-s | 384 | 512 | Default. Small, fast. |
| snowflake-arctic-embed-m-v1.5 | 768 | 512 | Recommended. Better quality. |
| jina-embeddings-v2-base-code | 768 | 8192 | Code-specialized, long context. |

Switch models with `rawq model download <name>` and `rawq model default <name>`.

## Environment variables

| Variable | Description |
|----------|-------------|
| `RAWQ_MODEL` | Override default model |
| `RAWQ_NO_GPU` | Force CPU mode (`=1`) |
| `RAWQ_NO_DAEMON` | Disable daemon (`=1`) |
| `RAWQ_NO_BAT` | Disable syntax highlighting (`=1`) |
| `RAWQ_NO_PAGER` | Disable paged output (`=1`) |
| `RAWQ_OFFLINE` | Skip network calls (`=1`) |
| `RAWQ_DML_DEVICE` | DirectML device index |
| `RAWQ_CUDA_DEVICE` | CUDA device index |
| `RAWQ_VRAM_BUDGET` | Override VRAM budget (bytes) |

## AI agent usage

Set [SKILL.md](SKILL.md) as context for your AI agent to teach it how to use rawq effectively — query strategies, filtering options, and common patterns.

## License

[MIT](LICENSE)
