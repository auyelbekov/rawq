# CLAUDE.md

## Project

rawq is a CLI context retrieval engine for AI agents. Semantic + lexical search over codebases, single Rust binary, fully offline, built for AI agents.

## Structure

Single crate. All modules under `src/`:

```
src/
  embed/     # ort ONNX session, tokenizer, embed(), model management + download (hf-hub), GPU detection
  index/     # file walking, chunking, embedding, vector+tantivy storage, indexing pipeline
  daemon/    # daemon client/server, NDJSON protocol, embedding server
  search/    # search/RRF logic, ModelSource coordination
  cli/       # clap CLI, output formatting, subcommands, daemon process spawning
testdata/    # sample files for integration tests (.rs, .py, .ts, .md, .go, .java)
tests/       # integration tests (embed, search)
scripts/     # install scripts (install.sh, install.ps1)
```

Module dependency: cli → search → {index, daemon} → embed.

## Build & Test

```bash
cargo check
cargo test
cargo clippy -- -D warnings   # must pass with zero warnings
cargo run -- chunk ./testdata  # NDJSON output to stdout, summary to stderr
cargo run -- embed "hello world"
cargo run -- index ./testdata  # builds index at ~/.cache/rawq/<hash>/
```

## Key Constraints

- **Windows**: Primary dev platform. MSVC toolchain. `ort` must use `download-binaries` feature (static linking). **Never use `load-dynamic`** — it picks up old System32 onnxruntime.dll and hangs.
- **Memory pattern disabled**: `with_memory_pattern(false)` required for GPU EPs — the memory planner conflicts with device memory management and causes hangs on repeated `session.run()` calls.
- **ort-managed tensors**: Use `Tensor::new()` + `extract_tensor_mut()` instead of `Tensor::from_array()`. GPU EPs may hold internal references to input memory between runs.
- **No `fastembed-rs`**: Raw `ort` + `tokenizers` gives full control (~150 lines).
- **rayon**: Used for parallel chunking in indexing pipeline.
- **tree-sitter versions**: Don't pin `tree-sitter` in deps. Let Cargo resolve from grammar crate requirements.
- **ndarray**: Must be 0.17 to match ort's version.
- **No `usearch`**: C++ headers fail on MSVC (MAP_FAILED is POSIX-only). Using flat brute-force vector store instead. Fine for <50k chunks; swap in HNSW later if needed.

## Conventions

- `cargo clippy -- -D warnings` must pass with zero warnings before committing.
- `cargo test` must pass before committing.
- Do not add Co-Authored-By lines in commits.
- Small atomic commits per working feature.
- Chunk type (`Chunk`, `Language`) lives in `index::chunk`.

## Current State (v0.1.0)

CLI subcommands: `search`, `index` (build/status/remove), `diff`, `map`, `watch`, `model` (download/list/remove/default), `daemon` (start/stop/status), `embed`, `chunk`, `completions`. Default fallback: `rawq "query" ./path` works without `search` subcommand. Commands ordered by importance: core workflow → infrastructure → debug/dev. Auto-downloads default embedding model (snowflake-arctic-embed-s) on first use; `RAWQ_OFFLINE=1` skips network calls.

CLI source is modular: `src/cli/commands/` (one file per subcommand), `args.rs` (clap types), `output.rs`, `util.rs`. Daemon server/client/protocol in `src/daemon/`.

Supported languages (tree-sitter): Rust, Python, TypeScript, JavaScript, Go, Java, C, C++, C#, Ruby, PHP, Swift, Bash, Lua, Scala, Dart. All other text files automatically indexed with `Language::Other(extension)` — `.sql`, `.yaml`, `.kt`, `.proto`, `.tf`, `.graphql`, etc. get their real extension as the language label in JSON output. Binary extensions (png, exe, pdf, wasm, etc.) are skipped.

Chunking uses NWS-based v2 algorithm: counts non-whitespace characters instead of lines. Universal budget: `MAX_NWS = 1500` (recurse threshold), `MIN_OWN_NWS = 150` (own-chunk threshold). Decision tree: `>1500` NWS → recurse into children; `150–1500` → own chunk; `<150` → buffer. Buffer flushes before any non-buffer node. Each recursion level has its own independent buffer. No per-language line limits — NWS naturally normalizes for language verbosity. Small named functions (<150 NWS) get buffered together instead of individual chunks. `count_nws()` measures node byte range only (excludes sibling doc comments). `emit_node_with_comments()` attaches doc comments after the sizing decision. `queries.rs` only maps Language → tree-sitter grammar; `get_config()` returns `Option<LanguageConfig>`. Context prefix (`// File: ... // Language: ... // Scope: ...`) prepended to chunk text before embedding — stored content stays clean. Adjacent doc comments attached via `take_adjacent_comments()` which walks the buffer backwards checking `kind().contains("comment")` — works across all 16 tree-sitter languages with no hardcoded lists. `last_content_row()` handles tree-sitter's exclusive end position. Comments separated by blank lines are not attached. Text chunker has the same 1500 NWS budget for long paragraphs.

Search uses adaptive RRF weighting: single-word identifiers and code-keyword queries (`fn main`, `class Foo`, `def parse`) favor BM25 (0.3 semantic), multi-word NL queries favor semantic (0.7). `--rrf-weight` overrides when set explicitly. `--rerank` enables a two-pass keyword overlap heuristic (skips for single-term queries). `--text-weight` controls text chunk penalty (default 0.5, set 1.0 for doc-heavy repos). Results are always sorted by confidence descending. Search auto-detects stale indexes and incrementally reindexes before querying via `check_index_freshness()` → `manifest.diff_with_options()`. `--no-reindex` flag to skip.

`SearchOptions` is `#[non_exhaustive]` — construct via `SearchOptions::default()` then set fields. `SearchOutput` includes `schema_version: u32` (currently 1) for agent schema detection. `EmbedError` typed via thiserror in `embed::error`; other modules use `anyhow::Result` only.

Search results include 3 context lines above/below by default (`--context 3`). JSON output separates context into `context_before` and `context_after` fields. Terminal output uses `┊` gutter to visually distinguish context lines from the matched chunk.

`search()` and `search_with_reader()` share a single `search_inner()` implementation — no code duplication. Threshold check happens before file I/O (enrichment). CamelCase token splitting filters single-char noise tokens. `normalize_path()` centralizes all `\r`-to-`/` path normalization.

`--exclude` glob patterns are applied in search via a pre-built `GlobSet` (compiled once per search, not per result). Source files are stripped of `\r` on read so stored content and JSON output are never CRLF-polluted. `chunk_content()` also strips CRLF for stdin mode.

Daemon mode (`src/daemon/`): auto-starts on first semantic/hybrid search or embed, holds ONNX model hot via IPC (named pipes on Windows, Unix sockets). Eliminates cold model reload on every operation. All model-using commands (search, embed, diff) benefit via `ModelSource` in `search::model_source`. Protocol v2: Embed, Status, Shutdown. `--no-daemon` or `RAWQ_NO_DAEMON=1` to skip. `connect_or_start()` takes spawner callback. Exponential backoff on connection retry (10ms → 500ms). Max request size 10MB. Unix startup checks PID liveness before removing stale socket. `DaemonClient::embed_batch()` uses per-item IPC calls for small incremental reindexes (<50 changed files). `build_index_and_ensure_daemon()` pre-checks the manifest diff and skips daemon for large builds or force-reindex, falling back to local pipeline with proper batch sizing and OOM recovery.

Watch mode loads the embedder once and reuses it across all re-index cycles via `build_index_with_embedder()` — no cold model load per cycle. `--no-daemon` flag available.

Index format is schema v5 with `_commit_ok` marker for crash safety. `Manifest::load_checked()` rejects mismatched schemas with a rebuild prompt. All index files (vectors.bin, manifest.json, chunks.jsonl) are fsynced before the commit marker. Index directory ownership verified on Unix (world-writable rejected). Orphaned vectors from crashed builds are pruned on next incremental index. `ManifestDiff` has `is_empty()` method; lists are sorted for deterministic output. `Manifest::total_chunks()` method replaces repeated sum expressions. `Manifest::new()` uses empty string instead of defaulting to "snowflake-arctic-embed-s" — fixes false "Model changed" on fresh builds. `index remove --all` prints "No cached indexes found." when empty. `StoredChunk.language` and `ChunkRecord.language` use `Language` enum (not String) — serde-compatible with existing chunks.jsonl.

Search module is a pure library — no auto-index side effects. `search_with_reader()` accepts a pre-opened `IndexReader` for multi-mode batch searches. `ModelSource` enum + `resolve_model_source()` centralize daemon-vs-local model coordination. Batch size computation uses checked arithmetic (no overflow panic). OOM recovery detects Windows DirectML errors, quarters batch size, and bails on device-lost.

Model catalog: single `MODEL_CATALOG` array in `src/embed/config.rs` is the source of truth for all model metadata (name, repo, dim, seq_len, query_prefix, onnx_file). `recommended_models()`, `known_model_config()`, and `known_repo_for_name()` all read from this array. Add new models by appending to the catalog.

GPU auto-detection: DXGI (DirectML), nvidia-smi (CUDA), sysctl (CoreML) enumerate adapters and pick the GPU with the most dedicated VRAM. Batch size formula: `per_item = dim * seq * 4 * activation_factor` where `activation_factor = (seq/6).max(4)` accounts for transformer attention memory. VRAM budget = 75% of detected VRAM. No hardcoded caps. `RAWQ_DML_DEVICE`, `RAWQ_CUDA_DEVICE`, `RAWQ_VRAM_BUDGET` override. OOM recovery quarters batch size, detects device-lost (887A0005/887A0006). `uses_gpu()` means "GPU EP was registered"; ort may silently fall back to CPU. Release CI builds with platform GPU features (directml/cuda/coreml). Memory pattern optimization disabled (`with_memory_pattern(false)`) — required for GPU EPs with dynamic batch sizes. Input tensors use ort-managed memory (`Tensor::new` + `extract_tensor_mut`) instead of wrapping caller-owned ndarrays. Sequence length always padded to `max_seq_len` for consistent tensor shapes. Batch size logged to stderr at embedding start.

Terminal output: `bat` (if on PATH) provides per-chunk syntax highlighting (including Kotlin, SQL, YAML, TOML, JSON); `less`/`bat` pager when stdout is TTY. `RAWQ_NO_BAT=1` and `RAWQ_NO_PAGER=1` disable these.

```bash
cargo run -- search "function" ./testdata            # hybrid search (default, adaptive RRF)
cargo run -- search -s "function" ./testdata          # semantic-only
cargo run -- search -e "def" ./testdata               # lexical BM25
cargo run -- search "function" ./testdata --json      # JSON output
cargo run -- search "function" ./testdata --stream    # NDJSON streaming output
cargo run -- search "function" ./testdata --rerank    # keyword overlap re-ranking
cargo run -- "function" ./testdata                    # default fallback
cargo run -- search "function" ./testdata --context 3
cargo run -- search "function" ./testdata --full-file
cargo run -- search "function" ./testdata --text-weight 1.0  # equal weight for text chunks
cargo run -- search "function" ./testdata --no-daemon # skip daemon, embed locally
echo "fn foo() {}" | cargo run -- search "foo" --stdin --lang-hint rust  # stdin pipe
cargo run -- diff "query" ./testdata                  # search within git diff scope
cargo run -- diff "query" ./testdata --staged         # include staged changes
cargo run -- diff "query" ./testdata --base main      # diff vs branch
cargo run -- index status ./testdata                  # index stats
cargo run -- index build --reindex ./testdata         # force full re-index
cargo run -- index remove ./testdata                  # delete index for path
cargo run -- index remove --all                       # delete all cached indexes
cargo run -- map ./testdata                           # show codebase structure
cargo run -- map ./testdata --json                    # JSON map output
cargo run -- search "function" ./testdata --no-reindex  # skip auto-reindex
cargo run -- watch ./testdata                         # auto-re-index on changes
cargo run -- watch ./testdata --interval 5            # poll every 5 seconds
cargo run -- watch ./testdata --no-daemon             # skip daemon in watch
cargo run -- daemon start                             # start daemon manually
cargo run -- daemon status                            # check daemon status
cargo run -- daemon stop                              # stop daemon
```

`rawq map` walks the tree-sitter AST directly (not chunks). Shows definitions with real hierarchy (impl > methods). Kind labels extracted programmatically from the first anonymous child token — actual language keywords (`fn`, `def`, `func`, `class`). Skips identifier/expression/parameter kinds universally. Default depth 2, `--depth` flag to control. Non-tree-sitter files omitted (no symbols to show). `index::map::extract_symbols()` is the public API.

Model catalog: `MODEL_CATALOG` includes snowflake-arctic-embed-s, snowflake-arctic-embed-m-v1.5 (recommended), and jina-embeddings-v2-base-code. ONNX download tries `model_quantized.onnx`, `model_int8.onnx`, `model.onnx` in priority order.

61 tests (35 unit + 19 search integration + 7 embed integration). Zero clippy warnings.
