# rawq — Agent Usage Guide

Context retrieval engine. Semantic + lexical hybrid search over codebases. Returns ranked code chunks with file paths, line ranges, scope labels, and confidence scores.

## When to use rawq

Use rawq when you don't know where to look. Use grep/read when you already know the file or exact string.

| Situation | Tool |
|-----------|------|
| "Where is retry logic implemented?" | `rawq search` |
| "Find the function named `parse_config`" | `rawq search` or `grep` |
| "Read line 42 of src/main.rs" | `read` |
| "What does this codebase do?" | `rawq map` |
| "What changed and how does it affect X?" | `rawq diff` |

## Query style matters

rawq blends semantic (embedding) and lexical (BM25) search. **How you phrase the query changes which mode dominates.**

**Use natural language for concepts** — this is where rawq beats grep:
```bash
rawq search "how does the app handle authentication failures" .
rawq search "database connection pooling and retry logic" .
rawq search "where are environment variables validated" .
```

**Use identifiers only for exact symbol lookup:**
```bash
rawq search "fn parse_config" .
rawq search "class DatabaseClient" .
```

**Do NOT use grep-style keyword queries with rawq.** These produce worse results:
```bash
# BAD — grep-style keywords, rawq can't infer intent
rawq search "auth error" .
rawq search "db pool" .

# GOOD — natural language, rawq understands the concept
rawq search "how does authentication error handling work" .
rawq search "database connection pool management" .
```

The more descriptive your query, the better semantic search works. Single keywords trigger lexical-dominant mode which is just BM25 — no better than grep.

## Use filtering options

Agents often search the entire codebase when they already know constraints. Use filters to narrow results and improve relevance:

```bash
# Filter by language — skip irrelevant file types
rawq search "parse config" . --lang rust
rawq search "API endpoint" . --lang typescript

# Exclude patterns — skip tests, generated code, vendored deps
rawq search "database" . --exclude "*.test.*" --exclude "vendor/*"

# Force search mode when you know what you need
rawq search -e "reconnect" .          # lexical only — exact keyword match
rawq search -s "how does caching work" .   # semantic only — concept search

# Re-rank for better precision on ambiguous queries
rawq search "error handling" . --rerank

# Text weight — boost docs/comments when searching for explanations
rawq search "how to configure" . --text-weight 1.0

# Token budget — control how much context is returned
rawq search "auth" . --token-budget 2000 --json
```

## Commands

### search — find relevant code
```bash
rawq search "query" [path]                    # hybrid search (default)
rawq search "query" [path] --json             # structured JSON for parsing
rawq search "query" [path] --lang rust        # only Rust files
rawq search "query" [path] --exclude "test*"  # skip test files
rawq search "query" [path] --top 5            # limit to 5 results
rawq "query" [path]                           # shorthand (no subcommand needed)
```

Key flags:
- `--top N` — number of results (default 10)
- `--context N` — surrounding context lines (default 3)
- `--json` — structured output with all fields
- `--stream` — NDJSON streaming (one result per line)
- `--lang X` — filter by language
- `--exclude "glob"` — skip matching files
- `-e` / `-s` — force lexical / semantic mode
- `--rerank` — two-pass keyword overlap re-ranking
- `--text-weight F` — weight for text/markdown chunks (default 0.5, use 1.0 for docs)
- `--token-budget N` — max tokens in results
- `--full-file` — include full file content in results

### map — codebase structure
```bash
rawq map .                  # definitions with hierarchy
rawq map . --depth 3        # deeper nesting
rawq map . --lang rust      # only Rust files
rawq map . --exclude "test*" # skip test directories
rawq map . --json           # structured output
```
Use to orient in an unfamiliar codebase before searching. **Filter with `--lang` and `--exclude`** to avoid noise from irrelevant files.

### diff — search within changes
```bash
rawq diff "query" .                # unstaged changes
rawq diff "query" . --staged       # staged changes
rawq diff "query" . --base main    # diff vs branch
```

## JSON output format

```json
{
  "schema_version": 1,
  "model": "snowflake-arctic-embed-s",
  "results": [
    {
      "file": "src/db.rs",
      "lines": [23, 41],
      "display_start_line": 23,
      "language": "rust",
      "scope": "DatabaseClient.reconnect",
      "confidence": 0.91,
      "content": "...",
      "context_before": "...",
      "context_after": "...",
      "token_count": 45
    }
  ],
  "query_ms": 8,
  "total_tokens": 45
}
```

## Workflow

1. `rawq map .` — understand the structure
2. `rawq search "descriptive query" . --json` — find relevant code
3. Read the top results' files for full context
4. Act on what you found

rawq narrows down which files matter. Read those files, not everything.
