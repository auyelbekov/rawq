pub mod chunk;
pub(crate) mod chunker;
pub mod lang;
pub mod lock;
pub mod manifest;
pub mod map;
pub mod pipeline;
pub mod store;
pub mod walker;

use std::path::Path;

use anyhow::Result;

pub use chunk::{Chunk, Language};
pub use manifest::{index_dir_for, Manifest, ManifestDiff, COMMIT_MARKER, CURRENT_SCHEMA_VERSION};
pub use pipeline::{build_index, build_index_force, build_index_with_options, build_index_with_embedder, build_index_with_embed_fn, EmbedFn, IndexOptions, IndexStats};
pub use store::{IndexReader, StoredChunk, build_code_analyzer, build_fuzzy_tantivy_query, register_code_tokenizer, split_camel_case, CODE_TOKENIZER_NAME};

/// Normalize a file path to use forward slashes consistently.
/// Call this at every point a path is stored, compared, or used as a HashMap key.
#[inline]
pub fn normalize_path(p: &str) -> String {
    p.replace('\\', "/")
}

/// Chunk in-memory content without walking a directory.
/// Convenience function for stdin/pipe modes.
/// Strips CRLF line endings for Windows compatibility.
pub fn chunk_content(filename: &str, source: &str, language: Language) -> Result<Vec<Chunk>> {
    let clean = source.replace('\r', "");
    chunker::chunk_file(filename, &clean, language)
}

pub fn walk_and_chunk(root: &Path) -> Result<Vec<Chunk>> {
    let entries = walker::walk_directory(root)?;
    let mut all_chunks = Vec::new();

    for entry in entries {
        let rel_path = normalize_path(
            &entry.path.strip_prefix(root).unwrap_or(&entry.path).to_string_lossy(),
        );

        let source = match std::fs::read_to_string(&entry.path) {
            Ok(s) => s.replace('\r', ""),
            Err(e) => {
                eprintln!("warning: skipping {rel_path}: {e}");
                continue;
            }
        };
        let chunks = match chunker::chunk_file(&rel_path, &source, entry.language) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("warning: failed to chunk {rel_path}: {e}");
                continue;
            }
        };
        all_chunks.extend(chunks);
    }

    Ok(all_chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    // RWQ-240: verify CRLF is stripped from chunk content
    #[test]
    fn test_crlf_stripped_from_chunks() {
        let source_crlf = "fn foo() {\r\n    println!(\"hello\");\r\n}\r\n";
        let chunks = chunk_content("test.rs", source_crlf, Language::Rust).unwrap();
        for chunk in &chunks {
            assert!(
                !chunk.content.contains('\r'),
                "Chunk content should not contain \\r: {:?}",
                &chunk.content[..chunk.content.len().min(50)]
            );
        }
    }
}
