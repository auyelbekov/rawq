pub(crate) mod code;
pub mod queries;
mod text;

use anyhow::Result;

use crate::index::chunk::{Chunk, Language};

/// Dispatch to the appropriate chunker based on language.
/// If tree-sitter grammar is available, use code chunker. Otherwise, text chunker.
pub fn chunk_file(file: &str, source: &str, language: Language) -> Result<Vec<Chunk>> {
    if queries::get_config(&language).is_some() {
        code::chunk_code(file, source, language)
    } else {
        let is_markdown = file.ends_with(".md") || file.ends_with(".markdown");
        Ok(text::chunk_text(file, source, is_markdown, &language))
    }
}
