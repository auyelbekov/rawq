use std::path::{Path, PathBuf};

use anyhow::Result;
use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::WalkBuilder;

use crate::index::chunk::Language;
use crate::index::lang::detect_language;

/// Default maximum file size (1 MB).
const DEFAULT_MAX_FILE_SIZE: u64 = 1_048_576;

/// Number of bytes to read for binary detection.
const BINARY_CHECK_SIZE: usize = 8192;

#[derive(Debug)]
pub struct FileEntry {
    pub path: PathBuf,
    pub language: Language,
}

/// Options for the file walker.
pub struct WalkOptions {
    /// Maximum file size in bytes. Files larger than this are skipped.
    pub max_file_size: u64,
    /// Whether to skip binary files.
    pub skip_binary: bool,
    /// Glob patterns to exclude from walking.
    pub exclude_patterns: Vec<String>,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            max_file_size: DEFAULT_MAX_FILE_SIZE,
            skip_binary: true,
            exclude_patterns: Vec::new(),
        }
    }
}

/// Build a GlobSet from a list of pattern strings.
pub fn build_exclude_set(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pat in patterns {
        builder.add(Glob::new(pat)?);
    }
    Ok(builder.build()?)
}

pub fn walk_directory(root: &Path) -> Result<Vec<FileEntry>> {
    walk_directory_with_options(root, &WalkOptions::default())
}

pub fn walk_directory_with_options(root: &Path, opts: &WalkOptions) -> Result<Vec<FileEntry>> {
    let mut entries = Vec::new();

    let exclude_set = if opts.exclude_patterns.is_empty() {
        None
    } else {
        Some(build_exclude_set(&opts.exclude_patterns)?)
    };

    for result in WalkBuilder::new(root).build() {
        let entry = result?;
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        // Check exclude patterns against relative path
        if let Some(ref excludes) = exclude_set {
            let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
            let rel_str = crate::index::normalize_path(&rel.to_string_lossy());
            if excludes.is_match(&rel_str) {
                continue;
            }
        }

        // Check file size
        if let Ok(meta) = entry.metadata() {
            if meta.len() > opts.max_file_size {
                eprintln!(
                    "warning: skipping {} ({}KB > {}KB limit)",
                    entry.path().display(),
                    meta.len() / 1024,
                    opts.max_file_size / 1024,
                );
                continue;
            }
        }

        // Check for binary content
        if opts.skip_binary && is_binary(entry.path()) {
            continue;
        }

        if let Some(lang) = detect_language(entry.path()) {
            entries.push(FileEntry {
                path: entry.into_path(),
                language: lang,
            });
        }
    }

    Ok(entries)
}

/// Detect if a file is binary by checking the first 8KB for NULL bytes.
pub fn is_binary(path: &Path) -> bool {
    use std::io::Read;
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return true,
    };
    let mut reader = std::io::BufReader::new(file);
    let mut buf = vec![0u8; BINARY_CHECK_SIZE];
    let n = match reader.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return true,
    };
    buf[..n].contains(&0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_testdata() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata");
        let entries = walk_directory(&root).unwrap();
        assert!(
            entries.len() >= 4,
            "expected at least 4 files, got {}",
            entries.len()
        );

        let extensions: Vec<&str> = entries
            .iter()
            .filter_map(|e| e.path.extension()?.to_str())
            .collect();
        assert!(extensions.contains(&"rs"));
        assert!(extensions.contains(&"py"));
        assert!(extensions.contains(&"ts"));
        assert!(extensions.contains(&"md"));
    }

    #[test]
    fn test_size_limit_skips_large() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata");
        let opts = WalkOptions {
            max_file_size: 1, // 1 byte — should skip everything
            skip_binary: true,
            exclude_patterns: Vec::new(),
        };
        let entries = walk_directory_with_options(&root, &opts).unwrap();
        assert!(entries.is_empty(), "all files should be skipped");
    }

    #[test]
    fn test_is_binary_on_text() {
        // A known text file should not be detected as binary
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join("readme.md");
        assert!(!is_binary(&root));
    }
}
