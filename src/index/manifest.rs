use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::index::walker;

/// Current index schema version. Bump when the on-disk format changes.
/// v2 → v3: tantivy content/scope fields no longer stored (index-only);
///           vectors.bin uses f16 instead of f32 (halves file size).
///      v5 — NWS-based chunking (v2): universal 1500/150 char budgets, no per-language line limits.
pub const CURRENT_SCHEMA_VERSION: u32 = 5;

/// Filename written as the last step of index build to mark a complete index.
pub const COMMIT_MARKER: &str = "_commit_ok";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version for forward-compatibility checks. Old manifests default to 0.
    #[serde(default)]
    pub schema_version: u32,
    pub model: String,
    pub next_chunk_id: u64,
    pub files: HashMap<String, FileRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub mtime_secs: u64,
    pub size: u64,
    pub chunk_ids: Vec<u64>,
    /// SHA-256 hash of file content (hex string). Added in v0.2.
    #[serde(default)]
    pub content_hash: Option<String>,
}

pub struct ManifestDiff {
    pub added: Vec<String>,
    pub changed: Vec<String>,
    pub removed: Vec<String>,
    pub unchanged: Vec<String>,
    /// Files whose mtime/size changed but content hash is the same (mtime-only update).
    pub mtime_only: Vec<String>,
}

impl ManifestDiff {
    /// Returns true if no files were added, changed, or removed.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.changed.is_empty() && self.removed.is_empty()
    }
}

impl Default for Manifest {
    fn default() -> Self {
        Self::new()
    }
}

impl Manifest {
    pub fn new() -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            model: String::new(),
            next_chunk_id: 0,
            files: HashMap::new(),
        }
    }

    /// Load manifest without schema validation.
    /// Returns `Ok(None)` if manifest.json does not exist (fresh/empty index).
    /// Returns `Err` if manifest exists but cannot be read or parsed.
    pub fn load(index_dir: &Path) -> Result<Option<Self>> {
        let path = index_dir.join("manifest.json");
        if !path.exists() {
            return Ok(None);
        }
        let data = fs::read_to_string(&path).context("read manifest.json")?;
        let manifest: Self = serde_json::from_str(&data).context("parse manifest.json")?;
        Ok(Some(manifest))
    }

    /// Load manifest with schema version validation.
    /// Returns `Err` if schema version doesn't match current (except v0 for legacy).
    pub fn load_checked(index_dir: &Path) -> Result<Option<Self>> {
        let manifest = Self::load(index_dir)?;
        if let Some(ref m) = manifest {
            if m.schema_version != CURRENT_SCHEMA_VERSION && m.schema_version != 0 {
                anyhow::bail!(
                    "Index schema v{} does not match current v{}. \
                     Rebuild with: rawq index build --reindex <path>",
                    m.schema_version,
                    CURRENT_SCHEMA_VERSION
                );
            }
        }
        Ok(manifest)
    }

    pub fn total_chunks(&self) -> usize {
        self.files.values().map(|r| r.chunk_ids.len()).sum()
    }

    pub fn save(&self, index_dir: &Path) -> Result<()> {
        fs::create_dir_all(index_dir)?;
        let path = index_dir.join("manifest.json");
        let data = serde_json::to_string_pretty(self)?;
        let file = fs::File::create(&path).context("create manifest.json")?;
        use std::io::Write;
        let mut w = std::io::BufWriter::new(&file);
        w.write_all(data.as_bytes())?;
        w.flush()?;
        file.sync_all().context("fsync manifest.json")?;
        Ok(())
    }

    /// Compare on-disk files against the manifest to find what changed.
    pub fn diff(&self, root: &Path) -> Result<ManifestDiff> {
        self.diff_with_options(root, &walker::WalkOptions::default())
    }

    /// Try to compute diff using git ls-files for faster change detection.
    /// Returns None if not a git repo or git commands fail (e.g. detached HEAD in CI).
    /// Falls back gracefully to stat-based diff via the caller.
    /// All git ls-files commands return paths relative to the root (current_dir).
    fn try_git_diff(&self, root: &Path, walk_opts: &walker::WalkOptions) -> Option<ManifestDiff> {
        use std::process::Command;

        // Helper: run a git command and return raw stdout
        let run_git_raw = |args: &[&str]| -> Option<String> {
            let output = Command::new("git")
                .args(args)
                .current_dir(root)
                .output()
                .ok()?;
            if !output.status.success() {
                return None;
            }
            Some(String::from_utf8_lossy(&output.stdout).to_string())
        };

        // 1) git ls-files — all tracked files
        let tracked_text = run_git_raw(&["ls-files"])?;
        let tracked: std::collections::HashSet<String> = tracked_text
            .lines()
            .map(|l| crate::index::normalize_path(l.trim()))
            .filter(|l| !l.is_empty())
            .collect();

        // 2) git status --porcelain — modified, deleted, untracked in one call
        let status_text = run_git_raw(&["status", "--porcelain"])?;
        let mut modified = std::collections::HashSet::new();
        let mut deleted = std::collections::HashSet::new();
        let mut untracked = std::collections::HashSet::new();
        for line in status_text.lines() {
            if line.len() < 4 {
                continue;
            }
            let status = &line[..2];
            let path = crate::index::normalize_path(line[3..].trim());
            // Handle rename entries ("R  old -> new")
            let path = if let Some(arrow) = path.find(" -> ") {
                path[arrow + 4..].to_string()
            } else {
                path
            };
            match status.trim() {
                "??" => {
                    untracked.insert(path);
                }
                "D" | " D" => {
                    deleted.insert(path);
                }
                _ => {
                    modified.insert(path);
                }
            }
        }

        let exclude_set = if walk_opts.exclude_patterns.is_empty() {
            None
        } else {
            walker::build_exclude_set(&walk_opts.exclude_patterns).ok()
        };

        // Filter helper: apply walker rules (exclude, extension, size, binary)
        // Extension check is first after exclude — it's a pure string match that
        // skips all known-binary extensions without any I/O.
        let passes_filters = |rel: &str| -> bool {
            if let Some(ref excludes) = exclude_set {
                if excludes.is_match(rel) {
                    return false;
                }
            }
            if crate::index::lang::detect_language(Path::new(rel)).is_none() {
                return false;
            }
            let abs = root.join(rel);
            if let Ok(meta) = fs::metadata(&abs) {
                if meta.len() > walk_opts.max_file_size {
                    return false;
                }
            } else {
                return false; // can't stat, skip
            }
            if walk_opts.skip_binary && crate::index::walker::is_binary(&abs) {
                return false;
            }
            true
        };

        // Build the full set of files that exist on disk
        let all_disk_files: std::collections::HashSet<String> = tracked
            .iter()
            .chain(untracked.iter())
            .filter(|f| !deleted.contains(*f))
            .filter(|f| passes_filters(f))
            .cloned()
            .collect();

        let mut added = Vec::new();
        let mut changed = Vec::new();
        let mut unchanged = Vec::new();

        for rel in &all_disk_files {
            if !self.files.contains_key(rel) {
                // New file (either untracked or tracked but not in manifest)
                added.push(rel.clone());
            } else if modified.contains(rel) || untracked.contains(rel) {
                // A file is reported as "modified" by `git status`, but it means that the file
                // has been changed since the last commit, not since the last time we build index.
                // So we still need to check the modified_time / size against the manifest to determine
                // of re-indexing is needed.
                let record = &self.files[rel];
                let abs = root.join(rel);
                // Use != to detect both newer AND older mtime.
                // because `git switch/checkout` can make file's mtime go backward.
                let needs_reindex = if let Ok(meta) = fs::metadata(&abs) {
                    let curr_mtime = meta
                        .modified()
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let curr_size = meta.len();
                    curr_mtime != record.mtime_secs || curr_size != record.size
                } else {
                    false
                };
                if needs_reindex {
                    changed.push(rel.clone());
                } else {
                    // Git says modified but mtime/size unchanged — check content hash
                    if let Some(ref old_hash) = record.content_hash {
                        if let Ok(new_hash) = compute_file_hash(&abs) {
                            if &new_hash != old_hash {
                                changed.push(rel.clone());
                            } else {
                                unchanged.push(rel.clone());
                            }
                        } else {
                            unchanged.push(rel.clone());
                        }
                    } else {
                        unchanged.push(rel.clone());
                    }
                }
            } else {
                // Git says clean — but verify mtime/size match manifest (catches branch switches)
                let record = &self.files[rel];
                let abs = root.join(rel);
                // Use != to detect both newer AND older mtime.
                // because `git switch/checkout` can make file's mtime go backward.
                let needs_reindex = if let Ok(meta) = fs::metadata(&abs) {
                    let mtime = meta
                        .modified()
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    mtime != record.mtime_secs || meta.len() != record.size
                } else {
                    false
                };
                if needs_reindex {
                    changed.push(rel.clone());
                } else {
                    unchanged.push(rel.clone());
                }
            }
        }

        // Files in manifest but not on disk are removed
        let mut removed: Vec<String> = self
            .files
            .keys()
            .filter(|k| !all_disk_files.contains(*k))
            .cloned()
            .collect();

        // Sort for deterministic output (RWQ-244)
        added.sort();
        changed.sort();
        removed.sort();
        unchanged.sort();

        Some(ManifestDiff {
            added,
            changed,
            removed,
            unchanged,
            mtime_only: Vec::new(),
        })
    }

    /// Compare on-disk files against the manifest with custom walk options.
    /// Tries git-based diff first for speed, falls back to mtime+SHA-256 walk.
    pub fn diff_with_options(
        &self,
        root: &Path,
        walk_opts: &walker::WalkOptions,
    ) -> Result<ManifestDiff> {
        // Try git-based diff first (faster for git repos)
        if let Some(diff) = self.try_git_diff(root, walk_opts) {
            return Ok(diff);
        }

        let entries = walker::walk_directory_with_options(root, walk_opts)?;
        let mut added = Vec::new();
        let mut changed = Vec::new();
        let mut unchanged = Vec::new();
        let mut mtime_only = Vec::new();

        let mut seen = HashMap::new();
        for entry in &entries {
            let rel = crate::index::normalize_path(
                &entry
                    .path
                    .strip_prefix(root)
                    .unwrap_or(&entry.path)
                    .to_string_lossy(),
            );
            seen.insert(rel.clone(), &entry.path);

            match self.files.get(&rel) {
                None => added.push(rel),
                Some(record) => {
                    // Use != to detect both newer AND older mtime.
                    // because `git switch/checkout` can make file's mtime go backward.
                    let meta = fs::metadata(&entry.path)?;
                    let mtime = meta
                        .modified()
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let size = meta.len();

                    if mtime != record.mtime_secs || size != record.size {
                        // mtime/size differ — check content hash to avoid re-chunking
                        if let Some(ref old_hash) = record.content_hash {
                            let new_hash = compute_file_hash(&entry.path)?;
                            if &new_hash == old_hash {
                                // Content unchanged, just mtime update
                                mtime_only.push(rel);
                                continue;
                            }
                        }
                        changed.push(rel);
                    } else {
                        unchanged.push(rel);
                    }
                }
            }
        }

        let mut removed: Vec<String> = self
            .files
            .keys()
            .filter(|k| !seen.contains_key(*k))
            .cloned()
            .collect();

        // Sort all lists for deterministic output across runs (RWQ-244)
        added.sort();
        changed.sort();
        removed.sort();
        unchanged.sort();
        mtime_only.sort();

        Ok(ManifestDiff {
            added,
            changed,
            removed,
            unchanged,
            mtime_only,
        })
    }
}

/// Compute SHA-256 hash of a file's contents, returning hex string.
pub fn compute_file_hash(path: &Path) -> Result<String> {
    let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let hash = Sha256::digest(&data);
    Ok(format!("{hash:x}"))
}

/// Compute the index directory for a given project root.
/// `~/.cache/rawq/<sha256(canonical_path)[:16]>/`
pub fn index_dir_for(root: &Path) -> Result<PathBuf> {
    let canonical = fs::canonicalize(root).context("canonicalize root path")?;
    let hash = Sha256::digest(canonical.to_string_lossy().as_bytes());
    let hex = format!("{hash:x}");
    let short = &hex[..16];

    let cache = dirs::cache_dir().context("could not determine cache directory")?;
    Ok(cache.join("rawq").join(short))
}

/// Verify that the index directory is not world-writable (Unix only).
/// On shared machines, a world-writable index dir could allow another user
/// to inject a malicious manifest.
#[cfg(unix)]
pub fn verify_index_dir_safety(index_dir: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(index_dir) {
        let mode = meta.permissions().mode();
        if mode & 0o002 != 0 {
            anyhow::bail!(
                "index directory {} is world-writable (mode {:o}), refusing to use",
                index_dir.display(),
                mode
            );
        }
    }
    Ok(())
}

/// No-op on Windows — POSIX permissions don't apply.
#[cfg(not(unix))]
pub fn verify_index_dir_safety(_index_dir: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trip() {
        let mut m = Manifest::new();
        m.next_chunk_id = 42;
        m.files.insert(
            "src/main.rs".to_string(),
            FileRecord {
                mtime_secs: 1000,
                size: 500,
                chunk_ids: vec![0, 1, 2],
                content_hash: Some("abc123".to_string()),
            },
        );

        let json = serde_json::to_string(&m).unwrap();
        let m2: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.next_chunk_id, 42);
        assert_eq!(m2.files["src/main.rs"].chunk_ids, vec![0, 1, 2]);
        assert_eq!(
            m2.files["src/main.rs"].content_hash.as_deref(),
            Some("abc123")
        );
    }

    #[test]
    fn manifest_backward_compat_no_hash() {
        // Old manifests without content_hash should deserialize fine
        let json = r#"{"model":"snowflake-arctic-embed-s","next_chunk_id":1,"files":{"a.rs":{"mtime_secs":100,"size":50,"chunk_ids":[0]}}}"#;
        let m: Manifest = serde_json::from_str(json).unwrap();
        assert!(m.files["a.rs"].content_hash.is_none());
    }

    #[test]
    fn manifest_diff_detects_added() {
        let m = Manifest::new();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
        let diff = m.diff(&root).unwrap();
        // Fresh manifest => all files are "added"
        assert!(!diff.added.is_empty());
        assert!(diff.changed.is_empty());
        assert!(diff.removed.is_empty());
        assert!(diff.unchanged.is_empty());
    }

    #[test]
    fn index_dir_is_deterministic() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let a = index_dir_for(root).unwrap();
        let b = index_dir_for(root).unwrap();
        assert_eq!(a, b);
        assert!(a.to_string_lossy().contains("rawq"));
    }

    #[test]
    fn manifest_diff_lists_are_sorted() {
        // RWQ-244: verify deterministic ordering
        let mut m = Manifest::new();
        m.files.insert(
            "z.rs".to_string(),
            FileRecord {
                mtime_secs: 100,
                size: 50,
                chunk_ids: vec![0],
                content_hash: None,
            },
        );
        m.files.insert(
            "a.rs".to_string(),
            FileRecord {
                mtime_secs: 100,
                size: 50,
                chunk_ids: vec![1],
                content_hash: None,
            },
        );
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
        let diff = m.diff(&root).unwrap();
        // removed should be sorted alphabetically
        if diff.removed.len() >= 2 {
            for w in diff.removed.windows(2) {
                assert!(w[0] <= w[1], "removed list not sorted: {:?}", diff.removed);
            }
        }
        // added should also be sorted
        if diff.added.len() >= 2 {
            for w in diff.added.windows(2) {
                assert!(w[0] <= w[1], "added list not sorted: {:?}", diff.added);
            }
        }
    }

    #[test]
    fn load_checked_rejects_wrong_schema() {
        // RWQ-189: verify schema enforcement
        let tmp = std::env::temp_dir().join("rawq_test_schema_check");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let mut m = Manifest::new();
        m.schema_version = 99; // future version
        m.save(&tmp).unwrap();

        let result = Manifest::load_checked(&tmp);
        assert!(result.is_err(), "Should reject schema v99");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("schema"), "Error should mention schema: {err}");

        let _ = fs::remove_dir_all(&tmp);
    }
}
