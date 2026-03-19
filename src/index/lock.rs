use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use fs2::FileExt;

/// Advisory file lock for index directory access.
/// Write lock for indexing, read lock for searching.
pub struct IndexLock {
    file: fs::File,
    path: PathBuf,
}

impl IndexLock {
    /// Acquire a shared (read) lock on the index directory.
    pub fn read(index_dir: &Path) -> Result<Self> {
        let path = index_dir.join("lock");
        fs::create_dir_all(index_dir)?;
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .context("open lock file")?;
        file.lock_shared().context("acquire shared lock")?;
        Ok(Self { file, path })
    }

    /// Acquire an exclusive (write) lock on the index directory.
    pub fn write(index_dir: &Path) -> Result<Self> {
        let path = index_dir.join("lock");
        fs::create_dir_all(index_dir)?;
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .context("open lock file")?;
        file.lock_exclusive().context("acquire exclusive lock")?;
        Ok(Self { file, path })
    }
}

impl Drop for IndexLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
        let _ = &self.path; // keep path alive for debugging
    }
}
