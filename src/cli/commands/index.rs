use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use crate::cli::util::{build_index_and_ensure_daemon, count_dir_contents, effective_model};

pub fn cmd_index(
    path: PathBuf,
    reindex: bool,
    model: Option<String>,
    batch_size: Option<usize>,
    json: bool,
    exclude: Vec<String>,
) -> Result<()> {
    let start = Instant::now();
    let opts = crate::index::IndexOptions {
        model: effective_model(&model),
        batch_size,
        exclude_patterns: exclude,
    };
    let stats = build_index_and_ensure_daemon(&path, reindex, &opts, false)?;
    let elapsed = start.elapsed().as_millis();

    if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        eprintln!(
            "Indexed {} files ({} chunks) in {elapsed}ms",
            stats.total_files, stats.total_chunks
        );
        eprintln!(
            "  added: {}, changed: {}, removed: {}",
            stats.added_files, stats.changed_files, stats.removed_files
        );
    }

    Ok(())
}

pub fn cmd_status(path: PathBuf, json: bool) -> Result<()> {
    let root = std::fs::canonicalize(&path)?;
    let index_dir = crate::index::index_dir_for(&root)?;

    if !index_dir.join("manifest.json").exists() {
        if json {
            println!("{}", serde_json::json!({"indexed": false}));
        } else {
            eprintln!("No index found for {}", root.display());
            eprintln!("Run: rawq index build {}", path.display());
        }
        return Ok(());
    }

    let manifest = crate::index::Manifest::load(&index_dir)?
        .ok_or_else(|| anyhow::anyhow!("manifest missing"))?;

    let total_files = manifest.files.len();
    let total_chunks = manifest.total_chunks();

    // Count languages by detecting from file extensions
    let mut lang_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for file in manifest.files.keys() {
        let lang = crate::index::lang::detect_language(std::path::Path::new(file))
            .map(|l| format!("{l:?}").to_lowercase())
            .unwrap_or_else(|| "unknown".to_string());
        *lang_counts.entry(lang).or_default() += 1;
    }

    if json {
        let mut langs: Vec<_> = lang_counts.into_iter().collect();
        langs.sort_by(|a, b| b.1.cmp(&a.1));
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "indexed": true,
            "index_dir": index_dir.display().to_string(),
            "model": manifest.model,
            "files": total_files,
            "chunks": total_chunks,
            "languages": langs.into_iter().collect::<std::collections::HashMap<_, _>>(),
        }))?);
    } else {
        let mut langs: Vec<_> = lang_counts.into_iter().collect();
        langs.sort_by(|a, b| b.1.cmp(&a.1));
        let lang_str: Vec<String> = langs.iter().map(|(l, c)| format!("{l} ({c})")).collect();

        println!("Index:     {}", index_dir.display());
        println!("Model:     {}", manifest.model);
        println!("Files:     {total_files}");
        println!("Chunks:    {total_chunks}");
        println!("Languages: {}", lang_str.join(", "));
    }

    Ok(())
}

pub fn cmd_unindex(path: PathBuf, all: bool) -> Result<()> {
    if all {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("cannot determine cache directory"))?
            .join("rawq");
        if !cache_dir.exists() {
            eprintln!("No cached indexes found.");
            return Ok(());
        }
        // Remove index directories but preserve the models/ directory
        let mut files = 0usize;
        let mut bytes = 0u64;
        for entry in std::fs::read_dir(&cache_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            if name == "models" {
                continue;
            }
            if entry.file_type()?.is_dir() {
                let (f, b) = count_dir_contents(&entry.path())?;
                files += f;
                bytes += b;
                std::fs::remove_dir_all(entry.path())?;
            }
        }
        if files == 0 {
            eprintln!("No cached indexes found.");
        } else {
            eprintln!(
                "Removed all cached indexes ({} files, {:.1} KB freed)",
                files,
                bytes as f64 / 1024.0,
            );
        }
    } else {
        let root = std::fs::canonicalize(&path)?;
        let index_dir = crate::index::index_dir_for(&root)?;
        if !index_dir.exists() {
            eprintln!("No index found for {}", root.display());
            return Ok(());
        }
        // Acquire write lock before deletion
        let _lock = crate::index::lock::IndexLock::write(&index_dir)?;
        let (files, bytes) = count_dir_contents(&index_dir)?;
        // Remove contents except lock file, then drop lock and remove dir
        for entry in std::fs::read_dir(&index_dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.file_name() == Some(std::ffi::OsStr::new("lock")) {
                continue;
            }
            if p.is_dir() {
                std::fs::remove_dir_all(&p)?;
            } else {
                std::fs::remove_file(&p)?;
            }
        }
        drop(_lock);
        let _ = std::fs::remove_dir_all(&index_dir);
        eprintln!(
            "Removed index for {} ({} files, {:.1} KB freed)",
            path.display(),
            files,
            bytes as f64 / 1024.0,
        );
    }
    Ok(())
}
