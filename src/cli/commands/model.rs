use std::time::Instant;

use anyhow::Result;

use crate::cli::args::ModelCommand;

pub fn cmd_model(command: ModelCommand) -> Result<()> {
    match command {
        ModelCommand::Download { repo_id, name } => cmd_model_download(&repo_id, name.as_deref()),
        ModelCommand::List { json } => cmd_model_list(json),
        ModelCommand::Remove { name, all } => cmd_model_remove(name.as_deref(), all),
        ModelCommand::Default { name } => cmd_model_default(name.as_deref()),
    }
}

fn cmd_model_download(repo_id: &str, name: Option<&str>) -> Result<()> {
    let had_default = crate::embed::config::get_default_model().is_some();
    let start = Instant::now();
    eprintln!("Downloading model from {}...", repo_id);

    let (dir, config) = crate::embed::config::download_model(repo_id, name)?;

    let elapsed = start.elapsed().as_secs_f64();
    let size = crate::embed::config::disk_size(&dir).unwrap_or(0);
    eprintln!(
        "Downloaded '{}' ({} dim, {:.1} MB) in {:.1}s",
        config.name,
        config.embed_dim,
        size as f64 / (1024.0 * 1024.0),
        elapsed
    );
    if !had_default {
        eprintln!("Set as default model");
    }
    eprintln!("Location: {}", dir.display());

    Ok(())
}

fn cmd_model_list(json: bool) -> Result<()> {
    let models = crate::embed::config::list_installed_models()?;
    let default_name = crate::embed::config::get_default_model().unwrap_or_default();

    if json {
        let entries: Vec<serde_json::Value> = models
            .iter()
            .map(|(path, config)| {
                let size = crate::embed::config::disk_size(path).unwrap_or(0);
                serde_json::json!({
                    "name": config.name,
                    "embed_dim": config.embed_dim,
                    "disk_size_bytes": size,
                    "default": config.name == default_name,
                    "path": path.display().to_string(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&entries)?);
    } else {
        if models.is_empty() {
            eprintln!("No models installed.");
            eprintln!();
            eprintln!("Recommended models:");
            for m in crate::embed::config::recommended_models() {
                eprintln!(
                    "  {:<48} {:>3} dim, {:>5} seq  ({})",
                    m.repo_id, m.embed_dim, m.max_seq_len, m.description
                );
            }
            eprintln!();
            eprintln!("Install with: rawq model download <repo>");
            return Ok(());
        }
        for (path, config) in &models {
            let size = crate::embed::config::disk_size(path).unwrap_or(0);
            let size_mb = size as f64 / (1024.0 * 1024.0);
            let default_marker = if config.name == default_name { "  [default]" } else { "" };
            println!(
                "{}  dim={}  {:.1} MB{}",
                config.name, config.embed_dim, size_mb, default_marker
            );
        }
    }

    Ok(())
}

fn cmd_model_remove(name: Option<&str>, all: bool) -> Result<()> {
    if all {
        let (count, bytes) = crate::embed::config::remove_all_models()?;
        if count > 0 {
            let _ = crate::embed::config::clear_default_model();
        }
        if count == 0 {
            eprintln!("No models installed.");
        } else {
            eprintln!(
                "Removed {} model(s), {:.1} MB freed",
                count,
                bytes as f64 / (1024.0 * 1024.0)
            );
        }
        return Ok(());
    }

    let name = name.ok_or_else(|| anyhow::anyhow!("specify a model name or use --all"))?;

    // Warn about indexes built with this model
    if let Some(cache_dir) = dirs::cache_dir() {
        let rawq_cache = cache_dir.join("rawq");
        if rawq_cache.exists() {
            let mut index_count = 0usize;
            if let Ok(entries) = std::fs::read_dir(&rawq_cache) {
                for entry in entries.flatten() {
                    let entry_name = entry.file_name();
                    if entry_name == "models" {
                        continue;
                    }
                    let manifest_path = entry.path().join("manifest.json");
                    if manifest_path.exists() {
                        if let Ok(data) = std::fs::read_to_string(&manifest_path) {
                            if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&data) {
                                if manifest.get("model").and_then(|v| v.as_str()) == Some(name) {
                                    index_count += 1;
                                }
                            }
                        }
                    }
                }
            }
            if index_count > 0 {
                eprintln!(
                    "warning: {} index(es) were built with model '{}' and may need rebuilding",
                    index_count, name
                );
            }
        }
    }

    // Check if removing the default model
    let is_default = crate::embed::config::get_default_model()
        .as_deref()
        == Some(name);

    let bytes = crate::embed::config::remove_model(name)?;

    if is_default {
        let _ = crate::embed::config::clear_default_model();
    }

    eprintln!(
        "Removed model '{}', {:.1} MB freed",
        name,
        bytes as f64 / (1024.0 * 1024.0)
    );

    // If exactly one model remains after deletion, auto-promote it to default
    if is_default {
        if let Ok(remaining) = crate::embed::config::list_installed_models() {
            if remaining.len() == 1 {
                let sole = &remaining[0].1.name;
                let _ = crate::embed::config::set_default_model(sole);
                eprintln!("'{}' is now the default model", sole);
            } else if !remaining.is_empty() {
                eprintln!("No default model set. Pick one with: rawq model default <name>");
            }
        }
    }

    Ok(())
}

fn cmd_model_default(name: Option<&str>) -> Result<()> {
    match name {
        Some(name) => {
            crate::embed::config::set_default_model(name)?;
            eprintln!("Default model set to '{name}'");
        }
        None => match crate::embed::config::get_default_model() {
            Some(name) => println!("{name}"),
            None => {
                let models = crate::embed::config::list_installed_models()?;
                if models.len() == 1 {
                    println!("{} (only installed model)", models[0].1.name);
                } else if models.is_empty() {
                    eprintln!("No models installed.");
                    eprintln!("Install with: rawq model download <repo>");
                } else {
                    eprintln!("No default model set. Installed models:");
                    for (_, config) in &models {
                        eprintln!("  {}", config.name);
                    }
                    eprintln!("\nSet with: rawq model default <name>");
                }
            }
        },
    }
    Ok(())
}
