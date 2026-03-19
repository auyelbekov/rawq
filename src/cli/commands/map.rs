use anyhow::Result;
use colored::Colorize;
use serde::Serialize;

use crate::cli::args::MapArgs;

#[derive(Serialize)]
struct FileMap {
    file: String,
    language: String,
    symbols: Vec<crate::index::map::SymbolEntry>,
}

pub fn cmd_map(args: MapArgs) -> Result<()> {
    let abs_path = std::fs::canonicalize(&args.path)?;

    let walk_opts = crate::index::walker::WalkOptions {
        exclude_patterns: args.exclude.clone(),
        ..Default::default()
    };
    let entries = crate::index::walker::walk_directory_with_options(&abs_path, &walk_opts)?;

    let mut file_maps: Vec<FileMap> = Vec::new();

    for entry in &entries {
        let rel_path = crate::index::normalize_path(
            &entry
                .path
                .strip_prefix(&abs_path)
                .unwrap_or(&entry.path)
                .to_string_lossy(),
        );

        let lang_str = entry.language.to_string();

        if let Some(ref lang) = args.lang {
            if !lang_str.eq_ignore_ascii_case(lang) {
                continue;
            }
        }

        let source = match std::fs::read_to_string(&entry.path) {
            Ok(s) => s.replace('\r', ""),
            Err(_) => continue,
        };

        let symbols =
            crate::index::map::extract_symbols(&source, &entry.language, args.depth).unwrap_or_default();

        if !symbols.is_empty() {
            file_maps.push(FileMap {
                file: rel_path,
                language: lang_str,
                symbols,
            });
        }
    }

    let omitted = if args.lang.is_none() {
        entries.len().saturating_sub(file_maps.len())
    } else {
        0
    };

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "files": file_maps,
                "omitted_files": omitted,
            }))?
        );
    } else {
        for fm in &file_maps {
            let lang_display = {
                let mut chars = fm.language.chars();
                match chars.next() {
                    Some(c) => c.to_uppercase().to_string() + chars.as_str(),
                    None => String::new(),
                }
            };
            println!("{} ({})", fm.file.green().bold(), lang_display);
            print_symbols(&fm.symbols, 1);
        }

        let total_files = file_maps.len();
        let total_symbols: usize = file_maps.iter().map(|f| count_symbols(&f.symbols)).sum();
        let omission_note = if omitted > 0 {
            format!(" ({omitted} text/unsupported file(s) not shown)")
        } else {
            String::new()
        };
        eprintln!("{total_files} files, {total_symbols} symbols{omission_note}");
    }

    Ok(())
}

fn print_symbols(symbols: &[crate::index::map::SymbolEntry], depth: usize) {
    let indent = "  ".repeat(depth);
    for sym in symbols {
        let lines = if sym.line == sym.end_line {
            format!("{}", sym.line)
        } else {
            format!("{}-{}", sym.line, sym.end_line)
        };
        println!(
            "{}{} {} ({})",
            indent,
            sym.kind.dimmed(),
            sym.name,
            lines,
        );
        if !sym.children.is_empty() {
            print_symbols(&sym.children, depth + 1);
        }
    }
}

fn count_symbols(symbols: &[crate::index::map::SymbolEntry]) -> usize {
    symbols
        .iter()
        .map(|s| 1 + count_symbols(&s.children))
        .sum()
}
