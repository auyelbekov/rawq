use std::time::Instant;

use anyhow::Result;

use crate::cli::args::WatchArgs;
use crate::cli::util::{effective_model, ensure_daemon_running, resolve_model_name};

pub fn cmd_watch(args: WatchArgs) -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let interval = std::time::Duration::from_secs(args.interval.max(1));
    let effective = effective_model(&args.model);

    // Ensure initial index exists (bulk local load — appropriate for first build)
    let opts = crate::index::IndexOptions {
        model: effective.clone(),
        batch_size: None,
        exclude_patterns: args.exclude.clone(),
    };
    eprintln!("Building initial index...");
    crate::index::build_index_with_options(&args.path, false, &opts)?;

    // Spawn daemon for other commands' benefit (search, embed, diff)
    let model_name = resolve_model_name(&effective);
    ensure_daemon_running(&model_name, args.no_daemon);

    // Load hot embedder once — reused across all re-index cycles (RWQ-259).
    // This eliminates the ~2s cold ONNX model load that previously happened
    // on every change detection cycle.
    let mut embedder = crate::embed::Embedder::from_managed(effective.as_deref())?;

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    if let Err(e) = ctrlc::try_set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }) {
        eprintln!("warning: could not set Ctrl+C handler: {e}");
    }

    eprintln!(
        "Watching {} (poll every {}s, Ctrl+C to stop)",
        args.path.display(),
        args.interval.max(1)
    );

    while running.load(Ordering::SeqCst) {
        std::thread::sleep(interval);
        if !running.load(Ordering::SeqCst) {
            break;
        }

        let root = match std::fs::canonicalize(&args.path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("warning: {e:#}");
                continue;
            }
        };
        let index_dir = match crate::index::index_dir_for(&root) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("warning: {e:#}");
                continue;
            }
        };

        let walk_opts = crate::index::walker::WalkOptions {
            exclude_patterns: args.exclude.clone(),
            ..Default::default()
        };
        let old_manifest = crate::index::Manifest::load(&index_dir)
            .ok()
            .flatten()
            .unwrap_or_default();

        let diff = match old_manifest.diff_with_options(&root, &walk_opts) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("warning: {e:#}");
                continue;
            }
        };

        if diff.is_empty() {
            continue;
        }

        let start = Instant::now();
        match crate::index::build_index_with_embedder(&args.path, false, &opts, &mut embedder) {
            Ok(stats) => {
                let ms = start.elapsed().as_millis();
                eprintln!(
                    "Re-indexed: +{} ~{} -{} files ({}ms)",
                    stats.added_files, stats.changed_files, stats.removed_files, ms
                );
            }
            Err(e) => {
                eprintln!("warning: re-index failed: {e:#}");
            }
        }
    }

    eprintln!("Watch stopped.");
    Ok(())
}
