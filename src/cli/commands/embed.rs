use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use crate::search::{ModelSource, resolve_model_source};
use crate::daemon::protocol::Response;
use crate::embed::Embedder;

use crate::cli::util::{effective_model, resolve_model_name, DAEMON_READY_MSG};

pub fn cmd_embed(
    text: String,
    query: bool,
    model: Option<String>,
    tokenizer: Option<PathBuf>,
    no_daemon: bool,
) -> Result<()> {
    let start = Instant::now();
    let effective = effective_model(&model);

    // Explicit paths bypass daemon entirely
    if let (Some(m), Some(t)) = (&model, &tokenizer) {
        let mut embedder = Embedder::from_paths(std::path::Path::new(m), t)?;
        let load_ms = start.elapsed().as_millis();
        eprintln!(
            "Loaded {} ({} dim, {} seq) in {load_ms}ms",
            embedder.model_name(),
            embedder.embed_dim(),
            embedder.max_seq_len()
        );
        return embed_and_print(&mut embedder, &text, query);
    }

    // Try daemon for fast embedding
    let model_name = resolve_model_name(&effective);

    let source = resolve_model_source(
        &model_name,
        no_daemon,
        Some(|name: &str| crate::cli::daemon_spawn::spawn_daemon(name)),
    );

    if let ModelSource::Daemon(ref client, freshly_started) = source {
        if freshly_started {
            eprintln!("{DAEMON_READY_MSG}");
        } else {
            eprintln!("Using daemon ({model_name}).");
        }
        if let Ok(Response::Embed {
            vector,
            model_name: name,
            embed_dim,
            ..
        }) = client.embed(&text, query)
        {
            let elapsed_ms = start.elapsed().as_millis();
            let floats: Vec<String> = vector.iter().map(|f| format!("{f:.6}")).collect();
            println!("[{}]", floats.join(", "));
            eprintln!("Embedded via daemon ({name}, {embed_dim} dim) in {elapsed_ms}ms");
            eprintln!("Dimensions: {}", vector.len());
            return Ok(());
        }
        // Daemon embed failed — fall through to local
    }

    // Local fallback
    let mut embedder = Embedder::from_managed(effective.as_deref())?;
    let load_ms = start.elapsed().as_millis();
    eprintln!(
        "Loaded {} ({} dim, {} seq) in {load_ms}ms",
        embedder.model_name(),
        embedder.embed_dim(),
        embedder.max_seq_len()
    );
    embed_and_print(&mut embedder, &text, query)
}

fn embed_and_print(
    embedder: &mut Embedder,
    text: &str,
    query: bool,
) -> Result<()> {
    let embed_start = Instant::now();
    let vec = if query {
        embedder.embed_query(text)?
    } else {
        embedder.embed_document(text)?
    };
    let embed_ms = embed_start.elapsed().as_micros() as f64 / 1000.0;

    let floats: Vec<String> = vec.iter().map(|f| format!("{f:.6}")).collect();
    println!("[{}]", floats.join(", "));

    eprintln!("Dimensions: {}", vec.len());
    eprintln!("Embed time: {embed_ms:.2}ms");

    Ok(())
}
