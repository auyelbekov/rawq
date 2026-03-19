use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

pub fn cmd_chunk(path: PathBuf) -> Result<()> {
    let start = Instant::now();
    let abs_path = std::fs::canonicalize(&path)?;

    let chunks = crate::index::walk_and_chunk(&abs_path)?;

    for chunk in &chunks {
        let json = serde_json::to_string(chunk)?;
        println!("{json}");
    }

    let elapsed = start.elapsed().as_millis();
    eprintln!("{} chunks in {elapsed}ms", chunks.len());

    Ok(())
}
