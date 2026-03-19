pub mod client;
pub mod protocol;
pub mod server;

pub use client::DaemonClient;
pub use protocol::{Request, Response, PROTOCOL_VERSION};

use std::path::PathBuf;

use anyhow::{Context, Result};

/// Sanitize a model name into a safe socket/file identifier.
fn safe_name(model_name: &str) -> String {
    model_name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

/// Return a platform-specific socket name for the daemon.
///
/// - Windows: a plain name like `rawq-daemon-<safe>` (interprocess prepends `\\.\pipe\`
///   via `GenericNamespaced`).
/// - Unix: a full path `~/.cache/rawq/daemon-<safe>.sock`.
pub fn daemon_socket_name(model_name: &str) -> String {
    let safe = safe_name(model_name);
    if cfg!(windows) {
        format!("rawq-daemon-{safe}")
    } else {
        let cache = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
        let sock = cache.join("rawq").join(format!("daemon-{safe}.sock"));
        sock.to_string_lossy().into_owned()
    }
}

/// Return the path to the PID file for a given model.
pub fn pid_file_path(model_name: &str) -> Result<PathBuf> {
    let safe = safe_name(model_name);
    let cache = dirs::cache_dir().context("could not determine cache directory")?;
    Ok(cache.join("rawq").join(format!("daemon-{safe}.pid")))
}
