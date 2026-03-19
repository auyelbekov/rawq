use anyhow::Result;
use crate::daemon::client::DaemonClient;
use crate::daemon::protocol::Response;

/// How to obtain an embedding vector — daemon (fast IPC) or local (cold ONNX load).
///
/// All commands that need embedding should use `resolve_model_source()` to get one.
/// The daemon path avoids cold model load (2-10s) by reusing a hot ONNX session via IPC.
pub enum ModelSource {
    /// Daemon is connected and ready. The bool indicates whether it was freshly started
    /// (true) or was already running (false), so callers can print context-specific hints.
    Daemon(DaemonClient, bool),
    /// No daemon available. Caller should load the embedder locally.
    Local,
}

/// Resolve model source: try daemon first (fast), fall back to local.
///
/// `spawner` is called if no daemon is running. Pass `None` to skip daemon entirely.
/// Prints shared messages (daemon startup / failure warning) for consistency across
/// all commands. Command-specific messages (e.g. "Subsequent searches will be faster")
/// are NOT printed here — the caller handles those based on the `freshly_started` flag.
pub fn resolve_model_source<F>(
    model_name: &str,
    no_daemon: bool,
    spawner: Option<F>,
) -> ModelSource
where
    F: FnOnce(&str) -> Result<u32>,
{
    if no_daemon || std::env::var("RAWQ_NO_DAEMON").is_ok() {
        return ModelSource::Local;
    }

    let Some(spawn_fn) = spawner else {
        return ModelSource::Local;
    };

    match DaemonClient::connect_or_start(model_name, spawn_fn) {
        Ok((client, freshly_started)) => ModelSource::Daemon(client, freshly_started),
        Err(e) => {
            eprintln!("warning: daemon unavailable ({e:#}), loading model locally");
            ModelSource::Local
        }
    }
}

/// Embed a query via the model source. Returns `(vector, model_name)` if daemon
/// succeeded, or `None` if local (caller should load the embedder and embed locally).
pub fn embed_query_via_source(
    source: &ModelSource,
    query: &str,
) -> Option<(Vec<f32>, String)> {
    match source {
        ModelSource::Daemon(client, _) => match client.embed(query, true) {
            Ok(Response::Embed {
                vector,
                model_name,
                ..
            }) => Some((vector, model_name)),
            Ok(Response::Error { message }) => {
                eprintln!("warning: daemon embed failed: {message}, falling back to local");
                None
            }
            _ => None,
        },
        ModelSource::Local => None,
    }
}
