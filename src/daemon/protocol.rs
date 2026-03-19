use serde::{Deserialize, Serialize};

/// Protocol version for CLI/daemon compatibility checks.
/// Increment when the wire format changes in incompatible ways.
/// v2: Watch removed — daemon is a pure embedding server.
pub const PROTOCOL_VERSION: u32 = 2;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum Request {
    Embed { text: String, query: bool },
    Status,
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Embed {
        vector: Vec<f32>,
        model_name: String,
        embed_dim: usize,
    },
    Status {
        model: String,
        embed_dim: usize,
        uptime_secs: u64,
        requests_served: u64,
        pid: u32,
        #[serde(default)]
        protocol_version: u32,
    },
    Shutdown {
        ok: bool,
    },
    Error {
        message: String,
    },
}
