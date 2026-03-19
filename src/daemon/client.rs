use std::io::{BufRead, BufReader, Write};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::daemon::protocol::{Request, Response};

pub struct DaemonClient {
    model_name: String,
}

impl DaemonClient {
    /// Try connecting to an existing daemon. Returns `Some` if the daemon is alive.
    pub fn try_connect(model_name: &str) -> Option<Self> {
        let client = Self {
            model_name: model_name.to_string(),
        };
        match client.send_request(&Request::Status) {
            Ok(_) => Some(client),
            Err(_) => None,
        }
    }

    /// Connect to a running daemon, or start one and wait for it to become ready.
    ///
    /// `spawner` is called to start the daemon process if none is running.
    /// Returns `(client, freshly_started)` — the bool indicates whether the daemon
    /// was just spawned (true) or was already running (false).
    pub fn connect_or_start(
        model_name: &str,
        spawner: impl FnOnce(&str) -> Result<u32>,
    ) -> Result<(Self, bool)> {
        // Try connecting first
        if let Some(client) = Self::try_connect(model_name) {
            return Ok((client, false));
        }

        // Spawn daemon
        eprint!("Starting rawq daemon ({model_name})... ");
        let start = Instant::now();
        spawner(model_name)?;

        // Retry with exponential backoff (10ms → 20ms → 40ms → ... capped at 500ms)
        let timeout = Duration::from_secs(30);
        let mut interval = Duration::from_millis(10);
        let max_interval = Duration::from_millis(500);

        loop {
            std::thread::sleep(interval);

            if let Some(client) = Self::try_connect(model_name) {
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!("ready in {elapsed:.1}s.");
                return Ok((client, true));
            }

            if start.elapsed() > timeout {
                anyhow::bail!(
                    "daemon for model '{model_name}' did not start within {}s",
                    timeout.as_secs()
                );
            }

            interval = (interval * 2).min(max_interval);
        }
    }

    /// Send an Embed request.
    pub fn embed(&self, text: &str, query: bool) -> Result<Response> {
        self.send_request(&Request::Embed {
            text: text.to_string(),
            query,
        })
    }

    /// Embed multiple texts via per-item IPC calls.
    /// Suitable for small incremental reindexes (< ~50 chunks).
    /// Returns `(vectors, model_name)`.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<(Vec<Vec<f32>>, String)> {
        let mut vectors = Vec::with_capacity(texts.len());
        let mut model_name = String::new();
        for text in texts {
            match self.embed(text, false)? {
                Response::Embed {
                    vector,
                    model_name: name,
                    ..
                } => {
                    model_name = name;
                    vectors.push(vector);
                }
                Response::Error { message } => {
                    anyhow::bail!("daemon embed error: {message}");
                }
                _ => {
                    anyhow::bail!("unexpected daemon response");
                }
            }
        }
        Ok((vectors, model_name))
    }

    /// Send a Status request.
    pub fn status(&self) -> Result<Response> {
        self.send_request(&Request::Status)
    }

    /// Send a Shutdown request.
    pub fn shutdown(&self) -> Result<Response> {
        self.send_request(&Request::Shutdown)
    }

    /// Open a new connection, send an NDJSON request, read an NDJSON response.
    fn send_request(&self, request: &Request) -> Result<Response> {
        let socket_name = crate::daemon::daemon_socket_name(&self.model_name);

        #[cfg(windows)]
        let stream = {
            use interprocess::local_socket::traits::Stream as _;
            use interprocess::local_socket::{GenericNamespaced, Stream, ToNsName};
            let name = socket_name.to_ns_name::<GenericNamespaced>()?;
            Stream::connect(name).context("connect to daemon socket")?
        };

        #[cfg(not(windows))]
        let stream = {
            use interprocess::local_socket::traits::Stream as _;
            use interprocess::local_socket::{GenericFilePath, Stream, ToFsName};
            let name = socket_name.to_fs_name::<GenericFilePath>()?;
            Stream::connect(name).context("connect to daemon socket")?
        };

        // Write request as NDJSON
        let mut json = serde_json::to_string(request)?;
        json.push('\n');
        (&stream).write_all(json.as_bytes())?;
        (&stream).flush()?;

        // Read response
        let mut reader = BufReader::new(&stream);
        let mut resp_line = String::new();
        reader.read_line(&mut resp_line)?;

        let response: Response =
            serde_json::from_str(resp_line.trim()).context("parse daemon response")?;
        Ok(response)
    }
}
