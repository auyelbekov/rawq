use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use interprocess::local_socket::traits::Listener as _;
use interprocess::local_socket::{ListenerNonblockingMode, ListenerOptions};

use crate::daemon::protocol::{Request, Response, PROTOCOL_VERSION};

/// Run the daemon: load embedding model, listen on local socket, serve requests.
///
/// `model_name` should already be resolved (the caller handles env vars / config).
/// The accept loop is non-blocking (200ms poll), connection handling is sequential
/// because the ONNX session (`Embedder::embed(&mut self)`) is not thread-safe.
pub fn run(model_name: Option<&str>, idle_timeout_mins: u64, background: bool) -> Result<()> {
    let mut embedder = crate::embed::Embedder::from_managed(model_name)?;
    let model = embedder.model_name().to_string();
    let embed_dim = embedder.embed_dim();

    // Write PID file
    let pid = std::process::id();
    let pid_path = crate::daemon::pid_file_path(&model)?;
    if let Some(parent) = pid_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&pid_path, pid.to_string())
        .with_context(|| format!("write PID file: {}", pid_path.display()))?;

    let socket_name = crate::daemon::daemon_socket_name(&model);

    #[cfg(not(windows))]
    {
        use interprocess::local_socket::{GenericFilePath, ToFsName};
        let name = socket_name.clone().to_fs_name::<GenericFilePath>()?;

        // Check if an existing daemon is still alive before removing the socket.
        if let Ok(old_pid_str) = std::fs::read_to_string(&pid_path) {
            if let Ok(old_pid) = old_pid_str.trim().parse::<u32>() {
                if old_pid != pid && std::path::Path::new(&format!("/proc/{old_pid}")).exists() {
                    anyhow::bail!(
                        "daemon already running (pid: {old_pid}). \
                         Stop it with: rawq daemon stop"
                    );
                }
            }
        }
        let _ = std::fs::remove_file(&socket_name);

        let listener = ListenerOptions::new()
            .name(name)
            .nonblocking(ListenerNonblockingMode::Accept)
            .create_sync()
            .with_context(|| format!("bind local socket: {socket_name}"))?;

        run_loop(
            listener, &mut embedder, &model, embed_dim, pid,
            idle_timeout_mins, background, &pid_path, Some(&socket_name),
        )
    }

    #[cfg(windows)]
    {
        use interprocess::local_socket::{GenericNamespaced, ToNsName};
        let name = socket_name.clone().to_ns_name::<GenericNamespaced>()?;

        let listener = ListenerOptions::new()
            .name(name)
            .nonblocking(ListenerNonblockingMode::Accept)
            .create_sync()
            .with_context(|| format!("bind local socket: {socket_name}"))?;

        run_loop(
            listener, &mut embedder, &model, embed_dim, pid,
            idle_timeout_mins, background, &pid_path, None,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_loop(
    listener: interprocess::local_socket::Listener,
    embedder: &mut crate::embed::Embedder,
    model_name: &str,
    embed_dim: usize,
    pid: u32,
    idle_timeout_mins: u64,
    background: bool,
    pid_path: &std::path::Path,
    _socket_file: Option<&str>,
) -> Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    if let Err(e) = ctrlc::try_set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }) {
        eprintln!("warning: could not set Ctrl+C handler: {e}");
    }

    if !background {
        eprintln!("rawq daemon ready (model: {model_name}, pid: {pid})");
    }

    let start_time = Instant::now();
    let mut last_activity = Instant::now();
    let mut requests_served: u64 = 0;
    let idle_timeout = Duration::from_secs(idle_timeout_mins * 60);

    while running.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok(stream) => {
                last_activity = Instant::now();
                requests_served += 1;

                let _ = handle_connection(
                    stream, embedder, model_name, embed_dim,
                    &start_time, requests_served, pid, &running,
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => {
                eprintln!("warning: accept error: {e}");
            }
        }

        if last_activity.elapsed() > idle_timeout {
            eprintln!("Idle timeout reached, shutting down daemon.");
            break;
        }

        std::thread::sleep(Duration::from_millis(200));
    }

    // Cleanup
    let _ = std::fs::remove_file(pid_path);
    #[cfg(not(windows))]
    if let Some(sock) = _socket_file {
        let _ = std::fs::remove_file(sock);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_connection(
    stream: interprocess::local_socket::Stream,
    embedder: &mut crate::embed::Embedder,
    model_name: &str,
    embed_dim: usize,
    start_time: &Instant,
    requests_served: u64,
    pid: u32,
    running: &Arc<AtomicBool>,
) -> Result<()> {
    const MAX_REQUEST_BYTES: usize = 10 * 1024 * 1024;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;

    if line.len() > MAX_REQUEST_BYTES {
        let resp = Response::Error {
            message: format!(
                "request too large ({} bytes, max {})",
                line.len(),
                MAX_REQUEST_BYTES
            ),
        };
        write_response(&stream, &resp)?;
        return Ok(());
    }

    let request: Request = match serde_json::from_str(line.trim()) {
        Ok(r) => r,
        Err(e) => {
            let resp = Response::Error {
                message: format!("invalid request: {e}"),
            };
            write_response(&stream, &resp)?;
            return Ok(());
        }
    };

    let response = match request {
        Request::Embed { ref text, query } => {
            let vec_result = if query {
                embedder.embed_query(text)
            } else {
                embedder.embed_document(text)
            };
            match vec_result {
                Ok(vec) => Response::Embed {
                    vector: vec.to_vec(),
                    model_name: model_name.to_string(),
                    embed_dim,
                },
                Err(e) => Response::Error {
                    message: format!("embed failed: {e:#}"),
                },
            }
        }
        Request::Status => {
            let uptime_secs = start_time.elapsed().as_secs();
            Response::Status {
                model: model_name.to_string(),
                embed_dim,
                uptime_secs,
                requests_served,
                pid,
                protocol_version: PROTOCOL_VERSION,
            }
        }
        Request::Shutdown => {
            running.store(false, Ordering::SeqCst);
            Response::Shutdown { ok: true }
        }
    };

    write_response(&stream, &response)?;
    Ok(())
}

fn write_response(
    mut stream: &interprocess::local_socket::Stream,
    response: &Response,
) -> Result<()> {
    let mut json = serde_json::to_string(response)?;
    json.push('\n');
    stream.write_all(json.as_bytes())?;
    stream.flush()?;
    Ok(())
}
