use anyhow::Result;

use crate::daemon::DaemonClient;
use crate::daemon::protocol::Response;

use crate::cli::util::{effective_model, resolve_model_name};

/// Start the daemon (foreground or background).
pub fn cmd_daemon_start(
    model: Option<String>,
    idle_timeout: u64,
    background: bool,
) -> Result<()> {
    let effective = effective_model(&model);
    let model_name = resolve_model_name(&effective);

    if DaemonClient::try_connect(&model_name).is_some() {
        eprintln!("Daemon for model '{model_name}' is already running.");
        return Ok(());
    }

    if background {
        let pid = crate::cli::daemon_spawn::spawn_daemon(&model_name)?;
        eprintln!("Daemon spawned (model: {model_name}, pid: {pid})");
        return Ok(());
    }

    // Run in foreground (blocking)
    crate::daemon::server::run(effective.as_deref(), idle_timeout, false)
}

/// Stop a running daemon.
pub fn cmd_daemon_stop(model: Option<String>) -> Result<()> {
    let model_name = resolve_model_name(&effective_model(&model));

    match DaemonClient::try_connect(&model_name) {
        Some(client) => {
            client.shutdown()?;
            eprintln!("Daemon for model '{model_name}' stopped.");
        }
        None => {
            eprintln!("No daemon running for model '{model_name}'.");
        }
    }
    Ok(())
}

/// Show daemon status.
pub fn cmd_daemon_status(model: Option<String>, json: bool) -> Result<()> {
    let model_name = resolve_model_name(&effective_model(&model));

    match DaemonClient::try_connect(&model_name) {
        Some(client) => {
            let resp = client.status()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp)?);
            } else if let Response::Status {
                model,
                embed_dim,
                uptime_secs,
                requests_served,
                pid,
                ..
            } = resp
            {
                eprintln!("Daemon running:");
                eprintln!("  Model:     {model}");
                eprintln!("  Embed dim: {embed_dim}");
                eprintln!("  PID:       {pid}");
                eprintln!("  Uptime:    {uptime_secs}s");
                eprintln!("  Requests:  {requests_served}");
            }
        }
        None => {
            if json {
                println!("{{\"running\": false}}");
            } else {
                eprintln!("No daemon running for model '{model_name}'.");
            }
        }
    }
    Ok(())
}
