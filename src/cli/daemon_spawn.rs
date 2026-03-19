use anyhow::{Context, Result};

/// Spawn the daemon as a detached background process.
///
/// Runs `rawq daemon start --model <name>` and returns the child PID.
/// The child has null stdio + CREATE_NO_WINDOW (Windows) so it is already detached.
///
/// This is CLI-specific — it spawns the rawq binary as a subprocess.
/// The daemon crate's `connect_or_start()` takes this as a callback.
pub fn spawn_daemon(model_name: &str) -> Result<u32> {
    let exe = std::env::current_exe().context("could not determine rawq executable path")?;

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let child = std::process::Command::new(&exe)
            .args(["daemon", "start", "--model", model_name])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .context("failed to spawn daemon process")?;
        Ok(child.id())
    }

    #[cfg(not(windows))]
    {
        let child = std::process::Command::new(&exe)
            .args(["daemon", "start", "--model", model_name])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("failed to spawn daemon process")?;
        Ok(child.id())
    }
}
