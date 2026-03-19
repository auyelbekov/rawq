//! GPU hardware detection: auto-select best device and query real VRAM.

/// Detected GPU hardware info.
pub struct GpuInfo {
    /// Device index to pass to the execution provider.
    pub device_id: i32,
    /// Dedicated VRAM in bytes (or usable GPU memory for unified architectures).
    pub vram_bytes: u64,
    /// Human-readable GPU name (for logging).
    pub name: String,
}

// ── DirectML (Windows) ─────────────────────────────────────────────────

/// Enumerate DXGI adapters and pick the one with the most dedicated VRAM.
/// Skips software/render-only adapters.
#[cfg(all(windows, feature = "directml"))]
pub fn detect_best_gpu() -> Option<GpuInfo> {
    use windows::Win32::Graphics::Dxgi::*;

    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut best_idx: i32 = 0;
        let mut best_vram: u64 = 0;
        let mut best_name = String::new();
        let mut i = 0u32;

        while let Ok(adapter) = factory.EnumAdapters1(i) {
            if let Ok(desc) = adapter.GetDesc1() {
                // Skip software adapters (e.g. Microsoft Basic Render Driver)
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                    i += 1;
                    continue;
                }
                let vram = desc.DedicatedVideoMemory as u64;
                let name = String::from_utf16_lossy(
                    &desc.Description[..desc.Description.iter().position(|&c| c == 0).unwrap_or(128)],
                );
                eprintln!("  GPU {i}: {name} — {:.1} GB VRAM", vram as f64 / (1024.0 * 1024.0 * 1024.0));
                if vram > best_vram {
                    best_vram = vram;
                    best_idx = i as i32;
                    best_name = name;
                }
            }
            i += 1;
        }

        if best_vram > 0 {
            Some(GpuInfo {
                device_id: best_idx,
                vram_bytes: best_vram,
                name: best_name,
            })
        } else {
            None
        }
    }
}

/// Query VRAM for a specific DXGI adapter by index.
#[cfg(all(windows, feature = "directml"))]
pub fn detect_gpu_vram(device_idx: u32) -> Option<u64> {
    use windows::Win32::Graphics::Dxgi::*;

    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let adapter = factory.EnumAdapters1(device_idx).ok()?;
        let desc = adapter.GetDesc1().ok()?;
        Some(desc.DedicatedVideoMemory as u64)
    }
}

// ── CUDA (Linux / Windows with NVIDIA) ──────────────────────────────────

/// Parse `nvidia-smi` output to find the GPU with the most VRAM.
#[cfg(feature = "cuda")]
pub fn detect_best_gpu() -> Option<GpuInfo> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,memory.total,name", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut best_idx: i32 = 0;
    let mut best_vram: u64 = 0;
    let mut best_name = String::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 3 {
            let idx = parts[0].parse::<i32>().ok()?;
            // nvidia-smi reports in MiB
            let vram_mib = parts[1].parse::<u64>().ok()?;
            let vram_bytes = vram_mib * 1024 * 1024;
            let name = parts[2].to_string();
            eprintln!("  GPU {idx}: {name} — {vram_mib} MiB VRAM");
            if vram_bytes > best_vram {
                best_vram = vram_bytes;
                best_idx = idx;
                best_name = name;
            }
        }
    }

    if best_vram > 0 {
        Some(GpuInfo {
            device_id: best_idx,
            vram_bytes: best_vram,
            name: best_name,
        })
    } else {
        None
    }
}

/// Query VRAM for a specific CUDA device by index.
#[cfg(feature = "cuda")]
pub fn detect_gpu_vram(device_idx: u32) -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            &format!("--id={device_idx}"),
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let vram_mib = stdout.trim().parse::<u64>().ok()?;
    Some(vram_mib * 1024 * 1024)
}

// ── CoreML (macOS) ──────────────────────────────────────────────────────

/// On macOS (Apple Silicon), GPU shares system RAM.
/// Query total physical memory via `sysctl hw.memsize` and use 75%.
#[cfg(feature = "coreml")]
pub fn detect_best_gpu() -> Option<GpuInfo> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let total_bytes = stdout.trim().parse::<u64>().ok()?;
    // Apple Silicon unified memory: allocate 75% as GPU-usable budget
    let gpu_budget = (total_bytes / 4) * 3;

    Some(GpuInfo {
        device_id: 0,
        vram_bytes: gpu_budget,
        name: "Apple Silicon (unified)".to_string(),
    })
}

/// CoreML has a single device, return the same budget.
#[cfg(feature = "coreml")]
pub fn detect_gpu_vram(_device_idx: u32) -> Option<u64> {
    detect_best_gpu().map(|info| info.vram_bytes)
}

// ── Fallback (no GPU feature) ───────────────────────────────────────────

#[cfg(not(any(
    all(windows, feature = "directml"),
    feature = "cuda",
    feature = "coreml"
)))]
pub fn detect_best_gpu() -> Option<GpuInfo> {
    None
}

#[cfg(not(any(
    all(windows, feature = "directml"),
    feature = "cuda",
    feature = "coreml"
)))]
pub fn detect_gpu_vram(_device_idx: u32) -> Option<u64> {
    None
}
