//! CPU batch size auto-calibration.
//!
//! On CPU, there is no fixed VRAM wall — the optimal batch size depends on
//! the specific CPU, cache hierarchy, and model. Instead of guessing with a
//! hardcoded memory budget, we measure: double the batch size until throughput
//! stops improving, then cache the result.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::Embedder;

/// Synthetic text that resembles a real code chunk (~200 tokens).
const CALIBRATION_TEXT: &str = "\
/// Process a batch of items with the given configuration.\n\
fn process_items(items: &[Item], config: &Config) -> Result<Vec<Output>> {\n\
    let mut results = Vec::with_capacity(items.len());\n\
    for item in items {\n\
        let validated = validate(item, &config.rules)?;\n\
        let transformed = transform(&validated, config)?;\n\
        results.push(transformed);\n\
    }\n\
    Ok(results)\n\
}\n";

/// Minimum improvement ratio to keep doubling. If throughput doesn't improve
/// by at least this factor, we've hit the plateau.
const IMPROVEMENT_THRESHOLD: f64 = 1.05;

/// Maximum batch size to try during calibration.
const MAX_CALIBRATION_BATCH: usize = 128;

/// Maximum wall-clock time for the entire calibration (seconds).
const MAX_CALIBRATION_SECS: f64 = 3.0;

// ── Cache ────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct CalibrationEntry {
    batch_size: usize,
    throughput: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct CalibrationCache {
    entries: HashMap<String, CalibrationEntry>,
}

fn cache_path() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("rawq").join("calibration.json"))
}

fn cache_key(model_name: &str, cpu_model: &str) -> String {
    format!("{model_name}::{cpu_model}")
}

fn load_cache() -> CalibrationCache {
    let Some(path) = cache_path() else {
        return CalibrationCache::default();
    };
    fs::read_to_string(path)
        .ok()
        .and_then(|data| serde_json::from_str(&data).ok())
        .unwrap_or_default()
}

fn save_cache(cache: &CalibrationCache) {
    let Some(path) = cache_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(data) = serde_json::to_string_pretty(cache) {
        let _ = fs::write(path, data);
    }
}

// ── CPU detection ────────────────────────────────────────────────────────

/// Detect the CPU model string for use as a cache key.
pub fn detect_cpu_model() -> String {
    detect_cpu_model_inner().unwrap_or_else(|| "unknown-cpu".to_string())
}

#[cfg(target_os = "windows")]
fn detect_cpu_model_inner() -> Option<String> {
    let output = std::process::Command::new("reg")
        .args([
            "query",
            r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            "/v",
            "ProcessorNameString",
        ])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("ProcessorNameString") {
            let value = line.rsplit("REG_SZ").next()?.trim();
            if !value.is_empty() {
                return Some(sanitize_cpu_name(value));
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn detect_cpu_model_inner() -> Option<String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in cpuinfo.lines() {
        if line.starts_with("model name") {
            let value = line.split(':').nth(1)?.trim();
            if !value.is_empty() {
                return Some(sanitize_cpu_name(value));
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn detect_cpu_model_inner() -> Option<String> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()?;
    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if name.is_empty() {
        None
    } else {
        Some(sanitize_cpu_name(&name))
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn detect_cpu_model_inner() -> Option<String> {
    None
}

/// Replace whitespace/special chars with underscores for safe cache keys.
fn sanitize_cpu_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        // Collapse repeated underscores
        .split('_')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

// ── Calibration ──────────────────────────────────────────────────────────

/// Measure throughput for a given batch size (median of `runs` attempts).
fn measure_throughput(embedder: &mut Embedder, bs: usize, runs: usize) -> Option<f64> {
    let batch: Vec<&str> = vec![CALIBRATION_TEXT; bs];
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        if embedder.embed(&batch).is_err() {
            return None;
        }
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    Some(bs as f64 / median)
}

/// Run the calibration loop: warmup, then double batch size until throughput
/// plateaus. Returns (optimal_batch_size, throughput_items_per_sec).
fn run_calibration(embedder: &mut Embedder) -> (usize, f64) {
    let cal_start = Instant::now();

    // Warmup: multiple runs to stabilize ONNX thread pool, memory allocator,
    // and OS page faults. A single warmup leaves batch=2+ artificially slow
    // because they touch new memory pages for the first time.
    for bs in [1, 2, 4] {
        let batch: Vec<&str> = vec![CALIBRATION_TEXT; bs];
        let _ = embedder.embed(&batch);
    }

    // Measure each batch size with 3 runs (take median to filter noise)
    let mut best_size = 1usize;
    let mut best_throughput = measure_throughput(embedder, 1, 3).unwrap_or(1.0);
    let mut plateaus = 0u32;

    let mut bs = 2usize;
    while bs <= MAX_CALIBRATION_BATCH {
        if cal_start.elapsed().as_secs_f64() > MAX_CALIBRATION_SECS {
            break;
        }

        match measure_throughput(embedder, bs, 3) {
            Some(throughput) => {
                if throughput > best_throughput * IMPROVEMENT_THRESHOLD {
                    best_size = bs;
                    best_throughput = throughput;
                    plateaus = 0;
                } else {
                    // Require two consecutive plateaus to confirm — a single
                    // flat step can be noise from OS scheduling or cache effects.
                    plateaus += 1;
                    if plateaus >= 2 {
                        break;
                    }
                }
            }
            None => {
                break;
            }
        }

        bs *= 2;
    }

    (best_size, best_throughput)
}

/// Return the calibrated CPU batch size for this embedder.
///
/// Checks cache first; runs calibration if no cached result or if
/// `RAWQ_RECALIBRATE=1` is set.
pub fn calibrated_batch_size(embedder: &mut Embedder) -> usize {
    let cpu = detect_cpu_model();
    let key = cache_key(embedder.model_name(), &cpu);
    let force = std::env::var("RAWQ_RECALIBRATE").is_ok();

    if !force {
        let cache = load_cache();
        if let Some(entry) = cache.entries.get(&key) {
            eprintln!(
                "  CPU batch size: {} (cached, {:.0} items/s)",
                entry.batch_size, entry.throughput
            );
            return entry.batch_size;
        }
    }

    eprintln!("  Calibrating CPU batch size...");
    let (batch_size, throughput) = run_calibration(embedder);
    eprintln!(
        "  CPU batch size: {} (calibrated, {:.0} items/s)",
        batch_size, throughput
    );

    let mut cache = load_cache();
    cache.entries.insert(
        key,
        CalibrationEntry {
            batch_size,
            throughput,
        },
    );
    save_cache(&cache);

    batch_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_model_detection() {
        let cpu = detect_cpu_model();
        assert!(!cpu.is_empty());
        assert!(!cpu.contains(' '));
    }

    #[test]
    fn sanitize_cpu_name_works() {
        assert_eq!(
            sanitize_cpu_name("Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz"),
            "Intel_R_Core_TM_i7-10750H_CPU_2.60GHz"
        );
        assert_eq!(sanitize_cpu_name("Apple M2 Pro"), "Apple_M2_Pro");
    }

    #[test]
    fn cache_key_format() {
        let key = cache_key("snowflake-arctic-embed-s", "Intel_Core_i7");
        assert_eq!(key, "snowflake-arctic-embed-s::Intel_Core_i7");
    }

    #[test]
    fn cache_roundtrip() {
        let mut cache = CalibrationCache::default();
        cache.entries.insert(
            "test::cpu".to_string(),
            CalibrationEntry {
                batch_size: 16,
                throughput: 42.5,
            },
        );
        let json = serde_json::to_string(&cache).unwrap();
        let loaded: CalibrationCache = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.entries["test::cpu"].batch_size, 16);
    }
}
