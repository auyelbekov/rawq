use colored::Colorize;
use serde::Serialize;

/// Cache whether `bat` is on PATH (checked once per process).
/// Installing `bat` after the first search won't take effect until rawq restarts.
static BAT_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn bat_on_path() -> bool {
    *BAT_AVAILABLE.get_or_init(|| {
        std::process::Command::new("bat")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

fn map_lang_to_bat(language: &str) -> &str {
    match language {
        "rust" => "Rust",
        "python" => "Python",
        "typescript" => "TypeScript",
        "javascript" => "JavaScript",
        "go" => "Go",
        "java" => "Java",
        "c" => "C",
        "cpp" => "C++",
        "csharp" => "C#",
        "ruby" => "Ruby",
        "php" => "PHP",
        "swift" => "Swift",
        "bash" => "Bash",
        "lua" => "Lua",
        "scala" => "Scala",
        "dart" => "Dart",
        "kotlin" => "Kotlin",
        "sql" => "SQL",
        "yaml" => "YAML",
        "toml" => "TOML",
        "json" => "JSON",
        _ => "Plain Text",
    }
}

/// Syntax-highlight `content` with bat. Returns per-line highlighted strings, or None if bat fails/absent.
fn bat_highlight(content: &str, language: &str) -> Option<Vec<String>> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let bat_lang = map_lang_to_bat(language);
    let mut child = Command::new("bat")
        .args([
            "--color=always",
            "--paging=never",
            "--plain",
            "--language",
            bat_lang,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(content.as_bytes());
    }

    let out = child.wait_with_output().ok()?;
    if out.status.success() {
        let highlighted = String::from_utf8_lossy(&out.stdout);
        Some(highlighted.lines().map(|l| l.to_string()).collect())
    } else {
        None
    }
}

#[derive(Serialize)]
pub struct ErrorOutput {
    pub error: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_ms: Option<u64>,
}

/// Check if --json or --stream is present in args (works even before clap parses).
pub fn wants_json() -> bool {
    std::env::args().any(|a| a == "--json" || a == "--stream")
}

pub fn print_json_error(code: &str, msg: &str, suggestion: Option<&str>) {
    let output = ErrorOutput {
        error: code.to_string(),
        message: msg.to_string(),
        suggestion: suggestion.map(String::from),
        query_ms: None,
    };
    let json = serde_json::to_string_pretty(&output)
        .unwrap_or_else(|e| format!(r#"{{"error":"serialization_error","message":"{e}"}}"#));
    println!("{json}");
}

pub fn print_json(output: &crate::search::SearchOutput) {
    let json = serde_json::to_string_pretty(output)
        .unwrap_or_else(|e| format!(r#"{{"error":"serialization_error","message":"{e}"}}"#));
    println!("{json}");
}

pub fn print_ndjson(output: &crate::search::SearchOutput) {
    for result in &output.results {
        let line = match serde_json::to_string(result) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("warning: failed to serialize result: {e}");
                continue;
            }
        };
        println!("{line}");
    }
    // Final metadata line
    let meta = serde_json::json!({
        "_meta": {
            "model": output.model,
            "query_ms": output.query_ms,
            "total_tokens": output.total_tokens,
            "count": output.results.len(),
        }
    });
    let meta_line = serde_json::to_string(&meta)
        .unwrap_or_else(|_| r#"{"_meta":{"error":"serialization_failed"}}"#.to_string());
    println!("{meta_line}");
}

/// Format search results as a human-readable string.
/// Uses `bat` for syntax highlighting when available and stdout is a TTY.
fn format_human(output: &crate::search::SearchOutput) -> String {
    let use_bat = is_tty_stdout()
        && bat_on_path()
        && std::env::var("RAWQ_NO_BAT").is_err();

    let mut buf = String::new();
    for (i, result) in output.results.iter().enumerate() {
        if i > 0 {
            buf.push_str(&format!("{}\n", "--".dimmed()));
        }

        // File path + line range
        let loc = format!(
            "{}:{}-{}",
            result.file, result.lines[0], result.lines[1]
        );
        buf.push_str(&format!("{}", loc.green().bold()));

        // Confidence
        buf.push_str(&format!(
            "  {}",
            format!("[{:.0}%]", result.confidence * 100.0).yellow()
        ));

        // Scope
        if !result.scope.is_empty() {
            buf.push_str(&format!("  {}", result.scope.dimmed()));
        }
        buf.push('\n');

        // Context before (dimmed with ┊ gutter)
        if !result.context_before.is_empty() {
            let ctx_start = result.display_start_line;
            for (offset, line) in result.context_before.lines().enumerate() {
                let line_no = ctx_start + offset;
                buf.push_str(&format!(
                    "{}{}\n",
                    format!(" ┊{line_no:>4} | ").dimmed(),
                    line.dimmed()
                ));
            }
        }

        // Numbered code lines — use bat for syntax highlighting if available
        let highlighted = if use_bat {
            bat_highlight(&result.content, &result.language)
        } else {
            None
        };
        let plain_lines: Vec<&str> = result.content.lines().collect();
        for (offset, plain_line) in plain_lines.iter().enumerate() {
            let line_no = result.lines[0] + offset;
            let code_line = highlighted
                .as_ref()
                .and_then(|hl| hl.get(offset).map(|s| s.as_str()))
                .unwrap_or(plain_line);

            if highlighted.is_some() {
                let gutter = if result.matched_lines.contains(&line_no) {
                    format!("{}", format!("{line_no:>6} > ").yellow())
                } else {
                    format!("{}", format!("{line_no:>6} | ").dimmed())
                };
                buf.push_str(&format!("{gutter}{code_line}\n"));
            } else {
                let is_matched = result.matched_lines.contains(&line_no);
                if is_matched {
                    buf.push_str(&format!(
                        "{}{}\n",
                        format!("{line_no:>6} | ").dimmed(),
                        plain_line.bold()
                    ));
                } else {
                    buf.push_str(&format!(
                        "{}{}\n",
                        format!("{line_no:>6} | ").dimmed(),
                        plain_line
                    ));
                }
            }
        }

        // Context after (dimmed with ┊ gutter)
        if !result.context_after.is_empty() {
            let ctx_start = result.lines[1] + 1;
            for (offset, line) in result.context_after.lines().enumerate() {
                let line_no = ctx_start + offset;
                buf.push_str(&format!(
                    "{}{}\n",
                    format!(" ┊{line_no:>4} | ").dimmed(),
                    line.dimmed()
                ));
            }
        }
    }
    buf
}

pub fn print_human(output: &crate::search::SearchOutput) {
    use std::io::Write;
    let text = format_human(output);
    // Use pager when stdout is a TTY, unless explicitly disabled
    if is_tty_stdout() && std::env::var("RAWQ_NO_PAGER").is_err() && try_pager(&text) {
        return;
    }
    let stdout = std::io::stdout();
    let _ = stdout.lock().write_all(text.as_bytes());
}

/// True when stdout is connected to a terminal.
fn is_tty_stdout() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

/// Try to pipe `text` through an available pager (`bat` or `less`).
/// Returns true if the pager ran successfully.
fn try_pager(text: &str) -> bool {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // Respect $PAGER env, then try less, then bat (as a plain pager — syntax highlight already done)
    let pager_env = std::env::var("PAGER").ok();
    let candidates: Vec<Vec<&str>> = if let Some(ref pager) = pager_env {
        vec![vec![pager.as_str()]]
    } else {
        vec![
            // less: pass through ANSI (-R), exit-if-one-screen (-F), no clear-screen (-X)
            vec!["less", "-R", "-F", "-X"],
            // bat as a plain pager (syntax highlight was already applied per-chunk)
            vec!["bat", "--color=always", "--plain", "--paging=always"],
        ]
    };

    for args in &candidates {
        let (program, rest) = match args.as_slice() {
            [prog, rest @ ..] => (*prog, rest),
            _ => continue,
        };
        let mut child = match Command::new(program)
            .args(rest)
            .stdin(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => continue,
        };

        if let Some(stdin) = child.stdin.take() {
            let mut w = stdin;
            let _ = w.write_all(text.as_bytes());
        }

        match child.wait() {
            Ok(s) if s.success() => return true,
            _ => {}
        }
    }
    false
}
