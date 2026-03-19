use anyhow::Result;
use tree_sitter::{Node, Parser};

use crate::index::chunk::{Chunk, Language};
use crate::index::chunker::queries::get_config;

/// Maximum non-whitespace characters per chunk. Nodes above this recurse.
const MAX_NWS: usize = 1500;
/// Minimum non-whitespace characters for a node to get its own chunk.
/// Nodes below this go into the buffer.
const MIN_OWN_NWS: usize = 150;

/// Count non-whitespace characters in a node's byte range.
fn count_nws(node: &Node, source: &[u8]) -> usize {
    source[node.start_byte()..node.end_byte()]
        .iter()
        .filter(|b| !b.is_ascii_whitespace())
        .count()
}

pub fn chunk_code(file: &str, source: &str, language: Language) -> Result<Vec<Chunk>> {
    let config = get_config(&language)
        .ok_or_else(|| anyhow::anyhow!("no tree-sitter grammar for {language}"))?;

    let mut parser = Parser::new();
    parser.set_language(&config.language)?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!(
            "tree-sitter parse failed for {file} ({language:?}); file may contain syntax errors"
        ))?;

    let source_bytes = source.as_bytes();
    let total_lines = source.lines().count();
    let all_lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();

    chunk_node(
        &tree.root_node(), source_bytes, file, &language,
        "", &mut chunks,
    );

    // Catch lines not covered by any named node (comments, blank-line-separated code)
    let gap_chunks = extract_uncovered_lines(&chunks, &all_lines, file, &language, total_lines);
    chunks.extend(gap_chunks);

    // Fallback: empty file or no recognized nodes
    if chunks.is_empty() && total_lines > 0 {
        chunks.push(Chunk {
            file: file.to_string(),
            lines: [1, total_lines],
            language: language.clone(),
            scope: String::new(),
            content: source.to_string(),
            kind: "file".to_string(),
        });
    }

    Ok(chunks)
}

/// Recursive NWS-based chunking (v2 algorithm).
///
/// - `>1500` NWS → recurse; `150–1500` NWS → own chunk; `<150` NWS → buffer.
/// - Each recursion level has its own independent buffer.
fn chunk_node(
    node: &Node,
    source: &[u8],
    file: &str,
    language: &Language,
    parent_scope: &str,
    chunks: &mut Vec<Chunk>,
) {
    let mut cursor = node.walk();
    let children: Vec<Node> = node.named_children(&mut cursor).collect();

    let mut buffer: Vec<Node> = Vec::new();
    let mut buffer_nws: usize = 0;

    for child in &children {
        let child_nws = count_nws(child, source);

        if child_nws > MAX_NWS && child.named_child_count() > 0 {
            // Over budget with children — flush buffer, then recurse
            flush_buffer(&buffer, source, file, language, parent_scope, chunks);
            buffer.clear();
            buffer_nws = 0;

            let name = extract_name(child, language, source);
            let child_scope = build_scope(parent_scope, &name);
            chunk_node(child, source, file, language, &child_scope, chunks);
        } else if child_nws > MAX_NWS {
            // Over budget, no children — take comments, flush buffer, emit as-is
            let comments = take_adjacent_comments(&mut buffer, child);
            flush_buffer(&buffer, source, file, language, parent_scope, chunks);
            buffer.clear();
            buffer_nws = 0;

            emit_node_with_comments(child, &comments, source, file, language, parent_scope, chunks);
        } else if child_nws >= MIN_OWN_NWS {
            // 150–1500 NWS — take comments, flush buffer, own chunk
            let comments = take_adjacent_comments(&mut buffer, child);
            flush_buffer(&buffer, source, file, language, parent_scope, chunks);
            buffer.clear();
            buffer_nws = 0;

            emit_node_with_comments(child, &comments, source, file, language, parent_scope, chunks);
        } else {
            // Under 150 NWS — add to buffer, flush first if would exceed budget
            if buffer_nws + child_nws > MAX_NWS {
                flush_buffer(&buffer, source, file, language, parent_scope, chunks);
                buffer.clear();
                buffer_nws = 0;
            }
            buffer.push(*child);
            buffer_nws += child_nws;
        }
    }

    // Flush remaining buffer
    flush_buffer(&buffer, source, file, language, parent_scope, chunks);
}

/// Emit a single node with optional preceding doc comments as one chunk.
fn emit_node_with_comments(
    node: &Node,
    comments: &[Node],
    source: &[u8],
    file: &str,
    language: &Language,
    parent_scope: &str,
    chunks: &mut Vec<Chunk>,
) {
    let name = extract_name(node, language, source);
    let scope = build_scope(parent_scope, &name);
    let child_start = node.start_position().row + 1;
    let child_end = node.end_position().row + 1;
    let chunk_start = comments
        .first()
        .map_or(child_start, |c| c.start_position().row + 1);
    let content_start = comments
        .first()
        .map_or(node.start_byte(), |c| c.start_byte());
    let content = std::str::from_utf8(&source[content_start..node.end_byte()])
        .unwrap_or_default()
        .to_string();
    chunks.push(Chunk {
        file: file.to_string(),
        lines: [chunk_start, child_end],
        language: language.clone(),
        scope,
        content,
        kind: node.kind().to_string(),
    });
}

/// Emit a chunk from buffered nodes.
/// Single node → uses its own name as scope.
/// Multiple nodes → merged, uses parent scope.
fn flush_buffer(
    buffer: &[Node],
    source: &[u8],
    file: &str,
    language: &Language,
    parent_scope: &str,
    chunks: &mut Vec<Chunk>,
) {
    if buffer.is_empty() {
        return;
    }

    let first = &buffer[0];
    let last = &buffer[buffer.len() - 1];
    let start_line = first.start_position().row + 1;
    let end_line = last.end_position().row + 1;

    // Content: full text from first node's start to last node's end
    let start_byte = first.start_byte();
    let end_byte = last.end_byte();
    let content = std::str::from_utf8(&source[start_byte..end_byte])
        .unwrap_or_default()
        .to_string();

    if content.trim().is_empty() {
        return;
    }

    let (scope, kind) = if buffer.len() == 1 {
        let name = extract_name(first, language, source);
        (build_scope(parent_scope, &name), first.kind().to_string())
    } else {
        (parent_scope.to_string(), "merged".to_string())
    };

    chunks.push(Chunk {
        file: file.to_string(),
        lines: [start_line, end_line],
        language: language.clone(),
        scope,
        content,
        kind,
    });
}

/// The row of the last actual character in a node.
/// tree-sitter `end_position()` is exclusive: if a node ends with `\n`,
/// end_position points to the start of the next line. Adjust for that.
fn last_content_row(node: &Node) -> usize {
    let end = node.end_position();
    if end.column == 0 {
        end.row.saturating_sub(1)
    } else {
        end.row
    }
}

/// Pull adjacent comment nodes from the end of `buffer` that sit directly above `target`.
/// Walks backwards: each comment must end on the row immediately before the next node starts.
/// Returns comments in source order (first = topmost).
fn take_adjacent_comments<'a>(buffer: &mut Vec<Node<'a>>, target: &Node) -> Vec<Node<'a>> {
    let mut comments = Vec::new();
    let mut expected_row = target.start_position().row;

    while let Some(last) = buffer.last() {
        if !last.kind().contains("comment") {
            break;
        }
        if last_content_row(last) + 1 == expected_row {
            expected_row = last.start_position().row;
            comments.push(buffer.pop().unwrap());
        } else {
            break;
        }
    }

    comments.reverse();
    comments
}

fn build_scope(parent: &str, name: &str) -> String {
    if name.is_empty() {
        parent.to_string()
    } else if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}.{name}")
    }
}

/// Extract chunks for lines not covered by any named node.
fn extract_uncovered_lines(
    chunks: &[Chunk],
    lines: &[&str],
    file: &str,
    language: &Language,
    total_lines: usize,
) -> Vec<Chunk> {
    let mut covered = vec![false; total_lines + 1]; // 1-indexed
    for chunk in chunks {
        let max = chunk.lines[1].min(total_lines);
        for item in covered.iter_mut().take(max + 1).skip(chunk.lines[0]) {
            *item = true;
        }
    }

    let mut gap_chunks = Vec::new();
    let mut gap_start: Option<usize> = None;

    for (idx, &is_covered) in covered.iter().enumerate().take(total_lines + 1).skip(1) {
        let line_text = lines.get(idx - 1).unwrap_or(&"");
        let is_blank = line_text.trim().is_empty();

        if !is_covered && !is_blank {
            if gap_start.is_none() {
                gap_start = Some(idx);
            }
        } else if let Some(start) = gap_start {
            let end = idx - 1;
            let content: String = lines[start - 1..end].join("\n");
            if !content.trim().is_empty() {
                gap_chunks.push(Chunk {
                    file: file.to_string(),
                    lines: [start, end],
                    language: language.clone(),
                    scope: String::new(),
                    content,
                    kind: "gap".to_string(),
                });
            }
            gap_start = None;
        }
    }

    if let Some(start) = gap_start {
        let content: String = lines[start - 1..total_lines].join("\n");
        if !content.trim().is_empty() {
            gap_chunks.push(Chunk {
                file: file.to_string(),
                lines: [start, total_lines],
                language: language.clone(),
                scope: String::new(),
                content,
                kind: "gap".to_string(),
            });
        }
    }

    gap_chunks
}

pub(crate) fn extract_name(node: &Node, language: &Language, source: &[u8]) -> String {
    let kind = node.kind();

    // Handle Python decorated_definition: unwrap to inner definition
    if kind == "decorated_definition" {
        if let Some(inner) = node.child_by_field_name("definition") {
            return extract_name(&inner, language, source);
        }
    }

    // For Rust impl_item, use the "type" field
    if *language == Language::Rust && kind == "impl_item" {
        if let Some(type_node) = node.child_by_field_name("type") {
            return type_node.utf8_text(source).unwrap_or_default().to_string();
        }
    }

    // C/C++ function_definition: name is inside the "declarator" field
    if (*language == Language::C || *language == Language::Cpp) && kind == "function_definition" {
        if let Some(decl) = node.child_by_field_name("declarator") {
            if let Some(inner) = decl.child_by_field_name("declarator") {
                return inner.utf8_text(source).unwrap_or_default().to_string();
            }
            return decl.utf8_text(source).unwrap_or_default().to_string();
        }
    }

    // Scala val/var: name is in the "pattern" field
    if *language == Language::Scala && (kind == "val_definition" || kind == "var_definition") {
        if let Some(pat) = node.child_by_field_name("pattern") {
            return pat.utf8_text(source).unwrap_or_default().to_string();
        }
    }

    // Default: use "name" field
    if let Some(name_node) = node.child_by_field_name("name") {
        return name_node.utf8_text(source).unwrap_or_default().to_string();
    }

    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_scope_containing(chunks: &[Chunk], needle: &str) -> bool {
        chunks.iter().any(|c| c.scope.contains(needle))
    }

    #[test]
    fn test_chunk_rust() {
        let src = r#"
fn hello() {
    println!("hello");
}

struct Config {
    name: String,
}

impl Config {
    fn validate(&self) -> bool {
        !self.name.is_empty()
    }
}
"#;
        let chunks = chunk_code("test.rs", src, Language::Rust).unwrap();
        assert!(!chunks.is_empty());
        // Small file — all nodes under 150 NWS, merged into buffer chunks.
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("fn hello"), "should capture hello fn");
        assert!(all_content.contains("struct Config"), "should capture Config struct");
        assert!(all_content.contains("fn validate"), "should capture validate method");
    }

    #[test]
    fn test_chunk_python() {
        let src = r#"
def greet(name):
    return f"Hello, {name}"

class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"#;
        let chunks = chunk_code("test.py", src, Language::Python).unwrap();
        assert!(!chunks.is_empty());
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("def greet"), "should capture greet");
        assert!(all_content.contains("class Calculator"), "should capture Calculator");
    }

    #[test]
    fn test_chunk_javascript_arrow_functions() {
        let src = r#"
const handler = (req, res) => {
    res.json({ ok: true });
};

function add(a, b) {
    return a + b;
}
"#;
        let chunks = chunk_code("test.js", src, Language::JavaScript).unwrap();
        assert!(!chunks.is_empty());
        let has_handler = chunks.iter().any(|c| c.content.contains("handler"));
        assert!(has_handler, "arrow function should be captured");
    }

    #[test]
    fn test_chunk_go() {
        let src = r#"
package main

func Add(a, b int) int {
    return a + b
}

func Multiply(a, b int) int {
    return a * b
}
"#;
        let chunks = chunk_code("test.go", src, Language::Go).unwrap();
        assert!(!chunks.is_empty());
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("func Add"), "should capture Add");
        assert!(all_content.contains("func Multiply"), "should capture Multiply");
    }

    #[test]
    fn test_small_siblings_merged() {
        // Small adjacent nodes should be merged into one chunk
        let src = "use std::io;\nuse std::fs;\nuse anyhow::Result;\n\nfn main() {}\n";
        let chunks = chunk_code("test.rs", src, Language::Rust).unwrap();
        // All nodes under 150 NWS — everything buffered together
        assert!(chunks.len() <= 2, "small siblings should merge, got {} chunks", chunks.len());
    }

    #[test]
    fn test_big_block_recursed() {
        // A big impl block (>1500 NWS) should be recursed into
        let mut src = String::from("impl BigStruct {\n");
        for i in 0..20 {
            src.push_str(&format!(
                "    fn method_{i}(&self, input: &str) -> Result<String> {{\n\
                 \x20       let value = self.process(input)?;\n\
                 \x20       let transformed = value.to_uppercase();\n\
                 \x20       if transformed.len() > self.max_length {{\n\
                 \x20           return Err(anyhow!(\"too long\"));\n\
                 \x20       }}\n\
                 \x20       Ok(transformed)\n\
                 \x20   }}\n\n"
            ));
        }
        src.push_str("}\n");

        let chunks = chunk_code("test.rs", &src, Language::Rust).unwrap();
        // Should produce multiple chunks (methods), not one blob
        assert!(chunks.len() >= 2, "big block should split, got {} chunks", chunks.len());
    }

    #[test]
    fn test_all_lines_covered() {
        let src = "let x = 42;\nlet y = 100;\n";
        let chunks = chunk_code("test.rs", src, Language::Rust).unwrap();
        assert!(!chunks.is_empty(), "should produce at least 1 chunk");
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("let x = 42"), "should capture all lines");
    }

    #[test]
    fn test_gap_chunks_captured() {
        let src = r#"use std::collections::HashMap;

const MAX_SIZE: usize = 1024;

fn process() {
    println!("processing");
}
"#;
        let chunks = chunk_code("test.rs", src, Language::Rust).unwrap();
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("HashMap"), "should capture imports");
        assert!(all_content.contains("MAX_SIZE"), "should capture constants");
    }

    #[test]
    fn test_doc_comments_attached() {
        // Function must be ≥150 NWS to get its own chunk (and thus doc comment attachment)
        let src = r#"
/// Validates the configuration settings.
/// Returns false if any setting is invalid.
fn validate(config: &Config) -> bool {
    if config.name.is_empty() {
        eprintln!("validation failed: name is empty");
        return false;
    }
    if config.timeout == 0 {
        eprintln!("validation failed: timeout is zero");
        return false;
    }
    if config.max_retries > 100 {
        eprintln!("validation failed: too many retries");
        return false;
    }
    config.name.len() < 256
}

// This is a stray comment (not adjacent to bar)

fn bar() {}
"#;
        let chunks = chunk_code("test.rs", src, Language::Rust).unwrap();
        let validate_chunk = chunks.iter().find(|c| c.scope.contains("validate")).unwrap();
        assert!(
            validate_chunk.content.contains("/// Validates"),
            "doc comment should be included in chunk: {:?}",
            validate_chunk.content
        );
        assert!(
            validate_chunk.content.contains("fn validate"),
            "function body should still be in chunk"
        );
        // The stray comment (separated by blank line) should NOT be in bar's chunk
        let bar_chunk = chunks.iter().find(|c| c.scope.contains("bar"));
        if let Some(bc) = bar_chunk {
            assert!(
                !bc.content.contains("stray comment"),
                "non-adjacent comment should not attach to bar"
            );
        }
    }

    #[test]
    fn test_doc_comments_multiline_python() {
        // Function must be ≥150 NWS to get its own chunk (and thus doc comment attachment)
        let src = r#"
# Computes the sum of two numbers.
# Handles negative values correctly.
def add(a, b):
    if not isinstance(a, (int, float)):
        raise TypeError(f"Expected number, got {type(a)}")
    if not isinstance(b, (int, float)):
        raise TypeError(f"Expected number, got {type(b)}")
    result = a + b
    if result > MAX_VALUE:
        raise OverflowError(f"Result {result} exceeds maximum")
    return result
"#;
        let chunks = chunk_code("test.py", src, Language::Python).unwrap();
        let add_chunk = chunks.iter().find(|c| c.scope.contains("add")).unwrap();
        assert!(
            add_chunk.content.contains("# Computes"),
            "Python comments should attach to function: {:?}",
            add_chunk.content
        );
    }

    #[test]
    fn test_scope_tracks_nesting() {
        // impl block must be >1500 NWS to recurse, giving methods nested scopes
        let mut src = String::from("impl Config {\n");
        for i in 0..15 {
            src.push_str(&format!(
                "    fn check_{i}(&self) -> bool {{\n\
                 \x20       if self.field_{i}.is_empty() {{ return false; }}\n\
                 \x20       if self.field_{i}.len() > self.max_len {{ return false; }}\n\
                 \x20       self.field_{i}.chars().all(|c| c.is_alphanumeric())\n\
                 \x20   }}\n\n"
            ));
        }
        src.push_str("}\n");

        let chunks = chunk_code("test.rs", &src, Language::Rust).unwrap();
        assert!(
            has_scope_containing(&chunks, "Config"),
            "scope should include parent name"
        );
    }

    #[test]
    fn test_small_named_function_buffered() {
        // Small named functions (<150 NWS) should be buffered together
        let src = r#"
fn is_valid(&self) -> bool { self.valid }
fn name(&self) -> &str { &self.name }
fn len(&self) -> usize { self.data.len() }
"#;
        let chunks = chunk_code("test.rs", src, Language::Rust).unwrap();
        // All three are tiny — should be merged into one buffer chunk
        let code_chunks: Vec<_> = chunks.iter().filter(|c| c.kind != "gap").collect();
        assert_eq!(code_chunks.len(), 1, "tiny named functions should merge into one buffer chunk, got {}", code_chunks.len());
        assert!(code_chunks[0].content.contains("is_valid"), "buffer should contain is_valid");
        assert!(code_chunks[0].content.contains("len"), "buffer should contain len");
    }

    #[test]
    fn test_nws_based_splitting() {
        // Verify that a function right at the MIN_OWN_NWS threshold gets its own chunk
        // while one below it gets buffered
        let src = r#"
use std::io;

fn big_enough(config: &Config) -> Result<bool> {
    if config.name.is_empty() {
        eprintln!("name is empty, rejecting config");
        return Ok(false);
    }
    if config.timeout_seconds == 0 {
        eprintln!("timeout is zero, rejecting config");
        return Ok(false);
    }
    Ok(true)
}

fn tiny() -> bool { true }
"#;
        let chunks = chunk_code("test.rs", src, Language::Rust).unwrap();
        // big_enough should get its own chunk with scope
        let big_chunk = chunks.iter().find(|c| c.scope.contains("big_enough"));
        assert!(big_chunk.is_some(), "big_enough should get its own chunk");
        // tiny should be in a buffer chunk (no scope or parent scope)
        // tiny is <150 NWS, but if it's the only node in the buffer it still gets
        // its own scope via flush_buffer single-node logic
        let has_tiny = chunks.iter().any(|c| c.content.contains("fn tiny"));
        assert!(has_tiny, "tiny should still be captured somewhere");
        // Verify they're separate — big_enough shouldn't contain tiny
        if let Some(bc) = big_chunk {
            assert!(!bc.content.contains("fn tiny"), "big_enough chunk should not contain tiny");
        }
    }
}
