use anyhow::Result;
use serde::Serialize;
use tree_sitter::{Node, Parser};

use crate::index::chunk::Language;
use crate::index::chunker::code::extract_name;
use crate::index::chunker::queries::get_config;

/// Default max depth for symbol extraction (root + one level of children).
pub const DEFAULT_DEPTH: usize = 2;

/// A named symbol found in the AST.
#[derive(Debug, Clone, Serialize)]
pub struct SymbolEntry {
    pub name: String,
    /// Language keyword extracted from the AST (e.g. "fn", "def", "class").
    pub kind: String,
    pub line: usize,
    pub end_line: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<SymbolEntry>,
}

/// Extract the language keyword from a node's anonymous children.
/// e.g. `function_item` in Rust has `fn` as its first anonymous child,
/// `function_definition` in Python has `def`, `class_declaration` has `class`.
/// Falls back to the raw tree-sitter kind if no keyword found.
fn node_keyword(node: &Node) -> String {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if !child.is_named() {
                let text = child.kind();
                if !text.is_empty() && text.chars().all(|c| c.is_alphabetic() || c == '_') {
                    return text.to_string();
                }
            }
        }
    }
    node.kind().to_string()
}

/// Extract named symbols from source code by walking the tree-sitter AST directly.
/// Returns a hierarchy: top-level definitions with nested children up to `max_depth`.
/// Non-tree-sitter files return an empty vec.
pub fn extract_symbols(source: &str, language: &Language, max_depth: usize) -> Result<Vec<SymbolEntry>> {
    let config = match get_config(language) {
        Some(c) => c,
        None => return Ok(Vec::new()),
    };

    let mut parser = Parser::new();
    parser.set_language(&config.language)?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("tree-sitter parse failed"))?;

    let source_bytes = source.as_bytes();
    let mut symbols = Vec::new();
    walk_node(&tree.root_node(), language, source_bytes, 0, max_depth, &mut symbols);
    Ok(symbols)
}

/// Recursively walk named children, collecting nodes where `extract_name()` is non-empty.
/// Stops recursing when `depth` reaches `max_depth`.
fn walk_node(
    node: &Node,
    language: &Language,
    source: &[u8],
    depth: usize,
    max_depth: usize,
    symbols: &mut Vec<SymbolEntry>,
) {
    if depth >= max_depth {
        return;
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        // Skip references, values, and params — not definitions.
        // Universal across all tree-sitter grammars:
        //   "identifier" = path segments, type references
        //   "expression" = value construction, function calls
        //   "parameter"  = function/type/lifetime params
        let kind = child.kind();
        if kind.contains("identifier") || kind.contains("expression") || kind.contains("parameter") {
            continue;
        }

        let name = extract_name(&child, language, source);
        if !name.is_empty() {
            let mut entry = SymbolEntry {
                name,
                kind: node_keyword(&child),
                line: child.start_position().row + 1,
                end_line: child.end_position().row + 1,
                children: Vec::new(),
            };
            // Recurse into this node's children for nested definitions
            walk_node(&child, language, source, depth + 1, max_depth, &mut entry.children);
            symbols.push(entry);
        } else {
            // No name but might contain named children (e.g. body wrappers)
            // Don't increment depth — wrappers are transparent
            walk_node(&child, language, source, depth, max_depth, symbols);
        }
    }
}
