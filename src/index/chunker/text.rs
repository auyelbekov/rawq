use crate::index::chunk::{Chunk, Language};

/// Maximum non-whitespace characters per text chunk.
const MAX_NWS: usize = 1500;

pub fn chunk_text(file: &str, source: &str, is_markdown: bool, language: &Language) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();
    let mut current_start: usize = 1;
    let mut current_heading = String::new();
    let mut current_nws: usize = 0;

    let lines: Vec<&str> = source.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        let line_num = i + 1;
        let line_nws = line.bytes().filter(|b| !b.is_ascii_whitespace()).count();

        if is_markdown && line.starts_with('#') {
            // Flush current chunk before heading
            flush_chunk(
                &mut chunks,
                file,
                language,
                &current_lines,
                current_start,
                &current_heading,
            );
            current_heading = line.trim_start_matches('#').trim().to_string();
            current_lines.clear();
            current_nws = 0;
            current_start = line_num;
            current_lines.push(line);
            current_nws += line_nws;
        } else if line.trim().is_empty() {
            // Paragraph break: flush current chunk
            flush_chunk(
                &mut chunks,
                file,
                language,
                &current_lines,
                current_start,
                &current_heading,
            );
            current_lines.clear();
            current_nws = 0;
            current_start = line_num + 1;
        } else {
            // NWS budget check: would adding this line exceed MAX_NWS?
            if current_nws + line_nws > MAX_NWS && !current_lines.is_empty() {
                flush_chunk(
                    &mut chunks,
                    file,
                    language,
                    &current_lines,
                    current_start,
                    &current_heading,
                );
                current_lines.clear();
                current_nws = 0;
                current_start = line_num;
            }
            if current_lines.is_empty() {
                current_start = line_num;
            }
            current_lines.push(line);
            current_nws += line_nws;
        }
    }

    // Flush remaining
    flush_chunk(
        &mut chunks,
        file,
        language,
        &current_lines,
        current_start,
        &current_heading,
    );

    chunks
}

fn flush_chunk(
    chunks: &mut Vec<Chunk>,
    file: &str,
    language: &Language,
    lines: &[&str],
    start: usize,
    heading: &str,
) {
    if lines.is_empty() {
        return;
    }
    let content = lines.join("\n");
    if content.trim().is_empty() {
        return;
    }
    let end = start + lines.len() - 1;
    chunks.push(Chunk {
        file: file.to_string(),
        lines: [start, end],
        language: language.clone(),
        scope: heading.to_string(),
        content,
        kind: "text".to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paragraph_split() {
        let source = "First paragraph line 1.\nFirst paragraph line 2.\n\nSecond paragraph.";
        let lang = Language::Other("md".to_string());
        let chunks = chunk_text("test.md", source, true, &lang);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].lines, [1, 2]);
        assert_eq!(chunks[1].lines, [4, 4]);
    }

    #[test]
    fn test_heading_scope() {
        let source = "# Introduction\n\nSome text here.\n\n# Details\n\nMore text.";
        let lang = Language::Other("md".to_string());
        let chunks = chunk_text("test.md", source, true, &lang);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].scope, "Introduction");
        assert_eq!(chunks[1].scope, "Introduction");
        assert_eq!(chunks[2].scope, "Details");
        assert_eq!(chunks[3].scope, "Details");
    }

    #[test]
    fn test_preserves_language() {
        let source = "SELECT * FROM users;";
        let lang = Language::Other("sql".to_string());
        let chunks = chunk_text("query.sql", source, false, &lang);
        assert_eq!(chunks[0].language, Language::Other("sql".to_string()));
    }

    #[test]
    fn test_nws_budget_splits_long_paragraph() {
        // Generate a single paragraph (no blank lines) that exceeds 1500 NWS chars
        let line = "abcdefghij abcdefghij abcdefghij abcdefghij abcdefghij"; // 50 NWS per line
        let lines: Vec<&str> = std::iter::repeat(line).take(40).collect(); // 40 * 50 = 2000 NWS
        let source = lines.join("\n");
        let lang = Language::Other("txt".to_string());
        let chunks = chunk_text("big.txt", &source, false, &lang);
        assert!(chunks.len() >= 2, "long paragraph should split at NWS budget, got {} chunks", chunks.len());
        // All content should be preserved
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert_eq!(all_content.matches("abcdefghij").count(), source.matches("abcdefghij").count());
    }
}
