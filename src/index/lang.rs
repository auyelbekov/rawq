use std::path::Path;

use crate::index::chunk::Language;

pub fn detect_language(path: &Path) -> Option<Language> {
    let ext = match path.extension().and_then(|e| e.to_str()) {
        Some(e) => e.to_lowercase(),
        None => return Some(Language::Other("text".to_string())),
    };
    match ext.as_str() {
        // Tree-sitter languages
        "rs" => Some(Language::Rust),
        "py" => Some(Language::Python),
        "ts" | "tsx" => Some(Language::TypeScript),
        "js" | "jsx" => Some(Language::JavaScript),
        "go" => Some(Language::Go),
        "java" => Some(Language::Java),
        "c" | "h" => Some(Language::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" => Some(Language::Cpp),
        "cs" => Some(Language::CSharp),
        "rb" => Some(Language::Ruby),
        "php" => Some(Language::Php),
        "swift" => Some(Language::Swift),
        "sh" | "bash" | "zsh" => Some(Language::Bash),
        "lua" => Some(Language::Lua),
        "scala" | "sc" => Some(Language::Scala),
        "dart" => Some(Language::Dart),
        // Known binary formats — skip entirely
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "ico" | "webp" | "svg" |
        "exe" | "dll" | "so" | "dylib" | "bin" |
        "zip" | "tar" | "gz" | "xz" | "7z" | "rar" | "bz2" |
        "woff" | "woff2" | "ttf" | "eot" | "otf" |
        "pdf" | "doc" | "docx" | "pptx" | "xlsx" |
        "o" | "obj" | "a" | "lib" | "pyc" | "pyo" | "class" |
        "wasm" | "map" | "min.js" | "min.css" |
        "mp3" | "mp4" | "wav" | "avi" | "mov" | "mkv" |
        "db" | "sqlite" | "sqlite3" => None,
        // Everything else — treat as text, label with extension
        _ => Some(Language::Other(ext)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        // Tree-sitter languages
        assert_eq!(detect_language(Path::new("foo.rs")), Some(Language::Rust));
        assert_eq!(detect_language(Path::new("bar.py")), Some(Language::Python));
        assert_eq!(detect_language(Path::new("baz.ts")), Some(Language::TypeScript));
        assert_eq!(detect_language(Path::new("app.js")), Some(Language::JavaScript));
        assert_eq!(detect_language(Path::new("main.go")), Some(Language::Go));
        assert_eq!(detect_language(Path::new("Main.java")), Some(Language::Java));
        assert_eq!(detect_language(Path::new("lib.c")), Some(Language::C));
        assert_eq!(detect_language(Path::new("lib.cpp")), Some(Language::Cpp));
        assert_eq!(detect_language(Path::new("Prog.cs")), Some(Language::CSharp));
        assert_eq!(detect_language(Path::new("app.rb")), Some(Language::Ruby));
        assert_eq!(detect_language(Path::new("index.php")), Some(Language::Php));
        assert_eq!(detect_language(Path::new("App.swift")), Some(Language::Swift));
        assert_eq!(detect_language(Path::new("run.sh")), Some(Language::Bash));
        assert_eq!(detect_language(Path::new("script.lua")), Some(Language::Lua));
        assert_eq!(detect_language(Path::new("Main.scala")), Some(Language::Scala));
        assert_eq!(detect_language(Path::new("main.dart")), Some(Language::Dart));

        // Binary extensions return None
        assert_eq!(detect_language(Path::new("image.png")), None);
        assert_eq!(detect_language(Path::new("app.exe")), None);
        assert_eq!(detect_language(Path::new("data.pdf")), None);

        // No-extension files → text
        assert_eq!(detect_language(Path::new("Makefile")), Some(Language::Other("text".to_string())));

        // Non-tree-sitter text files → Other with their extension
        assert_eq!(detect_language(Path::new("query.sql")), Some(Language::Other("sql".to_string())));
        assert_eq!(detect_language(Path::new("config.yaml")), Some(Language::Other("yaml".to_string())));
        assert_eq!(detect_language(Path::new("Main.kt")), Some(Language::Other("kt".to_string())));
        assert_eq!(detect_language(Path::new("Cargo.toml")), Some(Language::Other("toml".to_string())));
        assert_eq!(detect_language(Path::new("data.json")), Some(Language::Other("json".to_string())));
        assert_eq!(detect_language(Path::new("readme.md")), Some(Language::Other("md".to_string())));
        assert_eq!(detect_language(Path::new("schema.proto")), Some(Language::Other("proto".to_string())));
        assert_eq!(detect_language(Path::new("main.tf")), Some(Language::Other("tf".to_string())));
    }
}
