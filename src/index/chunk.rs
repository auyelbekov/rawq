use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Language of a chunk. Named variants have tree-sitter grammars.
/// Everything else that can be treated as text is `Other(extension)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Java,
    C,
    Cpp,
    CSharp,
    Ruby,
    Php,
    Swift,
    Bash,
    Lua,
    Scala,
    Dart,
    /// Text file identified by its extension (e.g. "yaml", "sql", "proto", "text").
    Other(String),
}

impl Serialize for Language {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Language {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.parse::<Language>() {
            Ok(lang) => Ok(lang),
            Err(_) => Ok(Language::Other(s)),
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Language::Rust => f.write_str("rust"),
            Language::Python => f.write_str("python"),
            Language::TypeScript => f.write_str("typescript"),
            Language::JavaScript => f.write_str("javascript"),
            Language::Go => f.write_str("go"),
            Language::Java => f.write_str("java"),
            Language::C => f.write_str("c"),
            Language::Cpp => f.write_str("cpp"),
            Language::CSharp => f.write_str("csharp"),
            Language::Ruby => f.write_str("ruby"),
            Language::Php => f.write_str("php"),
            Language::Swift => f.write_str("swift"),
            Language::Bash => f.write_str("bash"),
            Language::Lua => f.write_str("lua"),
            Language::Scala => f.write_str("scala"),
            Language::Dart => f.write_str("dart"),
            Language::Other(s) => f.write_str(s),
        }
    }
}

impl FromStr for Language {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rust" | "rs" => Ok(Language::Rust),
            "python" | "py" => Ok(Language::Python),
            "typescript" | "ts" => Ok(Language::TypeScript),
            "javascript" | "js" => Ok(Language::JavaScript),
            "go" => Ok(Language::Go),
            "java" => Ok(Language::Java),
            "c" => Ok(Language::C),
            "cpp" | "c++" | "cxx" => Ok(Language::Cpp),
            "csharp" | "c#" | "cs" => Ok(Language::CSharp),
            "ruby" | "rb" => Ok(Language::Ruby),
            "php" => Ok(Language::Php),
            "swift" => Ok(Language::Swift),
            "bash" | "sh" | "zsh" => Ok(Language::Bash),
            "lua" => Ok(Language::Lua),
            "scala" => Ok(Language::Scala),
            "dart" => Ok(Language::Dart),
            other => Ok(Language::Other(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub file: String,
    pub lines: [usize; 2],
    pub language: Language,
    pub scope: String,
    pub content: String,
    #[serde(default)]
    pub kind: String,
}
