use crate::index::chunk::Language;

pub struct LanguageConfig {
    pub language: tree_sitter::Language,
}

pub fn get_config(lang: &Language) -> Option<LanguageConfig> {
    match lang {
        Language::Rust => Some(LanguageConfig {
            language: tree_sitter_rust::LANGUAGE.into(),
        }),
        Language::Python => Some(LanguageConfig {
            language: tree_sitter_python::LANGUAGE.into(),
        }),
        Language::TypeScript => Some(LanguageConfig {
            language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        }),
        Language::JavaScript => Some(LanguageConfig {
            language: tree_sitter_javascript::LANGUAGE.into(),
        }),
        Language::Go => Some(LanguageConfig {
            language: tree_sitter_go::LANGUAGE.into(),
        }),
        Language::Java => Some(LanguageConfig {
            language: tree_sitter_java::LANGUAGE.into(),
        }),
        Language::C => Some(LanguageConfig {
            language: tree_sitter_c::LANGUAGE.into(),
        }),
        Language::Cpp => Some(LanguageConfig {
            language: tree_sitter_cpp::LANGUAGE.into(),
        }),
        Language::CSharp => Some(LanguageConfig {
            language: tree_sitter_c_sharp::LANGUAGE.into(),
        }),
        Language::Ruby => Some(LanguageConfig {
            language: tree_sitter_ruby::LANGUAGE.into(),
        }),
        Language::Php => Some(LanguageConfig {
            language: tree_sitter_php::LANGUAGE_PHP.into(),
        }),
        Language::Swift => Some(LanguageConfig {
            language: tree_sitter_swift::LANGUAGE.into(),
        }),
        Language::Bash => Some(LanguageConfig {
            language: tree_sitter_bash::LANGUAGE.into(),
        }),
        Language::Lua => Some(LanguageConfig {
            language: tree_sitter_lua::LANGUAGE.into(),
        }),
        Language::Scala => Some(LanguageConfig {
            language: tree_sitter_scala::LANGUAGE.into(),
        }),
        Language::Dart => Some(LanguageConfig {
            language: tree_sitter_dart::LANGUAGE.into(),
        }),
        Language::Other(_) => None,
    }
}
