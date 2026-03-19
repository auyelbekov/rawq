pub mod model_source;
pub mod engine;

pub use model_source::{ModelSource, resolve_model_source, embed_query_via_source};
pub use engine::{SearchMode, SearchOptions, SearchOutput, SearchResult, search_content, search_with_reader};
pub use crate::index::{Chunk, Language};

// Note: `IndexReader` from `rawq_index` is used in `search_with_reader()`.
// This is a deliberate dependency — rawq-core provides search logic while
// rawq-index owns the storage layer. A future `ChunkView` type in rawq-core
// would decouple the public API from rawq-index internals.
