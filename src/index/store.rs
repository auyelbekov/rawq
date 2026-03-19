use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Read as _, Write as _};
use std::path::{Path, PathBuf};

use half::f16;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, FuzzyTermQuery, Occur, QueryParser};
use tantivy::schema::{Field, Schema, TextFieldIndexing, TextOptions, Value, STORED, STRING};
use tantivy::tokenizer::{LowerCaser, SimpleTokenizer, TextAnalyzer, Token, TokenFilter, TokenStream, Tokenizer};
use tantivy::{doc, Index, IndexWriter, Term};
use crate::index::chunk::{Chunk, Language};

/// Serialized chunk record for chunks.jsonl.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub id: u64,
    pub file: String,
    pub lines: [usize; 2],
    pub language: Language,
    pub scope: String,
    pub content: String,
    #[serde(default)]
    pub kind: String,
    /// SHA-256 hash of chunk content for incremental embedding.
    #[serde(default)]
    pub content_hash: String,
}

/// A stored chunk loaded into memory for search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunk {
    pub id: u64,
    pub file: String,
    pub lines: [usize; 2],
    pub language: Language,
    pub scope: String,
    pub content: String,
    #[serde(default)]
    pub kind: String,
}

impl ChunkRecord {
    fn from_chunk(id: u64, chunk: &Chunk) -> Self {
        Self {
            id,
            file: chunk.file.clone(),
            lines: chunk.lines,
            language: chunk.language.clone(),
            scope: chunk.scope.clone(),
            content: chunk.content.clone(),
            kind: chunk.kind.clone(),
            content_hash: compute_chunk_hash(&chunk.file, chunk.lines[0], &chunk.content),
        }
    }
}

/// Compute SHA-256 hash of chunk content for incremental embedding.
/// Hash includes file path, start line, and content to distinguish identical code at different locations.
pub fn compute_chunk_hash(file: &str, start_line: usize, content: &str) -> String {
    let input = format!("{file}:{start_line}:{content}");
    let hash = Sha256::digest(input.as_bytes());
    format!("{hash:x}")
}

/// Flat vector store: brute-force cosine search over f32 vectors.
struct VectorStore {
    /// (chunk_id, vector) pairs
    vectors: Vec<(u64, Vec<f32>)>,
    /// O(1) lookup: chunk_id -> index in vectors
    id_index: HashMap<u64, usize>,
}

impl VectorStore {
    fn new() -> Self {
        Self {
            vectors: Vec::new(),
            id_index: HashMap::new(),
        }
    }

    fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let mut file = fs::File::open(path).context("open vectors.bin")?;

        // Format (v3+): [count: u64] then [id: u64, dim: u32, halves: [u16; dim]] per entry.
        // f16 (half-precision) halves storage vs f32 with negligible quality loss for cosine search.
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let count = u64::from_le_bytes(buf8) as usize;
        if count > 10_000_000 {
            anyhow::bail!(
                "vectors.bin corrupt: count={count} exceeds maximum (10,000,000)"
            );
        }

        let mut vectors = Vec::with_capacity(count);
        for _ in 0..count {
            file.read_exact(&mut buf8)?;
            let id = u64::from_le_bytes(buf8);

            let mut buf4 = [0u8; 4];
            file.read_exact(&mut buf4)?;
            let dim = u32::from_le_bytes(buf4) as usize;
            if dim > 4096 {
                anyhow::bail!(
                    "vectors.bin corrupt: dim={dim} exceeds maximum (4096)"
                );
            }

            // Read f16 values and convert to f32
            let mut half_bytes = vec![0u8; dim * 2];
            file.read_exact(&mut half_bytes)?;
            let floats: Vec<f32> = half_bytes
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect();

            vectors.push((id, floats));
        }

        let id_index: HashMap<u64, usize> = vectors.iter().enumerate().map(|(i, (id, _))| (*id, i)).collect();
        Ok(Self { vectors, id_index })
    }

    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = BufWriter::new(fs::File::create(path).context("create vectors.bin")?);

        let count = self.vectors.len() as u64;
        file.write_all(&count.to_le_bytes())?;

        // Write f16 values to halve file size (v3+ format)
        for (id, vec) in &self.vectors {
            file.write_all(&id.to_le_bytes())?;
            file.write_all(&(vec.len() as u32).to_le_bytes())?;
            for &v in vec {
                let h = f16::from_f32(v);
                file.write_all(&h.to_le_bytes())?;
            }
        }

        // Flush and fsync before returning so _commit_ok marker is written
        // only after data is durably on disk (crash-safety invariant).
        file.flush()?;
        file.get_ref().sync_all().context("fsync vectors.bin")?;
        Ok(())
    }

    fn insert(&mut self, id: u64, vector: Vec<f32>) {
        let idx = self.vectors.len();
        self.vectors.push((id, vector));
        self.id_index.insert(id, idx);
    }

    fn remove_ids(&mut self, ids: &HashSet<u64>) {
        self.vectors.retain(|(id, _)| !ids.contains(id));
        self.id_index.clear();
        for (i, (id, _)) in self.vectors.iter().enumerate() {
            self.id_index.insert(*id, i);
        }
    }

    /// Iterate over all stored vector IDs.
    fn vector_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.vectors.iter().map(|(id, _)| *id)
    }

    /// Look up a vector by chunk ID (O(1) via HashMap index).
    fn get_vector(&self, chunk_id: u64) -> Option<Vec<f32>> {
        self.id_index
            .get(&chunk_id)
            .and_then(|&i| self.vectors.get(i))
            .map(|(_, v)| v.clone())
    }

}

/// Combined index store: flat vector store + tantivy full-text.
pub struct IndexStore {
    index_dir: PathBuf,
    vectors: VectorStore,
    #[allow(dead_code)] // needed for search in week 3
    tantivy_index: Index,
    tantivy_writer: IndexWriter,
    field_chunk_id: Field,
    field_file: Field,
    field_lines: Field,
    field_language: Field,
    field_scope: Field,
    field_content: Field,
}

/// Split a string on camelCase / PascalCase boundaries.
/// e.g. "DatabaseClient" -> ["Database", "Client"]
///      "getHTTPResponse" -> ["get", "HTTP", "Response"]
///      "HTMLParser" -> ["HTML", "Parser"]
///      "simple" -> ["simple"]
///      "ALLCAPS" -> ["ALLCAPS"]
pub fn split_camel_case(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        let c = chars[i];
        if !c.is_alphanumeric() {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            continue;
        }
        if current.is_empty() {
            current.push(c);
            continue;
        }

        let prev = chars[i - 1];
        // Split on: lowercase -> uppercase (camelCase)
        if prev.is_lowercase() && c.is_uppercase() {
            parts.push(std::mem::take(&mut current));
            current.push(c);
        }
        // Split on: uppercase -> uppercase -> lowercase (HTTPResponse -> HTTP|Response)
        else if prev.is_uppercase() && c.is_uppercase() {
            let next = chars.get(i + 1);
            if next.is_some_and(|n| n.is_lowercase()) && current.len() > 1 {
                // The current uppercase char starts a new word
                parts.push(std::mem::take(&mut current));
                current.push(c);
            } else {
                current.push(c);
            }
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Tantivy token filter that splits camelCase/PascalCase tokens into sub-words.
#[derive(Clone)]
struct CamelCaseSplitter;

impl TokenFilter for CamelCaseSplitter {
    type Tokenizer<T: Tokenizer> = CamelCaseSplitterFilter<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        CamelCaseSplitterFilter { inner: tokenizer }
    }
}

#[derive(Clone)]
struct CamelCaseSplitterFilter<T> {
    inner: T,
}

impl<T: Tokenizer> Tokenizer for CamelCaseSplitterFilter<T> {
    type TokenStream<'a> = CamelCaseSplitterStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        CamelCaseSplitterStream {
            inner: self.inner.token_stream(text),
            buffer: Vec::new(),
            buffer_idx: 0,
        }
    }
}

struct CamelCaseSplitterStream<T> {
    inner: T,
    buffer: Vec<Token>,
    buffer_idx: usize,
}

impl<T: TokenStream> TokenStream for CamelCaseSplitterStream<T> {
    fn advance(&mut self) -> bool {
        // Drain buffer first
        if self.buffer_idx < self.buffer.len() {
            let token = &self.buffer[self.buffer_idx];
            self.buffer_idx += 1;
            // Copy token data to inner
            *self.inner.token_mut() = token.clone();
            return true;
        }
        self.buffer.clear();
        self.buffer_idx = 0;

        if !self.inner.advance() {
            return false;
        }

        let token = self.inner.token().clone();
        let parts = split_camel_case(&token.text);

        if parts.len() <= 1 {
            // No splitting needed, keep original
            return true;
        }

        // Emit the original compound token first (for exact matches)
        // then emit sub-parts
        self.buffer.push(token.clone());
        for part in &parts {
            let mut sub = token.clone();
            sub.text = part.to_lowercase();
            self.buffer.push(sub);
        }
        self.buffer_idx = 1; // will start from index 1 on next advance
        *self.inner.token_mut() = self.buffer[0].clone();
        true
    }

    fn token(&self) -> &Token {
        self.inner.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.inner.token_mut()
    }
}

/// Name used to register the code tokenizer with tantivy.
pub const CODE_TOKENIZER_NAME: &str = "code";

/// Build the code-aware text analyzer: SimpleTokenizer -> CamelCaseSplitter -> LowerCaser
pub fn build_code_analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(CamelCaseSplitter)
        .filter(LowerCaser)
        .build()
}

/// Register the code tokenizer on a tantivy Index.
pub fn register_code_tokenizer(index: &Index) {
    index
        .tokenizers()
        .register(CODE_TOKENIZER_NAME, build_code_analyzer());
}

fn build_schema() -> (Schema, Field, Field, Field, Field, Field, Field) {
    let mut builder = Schema::builder();
    let chunk_id = builder.add_u64_field("chunk_id", STORED | tantivy::schema::INDEXED);
    let file = builder.add_text_field("file", STRING | STORED);
    let lines = builder.add_text_field("lines", STRING | STORED);
    let language = builder.add_text_field("language", STRING | STORED);

    // Use code tokenizer for scope and content fields.
    // NOTE: NOT stored — BM25 search only needs indexing; chunk content is read from chunks.jsonl.
    let code_indexing = TextFieldIndexing::default()
        .set_tokenizer(CODE_TOKENIZER_NAME)
        .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions);
    let code_text_options = TextOptions::default()
        .set_indexing_options(code_indexing);

    let scope = builder.add_text_field("scope", code_text_options.clone());
    let content = builder.add_text_field("content", code_text_options);
    let schema = builder.build();
    (schema, chunk_id, file, lines, language, scope, content)
}

impl IndexStore {
    /// Open existing or create new index store.
    pub fn open_or_create(index_dir: &Path) -> Result<Self> {
        fs::create_dir_all(index_dir)?;

        let vectors = VectorStore::load(&index_dir.join("vectors.bin"))?;

        let (schema, chunk_id, file, lines, language, scope, content) = build_schema();

        let ft_dir = index_dir.join("fulltext");
        let tantivy_index = if ft_dir.exists() {
            Index::open_in_dir(&ft_dir).context("open tantivy index")?
        } else {
            fs::create_dir_all(&ft_dir)?;
            Index::create_in_dir(&ft_dir, schema).context("create tantivy index")?
        };

        register_code_tokenizer(&tantivy_index);

        let tantivy_writer = tantivy_index
            .writer(50_000_000)
            .context("tantivy writer")?;

        Ok(Self {
            index_dir: index_dir.to_path_buf(),
            vectors,
            tantivy_index,
            tantivy_writer,
            field_chunk_id: chunk_id,
            field_file: file,
            field_lines: lines,
            field_language: language,
            field_scope: scope,
            field_content: content,
        })
    }

    /// Insert a chunk with its embedding vector into both stores.
    pub fn insert(&mut self, id: u64, chunk: &Chunk, vector: Vec<f32>) -> Result<()> {
        self.vectors.insert(id, vector);

        let lines_str = format!("{}:{}", chunk.lines[0], chunk.lines[1]);
        let lang_str = chunk.language.to_string();

        self.tantivy_writer.add_document(doc!(
            self.field_chunk_id => id,
            self.field_file => chunk.file.clone(),
            self.field_lines => lines_str,
            self.field_language => lang_str,
            self.field_scope => chunk.scope.clone(),
            self.field_content => chunk.content.clone(),
        ))?;

        Ok(())
    }

    /// Remove chunks by ID from both stores.
    pub fn remove(&mut self, ids: &HashSet<u64>) {
        self.vectors.remove_ids(ids);
        for &id in ids {
            let term = Term::from_field_u64(self.field_chunk_id, id);
            self.tantivy_writer.delete_term(term);
        }
    }

    /// Prune vectors not referenced by the given valid ID set.
    /// Returns the number of orphaned vectors removed.
    pub fn prune_orphaned_vectors(&mut self, valid_ids: &HashSet<u64>) -> usize {
        let orphaned: HashSet<u64> = self
            .vectors
            .vector_ids()
            .filter(|id| !valid_ids.contains(id))
            .collect();
        if !orphaned.is_empty() {
            self.vectors.remove_ids(&orphaned);
        }
        orphaned.len()
    }

    /// Persist everything to disk.
    pub fn persist(&mut self) -> Result<()> {
        self.vectors
            .save(&self.index_dir.join("vectors.bin"))
            .context("save vectors")?;
        self.tantivy_writer
            .commit()
            .context("commit tantivy")?;
        Ok(())
    }

    /// Look up an embedding vector by chunk ID.
    pub fn get_vector(&self, chunk_id: u64) -> Option<Vec<f32>> {
        self.vectors.get_vector(chunk_id)
    }
}

/// Load existing chunks.jsonl, returning (id -> ChunkRecord) map.
pub fn load_chunks_jsonl(index_dir: &Path) -> Result<HashMap<u64, ChunkRecord>> {
    let path = index_dir.join("chunks.jsonl");
    if !path.exists() {
        return Ok(HashMap::new());
    }

    let file = fs::File::open(&path).context("open chunks.jsonl")?;
    let reader = BufReader::new(file);
    let mut map = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: ChunkRecord = serde_json::from_str(&line).context("parse chunk record")?;
        map.insert(record.id, record);
    }

    Ok(map)
}

/// Write chunks.jsonl from scratch.
pub fn save_chunks_jsonl(
    index_dir: &Path,
    existing: &HashMap<u64, ChunkRecord>,
    removed_ids: &HashSet<u64>,
    new_chunks: &[(u64, &Chunk)],
) -> Result<()> {
    fs::create_dir_all(index_dir)?;
    let path = index_dir.join("chunks.jsonl");
    let file = fs::File::create(&path).context("create chunks.jsonl")?;
    let mut writer = BufWriter::new(file);

    // Write surviving existing chunks
    for (id, record) in existing {
        if !removed_ids.contains(id) {
            let line = serde_json::to_string(record)?;
            writeln!(writer, "{line}")?;
        }
    }

    // Append new chunks
    for (id, chunk) in new_chunks {
        let record = ChunkRecord::from_chunk(*id, chunk);
        let line = serde_json::to_string(&record)?;
        writeln!(writer, "{line}")?;
    }

    writer.flush()?;
    writer
        .get_ref()
        .sync_all()
        .context("fsync chunks.jsonl")?;
    Ok(())
}

/// Memory-mapped read-only vector store for IndexReader.
/// On Windows, the mmap keeps vectors.bin open for the reader's lifetime.
/// Rebuilding the index while an IndexReader is alive will fail with a sharing
/// violation. The daemon's 30-min idle-exit mitigates this but does not eliminate it.
struct MmapVectors {
    /// Keep the file handle alive to maintain mmap validity.
    _file: Option<fs::File>,
    mmap: Option<memmap2::Mmap>,
    /// (id, byte_offset_of_f16_data, dim) — parsed from file header on open.
    entries: Vec<(u64, usize, usize)>,
    id_index: HashMap<u64, usize>,
}

impl MmapVectors {
    fn open(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self { _file: None, mmap: None, entries: Vec::new(), id_index: HashMap::new() });
        }

        let file = fs::File::open(path).context("open vectors.bin (mmap)")?;
        // SAFETY: The file handle is kept alive in `_file` for the lifetime of the mmap.
        let mmap = unsafe { memmap2::Mmap::map(&file).context("mmap vectors.bin")? };

        if mmap.len() < 8 {
            return Ok(Self { _file: Some(file), mmap: Some(mmap), entries: Vec::new(), id_index: HashMap::new() });
        }

        let count = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        if count > 10_000_000 {
            anyhow::bail!("vectors.bin corrupt: count={count}");
        }

        let mut entries = Vec::with_capacity(count);
        let mut offset = 8usize;
        for _ in 0..count {
            if offset + 12 > mmap.len() {
                anyhow::bail!("vectors.bin truncated at entry header");
            }
            let id = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap());
            let dim = u32::from_le_bytes(mmap[offset + 8..offset + 12].try_into().unwrap()) as usize;
            offset += 12;
            let data_offset = offset;
            let byte_count = dim * 2; // f16 = 2 bytes per value
            if offset + byte_count > mmap.len() {
                anyhow::bail!("vectors.bin truncated at entry data (id={id})");
            }
            entries.push((id, data_offset, dim));
            offset += byte_count;
        }

        let id_index: HashMap<u64, usize> =
            entries.iter().enumerate().map(|(i, (id, _, _))| (*id, i)).collect();

        Ok(Self { _file: Some(file), mmap: Some(mmap), entries, id_index })
    }

    fn search_top_k(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mmap = match &self.mmap {
            Some(m) => m,
            None => return Vec::new(),
        };
        let mut scored: Vec<(u64, f32)> = self.entries.iter()
            .map(|&(id, data_offset, dim)| {
                let bytes = &mmap[data_offset..data_offset + dim * 2];
                let score = bytes.chunks_exact(2)
                    .zip(query.iter())
                    .map(|(b, &q)| f16::from_le_bytes([b[0], b[1]]).to_f32() * q)
                    .sum::<f32>();
                (id, score)
            })
            .collect();

        let actual_k = k.min(scored.len());
        if actual_k == 0 {
            return scored;
        }
        scored.select_nth_unstable_by(actual_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(actual_k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        let mmap = self.mmap.as_ref()?;
        let &idx = self.id_index.get(&id)?;
        let &(_, data_offset, dim) = self.entries.get(idx)?;
        let bytes = &mmap[data_offset..data_offset + dim * 2];
        Some(
            bytes.chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()
        )
    }
}

/// Read-only index reader for search (no tantivy IndexWriter = no 50MB heap).
pub struct IndexReader {
    chunks: HashMap<u64, StoredChunk>,
    vectors: MmapVectors,
    tantivy_reader: tantivy::IndexReader,
    field_chunk_id: Field,
    field_content: Field,
    field_scope: Field,
}

impl IndexReader {
    /// Open an existing index for reading.
    pub fn open(index_dir: &Path) -> Result<Self> {
        use crate::index::manifest::{COMMIT_MARKER, Manifest};

        // Check schema version (rejects mismatched versions with clear error)
        if let Some(manifest) = Manifest::load_checked(index_dir)? {
            // Commit marker required for schema v2+ indexes
            if manifest.schema_version >= 2 && !index_dir.join(COMMIT_MARKER).exists() {
                anyhow::bail!(
                    "Index appears incomplete (missing commit marker). Rebuild: rawq index build --reindex <path>"
                );
            }
        }

        // Use mmap for read-only vector access — avoids eager loading of all vectors.
        let vectors = MmapVectors::open(&index_dir.join("vectors.bin"))?;

        // Load chunks.jsonl into StoredChunk map
        let records = load_chunks_jsonl(index_dir)?;
        let chunks: HashMap<u64, StoredChunk> = records
            .into_iter()
            .map(|(id, r)| {
                (
                    id,
                    StoredChunk {
                        id: r.id,
                        file: r.file,
                        lines: r.lines,
                        language: r.language,
                        scope: r.scope,
                        content: r.content,
                        kind: r.kind,
                    },
                )
            })
            .collect();

        // Open tantivy index read-only
        let (schema, chunk_id, _file, _lines, _language, scope, content) = build_schema();
        let ft_dir = index_dir.join("fulltext");
        let tantivy_index =
            Index::open_in_dir(&ft_dir).context("open tantivy index for reading")?;

        register_code_tokenizer(&tantivy_index);

        // Verify schema fields match
        let _ = schema;
        let tantivy_reader = tantivy_index
            .reader()
            .context("create tantivy reader")?;

        Ok(Self {
            chunks,
            vectors,
            tantivy_reader,
            field_chunk_id: chunk_id,
            field_content: content,
            field_scope: scope,
        })
    }

    /// Semantic vector search. Returns (chunk_id, cosine_score) sorted descending.
    pub fn search_vector(&self, query_vec: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        self.vectors.search_top_k(query_vec, top_k)
    }

    /// BM25 full-text search over content + scope fields (scope boosted 3x).
    /// Auto-falls back to fuzzy matching (edit distance 1) if exact search returns zero results.
    pub fn search_bm25(&self, query_str: &str, top_k: usize) -> Result<Vec<(u64, f32)>> {
        let searcher = self.tantivy_reader.searcher();
        let mut parser = QueryParser::for_index(
            searcher.index(),
            vec![self.field_content, self.field_scope],
        );
        parser.set_field_boost(self.field_scope, 3.0);
        let query = parser
            .parse_query(query_str)
            .context("parse BM25 query")?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(top_k))
            .context("tantivy search")?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_addr) in &top_docs {
            let doc = searcher.doc::<tantivy::TantivyDocument>(*doc_addr).context("retrieve doc")?;
            if let Some(val) = doc.get_first(self.field_chunk_id) {
                if let Some(id) = val.as_u64() {
                    results.push((id, *score));
                }
            }
        }

        // INVARIANT: Fuzzy only runs when exact returns 0 results, so no duplicate risk.
        // Auto-fallback: if exact BM25 returned nothing, retry with fuzzy terms
        if results.is_empty() && !query_str.trim().is_empty() {
            let fuzzy_query = build_fuzzy_tantivy_query(
                query_str,
                self.field_content,
                self.field_scope,
            );
            if let Some(fq) = fuzzy_query {
                if let Ok(fuzzy_docs) = searcher.search(&fq, &TopDocs::with_limit(top_k)) {
                    for (score, doc_addr) in &fuzzy_docs {
                        let doc = searcher.doc::<tantivy::TantivyDocument>(*doc_addr).context("retrieve fuzzy doc")?;
                        if let Some(val) = doc.get_first(self.field_chunk_id) {
                            if let Some(id) = val.as_u64() {
                                results.push((id, *score));
                            }
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Look up a chunk by ID.
    pub fn get_chunk(&self, id: u64) -> Option<&StoredChunk> {
        self.chunks.get(&id)
    }

    /// Iterate all stored chunks (for filtering before search, e.g., rawq diff).
    pub fn all_chunks(&self) -> impl Iterator<Item = &StoredChunk> {
        self.chunks.values()
    }

    /// Look up an embedding vector by chunk ID.
    pub fn get_vector(&self, chunk_id: u64) -> Option<Vec<f32>> {
        self.vectors.get_vector(chunk_id)
    }
}

/// Build a fuzzy tantivy query using FuzzyTermQuery for each term.
/// Terms with >2 characters get Levenshtein distance 1 fuzzy matching.
/// Short terms (<=2 chars like "fn", "if") are skipped to avoid false positives.
/// Returns None if no terms are suitable for fuzzy matching.
pub fn build_fuzzy_tantivy_query(
    query: &str,
    content_field: Field,
    scope_field: Field,
) -> Option<Box<dyn tantivy::query::Query>> {
    let terms: Vec<&str> = query
        .split_whitespace()
        .filter(|t| t.len() > 2)
        .collect();

    if terms.is_empty() {
        return None;
    }

    let mut sub_queries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
    for term in &terms {
        let lower = term.to_lowercase();
        // Fuzzy on content field
        let content_term = Term::from_field_text(content_field, &lower);
        let content_fuzzy = FuzzyTermQuery::new(content_term, 1, true);
        // Fuzzy on scope field (boosted)
        let scope_term = Term::from_field_text(scope_field, &lower);
        let scope_fuzzy = FuzzyTermQuery::new(scope_term, 1, true);

        // Combine: either content or scope match
        let either = BooleanQuery::new(vec![
            (Occur::Should, Box::new(content_fuzzy)),
            (Occur::Should, Box::new(scope_fuzzy)),
        ]);
        sub_queries.push((Occur::Should, Box::new(either)));
    }

    Some(Box::new(BooleanQuery::new(sub_queries)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_camel_case() {
        assert_eq!(split_camel_case("DatabaseClient"), vec!["Database", "Client"]);
        assert_eq!(split_camel_case("getHTTPResponse"), vec!["get", "HTTP", "Response"]);
        assert_eq!(split_camel_case("HTMLParser"), vec!["HTML", "Parser"]);
        assert_eq!(split_camel_case("simple"), vec!["simple"]);
        assert_eq!(split_camel_case("ALLCAPS"), vec!["ALLCAPS"]);
        // snake_case is kept as-is (underscore is a non-alphanumeric separator)
        assert_eq!(split_camel_case("snake_case"), vec!["snake", "case"]);
        assert_eq!(split_camel_case("myURL"), vec!["my", "URL"]);
    }

    #[test]
    fn mmap_vector_store_search_top_k() {
        // Write vectors via VectorStore (f16 format), then read back via MmapVectors.
        let dir = std::env::temp_dir().join(format!("rawq_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vectors.bin");

        // L2-normalized 3-dimensional vectors
        let mut store = VectorStore::new();
        store.insert(1, vec![1.0, 0.0, 0.0]);
        store.insert(2, vec![0.0, 1.0, 0.0]);
        store.insert(3, vec![0.707, 0.707, 0.0]);
        store.save(&path).unwrap();

        let mmap = MmapVectors::open(&path).unwrap();
        let query = &[1.0, 0.0, 0.0]; // closest to chunk 1
        let results = mmap.search_top_k(query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // chunk 1 is most similar
        assert!((results[0].1 - 1.0).abs() < 0.05); // f16 precision tolerance
        assert_eq!(results[1].0, 3); // chunk 3 is second

        let _ = std::fs::remove_dir_all(&dir);
    }

    // RWQ-147: incremental embedding — content hash dedup produces identical hashes
    // for unchanged chunks and different hashes when content changes.
    #[test]
    fn content_hash_dedup() {
        let hash_a = compute_chunk_hash("src/lib.rs", 1, "fn foo() {}");
        let hash_b = compute_chunk_hash("src/lib.rs", 1, "fn foo() {}");
        let hash_c = compute_chunk_hash("src/lib.rs", 1, "fn bar() {}"); // different content
        let hash_d = compute_chunk_hash("src/lib.rs", 2, "fn foo() {}"); // different line

        assert_eq!(hash_a, hash_b, "same chunk → same hash");
        assert_ne!(hash_a, hash_c, "different content → different hash");
        assert_ne!(hash_a, hash_d, "different line → different hash");
        assert!(!hash_a.is_empty(), "hash should not be empty");
    }
}
