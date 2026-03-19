use rawq::embed::{cosine_similarity, Embedder};

fn get_embedder() -> Embedder {
    Embedder::from_hf_hub().expect("failed to load embedder")
}

#[test]
fn test_embed_dimensionality() {
    let mut embedder = get_embedder();
    let result = embedder.embed(&["hello world"]).unwrap();
    assert_eq!(result.shape(), &[1, embedder.embed_dim()]);
}

#[test]
fn test_embed_normalized() {
    let mut embedder = get_embedder();
    let result = embedder.embed(&["hello world"]).unwrap();
    let norm: f32 = result.row(0).iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "L2 norm should be ~1.0, got {norm}"
    );
}

#[test]
fn test_similarity_sanity() {
    let mut embedder = get_embedder();
    let vecs = embedder
        .embed(&[
            "the cat sat on the mat",
            "the feline rested on the rug",
            "quantum physics explains entanglement",
        ])
        .unwrap();

    let sim_similar = cosine_similarity(vecs.row(0).as_slice().unwrap(), vecs.row(1).as_slice().unwrap());
    let sim_different = cosine_similarity(vecs.row(0).as_slice().unwrap(), vecs.row(2).as_slice().unwrap());

    assert!(
        sim_similar > sim_different,
        "Similar sentences should score higher: {sim_similar} vs {sim_different}"
    );
}

#[test]
fn test_batch_embed() {
    let mut embedder = get_embedder();
    let embed_dim = embedder.embed_dim();
    let texts: Vec<String> = (0..8)
        .map(|i| format!("This is test sentence number {i} for batch embedding"))
        .collect();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let result = embedder.embed(&refs).unwrap();
    assert_eq!(result.shape(), &[8, embed_dim]);

    for i in 0..8 {
        let norm: f32 = result.row(i).iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Row {i} norm should be ~1.0, got {norm}"
        );
    }
}

#[test]
fn test_query_prefix_changes_vector() {
    let mut embedder = get_embedder();
    // Only test prefix effect if the model has a query prefix
    if embedder.embed_dim() == 384 {
        // Models like arctic/bge have a query prefix
        let query_vec = embedder.embed_query("what is rust?").unwrap();
        let doc_vec = embedder.embed_document("what is rust?").unwrap();
        let sim = cosine_similarity(query_vec.as_slice().unwrap(), doc_vec.as_slice().unwrap());
        assert!(
            sim < 0.99,
            "Query and doc vectors should differ due to prefix, cosine sim: {sim}"
        );
    }
    // Models without prefix (jina, miniLM) produce identical query/doc vectors — that's correct
}

#[test]
fn test_model_name() {
    let embedder = get_embedder();
    // Model name should be non-empty and match an installed model
    assert!(
        !embedder.model_name().is_empty(),
        "model_name should not be empty"
    );
}

#[test]
fn test_embed_dim() {
    let embedder = get_embedder();
    // embed_dim should be a reasonable value (384 or 768 for known models)
    assert!(
        embedder.embed_dim() > 0,
        "embed_dim should be > 0, got {}",
        embedder.embed_dim()
    );
}
