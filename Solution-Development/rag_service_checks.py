"""
Autograder for the standalone "RAG Service with Real Embeddings" practice lab
(exercises_rag_service_real_embeddings.ipynb).

Unlike the main lab (which uses random simulated embeddings), these exercises
call a real embedding model. Real embedding vectors are not reproducible to
exact values across models/versions, so these checks validate STRUCTURE
(dimensions, types, determinism) and SEMANTIC SANITY (topically related chunks
should be more similar than unrelated ones) rather than fixed expected vectors.

Import as: from tests import rag_service_checks
"""

import math


def _ok(msg: str):
    print(f"✅ {msg}")


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Exercise 1 — parse_knowledge_base
# ---------------------------------------------------------------------------
def check_exercise_1(chunks):
    assert isinstance(chunks, list), "parse_knowledge_base should return a list of chunk dicts."
    assert len(chunks) == 25, (
        f"Expected 25 chunks from data/rag_knowledge_base.txt (entries separated by '---'), got {len(chunks)}. "
        "Make sure you're splitting on the '---' delimiter and skipping empty blocks."
    )

    for c in chunks:
        assert "text" in c and "metadata" in c, "Each chunk needs 'text' and 'metadata' keys."
        meta = c["metadata"]
        for key in ("title", "section", "chunk_id"):
            assert key in meta, f"metadata is missing '{key}'."
        assert isinstance(c["text"], str) and len(c["text"]) > 0, "chunk 'text' should be a non-empty string."
        assert "TITLE:" not in c["text"] and "SECTION:" not in c["text"], (
            "chunk 'text' still contains the 'TITLE:'/'SECTION:' prefix lines — "
            "those should be parsed into metadata, not left in the text body."
        )

    titles = [c["metadata"]["title"] for c in chunks]
    for expected_title in ("RAG Fundamentals", "HNSW Indexing", "RAG Security and Prompt Injection"):
        assert expected_title in titles, f"Expected to find a chunk titled '{expected_title}' in the parsed data."

    hnsw = next(c for c in chunks if c["metadata"]["title"] == "HNSW Indexing")
    assert hnsw["metadata"]["section"] == "algorithm", (
        f"HNSW Indexing chunk should have section 'algorithm', got '{hnsw['metadata']['section']}'."
    )
    assert "Hierarchical Navigable Small World" in hnsw["text"], (
        "HNSW Indexing chunk's text looks wrong — check that you're joining the text lines correctly."
    )

    _ok("Exercise 1 passed — parse_knowledge_base correctly parses all 25 chunks with title/section/text.")


# ---------------------------------------------------------------------------
# Exercise 2 — embed_chunks (real embedding model)
# ---------------------------------------------------------------------------
def check_exercise_2(chunks):
    assert isinstance(chunks, list) and len(chunks) > 0, "chunks should be a non-empty list."
    for c in chunks:
        assert "embedding" in c, (
            "Every chunk should have an 'embedding' key after calling your embed_chunks function."
        )
        emb = c["embedding"]
        assert isinstance(emb, list) and len(emb) > 0, "Each embedding should be a non-empty list of floats."
        assert all(isinstance(x, (int, float)) for x in emb), "Embedding values should all be numbers."

    dims = {len(c["embedding"]) for c in chunks}
    assert len(dims) == 1, f"All embeddings should have the same dimension, found dimensions: {dims}."
    dim = dims.pop()
    assert dim >= 64, (
        f"Embedding dimension is only {dim}, which is suspiciously small for a real sentence embedding model "
        "(e.g. all-MiniLM-L6-v2 produces 384-dim vectors). Did embed_chunks actually call the model?"
    )

    # Rule out a placeholder / constant embedding
    unique_vectors = {tuple(round(x, 6) for x in c["embedding"]) for c in chunks}
    assert len(unique_vectors) > 1, (
        "All chunks have the identical embedding vector — this looks like a placeholder, "
        "not real per-text embeddings. Make sure you're encoding each chunk's own text."
    )

    by_title = {c["metadata"]["title"]: c["embedding"] for c in chunks}
    required = {"RAG Fundamentals", "RAG vs Fine-tuning", "Vector Quantization"}
    missing = required - set(by_title)
    assert not missing, f"Expected chunks (by title) missing from input: {missing}."

    sim_related = _cosine(by_title["RAG Fundamentals"], by_title["RAG vs Fine-tuning"])
    sim_unrelated = _cosine(by_title["RAG Fundamentals"], by_title["Vector Quantization"])

    assert sim_related > sim_unrelated, (
        f"Semantic sanity check failed: cosine similarity between two RAG-topic chunks "
        f"('RAG Fundamentals' vs 'RAG vs Fine-tuning' = {sim_related:.4f}) should be higher than between "
        f"'RAG Fundamentals' and an unrelated chunk ('Vector Quantization' = {sim_unrelated:.4f}). "
        "This usually means the embeddings aren't coming from a real semantic embedding model — "
        "check that embed_chunks is actually calling model.encode(...) on each chunk's text."
    )

    _ok("Exercise 2 passed — embed_chunks produces real, dimensionally-consistent embeddings with sane topic clustering.")


# ---------------------------------------------------------------------------
# Exercise 3 — populate_and_verify_store
# ---------------------------------------------------------------------------
def check_exercise_3(stats, expected_count=25):
    assert isinstance(stats, dict) and "count" in stats, (
        "Pass the dict returned by store.get_stats() (should have a 'count' key)."
    )
    assert stats["count"] == expected_count, (
        f"Expected the collection to contain {expected_count} documents after adding all parsed chunks, "
        f"got {stats['count']}. Did you call store.add_documents(chunks) with the full, embedded chunk list?"
    )
    _ok(f"Exercise 3 passed — ChromaDB collection holds all {expected_count} real-embedded chunks.")


# ---------------------------------------------------------------------------
# Exercise 4 — embed_query (real embedding model, single text)
# ---------------------------------------------------------------------------
def check_exercise_4(query_embedding, doc_embedding_dim, repeat_embedding=None):
    assert isinstance(query_embedding, list) and len(query_embedding) > 0, (
        "embed_query should return a non-empty list of floats."
    )
    assert all(isinstance(x, (int, float)) for x in query_embedding), "Query embedding values should be numbers."
    assert len(query_embedding) == doc_embedding_dim, (
        f"Query embedding dimension ({len(query_embedding)}) should match the document embedding "
        f"dimension ({doc_embedding_dim}) — you must use the SAME model for both queries and documents."
    )

    if repeat_embedding is not None:
        sim = _cosine(query_embedding, repeat_embedding)
        assert sim > 0.999, (
            f"Embedding the same query twice should give (nearly) identical vectors — real embedding models "
            f"are deterministic, unlike the lab's random simulated embeddings. Got cosine similarity {sim:.6f} "
            "between the two calls; expected > 0.999."
        )

    _ok("Exercise 4 passed — embed_query produces a real, dimension-matched, deterministic query embedding.")


# ---------------------------------------------------------------------------
# Exercise 5 — semantic_search (paraphrase test)
# ---------------------------------------------------------------------------
def check_exercise_5(results_by_query, acceptable_titles_by_query):
    assert isinstance(results_by_query, dict), (
        "Pass a dict mapping each test query to its list of search results "
        "(each result a dict with a 'metadata' field, as returned by store.search)."
    )
    for query, acceptable_titles in acceptable_titles_by_query.items():
        assert query in results_by_query, f"No results were provided for query: '{query}'"
        results = results_by_query[query]
        assert isinstance(results, list) and len(results) > 0, f"No results returned for query: '{query}'"

        top_titles = [r["metadata"].get("title") for r in results[:3]]
        assert any(t in acceptable_titles for t in top_titles), (
            f"For the paraphrased query '{query}', none of the top-3 results "
            f"({top_titles}) matched an acceptable topic ({sorted(acceptable_titles)}). "
            "This is the whole point of using real embeddings: they should match on MEANING, "
            "not just shared keywords. If this fails, double check embed_query and embed_chunks "
            "are using the same real model."
        )

    _ok("Exercise 5 passed — semantic search correctly retrieves paraphrased queries by meaning, not keyword overlap.")


# ---------------------------------------------------------------------------
# Exercise 6 — filtered_semantic_search (real-score thresholding)
# ---------------------------------------------------------------------------
def check_exercise_6(loose_results, strict_results, min_score):
    assert isinstance(loose_results, list) and isinstance(strict_results, list), (
        "Both loose_results and strict_results should be lists of result dicts."
    )
    for r in strict_results:
        assert r["score"] >= min_score, (
            f"strict_results contains a result with score {r['score']:.4f}, below min_score={min_score}. "
            "Every returned result must satisfy score >= min_score."
        )
    assert len(strict_results) <= len(loose_results), (
        "Filtering by a minimum score should never return MORE results than the unfiltered search."
    )
    if loose_results and not strict_results:
        assert all(r["score"] < min_score for r in loose_results), (
            "strict_results is empty, but some loose_results actually clear min_score — "
            "they should have been kept."
        )

    _ok("Exercise 6 passed — min_score filtering correctly narrows results using real similarity scores.")
