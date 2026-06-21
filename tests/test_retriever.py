"""
Tests for the retrieval pipeline (Milestone 3).

Markers:
  integration — requires a running Qdrant instance (localhost:6333)
                and both documents indexed (mohasaba + qawaim)
  slow        — additionally loads BGE-M3 and bge-reranker-v2-m3

Run all:          pytest tests/test_retriever.py -m "integration and slow"
Run fast subset:  pytest tests/test_retriever.py -m "integration and not slow"
"""

import pytest

from diwan.retrieval.retriever import (
    POST_RERANK_K,
    PRE_RERANK_K,
    _get_embed_model,
    _get_qdrant,
    _get_reranker,
    retrieve,
)

ARABIC_QUERY     = "ما هي شروط الاعتراف بالإيراد؟"
ARABIC_PURPOSE   = "تحديد معايير الاعتراف بالإيراد في المعايير المحاسبية"
FINANCIAL_QUERY  = "صافي الربح للعام"
FINANCIAL_PURPOSE = "استخراج صافي الربح من القوائم المالية"

REQUIRED_CHUNK_FIELDS = {"chunk_id", "doc_name", "chunk_type", "markdown", "hook", "bboxes", "score"}
REQUIRED_OUTPUT_FIELDS = {"sub_query", "purpose", "chunks"}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mohasaba_result():
    """Session-scoped: load models once, run one mohasaba query."""
    return retrieve(ARABIC_QUERY, ARABIC_PURPOSE)


@pytest.fixture(scope="session")
def financial_result():
    """Session-scoped: reuses already-loaded models for a qawaim query."""
    return retrieve(FINANCIAL_QUERY, FINANCIAL_PURPOSE)


# ── Output structure ──────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
def test_output_has_required_keys(mohasaba_result):
    assert REQUIRED_OUTPUT_FIELDS == mohasaba_result.keys(), (
        f"Missing keys: {REQUIRED_OUTPUT_FIELDS - mohasaba_result.keys()}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_sub_query_passthrough(mohasaba_result):
    assert mohasaba_result["sub_query"] == ARABIC_QUERY


@pytest.mark.integration
@pytest.mark.slow
def test_purpose_passthrough(mohasaba_result):
    assert mohasaba_result["purpose"] == ARABIC_PURPOSE


@pytest.mark.integration
@pytest.mark.slow
def test_chunks_is_list(mohasaba_result):
    assert isinstance(mohasaba_result["chunks"], list)


# ── k-limits ─────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
def test_at_most_post_rerank_k_chunks(mohasaba_result):
    assert len(mohasaba_result["chunks"]) <= POST_RERANK_K, (
        f"Expected at most {POST_RERANK_K} chunks, got {len(mohasaba_result['chunks'])}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_returns_nonzero_chunks(mohasaba_result):
    assert len(mohasaba_result["chunks"]) > 0, "Query returned no chunks"


# ── Chunk payload ─────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
def test_chunk_fields_present(mohasaba_result):
    for chunk in mohasaba_result["chunks"]:
        missing = REQUIRED_CHUNK_FIELDS - chunk.keys()
        assert not missing, f"Chunk missing fields: {missing}"


@pytest.mark.integration
@pytest.mark.slow
def test_no_empty_markdown(mohasaba_result):
    for chunk in mohasaba_result["chunks"]:
        assert chunk["markdown"].strip(), f"Chunk {chunk['chunk_id']} has empty markdown"


@pytest.mark.integration
@pytest.mark.slow
def test_chunk_type_valid(mohasaba_result):
    valid = {"text", "table", "figure", "mixed"}
    for chunk in mohasaba_result["chunks"]:
        assert chunk["chunk_type"] in valid, (
            f"Chunk {chunk['chunk_id']} has invalid chunk_type: {chunk['chunk_type']}"
        )


@pytest.mark.integration
@pytest.mark.slow
def test_bboxes_non_empty(mohasaba_result):
    for chunk in mohasaba_result["chunks"]:
        assert chunk["bboxes"], f"Chunk {chunk['chunk_id']} has empty bboxes"


@pytest.mark.integration
@pytest.mark.slow
def test_doc_name_non_empty(mohasaba_result):
    for chunk in mohasaba_result["chunks"]:
        assert chunk["doc_name"], f"Chunk {chunk['chunk_id']} has empty doc_name"


# ── Reranker scores ───────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
def test_scores_normalized(mohasaba_result):
    for chunk in mohasaba_result["chunks"]:
        assert 0.0 <= chunk["score"] <= 1.0, (
            f"Score out of [0, 1] range: {chunk['score']}"
        )


@pytest.mark.integration
@pytest.mark.slow
def test_scores_descending(mohasaba_result):
    scores = [c["score"] for c in mohasaba_result["chunks"]]
    assert scores == sorted(scores, reverse=True), (
        f"Chunks are not sorted by score (desc): {scores}"
    )


# ── Cross-document retrieval ──────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
def test_financial_query_returns_results(financial_result):
    assert len(financial_result["chunks"]) > 0, "Financial query returned no chunks"


@pytest.mark.integration
@pytest.mark.slow
def test_different_queries_differ(mohasaba_result, financial_result):
    ids_a = {c["chunk_id"] for c in mohasaba_result["chunks"]}
    ids_b = {c["chunk_id"] for c in financial_result["chunks"]}
    assert ids_a != ids_b, "Two unrelated queries returned identical chunk sets"


# ── Singleton loading ─────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
def test_embed_model_singleton(mohasaba_result):
    m1 = _get_embed_model()
    m2 = _get_embed_model()
    assert m1 is m2, "Embed model singleton returned different objects"


@pytest.mark.integration
@pytest.mark.slow
def test_reranker_singleton(mohasaba_result):
    r1 = _get_reranker()
    r2 = _get_reranker()
    assert r1 is r2, "Reranker singleton returned different objects"


@pytest.mark.integration
def test_qdrant_singleton():
    c1 = _get_qdrant()
    c2 = _get_qdrant()
    assert c1 is c2, "Qdrant client singleton returned different objects"


# ── Dense vector dimension sanity check ──────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
def test_dense_vector_dimension():
    model = _get_embed_model()
    out = model.encode(
        ["test"],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    assert out["dense_vecs"].shape[1] == 1024, (
        f"Expected 1024-dim dense vector, got {out['dense_vecs'].shape[1]}"
    )
