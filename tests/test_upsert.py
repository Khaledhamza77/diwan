# -*- coding: utf-8 -*-
"""
Smoke tests for the Qdrant upsert pipeline — both documents.

Covers:
  Milestone 2A — mohasaba (معايير_المحاسبة_المصرية_2020): 152 pre-chunked sections
  Milestone 2B — qawaim (CIB Q1 2026 financial statements): 62 LLM-chunked sections

Markers:
  integration — requires a running Qdrant instance (localhost:6333)
  slow        — additionally loads BGE-M3 for live hybrid-search tests

Run all fast:   pytest tests/test_upsert.py -m "integration and not slow"
Run all:        pytest tests/test_upsert.py -m integration
"""

import re

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

COLLECTION = "diwan"

MOHASABA_DOC      = "معايير_المحاسبة_المصرية_2020"
QAWAIM_DOC        = "qawaim"
KNOWN_DOC_NAMES   = {MOHASABA_DOC, QAWAIM_DOC}
KNOWN_ID_PREFIXES = {"mohasaba__", "qawaim__"}

MOHASABA_EXPECTED = 152
QAWAIM_EXPECTED   = 62
TOTAL_EXPECTED    = MOHASABA_EXPECTED + QAWAIM_EXPECTED

ANCHOR_RE = re.compile(r"<a\s+id='[^']*'></a>")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qdrant():
    return QdrantClient(url="http://localhost:6333")


@pytest.fixture(scope="session")
def all_points(qdrant):
    """Scroll the entire collection — handles any point count via pagination."""
    collected = []
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        collected.extend(batch)
        if offset is None:
            break
    return collected


@pytest.fixture(scope="session")
def mohasaba_points(all_points):
    return [pt for pt in all_points if pt.payload.get("doc_name") == MOHASABA_DOC]


@pytest.fixture(scope="session")
def qawaim_points(all_points):
    return [pt for pt in all_points if pt.payload.get("doc_name") == QAWAIM_DOC]


@pytest.fixture(scope="session")
def embed_model():
    import torch
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=torch.cuda.is_available())


# ── Collection-level tests ────────────────────────────────────────────────────

@pytest.mark.integration
def test_collection_exists(qdrant):
    names = {c.name for c in qdrant.get_collections().collections}
    assert COLLECTION in names, f"Collection '{COLLECTION}' not found in Qdrant"


@pytest.mark.integration
def test_total_point_count(qdrant):
    count = qdrant.count(collection_name=COLLECTION).count
    assert count == TOTAL_EXPECTED, f"Expected {TOTAL_EXPECTED} points, got {count}"


@pytest.mark.integration
def test_mohasaba_point_count(mohasaba_points):
    assert len(mohasaba_points) == MOHASABA_EXPECTED, (
        f"Expected {MOHASABA_EXPECTED} mohasaba points, got {len(mohasaba_points)}"
    )


@pytest.mark.integration
def test_qawaim_point_count(qawaim_points):
    assert len(qawaim_points) == QAWAIM_EXPECTED, (
        f"Expected {QAWAIM_EXPECTED} qawaim points, got {len(qawaim_points)}"
    )


@pytest.mark.integration
def test_collection_config(qdrant):
    info    = qdrant.get_collection(COLLECTION)
    vectors = info.config.params.vectors
    sparse  = info.config.params.sparse_vectors

    assert "dense" in vectors, "Missing 'dense' named vector"
    assert vectors["dense"].size == 1024, f"Expected 1024-dim dense, got {vectors['dense'].size}"
    assert str(vectors["dense"].distance).lower() == "cosine", "Dense vector distance should be cosine"
    assert "sparse" in sparse, "Missing 'sparse' named vector"


# ── Payload tests — all points ────────────────────────────────────────────────

@pytest.mark.integration
def test_payload_fields_present(all_points):
    required = {"chunk_id", "doc_name", "chunk_type", "markdown", "hook", "bboxes"}
    for pt in all_points:
        missing = required - pt.payload.keys()
        assert not missing, f"Point {pt.id} missing payload fields: {missing}"


@pytest.mark.integration
def test_doc_names_valid(all_points):
    wrong = [
        (pt.id, pt.payload.get("doc_name"))
        for pt in all_points
        if pt.payload.get("doc_name") not in KNOWN_DOC_NAMES
    ]
    assert not wrong, f"{len(wrong)} points have unknown doc_name: {wrong[:3]}"


@pytest.mark.integration
def test_chunk_id_prefixes_valid(all_points):
    wrong = [
        (pt.id, pt.payload.get("chunk_id"))
        for pt in all_points
        if not any(pt.payload.get("chunk_id", "").startswith(p) for p in KNOWN_ID_PREFIXES)
    ]
    assert not wrong, f"{len(wrong)} points have unrecognised chunk_id prefix: {wrong[:3]}"


@pytest.mark.integration
def test_chunk_type_values(all_points):
    valid = {"text", "table", "figure", "mixed"}
    invalid = [
        (pt.id, pt.payload.get("chunk_type"))
        for pt in all_points
        if pt.payload.get("chunk_type") not in valid
    ]
    assert not invalid, f"Invalid chunk_type values: {invalid[:5]}"


@pytest.mark.integration
def test_no_empty_markdown(all_points):
    empty = [pt.id for pt in all_points if not pt.payload.get("markdown", "").strip()]
    assert not empty, f"{len(empty)} points have empty markdown: {empty}"


@pytest.mark.integration
def test_no_anchor_leaks(all_points):
    leaks = [pt.id for pt in all_points if ANCHOR_RE.search(pt.payload.get("markdown", ""))]
    assert not leaks, f"{len(leaks)} points have anchor tags in markdown: {leaks[:3]}"


@pytest.mark.integration
def test_bboxes_non_empty(all_points):
    empty = [pt.id for pt in all_points if not pt.payload.get("bboxes")]
    assert not empty, f"{len(empty)} points have empty bboxes: {empty}"


@pytest.mark.integration
def test_bboxes_schema(all_points):
    for pt in all_points:
        for bbox in pt.payload.get("bboxes", []):
            assert "page" in bbox, f"Point {pt.id}: bbox missing 'page'"
            assert "box" in bbox,  f"Point {pt.id}: bbox missing 'box'"
            for key in ("top", "bottom", "left", "right"):
                assert key in bbox["box"], f"Point {pt.id}: box missing '{key}'"


# ── Payload tests — per-document ──────────────────────────────────────────────

@pytest.mark.integration
def test_mohasaba_hooks_present(mohasaba_points):
    """Every mohasaba point must have a non-empty hook (pre-written retrieval hint)."""
    missing = [pt.id for pt in mohasaba_points if not pt.payload.get("hook")]
    assert not missing, f"{len(missing)} mohasaba points missing hook: {missing[:3]}"


@pytest.mark.integration
def test_qawaim_hook_is_null(qawaim_points):
    """qawaim points have no pre-written hooks — hook payload key must be null/None."""
    non_null = [pt.id for pt in qawaim_points if pt.payload.get("hook") is not None]
    assert not non_null, f"{len(non_null)} qawaim points have unexpected non-null hook: {non_null[:3]}"


@pytest.mark.integration
def test_mohasaba_chunk_id_prefix(mohasaba_points):
    wrong = [pt.id for pt in mohasaba_points if not pt.payload.get("chunk_id", "").startswith("mohasaba__")]
    assert not wrong, f"{len(wrong)} mohasaba points have wrong chunk_id prefix: {wrong[:3]}"


@pytest.mark.integration
def test_qawaim_chunk_id_prefix(qawaim_points):
    wrong = [pt.id for pt in qawaim_points if not pt.payload.get("chunk_id", "").startswith("qawaim__")]
    assert not wrong, f"{len(wrong)} qawaim points have wrong chunk_id prefix: {wrong[:3]}"


# ── Live retrieval tests ──────────────────────────────────────────────────────

def _hybrid_search(qdrant, model, query: str, limit: int = 3):
    out = model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense  = out["dense_vecs"][0].tolist()
    sparse = {int(k): float(v) for k, v in out["lexical_weights"][0].items()}
    return qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense, using="dense", limit=10),
            Prefetch(
                query=SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
                using="sparse",
                limit=10,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points


@pytest.mark.integration
@pytest.mark.slow
def test_query_returns_results(qdrant, embed_model):
    hits = _hybrid_search(qdrant, embed_model, "ما هي شروط الاعتراف بالإيراد؟")
    assert len(hits) > 0, "Query returned no results"


@pytest.mark.integration
@pytest.mark.slow
def test_query_results_from_known_docs(qdrant, embed_model):
    hits = _hybrid_search(qdrant, embed_model, "متطلبات الإفصاح عن الأدوات المالية")
    for h in hits:
        assert h.payload["doc_name"] in KNOWN_DOC_NAMES, (
            f"Result from unknown doc: {h.payload['doc_name']}"
        )


@pytest.mark.integration
@pytest.mark.slow
def test_query_scores_positive(qdrant, embed_model):
    hits = _hybrid_search(qdrant, embed_model, "معالجة عقود الإيجار")
    for h in hits:
        assert h.score > 0, f"Non-positive RRF score: {h.score}"


@pytest.mark.integration
@pytest.mark.slow
def test_query_results_have_non_empty_markdown(qdrant, embed_model):
    hits = _hybrid_search(qdrant, embed_model, "القوائم المالية والإفصاح")
    for h in hits:
        assert h.payload.get("markdown", "").strip(), f"Empty markdown in result {h.id}"


@pytest.mark.integration
@pytest.mark.slow
def test_mohasaba_query_hits_mohasaba(qdrant, embed_model):
    """A mohasaba-specific query should return at least one mohasaba chunk in top-3."""
    hits = _hybrid_search(qdrant, embed_model, "معيار المحاسبة المصري رقم 25 الأدوات المالية")
    docs = {h.payload["doc_name"] for h in hits}
    assert MOHASABA_DOC in docs, f"No mohasaba result for mohasaba-specific query; got: {docs}"


@pytest.mark.integration
@pytest.mark.slow
def test_qawaim_query_hits_qawaim(qdrant, embed_model):
    """A qawaim-specific query should return at least one qawaim chunk in top-3."""
    hits = _hybrid_search(qdrant, embed_model, "صافي ربح بنك CIB الربع الأول 2026")
    docs = {h.payload["doc_name"] for h in hits}
    assert QAWAIM_DOC in docs, f"No qawaim result for qawaim-specific query; got: {docs}"
