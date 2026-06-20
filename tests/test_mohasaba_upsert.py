"""
Smoke tests for the mohasaba upsert pipeline (Milestone 2A).

Markers:
  integration — requires a running Qdrant instance (localhost:6333)
  slow        — additionally loads BGE-M3 for live query tests

Run all:          pytest tests/test_mohasaba_upsert.py -m integration
Run without slow: pytest tests/test_mohasaba_upsert.py -m "integration and not slow"
"""

import re

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

COLLECTION = "diwan"
DOC_NAME   = "معايير_المحاسبة_المصرية_2020"
EXPECTED_POINTS = 152
ANCHOR_RE = re.compile(r"<a\s+id='[^']*'></a>")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qdrant():
    return QdrantClient(url="http://localhost:6333")


@pytest.fixture(scope="session")
def all_points(qdrant):
    results, _ = qdrant.scroll(
        collection_name=COLLECTION,
        limit=EXPECTED_POINTS + 50,
        with_payload=True,
        with_vectors=False,
    )
    return results


@pytest.fixture(scope="session")
def embed_model():
    import torch
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=torch.cuda.is_available())


# ── Collection tests ──────────────────────────────────────────────────────────

@pytest.mark.integration
def test_collection_exists(qdrant):
    names = {c.name for c in qdrant.get_collections().collections}
    assert COLLECTION in names, f"Collection '{COLLECTION}' not found in Qdrant"


@pytest.mark.integration
def test_point_count(qdrant):
    count = qdrant.count(collection_name=COLLECTION).count
    assert count == EXPECTED_POINTS, f"Expected {EXPECTED_POINTS} points, got {count}"


@pytest.mark.integration
def test_collection_config(qdrant):
    info = qdrant.get_collection(COLLECTION)
    vectors = info.config.params.vectors
    sparse  = info.config.params.sparse_vectors

    assert "dense" in vectors, "Missing 'dense' named vector"
    assert vectors["dense"].size == 1024, f"Expected 1024-dim dense, got {vectors['dense'].size}"
    assert str(vectors["dense"].distance).lower() == "cosine", "Dense vector distance should be cosine"
    assert "sparse" in sparse, "Missing 'sparse' named vector"


# ── Payload tests ─────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_payload_fields_present(all_points):
    required = {"chunk_id", "doc_name", "chunk_type", "markdown", "hook", "bboxes"}
    for pt in all_points:
        missing = required - pt.payload.keys()
        assert not missing, f"Point {pt.id} missing payload fields: {missing}"


@pytest.mark.integration
def test_doc_name(all_points):
    wrong = [pt.id for pt in all_points if pt.payload.get("doc_name") != DOC_NAME]
    assert not wrong, f"{len(wrong)} points have wrong doc_name: {wrong[:3]}"


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
def test_all_hooks_present(all_points):
    missing = [pt.id for pt in all_points if not pt.payload.get("hook")]
    assert not missing, f"{len(missing)} points missing hook: {missing[:3]}"


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
            box = bbox["box"]
            for key in ("top", "bottom", "left", "right"):
                assert key in box, f"Point {pt.id}: box missing '{key}'"


@pytest.mark.integration
def test_chunk_id_prefix(all_points):
    wrong = [pt.id for pt in all_points if not pt.payload.get("chunk_id", "").startswith("mohasaba__")]
    assert not wrong, f"{len(wrong)} points have wrong chunk_id prefix: {wrong[:3]}"


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
def test_query_results_have_correct_doc(qdrant, embed_model):
    hits = _hybrid_search(qdrant, embed_model, "متطلبات الإفصاح عن الأدوات المالية")
    for h in hits:
        assert h.payload["doc_name"] == DOC_NAME


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
