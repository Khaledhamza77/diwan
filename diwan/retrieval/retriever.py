"""
Retrieval pipeline for a single sub-query.

Steps:
  1. Embed sub_query with BGE-M3 (dense + sparse)
  2. Hybrid search Qdrant with RRF fusion (top-20 candidates)
  3. Rerank with bge-reranker-v2-m3 cross-encoder (top-5)
  4. Return top-5 chunks with full payload
"""

import os
from typing import TypedDict

import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

load_dotenv()

COLLECTION    = "diwan"
QDRANT_URL    = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL   = "BAAI/bge-m3"
RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"
PRE_RERANK_K  = 20
POST_RERANK_K = 5

# Lazy singletons — loaded once on first call, reused for every subsequent retrieve().
_embed_model: BGEM3FlagModel | None = None
_reranker: FlagReranker | None = None
_qdrant: QdrantClient | None = None


def _get_embed_model() -> BGEM3FlagModel:
    global _embed_model
    if _embed_model is None:
        _embed_model = BGEM3FlagModel(EMBED_MODEL, use_fp16=torch.cuda.is_available())
    return _embed_model


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(RERANK_MODEL, use_fp16=torch.cuda.is_available())
    return _reranker


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant


class RetrievedChunk(TypedDict):
    chunk_id: str
    doc_name: str
    chunk_type: str
    markdown: str
    hook: str | None
    bboxes: list[dict]
    score: float


class RetrieverOutput(TypedDict):
    sub_query: str
    purpose: str
    chunks: list[RetrievedChunk]


def retrieve(sub_query: str, purpose: str) -> RetrieverOutput:
    """Run the full retrieval pipeline for one sub-query."""
    model   = _get_embed_model()
    reranker = _get_reranker()
    client  = _get_qdrant()

    # 1. Embed sub-query — dense + sparse in one forward pass.
    out = model.encode(
        [sub_query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        max_length=8192,
    )
    dense       = out["dense_vecs"][0].tolist()
    sparse_dict = {int(k): float(v) for k, v in out["lexical_weights"][0].items()}

    # 2. Hybrid search — dense + sparse fused with RRF natively by Qdrant.
    hits = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense, using="dense", limit=PRE_RERANK_K),
            Prefetch(
                query=SparseVector(
                    indices=list(sparse_dict.keys()),
                    values=list(sparse_dict.values()),
                ),
                using="sparse",
                limit=PRE_RERANK_K,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=PRE_RERANK_K,
        with_payload=True,
    ).points

    if not hits:
        return {"sub_query": sub_query, "purpose": purpose, "chunks": []}

    # 3. Rerank — cross-encoder scores each candidate against the query directly.
    passages = [h.payload["markdown"] for h in hits]
    pairs    = [[sub_query, p] for p in passages]
    scores   = reranker.compute_score(pairs, normalize=True)
    # compute_score returns list[float] for multiple pairs; guard against a bare float.
    if isinstance(scores, (int, float)):
        scores = [float(scores)]
    else:
        scores = [float(s) for s in scores]

    ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    top    = ranked[:POST_RERANK_K]

    # 4. Assemble output.
    chunks: list[RetrievedChunk] = [
        {
            "chunk_id":   h.payload["chunk_id"],
            "doc_name":   h.payload["doc_name"],
            "chunk_type": h.payload["chunk_type"],
            "markdown":   h.payload["markdown"],
            "hook":       h.payload.get("hook"),
            "bboxes":     h.payload["bboxes"],
            "score":      score,
        }
        for score, h in top
    ]

    return {"sub_query": sub_query, "purpose": purpose, "chunks": chunks}
