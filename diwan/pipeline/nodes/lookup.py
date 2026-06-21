"""
Lookup function — assembles the Synthesizer context from Extractor outputs.

Pure function: no LLM, no Qdrant call.
Joins extractor-selected chunk IDs against full payloads already held in retriever outputs.
Deduplicates across parallel workers (same chunk may be selected by multiple sub-queries).
"""

from typing import TypedDict

from diwan.retrieval.retriever import RetrieverOutput
from diwan.pipeline.nodes.extractor import ExtractorOutput


class ContextChunk(TypedDict):
    chunk_id: str
    doc_name: str
    markdown: str
    bboxes: list[dict]
    sub_query: str
    purpose: str


def lookup(
    extractor_results: list[ExtractorOutput],
    retriever_outputs: list[RetrieverOutput],
) -> list[ContextChunk]:
    """
    Build the Synthesizer context by joining extractor chunk_ids with full payloads
    from retriever outputs.

    Order: extractor_results[i] corresponds to retriever_outputs[i] (same sub-query).
    When the same chunk is selected by multiple sub-queries, the first occurrence wins
    (preserving the sub-query association of the earliest worker).
    """
    # Build chunk_id → full payload map across all retriever outputs.
    chunk_map: dict[str, dict] = {}
    for ro in retriever_outputs:
        for chunk in ro["chunks"]:
            chunk_map[chunk["chunk_id"]] = chunk

    context: list[ContextChunk] = []
    seen: set[str] = set()

    for er in extractor_results:
        for cid in er["chunk_ids"]:
            if cid in seen:
                continue
            seen.add(cid)
            chunk = chunk_map.get(cid)
            if chunk is None:
                continue
            context.append({
                "chunk_id": cid,
                "doc_name": chunk["doc_name"],
                "markdown": chunk["markdown"],
                "bboxes":   chunk["bboxes"],
                "sub_query": er["sub_query"],
                "purpose":   er["purpose"],
            })

    return context
