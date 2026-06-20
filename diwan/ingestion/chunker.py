"""
Chunk and embed qawaim (financial statements) document into Qdrant.

Input:  data/qawaim_parsed/output.json
Output: Qdrant collection "diwan", one point per semantic chunk.

LangGraph graph:
  prepare → group_page (loop) → embed_and_upsert → END

Each group_page LLM call is recorded as a chunker/page_{n} generation in Langfuse.

Run pull_models.py first if BAAI/bge-m3 is not yet cached locally.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import TypedDict

import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from langfuse import Langfuse
from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

ROOT           = Path(__file__).parent.parent.parent
OUTPUT_JSON    = ROOT / "data/qawaim_parsed/output.json"
CHECKPOINT_JSON = ROOT / "data/qawaim_parsed/chunks_checkpoint.json"
DOC_NAME    = "qawaim"
COLLECTION  = "diwan"
QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = "BAAI/bge-m3"
GROUP_MODEL = "gpt-5.4-mini"
BATCH_SIZE  = 16

NOISE_TYPES = {"logo", "card", "attestation", "scan_code", "marginalia"}

# Module-level clients — set once in main() and used by graph nodes.
_oai: OpenAI | None = None
_lf: Langfuse | None = None
_trace = None

# Accumulators updated across the group_page loop.
_grouping_start:       float = 0.0
_total_input_tokens:   int   = 0
_total_output_tokens:  int   = 0
_pipeline_start:       float = 0.0


# ── LangGraph state ────────────────────────────────────────────────────────────

class ChunkerState(TypedDict):
    page_nums: list[int]
    pages: dict[str, list[dict]]   # str keys for JSON-safe serialisation
    current_page_idx: int
    pending_group: list[dict] | None
    completed_chunks: list[dict]


# ── LLM output schema ──────────────────────────────────────────────────────────

class BboxGroup(BaseModel):
    indices: list[int]

class GroupingOutput(BaseModel):
    groups: list[BboxGroup]
    last_group_complete: bool


# ── Helpers ────────────────────────────────────────────────────────────────────

_ANCHOR_RE = re.compile(r"<a\s+id='[^']*'></a>\n*")


def strip_anchors(text: str) -> str:
    return _ANCHOR_RE.sub("", text).strip()


def resolve_chunk_type(types: list[str]) -> str:
    unique = set(types)
    return types[0] if len(unique) == 1 else "mixed"


def make_chunk_id(bbox_ids: list[str]) -> str:
    key = "qawaim__" + "__".join(bbox_ids)
    return f"qawaim__{uuid.uuid5(uuid.NAMESPACE_DNS, key).hex[:12]}"


def ensure_collection(client: QdrantClient, name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    print(f"Created collection '{name}'.")


# ── Grouping LLM ───────────────────────────────────────────────────────────────

_GROUPING_SYSTEM = """You are a document chunking assistant for Arabic bank financial statements. Your job is to group text elements extracted from one page into coherent retrieval units.

You will receive a numbered list of bounding boxes, each with its element type and a text snippet. Assign EVERY bbox to exactly one group. No bounding box may be omitted.

## Grouping rules

**Rule 1 — Headings must attach to their body**
Any bbox whose content is a section title, subsection title, or standalone heading (e.g. "١. معلومات عامة", "ملخص الإيضاحات المتممة...") MUST be placed in the same group as the bbox(es) that form its body. A heading with no body on this page is an incomplete group — carry it forward via last_group_complete = false.

**Rule 2 — Captions attach to their element**
A text caption or label for a table or figure must be in the same group as that table/figure bbox.

**Rule 3 — Keep financial tables intact**
All parts of a single financial table (header row, data rows, subtotal rows, footnote lines) belong in one group. If a table continues across pages it is incomplete.

**Rule 4 — Split only on clear topic boundaries**
Separate groups only when there is an unambiguous topic shift: a new numbered note, a new named financial statement, or a clearly self-contained paragraph. Do not split a single logical unit across multiple groups.

## Deciding last_group_complete

Set last_group_complete = FALSE when the last group on this page:
- Is a heading or title whose body has not yet appeared
- Ends with a colon or list opener (e.g. "وتشمل ما يلي :", "كما يلي :", "على النحو التالي :")
- Is a paragraph that is visibly cut off mid-sentence or mid-thought
- Is a table that has no totals row yet, or whose header contains "تابع" (continued)

Set last_group_complete = TRUE when the last group:
- Is a fully self-contained paragraph ending with a complete sentence
- Is a financial table that includes its totals/closing row
- Is a figure with its caption fully present

## Output format
Return `groups` as an ordered array; each group contains the `indices` of its member bboxes. Then set `last_group_complete`."""


def _build_user_prompt(bboxes: list[dict]) -> str:
    lines = []
    for i, b in enumerate(bboxes):
        snippet = strip_anchors(b["markdown"])[:300].replace("\n", " ")
        top = b["grounding"]["box"]["top"]
        lines.append(f"[{i}] ({b['type']}, y={top:.2f}) {snippet}")
    return "\n".join(lines)


def _call_grouping_llm(bboxes: list[dict], page_num: int) -> GroupingOutput:
    global _total_input_tokens, _total_output_tokens

    user_prompt = _build_user_prompt(bboxes)

    gen = _trace.generation(
        name=f"chunker/page_{page_num}",
        model=GROUP_MODEL,
        input=[
            {"role": "system", "content": _GROUPING_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
    )

    response = _oai.beta.chat.completions.parse(
        model=GROUP_MODEL,
        messages=[
            {"role": "system", "content": _GROUPING_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        response_format=GroupingOutput,
        temperature=0,
        seed=42,
    )

    result: GroupingOutput = response.choices[0].message.parsed
    usage = response.usage

    _total_input_tokens  += usage.prompt_tokens
    _total_output_tokens += usage.completion_tokens

    gen.end(
        output={"groups": len(result.groups), "last_group_complete": result.last_group_complete},
        usage={"input": usage.prompt_tokens, "output": usage.completion_tokens},
    )

    return result


# ── Graph nodes ────────────────────────────────────────────────────────────────

def prepare(state: ChunkerState) -> ChunkerState:
    print("[1/3] Filtering and sorting bboxes ...")
    t0 = time.time()

    with open(OUTPUT_JSON, encoding="utf-8") as f:
        data = json.load(f)

    raw = data["chunks"]
    print(f"      {len(raw)} raw bboxes loaded from {OUTPUT_JSON.name}")

    # Remove noise types
    after_noise = [c for c in raw if c["type"] not in NOISE_TYPES]
    n_noise = len(raw) - len(after_noise)

    # Find text repeated verbatim across multiple pages (headers / footers)
    text_to_pages: dict[str, set[int]] = {}
    for c in after_noise:
        if c["type"] == "text":
            m = strip_anchors(c["markdown"])
            p = c["grounding"]["page"]
            text_to_pages.setdefault(m, set()).add(p)
    repeated_across_pages = {m for m, pages in text_to_pages.items() if len(pages) > 1}

    n_margin = n_repeat = 0

    def is_furniture(c: dict) -> bool:
        nonlocal n_margin, n_repeat
        box = c["grounding"]["box"]
        if box["top"] < 0.05 or box["bottom"] > 0.95:
            if c["type"] in ("text", "figure"):
                n_margin += 1
                return True
        if c["type"] != "text":
            return False
        if strip_anchors(c["markdown"]) in repeated_across_pages:
            n_repeat += 1
            return True
        return False

    filtered = [c for c in after_noise if not is_furniture(c)]

    # Reading order: page asc, then top asc
    filtered.sort(key=lambda c: (c["grounding"]["page"], c["grounding"]["box"]["top"]))

    # Group by page (str key for JSON safety)
    pages: dict[str, list[dict]] = {}
    for c in filtered:
        key = str(c["grounding"]["page"])
        pages.setdefault(key, []).append(c)

    page_nums = sorted(int(k) for k in pages)

    print(f"      Removed: {n_noise:>3} noise bboxes  (logo, marginalia, attestation, …)")
    print(f"               {n_margin:>3} margin bboxes  (top < 5% or bottom > 95%)")
    print(f"               {n_repeat:>3} repeated text  (cross-page headers/footers)")
    print(f"      {'─' * 44}")
    print(f"      {len(filtered):>3} bboxes remaining across {len(page_nums)} pages  "
          f"(filtered {len(raw) - len(filtered)} of {len(raw)})")

    per_page = ", ".join(
        f"p{p}→{len(pages[str(p)])}" for p in page_nums
    )
    print(f"      Distribution: {per_page}")
    print(f"      Done in {time.time() - t0:.2f}s.\n")

    return {
        "page_nums": page_nums,
        "pages": pages,
        "current_page_idx": 0,
        "pending_group": None,
        "completed_chunks": [],
    }


def group_page(state: ChunkerState) -> ChunkerState:
    global _grouping_start

    idx       = state["current_page_idx"]
    n_pages   = len(state["page_nums"])
    page_num  = state["page_nums"][idx]
    is_last   = (idx == n_pages - 1)

    if idx == 0:
        _grouping_start = time.time()
        print(f"[2/3] Grouping {n_pages} pages via {GROUP_MODEL} ...\n"
              f"      {'page':>4}  {'in':>4}  {'out':>4}  {'pend':>4}  running")

    page_bboxes = list(state["pages"][str(page_num)])
    all_bboxes  = (state["pending_group"] or []) + page_bboxes

    result = _call_grouping_llm(all_bboxes, page_num)

    # Decide which groups to emit now vs. carry forward
    if result.last_group_complete or is_last:
        groups_to_emit = result.groups
        new_pending    = None
    else:
        groups_to_emit  = result.groups[:-1]
        pending_indices = result.groups[-1].indices
        new_pending     = [all_bboxes[i] for i in pending_indices if i < len(all_bboxes)]

    # Assemble chunk records for emitted groups
    completed = list(state["completed_chunks"])
    for group in groups_to_emit:
        members = [all_bboxes[i] for i in group.indices if i < len(all_bboxes)]
        if not members:
            continue
        markdown    = strip_anchors("\n\n".join(b["markdown"] for b in members))
        chunk_type  = resolve_chunk_type([b["type"] for b in members])
        bboxes_list = [
            {"page": b["grounding"]["page"], "box": b["grounding"]["box"]}
            for b in members
        ]
        completed.append({
            "chunk_id":   make_chunk_id([b["id"] for b in members]),
            "doc_name":   DOC_NAME,
            "chunk_type": chunk_type,
            "markdown":   markdown,
            "hook":       None,
            "bboxes":     bboxes_list,
        })

    pending_count = len(new_pending) if new_pending else 0
    pending_tag   = f"{pending_count:>4}" if pending_count else "   —"
    print(f"  [{idx + 1:>2}/{n_pages}] "
          f"p{page_num:<3} "
          f"{len(all_bboxes):>4} bboxes  "
          f"→ {len(groups_to_emit):>3} chunks  "
          f"{pending_tag} pending  "
          f"total {len(completed):>3}")

    if is_last:
        elapsed = time.time() - _grouping_start
        from collections import Counter
        type_counts = Counter(c["chunk_type"] for c in completed)
        type_str = "  ".join(f"{t}: {n}" for t, n in sorted(type_counts.items()))
        print(f"\n      Grouping complete in {elapsed:.1f}s — "
              f"{len(completed)} chunks  ({type_str})")
        print(f"      Tokens — in: {_total_input_tokens:,}  "
              f"out: {_total_output_tokens:,}\n")

        # Sanity check
        errors = [c["chunk_id"] for c in completed if not c["markdown"].strip()]
        if errors:
            print(f"      [ABORT] {len(errors)} chunk(s) have empty markdown:")
            for e in errors:
                print(f"        {e}")
            raise SystemExit(1)
        print(f"      Sanity check: all {len(completed)} chunks non-empty ✓\n")

        CHECKPOINT_JSON.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_JSON.write_text(
            json.dumps(completed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"      Checkpoint saved → {CHECKPOINT_JSON.relative_to(ROOT)}\n")

    return {
        **state,
        "current_page_idx": idx + 1,
        "pending_group":    new_pending,
        "completed_chunks": completed,
    }


def embed_and_upsert(state: ChunkerState) -> ChunkerState:
    chunks = state["completed_chunks"]

    use_fp16 = torch.cuda.is_available()
    mode     = "fp16 on GPU" if use_fp16 else "fp32 on CPU"
    print(f"[3/3] Loading {EMBED_MODEL} ({mode}) ...")
    t0    = time.time()
    model = BGEM3FlagModel(EMBED_MODEL, use_fp16=use_fp16)
    print(f"      Model loaded in {time.time() - t0:.1f}s.\n")

    texts = [c["markdown"] for c in chunks]

    all_dense:  list[list[float]]      = []
    all_sparse: list[dict[int, float]] = []

    print(f"      Embedding {len(chunks)} chunks (batch_size={BATCH_SIZE}) ...")
    t0 = time.time()
    with tqdm(total=len(chunks), unit="chunk", desc="      ") as bar:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch  = texts[i : i + BATCH_SIZE]
            output = model.encode(
                batch,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
                max_length=8192,
            )
            all_dense.extend(output["dense_vecs"].tolist())
            for sparse_dict in output["lexical_weights"]:
                all_sparse.append({int(k): float(v) for k, v in sparse_dict.items()})
            bar.update(len(batch))
    print(f"      Embedded in {time.time() - t0:.1f}s.\n")

    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, COLLECTION)

    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="doc_name", match=MatchValue(value=DOC_NAME))]
        ),
    )
    print(f"      Deleted existing '{DOC_NAME}' points from '{COLLECTION}'.")

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"])),
            vector={
                "dense":  dense,
                "sparse": SparseVector(
                    indices=list(sparse.keys()),
                    values=list(sparse.values()),
                ),
            },
            payload={
                "chunk_id":   chunk["chunk_id"],
                "doc_name":   chunk["doc_name"],
                "chunk_type": chunk["chunk_type"],
                "markdown":   chunk["markdown"],
                "hook":       chunk["hook"],
                "bboxes":     chunk["bboxes"],
            },
        )
        for chunk, dense, sparse in zip(chunks, all_dense, all_sparse)
    ]

    print(f"      Upserting {len(points)} points to '{COLLECTION}' ...")
    t0 = time.time()
    with tqdm(total=len(points), unit="pt", desc="      ") as bar:
        for i in range(0, len(points), BATCH_SIZE):
            client.upsert(
                collection_name=COLLECTION,
                points=points[i : i + BATCH_SIZE],
            )
            bar.update(len(points[i : i + BATCH_SIZE]))
    print(f"      Upserted in {time.time() - t0:.1f}s.")

    total_count = client.count(collection_name=COLLECTION).count
    print(f"\n      Collection '{COLLECTION}' now has {total_count} points total.")

    elapsed = time.time() - _pipeline_start
    print(f"\nDone. Total pipeline time: {elapsed:.1f}s.")

    _lf.flush()
    return state


# ── Graph wiring ───────────────────────────────────────────────────────────────

def _should_continue(state: ChunkerState) -> str:
    if state["current_page_idx"] < len(state["page_nums"]):
        return "group_page"
    return "embed_and_upsert"


def _build_graph():
    builder = StateGraph(ChunkerState)
    builder.add_node("prepare",         prepare)
    builder.add_node("group_page",      group_page)
    builder.add_node("embed_and_upsert", embed_and_upsert)

    builder.set_entry_point("prepare")
    builder.add_edge("prepare", "group_page")
    builder.add_conditional_edges(
        "group_page",
        _should_continue,
        {"group_page": "group_page", "embed_and_upsert": "embed_and_upsert"},
    )
    builder.add_edge("embed_and_upsert", END)

    return builder.compile()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global _oai, _lf, _trace, _pipeline_start

    print("=== Qawaim Chunking Pipeline ===\n")
    _pipeline_start = time.time()

    _oai   = OpenAI()
    _lf    = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    )
    _trace = _lf.trace(name="qawaim-chunking")

    graph = _build_graph()
    graph.invoke({
        "page_nums":        [],
        "pages":            {},
        "current_page_idx": 0,
        "pending_group":    None,
        "completed_chunks": [],
    })


if __name__ == "__main__":
    main()
