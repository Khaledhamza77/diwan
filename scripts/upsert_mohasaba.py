"""
Embed and upsert pre-chunked mohasaba sections into Qdrant.

Input:  data/indecies/mohasaba/section_chunks.json
        data/mohasaba_parsed/output.json  (for bbox type lookup)
Output: Qdrant collection "diwan", one point per section.

Embedding input per chunk: hook + "\n\n" + markdown
BGE-M3 auto-truncates at 8192 tokens; hook is placed first so it is
always embedded in full even when the markdown tail is cut.

Run pull_models.py first if BAAI/bge-m3 is not yet cached locally.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path

import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

ROOT           = Path(__file__).parent.parent
SECTION_CHUNKS = ROOT / "data/indecies/mohasaba/section_chunks.json"
OUTPUT_JSON    = ROOT / "data/mohasaba_parsed/output.json"
DOC_NAME       = "معايير_المحاسبة_المصرية_2020"
COLLECTION     = "diwan"
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL    = "BAAI/bge-m3"
BATCH_SIZE     = 16   # BGE-M3 is memory-intensive; keep batches small


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_type_lookup(output_json: Path) -> dict[str, str]:
    """Return {chunk_id: type} from LandingAI output.json."""
    with open(output_json, encoding="utf-8") as f:
        data = json.load(f)
    return {c["id"]: c["type"] for c in data["chunks"]}


def resolve_chunk_type(bbox_types: list[str]) -> str:
    unique = set(bbox_types)
    return bbox_types[0] if len(unique) == 1 else "mixed"


_ANCHOR_RE = re.compile(r"<a\s+id='[^']*'></a>\n*")


def strip_anchors(text: str) -> str:
    return _ANCHOR_RE.sub("", text).strip()


def build_embed_text(hook: str | None, markdown: str) -> str:
    if hook:
        return hook + "\n\n" + markdown
    return markdown


def ensure_collection(client: QdrantClient, name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        print(f"Collection '{name}' already exists — skipping creation.")
        return

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    print(f"Created collection '{name}'.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    type_lookup = load_type_lookup(OUTPUT_JSON)

    with open(SECTION_CHUNKS, encoding="utf-8") as f:
        sections: dict = json.load(f)

    # ── Step 1: Assemble sections ─────────────────────────────────────────────
    print("[1/4] Assembling sections ...")
    chunks = []
    for section_id, sec in sections.items():
        hook      = sec.get("hook") or None
        raw_bboxes = sec.get("chunks", [])
        if not raw_bboxes:
            continue

        markdown = strip_anchors("\n\n".join(c["markdown"] for c in raw_bboxes))
        bboxes   = [{"page": c["page"], "box": c["box"]} for c in raw_bboxes]
        bbox_types = [
            type_lookup.get(c["chunk_id"], "text") for c in raw_bboxes
        ]

        chunks.append({
            "chunk_id":   f"mohasaba__{section_id}",
            "doc_name":   DOC_NAME,
            "chunk_type": resolve_chunk_type(bbox_types),
            "markdown":   markdown,
            "hook":       hook,
            "bboxes":     bboxes,
            "embed_text": build_embed_text(hook, markdown),
        })

    print(f"[1/4] Assembled {len(chunks)} sections.")

    # ── Sanity check ──────────────────────────────────────────────────────────
    errors = []
    for c in chunks:
        if not c["markdown"].strip():
            errors.append(f"  {c['chunk_id']}: empty markdown")
        if not c["embed_text"].strip():
            errors.append(f"  {c['chunk_id']}: empty embed_text")
        if not c["bboxes"]:
            errors.append(f"  {c['chunk_id']}: no bboxes")

    if errors:
        print(f"\n[ABORT] {len(errors)} section(s) failed sanity check:")
        for e in errors:
            print(e)
        raise SystemExit(1)

    hooks_present = sum(1 for c in chunks if c["hook"])
    print(f"       Sanity OK — {hooks_present}/{len(chunks)} sections have hooks, "
          f"{len(chunks) - hooks_present} embed markdown only.")

    # ── Step 2: Load model ────────────────────────────────────────────────────
    use_fp16 = torch.cuda.is_available()
    mode = "fp16 on GPU" if use_fp16 else "fp32 on CPU"
    print(f"\n[2/4] Loading {EMBED_MODEL} ({mode}) ...")
    t0 = time.time()
    model = BGEM3FlagModel(EMBED_MODEL, use_fp16=use_fp16)
    print(f"      Model loaded in {time.time() - t0:.1f}s.")

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    print(f"\n[3/4] Embedding {len(chunks)} chunks (batch_size={BATCH_SIZE}) ...")
    all_dense:  list[list[float]]      = []
    all_sparse: list[dict[int, float]] = []

    t0 = time.time()
    with tqdm(total=len(chunks), unit="chunk", desc="  embedding") as bar:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_texts = [c["embed_text"] for c in chunks[i : i + BATCH_SIZE]]
            output = model.encode(
                batch_texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
                max_length=8192,
            )
            all_dense.extend(output["dense_vecs"].tolist())
            for sparse_dict in output["lexical_weights"]:
                all_sparse.append({int(k): float(v) for k, v in sparse_dict.items()})
            bar.update(len(batch_texts))
    print(f"      Embedded in {time.time() - t0:.1f}s.")

    # ── Step 4: Upsert ────────────────────────────────────────────────────────
    print(f"\n[4/4] Upserting {len(chunks)} points to '{COLLECTION}' ...")
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, COLLECTION)

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

    t0 = time.time()
    with tqdm(total=len(points), unit="point", desc="  upserting") as bar:
        for i in range(0, len(points), BATCH_SIZE):
            client.upsert(
                collection_name=COLLECTION,
                points=points[i : i + BATCH_SIZE],
            )
            bar.update(len(points[i : i + BATCH_SIZE]))
    print(f"      Upserted in {time.time() - t0:.1f}s.")

    count = client.count(collection_name=COLLECTION).count
    print(f"\nDone. Collection '{COLLECTION}' now has {count} points.")


if __name__ == "__main__":
    main()
