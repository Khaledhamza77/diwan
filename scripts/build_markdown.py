"""
Merge all chunks from <out_dir>/output.json into markdown files.
Chunks are sorted by page then by vertical position (grounding.box.top)
to preserve reading order.

Usage:
    python scripts/build_markdown.py [--out-dir <dir>]

    --out-dir    Directory containing output.json (and where markdown/ is written).
                 Defaults to data/parsed/ relative to the project root.

Outputs:
  <out_dir>/markdown/document.md        — full document, all pages
  <out_dir>/markdown/pages/page_N.md   — one file per page
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

_ANCHOR_RE = re.compile(r"<a id='[^']*'></a>\n*")


def _clean_markdown(md: str) -> str:
    return _ANCHOR_RE.sub("", md).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown files from output.json.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "parsed"), help="Directory containing output.json.")
    args = parser.parse_args()

    INPUT_PATH = Path(args.out_dir) / "output.json"
    MARKDOWN_DIR = Path(args.out_dir) / "markdown"
    PAGES_DIR = MARKDOWN_DIR / "pages"
    DOCUMENT_PATH = MARKDOWN_DIR / "document.md"

    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found — run run_ingestion.py first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    chunks = data.get("chunks") or []

    if not chunks:
        print("Warning: no chunks found in output.json.", file=sys.stderr)
        sys.exit(1)

    # Normalize page numbers to 1-indexed if the data is 0-indexed
    raw_pages = [
        c["grounding"]["page"]
        for c in chunks
        if c.get("grounding") and c["grounding"].get("page") is not None
    ]
    page_offset = 1 if (raw_pages and min(raw_pages) == 0) else 0
    if page_offset:
        print("Detected 0-indexed pages — applying +1 offset.")

    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    if PAGES_DIR.exists():
        for f in PAGES_DIR.glob("*.md"):
            f.unlink()
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by page, then by vertical position within the page
    def sort_key(chunk: dict) -> tuple:
        grounding = chunk.get("grounding") or {}
        page = grounding.get("page") if grounding.get("page") is not None else float("inf")
        top = (grounding.get("box") or {}).get("top", float("inf"))
        return (page, top)

    chunks_sorted = sorted(chunks, key=sort_key)

    # Group chunks by page
    pages: dict[int, list[str]] = {}
    skipped_no_page = 0
    for chunk in chunks_sorted:
        grounding = chunk.get("grounding") or {}
        page = grounding.get("page")
        if page is not None:
            page += page_offset
        markdown = _clean_markdown(chunk.get("markdown") or "")
        if not markdown:
            continue
        if page is None:
            skipped_no_page += 1
            continue
        pages.setdefault(page, []).append(markdown)

    if skipped_no_page:
        print(f"Warning: skipped {skipped_no_page} chunk(s) with no page number.")

    # Write individual page files
    for page_num, page_chunks in sorted(pages.items()):
        page_path = PAGES_DIR / f"page_{page_num}.md"
        page_path.write_text("\n\n".join(page_chunks), encoding="utf-8")

    # Write full document
    doc_lines = []
    for page_num, page_chunks in sorted(pages.items()):
        if doc_lines:
            doc_lines.append("\n---\n")
        doc_lines.append(f"<!-- page {page_num} -->\n")
        doc_lines.append("\n\n".join(page_chunks))

    DOCUMENT_PATH.write_text("\n".join(doc_lines), encoding="utf-8")

    print(f"Pages written : {len(pages)} → {PAGES_DIR}")
    print(f"Full document : {DOCUMENT_PATH}")


if __name__ == "__main__":
    main()
