"""
Stage 1: Parse a PDF using LandingAI ADE and save to <out_dir>/output.json.
Idempotent — exits early if output.json already exists.

LandingAI free tier caps at 100 pages per request. The PDF is split into
chunks of MAX_PAGES, each chunk is parsed separately, and the results are
merged. Per-chunk checkpoints (<out_dir>/part_N.json) are kept so a
mid-run failure does not re-burn credits for already-parsed chunks.

Usage:
    python scripts/run_ingestion.py <pdf_path> [--out-dir <dir>]

    <pdf_path>   Path to the PDF to parse.
    --out-dir    Directory for output.json and part checkpoints.
                 Defaults to data/parsed/ relative to the project root.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import fitz  # pymupdf
from dotenv import load_dotenv
from landingai_ade import LandingAIADE

load_dotenv()

ROOT = Path(__file__).parent.parent

MAX_PAGES = 100  # LandingAI free-tier limit


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PDF with LandingAI ADE.")
    parser.add_argument("pdf", help="Path to the PDF file.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "parsed"), help="Output directory for output.json and checkpoints.")
    args = parser.parse_args()

    PDF_PATH = Path(args.pdf)
    PARSED_DIR = Path(args.out_dir)
    OUTPUT_PATH = PARSED_DIR / "output.json"

    if OUTPUT_PATH.exists():
        print(f"output.json already exists at {OUTPUT_PATH} — skipping.")
        sys.exit(0)

    api_key = os.environ.get("LANDINGAI_API_KEY")
    if not api_key:
        print("Error: LANDINGAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    if not PDF_PATH.exists():
        print(f"Error: PDF not found at {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    client = LandingAIADE(apikey=api_key)

    doc = fitz.open(PDF_PATH)
    try:
        total_pages = doc.page_count
        print(f"PDF has {total_pages} pages — splitting into chunks of {MAX_PAGES}.")

        # Build list of (start, end) page ranges, 0-indexed inclusive
        ranges = []
        start = 0
        while start < total_pages:
            end = min(start + MAX_PAGES - 1, total_pages - 1)
            ranges.append((start, end))
            start = end + 1

        all_chunks = []

        for part_idx, (start, end) in enumerate(ranges):
            checkpoint = PARSED_DIR / f"part_{part_idx}.json"

            if checkpoint.exists():
                print(f"Part {part_idx} (pages {start + 1}–{end + 1}): checkpoint found, loading.")
                try:
                    part_data = json.loads(checkpoint.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    print(
                        f"Error: checkpoint {checkpoint.name} is corrupt (partial write).\n"
                        f"Delete it and re-run to re-parse this chunk.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                # Checkpoint was already saved with page offset applied — load as-is.
                all_chunks.extend(part_data.get("chunks", []))
            else:
                print(f"Part {part_idx} (pages {start + 1}–{end + 1}): parsing ...")
                part_data = _parse_chunk(client, doc, start, end, part_idx, PARSED_DIR)

                # Normalize to 1-indexed full-document page numbers.
                # Auto-detect LandingAI's base (0 or 1-indexed) from the minimum
                # page seen in this chunk rather than assuming a fixed base.
                chunks = part_data.get("chunks", [])
                raw_pages = [
                    c["grounding"]["page"]
                    for c in chunks
                    if c.get("grounding") and c["grounding"].get("page") is not None
                ]
                if raw_pages:
                    min_raw = min(raw_pages)
                    target_start = start + 1  # 1-indexed position in the full document
                    page_offset = target_start - min_raw
                    for chunk in chunks:
                        if chunk.get("grounding") and chunk["grounding"].get("page") is not None:
                            chunk["grounding"]["page"] += page_offset
                part_data["chunks"] = chunks

                _atomic_write(checkpoint, json.dumps(part_data, ensure_ascii=False, indent=2))
                print(f"  Saved checkpoint → {checkpoint.name}")
                all_chunks.extend(chunks)
    finally:
        doc.close()

    merged = {"chunks": all_chunks}
    _atomic_write(OUTPUT_PATH, json.dumps(merged, ensure_ascii=False, indent=2))

    _log_summary(merged)
    print(f"\nSaved to {OUTPUT_PATH}")


def _atomic_write(path: Path, text: str) -> None:
    """Write text to a .tmp file then rename into place to avoid partial writes."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _parse_chunk(client: LandingAIADE, doc: fitz.Document, start: int, end: int, part_idx: int, parsed_dir: Path) -> dict:
    """Write pages [start, end] to a temp PDF, parse it, return the response as a dict."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        chunk_doc = fitz.open()
        try:
            chunk_doc.insert_pdf(doc, from_page=start, to_page=end)
            chunk_doc.save(tmp_path)
        finally:
            chunk_doc.close()

        response = client.parse(document=tmp_path, model="dpt-2")

        try:
            if hasattr(response, "model_dump"):
                return response.model_dump()
            elif hasattr(response, "dict"):
                return response.dict()
            elif isinstance(response, dict):
                return response
            else:
                raise TypeError(f"Unexpected response type: {type(response)}")
        except Exception as exc:
            fallback = parsed_dir / f"part_{part_idx}.raw.txt"
            fallback.write_text(repr(response), encoding="utf-8")
            print(f"Error serializing part {part_idx}: {exc}", file=sys.stderr)
            print(f"Raw response saved to {fallback} — inspect it and re-run.", file=sys.stderr)
            sys.exit(1)
    finally:
        tmp_path.unlink(missing_ok=True)


def _log_summary(data: dict) -> None:
    chunks = data.get("chunks", [])
    if not chunks:
        print("Warning: no 'chunks' key found in response — inspect output.json to verify structure.")
        return

    pages = {
        c["grounding"]["page"]
        for c in chunks
        if c.get("grounding") and c["grounding"].get("page") is not None
    }
    chunk_types: dict[str, int] = {}
    for c in chunks:
        ct = c.get("chunk_type", "unknown")
        chunk_types[ct] = chunk_types.get(ct, 0) + 1

    page_range = f"{min(pages)} – {max(pages)}" if pages else "unknown"
    print(f"Total chunks : {len(chunks)}")
    print(f"Pages found  : {len(pages)} ({page_range})")
    print("Chunk types  :")
    for ct, count in sorted(chunk_types.items(), key=lambda x: -x[1]):
        print(f"  {ct:<20} {count}")


if __name__ == "__main__":
    main()
