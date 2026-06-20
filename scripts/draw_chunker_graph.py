"""
Generate a PNG illustration of the qawaim chunking LangGraph.

Output: docs/chunker_graph.png

Uses the Mermaid Ink API (no local graphviz needed).
"""

from pathlib import Path

from diwan.ingestion.chunker import _build_graph

OUT = Path(__file__).parent.parent / "docs" / "chunker_graph.png"

graph = _build_graph()
png_bytes = graph.get_graph().draw_mermaid_png()
OUT.write_bytes(png_bytes)
print(f"Saved → {OUT}")
