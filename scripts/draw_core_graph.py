"""
Generate a PNG illustration of the core RAG pipeline LangGraph.

Output: docs/core_graph.png

Uses the Mermaid Ink API (no local graphviz needed).
"""

from pathlib import Path

from diwan.pipeline.graph import _build_graph

OUT = Path(__file__).parent.parent / "docs" / "core_graph.png"

graph = _build_graph().compile()
png_bytes = graph.get_graph().draw_mermaid_png()
OUT.write_bytes(png_bytes)
print(f"Saved: {OUT}")
