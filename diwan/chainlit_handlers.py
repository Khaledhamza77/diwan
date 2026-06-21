"""
Chainlit event handlers — wired into diwan/main.py.

on_chat_start : compile graph once (shared across sessions), map session → thread_id
on_message    : run Diwan pipeline in background thread, stream result to UI
"""
from __future__ import annotations

import asyncio
import logging
import os
import re

import chainlit as cl
from dotenv import load_dotenv

from diwan.pipeline.graph import create_graph, run

load_dotenv()

logger = logging.getLogger(__name__)

_ASSISTANT_NAME = "ديوان"

# ── Singletons ────────────────────────────────────────────────────────────────

_graph = None
_langfuse_client = None


def _get_langfuse():
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    if pk and sk and not pk.startswith("pk-lf-..."):
        try:
            from langfuse import Langfuse
            _langfuse_client = Langfuse(public_key=pk, secret_key=sk, host=host)
            logger.info("Langfuse tracing enabled → %s", host)
        except Exception as exc:
            logger.warning("Langfuse init failed: %s", exc)
    return _langfuse_client


# ── Startup ───────────────────────────────────────────────────────────────────

async def on_chat_start() -> None:
    global _graph
    if _graph is None:
        try:
            _graph = await asyncio.to_thread(create_graph)
        except Exception as exc:
            logger.error("Graph initialisation failed: %s", exc)
            await cl.Message(
                content=f"⚠️ خطأ في تهيئة الخدمة: {exc}",
                author=_ASSISTANT_NAME,
            ).send()
            return
    cl.user_session.set("graph", _graph)
    cl.user_session.set("thread_id", cl.context.session.id)


# ── Message handler ───────────────────────────────────────────────────────────

async def on_message(message: cl.Message) -> None:
    user_text = message.content.strip()
    graph = cl.user_session.get("graph")
    thread_id: str = cl.user_session.get("thread_id")

    if not graph:
        await cl.Message(
            content="الخدمة غير جاهزة. يرجى تحديث الصفحة والمحاولة مرة أخرى.",
            author=_ASSISTANT_NAME,
        ).send()
        return

    # Send placeholder immediately so the loading indicator appears
    response_msg = cl.Message(content="", author=_ASSISTANT_NAME)
    await response_msg.send()

    # Open a Langfuse trace for this request
    langfuse = _get_langfuse()
    lf_trace = None
    if langfuse:
        try:
            lf_trace = langfuse.trace(
                name="diwan-query",
                session_id=thread_id,
                input={"query": user_text},
            )
        except Exception as exc:
            logger.debug("Langfuse trace start failed: %s", exc)

    # Run the synchronous graph in a background thread
    try:
        answer: str = await asyncio.to_thread(
            run,
            query=user_text,
            thread_id=thread_id,
            graph=graph,
            trace=lf_trace,
        )
    except Exception as exc:
        logger.exception("Graph execution error")
        if lf_trace:
            try:
                lf_trace.update(output=f"ERROR: {exc}", level="ERROR")
                langfuse.flush()
            except Exception:
                pass
        response_msg.content = f"⚠️ حدث خطأ: {exc}"
        await response_msg.update()
        return

    # Apply text formatting
    display_text = _promote_first_sentence(answer)

    # Build sidebar citation elements from the footnote footer
    citation_els = _build_citation_elements(answer)
    if citation_els:
        response_msg.elements = citation_els

    # Complete Langfuse trace
    if lf_trace:
        try:
            lf_trace.update(output={"answer": answer})
            langfuse.flush()
        except Exception as exc:
            logger.debug("Langfuse trace update failed: %s", exc)

    response_msg.content = display_text
    await response_msg.update()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_title(text: str) -> str:
    return re.sub(r"[*_`]+", "", text).rstrip(" .,;:!?؟")


def _clamp_before_table(zone: str, zone_end: int) -> int:
    table_m = re.search(r"(?m)^[ \t]*\|", zone[:zone_end])
    if table_m:
        return min(zone_end, table_m.start())
    return zone_end


def _promote_first_sentence(text: str) -> str:
    """Promote the first sentence to a ### heading for frontend rendering."""
    stripped = text.lstrip()
    if stripped.startswith("#") or stripped.startswith("```"):
        return text
    if re.match(r"[\d١-٩][\.\)]\s", stripped):
        return text

    fence_pos = stripped.find("```")
    zone_end = fence_pos if fence_pos != -1 else len(stripped)
    list_m = re.search(r'(?m)^[ \t]*(?:\d+|[١-٩][٠-٩]*)[\.\)]\s', stripped[:zone_end])
    if list_m:
        zone_end = list_m.start()
    bullet_m = re.search(r'(?m)^[ \t]*[-*]\s', stripped[:zone_end])
    if bullet_m:
        zone_end = min(zone_end, bullet_m.start())
    zone_end = _clamp_before_table(stripped, zone_end)
    search_zone = stripped[:zone_end]

    m = re.search(r"(?<!\d)[.?!؟](?!\d)", search_zone)
    if not m:
        if zone_end == 0:
            return text
        lines = search_zone.split("\n", 1)
        title = lines[0].strip("* \t")
        if not title:
            return text
        rest = stripped[len(lines[0]):].lstrip()
        return f"### {title}\n\n{rest}".rstrip() if rest else f"### {title}"

    end = m.end()
    title = _clean_title(search_zone[:end].strip("* \t"))
    rest = stripped[end:].lstrip()
    return f"### {title}\n\n{rest}" if rest else f"### {title}"


def _parse_citations(answer: str) -> list[tuple[str, str, str | None]]:
    """
    Parse Diwan's footnote footer into (label, content, page) tuples.
    Pure function — no Chainlit dependency, fully testable.

    Footer format:
        ---
        [1] doc_name، ص. X
        [2] doc_name، ص. Y
    """
    m = re.search(r'\n---\n([\s\S]+)$', answer)
    if not m:
        return []

    results: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()

    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        lm = re.match(r'\[\d+\]\s*(.+?)(?:،\s*ص\.\s*(\d+))?\s*$', line)
        if not lm:
            continue
        doc_name = lm.group(1).strip()
        page = lm.group(2)
        label = f"{doc_name} | ص. {page}" if page else doc_name
        if label in seen:
            continue
        seen.add(label)
        content = f"المصدر: {doc_name}" + (f"\nالصفحة: {page}" if page else "")
        results.append((label, content, page))

    return results


def _build_citation_elements(answer: str) -> list[cl.Text]:
    """Build Chainlit sidebar Text elements from the synthesizer footnote footer."""
    return [
        cl.Text(name=label, content=content, display="side")
        for label, content, _ in _parse_citations(answer)
    ]
