"""
Planner node — decomposes an in-scope query into the minimum set of disjoint sub-queries.

Model: gpt-5.4 (reasoning quality matters; CoT is constrained, not free-form)
Prompt: English instructions; sub-queries and purposes are written in Arabic (searched against Arabic corpus)
"""

from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

PLANNER_MODEL = "gpt-5.4"

_oai: OpenAI | None = None


def _get_oai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI()
    return _oai


class SubQuery(TypedDict):
    query: str
    purpose: str


class PlannerOutput(TypedDict):
    sub_queries: list[SubQuery]


class _SubQuerySchema(BaseModel):
    query: str
    purpose: str


class _PlannerSchema(BaseModel):
    thinking: str
    sub_queries: list[_SubQuerySchema]


_SYSTEM = """\
You are a query planner for an Arabic financial document Q&A system.

## Available documents
- **Egyptian Accounting Standards 2020** (معايير المحاسبة المصرية): regulatory standards, \
articles, definitions, measurement and disclosure rules.
- **CIB Bank standalone financial statements** (القوائم المالية المنفردة): balance sheet, \
income statement, notes, disclosures.

## Objective
Identify the **minimum number of distinct sub-queries** that together cover the question \
completely, with no overlap between them.

## When to split
Split when the question has distinct information needs that don't naturally co-occur in \
the same section or passage — meaning a reader looking for one part would have to look \
somewhere completely different in the documents to find another part. This happens when:
- The question asks about two or more entities, periods, or figures that each live in \
different sections (e.g. comparing line items across years, or across statements).
- The question is composite: it contains sub-questions that point to different topics or \
areas of the documents, such that no single focused search would surface all of them.
- One part of the answer comes from a fundamentally different context than another \
(e.g. a definition from the standards and a figure from the financial statements).

## When to keep a single sub-query
- All parts of the question point to the same topic, section, or area of the documents.
- The question is verbose but a single focused search would surface everything needed.
- The sub-parts are so closely related they would naturally appear together in one passage.

## Required thinking step
Before emitting sub-queries, reason in the `thinking` field:
1. What exactly is the question asking for?
2. Does answering it require different retrieval targets (different entities, documents, or time periods)?
3. What is the minimum set of sub-queries that achieves full coverage?

Bias toward fewer sub-queries. One well-formed sub-query is better than two overlapping ones.

## Sub-query rules
- Each sub-query must be independently retrievable against the Arabic corpus.
- Write `query` in Arabic — it will be embedded and searched against Arabic text.
- Write `purpose` in Arabic — it tells the downstream extractor what specific information to look for.
- Make `purpose` specific: name the exact piece of information being sought.

## Output format
```json
{
  "thinking": "brief reasoning in English",
  "sub_queries": [
    { "query": "...", "purpose": "..." }
  ]
}
```
"""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior conversation)"
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Bot"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def plan(query: str, history: list[dict], trace=None) -> PlannerOutput:
    """Decompose the query into the minimum set of disjoint sub-queries."""
    client = _get_oai()

    history_text = _format_history(history)
    user_content = f"## Conversation history\n{history_text}\n\n## Current query\n{query}"

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    if trace is not None:
        gen = trace.generation(
            name="planner",
            model=PLANNER_MODEL,
            input=messages,
        )

    response = client.beta.chat.completions.parse(
        model=PLANNER_MODEL,
        messages=messages,
        response_format=_PlannerSchema,
        temperature=0,
    )

    parsed = response.choices[0].message.parsed
    usage  = response.usage

    if trace is not None:
        gen.end(
            output=parsed.model_dump(),
            usage={"input": usage.prompt_tokens, "output": usage.completion_tokens},
        )

    return {
        "sub_queries": [
            {"query": sq.query, "purpose": sq.purpose}
            for sq in parsed.sub_queries
        ]
    }
