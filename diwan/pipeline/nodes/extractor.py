"""
Extractor node — selects the relevant chunk IDs from the retriever's top-k candidates.

Model: gpt-5.4-mini (high-volume, mechanical selection task)
Prompt: English instructions; chunk content is Arabic
Output: list of chunk_ids that directly address the sub-query
"""

from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from diwan.retrieval.retriever import RetrievedChunk

load_dotenv()

EXTRACTOR_MODEL = "gpt-5.4-mini"

_oai: OpenAI | None = None


def _get_oai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI()
    return _oai


class ExtractorOutput(TypedDict):
    sub_query: str
    purpose: str
    chunk_ids: list[str]


class _ExtractorSchema(BaseModel):
    indices: list[int]


_SYSTEM = """\
You are a relevance filter for an Arabic financial document Q&A system.

You will receive a sub-query (with its stated purpose) and a numbered list of candidate \
chunks retrieved from the document corpus. The chunks are in Arabic.

Your job: return the numbers of chunks that DIRECTLY address the sub-query.

## Selection rules
- Include a chunk only if it contains information that directly answers or substantively \
contributes to answering the sub-query.
- Do NOT include a chunk merely because it is topically related or shares keywords — \
it must contain relevant facts or definitions.
- If the same information appears in multiple chunks, include all of them.
- If none of the chunks are relevant, return an empty list.

## What you must NOT do
- Do NOT generate, infer, paraphrase, or summarize content.
- Do NOT return numbers outside the range of the provided list.
- Do NOT reason about what the answer should be.

## Output
{ "indices": [1, 3] }
"""


def _build_user_prompt(sub_query: str, purpose: str, chunks: list[RetrievedChunk]) -> str:
    lines = [
        f"## Sub-query\n{sub_query}",
        f"\n## Purpose\n{purpose}",
        "\n## Candidate chunks",
    ]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"\n[{i}]\n{chunk['markdown']}")
    return "\n".join(lines)


def extract(
    sub_query: str,
    purpose: str,
    chunks: list[RetrievedChunk],
    trace=None,
) -> ExtractorOutput:
    """Select chunks relevant to the sub-query; model returns indices, we map to IDs."""
    client = _get_oai()

    user_content = _build_user_prompt(sub_query, purpose, chunks)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    span_name = f"extractor/{sub_query[:60]}"

    if trace is not None:
        gen = trace.generation(
            name=span_name,
            model=EXTRACTOR_MODEL,
            input=messages,
        )

    response = client.beta.chat.completions.parse(
        model=EXTRACTOR_MODEL,
        messages=messages,
        response_format=_ExtractorSchema,
        temperature=0,
    )

    parsed = response.choices[0].message.parsed
    usage  = response.usage

    # Map 1-based indices to chunk IDs; drop any out-of-range values.
    chunk_ids = [
        chunks[i - 1]["chunk_id"]
        for i in parsed.indices
        if 1 <= i <= len(chunks)
    ]

    if trace is not None:
        gen.end(
            output={"indices": parsed.indices, "chunk_ids": chunk_ids},
            usage={"input": usage.prompt_tokens, "output": usage.completion_tokens},
        )

    return {
        "sub_query": sub_query,
        "purpose":   purpose,
        "chunk_ids": chunk_ids,
    }
