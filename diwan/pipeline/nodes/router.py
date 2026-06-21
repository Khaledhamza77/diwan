"""
Router node — classifies every incoming query before any retrieval runs.

Routes:
  اجتماعي      — greetings, small talk       → direct LLM response
  عام           — questions about the system  → direct LLM response
  مالي_مباشر   — answer already in history   → direct LLM response
  مالي_استرجاع — requires corpus retrieval   → planner
  خارج_النطاق  — off-domain                 → polite redirect
"""

from typing import Literal, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

ROUTER_MODEL = "gpt-5.4-mini"

_oai: OpenAI | None = None


def _get_oai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI()
    return _oai


Route = Literal["اجتماعي", "عام", "مالي_مباشر", "مالي_استرجاع", "خارج_النطاق"]


class RouterOutput(TypedDict):
    route: Route
    reasoning: str
    confidence: float


class _RouterSchema(BaseModel):
    route: Literal["اجتماعي", "عام", "مالي_مباشر", "مالي_استرجاع", "خارج_النطاق"]
    reasoning: str
    confidence: float


_SYSTEM = """\
You are a query router for an Arabic-language financial document Q&A system.

The system covers two document types:
- Egyptian Accounting Standards 2020 (معايير المحاسبة المصرية): regulatory standards, articles, \
definitions, measurement and disclosure rules.
- CIB Bank standalone financial statements (القوائم المالية المنفردة): balance sheet, income \
statement, notes, disclosures.

Your job: classify the user's current query into exactly one route.

## Routes

**اجتماعي**
Greetings, thanks, small talk, pleasantries. No financial content.
Examples: "مرحبا", "شكراً", "كيف حالك"

**عام**
Questions about this system — what it covers, which documents are available, how it works.
Examples: "ما هي الوثائق المتاحة؟", "هل تغطي معايير المحاسبة؟", "ماذا يمكنك أن تفعل؟"

**مالي_مباشر**
An in-scope financial or regulatory question whose answer has already been stated explicitly \
in a prior bot turn in the conversation history.
Use this route ONLY when the exact answer appears as a concrete statement in a prior bot turn — \
not merely when the topic was mentioned.

**مالي_استرجاع**
An in-scope financial or regulatory question that requires searching the document corpus.
Use for any question about accounting standards, financial figures, notes, disclosures, or \
regulatory rules — even if the topic appeared in history but the precise answer was not stated.

**خارج_النطاق**
Off-domain, non-financial, or questions about entities/topics not covered by the two documents.
Examples: live stock prices, economic forecasts, competitor banks, general knowledge.

## Decision rules
1. Never route مالي_مباشر unless the concrete answer is already in a prior bot turn.
2. A follow-up referencing a new entity or new aspect of a prior topic → مالي_استرجاع.
3. A question that sounds financial but is not about the covered documents → خارج_النطاق.
4. When in doubt between مالي_مباشر and مالي_استرجاع, choose مالي_استرجاع.

## Examples

User: "كيف حالك؟"
→ { "route": "اجتماعي", "reasoning": "تحية اجتماعية بلا محتوى مالي.", "confidence": 0.99 }

User: "ما القيمة العادلة لأصول البنك التجاري الدولي في 2023؟" | History: (empty)
→ { "route": "مالي_استرجاع", "reasoning": "سؤال عن رقم مالي يستلزم البحث في القوائم المالية.", "confidence": 0.97 }

User: "كررها مرة أخرى" | Prior bot turn: "صافي الربح 5.2 مليار جنيه"
→ { "route": "مالي_مباشر", "reasoning": "الإجابة مذكورة صراحةً في رد سابق من النظام.", "confidence": 0.95 }

User: "ما هو سعر سهم CIB اليوم؟"
→ { "route": "خارج_النطاق", "reasoning": "أسعار الأسهم الآنية خارج نطاق الوثائق المتاحة.", "confidence": 0.98 }

User: "ما تعريف الأداة المالية حسب المعيار المصري؟" | History: bot mentioned معيار 25 but gave no definition
→ { "route": "مالي_استرجاع", "reasoning": "التعريف لم يُذكر صراحةً في المحادثة ويستلزم البحث.", "confidence": 0.96 }

User: "هل تغطي نتائج 2022؟" | History: bot said "أغطي القوائم المالية لعام 2023 فقط"
→ { "route": "مالي_مباشر", "reasoning": "النظام أجاب صراحةً عن نطاق التغطية الزمنية في رد سابق.", "confidence": 0.93 }
"""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior conversation)"
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Bot"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def route(query: str, history: list[dict], trace=None) -> RouterOutput:
    """Classify the query into one of five routes."""
    client = _get_oai()

    history_text = _format_history(history)
    user_content = f"## Conversation history\n{history_text}\n\n## Current query\n{query}"

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    if trace is not None:
        gen = trace.generation(
            name="router",
            model=ROUTER_MODEL,
            input=messages,
        )

    response = client.beta.chat.completions.parse(
        model=ROUTER_MODEL,
        messages=messages,
        response_format=_RouterSchema,
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
        "route":      parsed.route,
        "reasoning":  parsed.reasoning,
        "confidence": parsed.confidence,
    }
