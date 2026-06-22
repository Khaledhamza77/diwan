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
    response: str | None  # populated only when route is خارج_النطاق


class _RouterSchema(BaseModel):
    route: Literal["اجتماعي", "عام", "مالي_مباشر", "مالي_استرجاع", "خارج_النطاق"]
    reasoning: str
    confidence: float
    response: str | None  # populated only when route is خارج_النطاق


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
NEVER use عام for elaboration requests ("ياريت", "وضح أكتر", "كمّل", "شرح أكتر") when the \
prior bot turn was about a financial concept — those belong to مالي_استرجاع.

**مالي_مباشر**
Use this route ONLY when ALL THREE conditions are simultaneously true:
  (a) The question is in-scope (about the two covered documents).
  (b) A prior bot turn contains the concrete CITED answer to THIS SPECIFIC question as an \
      explicit statement — not merely a turn that mentions the topic in passing or answers \
      a different question.
  (c) That SAME prior bot turn contains at least one literal [N] citation marker \
      (e.g., [1] or [2]) embedded in its text — proving the specific answer was retrieved from documents.

If ANY condition is missing — especially (b) or (c) — you MUST route to مالي_استرجاع.
Mentioning a topic, listing document names, or giving a general explanation without [N] \
markers does NOT satisfy condition (c). A bot turn that describes what documents are \
available is a system explanation, not a cited answer.
Citations from a bot turn that answered a DIFFERENT question do NOT satisfy condition (b) \
even if those citations are present in history. The cited turn must answer THIS question specifically.

**مالي_استرجاع**
An in-scope financial or regulatory question that requires searching the document corpus.
Use for any question about accounting standards, financial figures, notes, disclosures, \
regulatory rules, or definitions — even if the topic appeared in history — unless the \
prior bot turn already answered it WITH [N] citations (see مالي_مباشر above).

**خارج_النطاق**
Off-domain, non-financial, or questions about entities/topics not covered by the two documents.
Examples: live stock prices, economic forecasts, competitor banks, general knowledge.

## Decision rules
1. CITATION GATE: Before choosing مالي_مباشر, find the prior bot turn that specifically \
   answered THIS question. That turn must contain a literal [N] pattern (e.g., [1], [2], [3]). \
   Citations in other turns that answered different questions do NOT count — مالي_مباشر is \
   forbidden unless the turn answering THIS specific question has [N] markers.
2. A follow-up referencing a new entity or new aspect of a prior topic → مالي_استرجاع.
3. A question that sounds financial but is not about the covered documents → خارج_النطاق.
4. When in doubt between مالي_مباشر and مالي_استرجاع, choose مالي_استرجاع.
5. ELABORATION GATE: An elaboration request ("ياريت", "وضح أكتر", "كمّل", "شرح أكتر") \
   follows the most recent financial topic in conversation. If the most recent bot turn on \
   that topic lacks [N] citations, route to مالي_استرجاع — never to عام.

## Response field
- When route is **خارج_النطاق**: populate `response` with a single polite Arabic sentence \
that declines the question and redirects the user to the two covered documents.
- For all other routes: set `response` to null.

## Examples

User: "كيف حالك؟"
→ { "route": "اجتماعي", "reasoning": "تحية اجتماعية بلا محتوى مالي.", "confidence": 0.99 }

User: "ما القيمة العادلة لأصول البنك التجاري الدولي في 2023؟" | History: (empty)
→ { "route": "مالي_استرجاع", "reasoning": "سؤال عن رقم مالي يستلزم البحث في القوائم المالية.", "confidence": 0.97 }

User: "كررها مرة أخرى" | Prior bot turn: "صافي الربح 5.2 مليار جنيه [1]\n---\n[1] القوائم المالية، ص. 5"
→ { "route": "مالي_مباشر", "reasoning": "الإجابة مذكورة صراحةً في رد سابق مستشهَد به [1].", "confidence": 0.95 }

User: "كررها مرة أخرى" | Prior bot turn: "صافي الربح 5.2 مليار جنيه" (no citations)
→ { "route": "مالي_استرجاع", "reasoning": "الرد السابق لا يحتوي على استشهادات موثقة فلا يُعتدّ به.", "confidence": 0.90 }

User: "ما هو سعر سهم CIB اليوم؟"
→ { "route": "خارج_النطاق", "reasoning": "أسعار الأسهم الآنية خارج نطاق الوثائق المتاحة.", "confidence": 0.98, "response": "عذراً، لا أستطيع الإجابة عن أسعار الأسهم، فأنا متخصص في معايير المحاسبة المصرية 2020 والقوائم المالية المنفردة لبنك CIB." }

User: "ما تعريف الأداة المالية حسب المعيار المصري؟" | History: bot mentioned معيار 25 but gave no definition
→ { "route": "مالي_استرجاع", "reasoning": "التعريف لم يُذكر صراحةً في المحادثة ويستلزم البحث.", "confidence": 0.96 }

User: "هل تغطي نتائج 2022؟" | History: bot said "أغطي القوائم المالية لعام 2023 فقط"
→ { "route": "مالي_استرجاع", "reasoning": "الرد السابق لا يحتوي على استشهادات [N] موثقة، لذا يُعاد البحث.", "confidence": 0.88 }

User: "يعني ايه قوائم مالية؟" | History: bot listed document coverage including "قائمة المركز المالي، قائمة الدخل" with no [N] markers
→ { "route": "مالي_استرجاع", "reasoning": "الرد السابق وصف تغطية النظام فقط دون استشهادات [N]، والسؤال يطلب تعريفاً يستلزم البحث في الوثائق.", "confidence": 0.97 }

User: "ايه القوائم المالية دي؟" | History: bot answered "معايير المحاسبة هي... [1][2][3][4]" (cited, but about a DIFFERENT topic)
→ { "route": "مالي_استرجاع", "reasoning": "الاستشهادات [N] الموجودة في التاريخ تخص سؤالاً مختلفاً عن معايير المحاسبة، وليس عن القوائم المالية. لا يوجد رد مستشهَد به يُجيب على هذا السؤال تحديداً.", "confidence": 0.97 }

User: "ياريت" | History: bot answered "معايير المحاسبة هي... [1][2][3][4]" then bot answered "القوائم المالية هي تقارير مالية..." (no citations)
→ { "route": "مالي_استرجاع", "reasoning": "الرد الأخير عن القوائم المالية لا يحتوي على استشهادات [N]. طلب التوسع يخص هذا الرد غير الموثق، فيجب إعادة البحث.", "confidence": 0.95 }
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
        "response":   parsed.response,
    }
