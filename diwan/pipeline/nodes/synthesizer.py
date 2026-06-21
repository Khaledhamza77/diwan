"""
Synthesizer node — produces the final Arabic answer with inline citations.

Model: gpt-5.4 (composition quality; strongest model)
Prompt: Arabic, strict grounding constraint, numbered footnote per factual claim
Output: free-text Arabic prose + numbered source list
"""

from dotenv import load_dotenv
from openai import OpenAI

from diwan.pipeline.nodes.lookup import ContextChunk

load_dotenv()

SYNTHESIZER_MODEL = "gpt-5.4"

_oai: OpenAI | None = None


def _get_oai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI()
    return _oai


_SYSTEM = """\
أنت مساعد متخصص في الإجابة عن الأسئلة المالية والمحاسبية بناءً على وثائق محددة.

## قيد التأسيس
أجب **فقط** بناءً على المقتطفات المقدمة إليك. لا تستخدم أي معلومات خارجية حتى لو كنت متأكداً منها.

## قواعد الاستشهاد
- كل ادعاء أو معلومة مستمدة من المقتطفات يحمل رقماً مرجعياً بين قوسين معقوفين مثل [1] أو [2]
- إذا دعم أكثر من مقتطع نفس الادعاء، اذكر جميع الأرقام مثل [1][3]
- أدرج قائمة المصادر رقمياً في نهاية الإجابة

## قواعد الإجابة الجزئية
- إذا لم تغطِ المقتطفات جوانب من السؤال، صرّح بذلك صراحةً: "لا تتوفر في الوثائق المتاحة معلومات عن..."
- لا تملأ الفراغات باستنتاجات أو معلومات خارجية

## تنسيق الإجابة
اكتب إجابتك في شكل نثر عربي متماسك، ثم اختم بـ:

---
[1] اسم الوثيقة، ص. X
[2] اسم الوثيقة، ص. Y
"""


def _build_user_prompt(query: str, context: list[ContextChunk]) -> str:
    lines = [f"## السؤال\n{query}", "\n## المقتطفات"]
    for i, chunk in enumerate(context, start=1):
        pages = sorted({b["page"] for b in chunk["bboxes"]})
        page_str = "، ".join(f"ص. {p}" for p in pages)
        lines.append(
            f"\n[{i}] {chunk['doc_name']} — {page_str}"
            f"\n(الاستعلام الفرعي: {chunk['sub_query']})"
            f"\n{chunk['markdown']}"
        )
    return "\n".join(lines)


def synthesize(query: str, context: list[ContextChunk], trace=None) -> str:
    """Generate the final Arabic answer with inline citations."""
    client = _get_oai()

    user_content = _build_user_prompt(query, context)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    if trace is not None:
        gen = trace.generation(
            name="synthesizer",
            model=SYNTHESIZER_MODEL,
            input=messages,
        )

    response = client.chat.completions.create(
        model=SYNTHESIZER_MODEL,
        messages=messages,
        temperature=0,
    )

    answer = response.choices[0].message.content
    usage  = response.usage

    if trace is not None:
        gen.end(
            output=answer,
            usage={"input": usage.prompt_tokens, "output": usage.completion_tokens},
        )

    return answer
