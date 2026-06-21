"""
LangGraph pipeline graph for Diwan.

Flow:
  START → router_node
    مالي_استرجاع  → planner_node → Send × N → retrieve_and_extract_node → synthesizer_node → END
    اجتماعي / عام / مالي_مباشر   → direct_response_node → END
    خارج_النطاق                  → router responds inline → END

Chat history is persisted in Redis via RedisSaver, keyed by thread_id.
Langfuse trace is passed through config["configurable"]["trace"] so it spans the full run
without being serialised into Redis state.
"""

import os
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from openai import OpenAI

from diwan.pipeline.nodes.extractor import ExtractorOutput, extract
from diwan.pipeline.nodes.lookup import lookup
from diwan.pipeline.nodes.planner import SubQuery, plan
from diwan.pipeline.nodes.router import route
from diwan.pipeline.nodes.synthesizer import synthesize
from diwan.retrieval.retriever import RetrieverOutput, retrieve

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DIRECT_RESPONSE_MODEL = "gpt-5.4-mini"

_oai: OpenAI | None = None


def _get_oai() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI()
    return _oai


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _reset_or_add(existing: list, new: list | None) -> list:
    """Reducer for per-turn accumulators.

    Pass None to reset the field at the start of a new turn (clears the
    value carried over from the previous turn's checkpoint). Pass a list to
    append — used by parallel workers during fan-in within the same turn.
    """
    if new is None:
        return []
    return (existing or []) + new


class GraphState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    query:             str
    route:             str | None
    sub_queries:       list[SubQuery]
    retriever_outputs: Annotated[list[RetrieverOutput], _reset_or_add]
    extractor_results: Annotated[list[ExtractorOutput], _reset_or_add]
    final_answer:      str | None


class WorkerInput(TypedDict):
    """Payload passed to each parallel retrieve_and_extract worker via Send."""
    sub_query: str
    purpose:   str


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _get_trace(config: RunnableConfig) -> Any | None:
    return (config.get("configurable") or {}).get("trace")


def _history_dicts(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to the {role, content} dicts expected by node prompts."""
    result = []
    for m in messages:
        if isinstance(m, HumanMessage):
            result.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            result.append({"role": "assistant", "content": m.content})
    return result


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def router_node(state: GraphState, config: RunnableConfig) -> dict:
    trace = _get_trace(config)
    history = _history_dicts(state.get("messages") or [])
    result = route(state["query"], history, trace=trace)
    update: dict = {"route": result["route"]}
    if result["route"] == "خارج_النطاق":
        answer = result["response"] or (
            "عذراً، هذا السؤال خارج نطاق ما يغطيه النظام. "
            "أنا متخصص في معايير المحاسبة المصرية 2020 والقوائم المالية المنفردة لبنك CIB."
        )
        update["final_answer"] = answer
        update["messages"] = [HumanMessage(content=state["query"]), AIMessage(content=answer)]
    return update


def planner_node(state: GraphState, config: RunnableConfig) -> dict:
    trace = _get_trace(config)
    history = _history_dicts(state.get("messages") or [])
    result = plan(state["query"], history, trace=trace)
    return {"sub_queries": result["sub_queries"]}


def retrieve_and_extract_node(worker: WorkerInput, config: RunnableConfig) -> dict:
    """Single parallel worker: retrieve → extract for one sub-query."""
    trace = _get_trace(config)
    ro: RetrieverOutput = retrieve(worker["sub_query"], worker["purpose"])
    er: ExtractorOutput = extract(
        worker["sub_query"],
        worker["purpose"],
        ro["chunks"],
        trace=trace,
    )
    return {
        "retriever_outputs": [ro],
        "extractor_results": [er],
    }


def synthesizer_node(state: GraphState, config: RunnableConfig) -> dict:
    trace = _get_trace(config)
    context = lookup(state["extractor_results"], state["retriever_outputs"])
    answer = synthesize(state["query"], context, trace=trace)
    return {
        "final_answer": answer,
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=answer)],
    }


_GUARDRAILS = """\

**مبادئ السلوك:**
- تحلَّ دائماً بالاحترام والأدب واللباقة، بصرف النظر عن طبيعة السؤال أو أسلوب المستخدم.
- لا تُلفِّق معلومات أو أرقاماً أو اقتباسات تحت أي ظرف؛ إذا لم تكن المعلومة موثوقة من المصدر أو السياق، صرّح بذلك صراحةً.
- ردودك ذات طابع استرشادي وليست مشورة قانونية أو مالية أو مهنية ملزمة.
- إذا تضمّن الطلب ما قد يُفضي إلى غش أو تضليل أو ضرر، اعتذر بأدب وامتنع عن المساعدة."""

_PERSONA = """\
## الشخصية والأسلوب
أنت مساعد متخصص في الوثائق المالية والمحاسبية. شخصيتك تقوم على:
- **الرسمية المريحة**: لغتك فصيحة واضحة بعيدة عن العامية والمصطلحات المتكلِّفة — دافئة لكن مهنية في جميع الأحوال.
- **الإيجاز الذكي**: حجم ردّك يتناسب مع حجم السؤال؛ التحية تُقابَل بجملة أو جملتين، والسؤال المركّب يستحق إجابة مفصَّلة.
- **الوضوح أولاً**: ابدأ بجوهر الإجابة مباشرةً دون مقدمات، ثم أضف السياق إن لزم.
- **التركيز على النطاق**: لا تتطوع بمعلومات خارج اختصاصك، وعندما تقدّم نفسك أو تصف ما تغطيه اكتفِ بما يخدم حاجة المستخدم الآنية.\
"""

_DIRECT_RESPONSE_PROMPTS: dict[str, str] = {
    "اجتماعي": f"""\
{_PERSONA}

الرسالة الحالية اجتماعية أو غير رسمية. رد بما يناسب سياقها مع الحفاظ على الطابع المهني.
{_GUARDRAILS}""",

    "عام": f"""\
{_PERSONA}

أنت متخصص في وثيقتين فقط:

**معايير المحاسبة المصرية 2020**
صادرة بقرار رئيس الجهاز المركزي للمحاسبات رقم (732) لسنة 2020، منشورة في الوقائع المصرية العدد 143 بتاريخ 24 يونيو 2020.
تغطي إطار إعداد وعرض القوائم المالية، ومجموعة شاملة من المعايير تتناول: عرض القوائم المالية، المخزون، التدفقات النقدية، السياسات المحاسبية، الأصول الثابتة والأصول غير الملموسة، الأدوات المالية (عرضاً واعترافاً وقياساً)، ضرائب الدخل، المخصصات، تجميع الأعمال، القوائم المالية الدورية، اضمحلال قيمة الأصول، الأطراف ذوي العلاقة، العملات الأجنبية، تكاليف الاقتراض، الاستثمار العقاري، وغيرها.

**القوائم المالية المستقلة الدورية المختصرة لبنك CIB (الربع الأول 2026)**
البنك التجاري الدولي - مصر (CIB)، عن الفترة المنتهية في 31 مارس 2026، معدة وفق قواعد البنك المركزي المصري.
تشمل قائمة المركز المالي، قائمة الدخل والدخل الشامل، قائمة التدفقات النقدية، قائمة التغير في حقوق الملكية، وإيضاحات تفصيلية على جميع بنود القوائم.

أجب على سؤال المستخدم بما يتناسب مع عمق سؤاله: سؤال عام يستحق إجابة موجزة، وسؤال تفصيلي عن التغطية يستحق الوصف الكامل أعلاه.
{_GUARDRAILS}""",

    "مالي_مباشر": f"""\
أنت مساعد مالي متخصص في وثيقتين: معايير المحاسبة المصرية 2020 والقوائم المالية المستقلة الدورية المختصرة لبنك CIB عن الربع الأول 2026.
الإجابة على سؤال المستخدم موجودة بالفعل في سياق المحادثة السابقة.

## التعليمات
- استخرج الإجابة من رد النظام السابق وأعد صياغتها بشكل واضح ومباشر باللغة العربية.
- احتفظ بأرقام الاستشهادات المضمّنة في الإجابة الأصلية (مثل [1] أو [2]) في مواضعها الصحيحة داخل النص.
- اختم إجابتك بقائمة المصادر كاملةً كما وردت في رد النظام السابق، بالتنسيق الآتي:

---
[1] اسم الوثيقة، ص. X
[2] اسم الوثيقة، ص. Y

- لا تخترع أرقام صفحات أو مصادر غير موجودة في السياق.
- لا تُضف معلومات جديدة لم ترد في الرد الأصلي.
{_GUARDRAILS}""",
}


def direct_response_node(state: GraphState, config: RunnableConfig) -> dict:
    route_key = state["route"]
    system_prompt = _DIRECT_RESPONSE_PROMPTS.get(route_key, _DIRECT_RESPONSE_PROMPTS["اجتماعي"])

    history = _history_dicts(state.get("messages") or [])
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append(turn)
    messages.append({"role": "user", "content": state["query"]})

    trace = _get_trace(config)
    if trace is not None:
        gen = trace.generation(
            name=f"direct_response/{route_key}",
            model=DIRECT_RESPONSE_MODEL,
            input=messages,
        )

    response = _get_oai().chat.completions.create(
        model=DIRECT_RESPONSE_MODEL,
        messages=messages,
        temperature=0,
    )
    answer = response.choices[0].message.content
    usage = response.usage

    if trace is not None:
        gen.end(
            output=answer,
            usage={"input": usage.prompt_tokens, "output": usage.completion_tokens},
        )

    return {
        "final_answer": answer,
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_after_router(state: GraphState) -> str:
    r = state["route"]
    if r == "مالي_استرجاع":
        return "planner"
    if r == "خارج_النطاق":
        return END
    return "direct_response"


def _fan_out_after_planner(state: GraphState) -> list[Send]:
    return [
        Send("retrieve_and_extract", {"sub_query": sq["query"], "purpose": sq["purpose"]})
        for sq in state["sub_queries"]
    ]


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    g.add_node("router",               router_node)
    g.add_node("planner",              planner_node)
    g.add_node("retrieve_and_extract", retrieve_and_extract_node)
    g.add_node("synthesizer",          synthesizer_node)
    g.add_node("direct_response",      direct_response_node)

    g.add_edge(START, "router")

    g.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "planner":         "planner",
            "direct_response": "direct_response",
            END:               END,
        },
    )

    g.add_conditional_edges(
        "planner",
        _fan_out_after_planner,
        ["retrieve_and_extract"],
    )

    g.add_edge("retrieve_and_extract", "synthesizer")
    g.add_edge("synthesizer",          END)
    g.add_edge("direct_response",      END)

    return g


def create_graph(redis_url: str = REDIS_URL):
    """Build and compile the graph with a Redis checkpointer."""
    g = _build_graph()
    # Instantiate directly — from_conn_string is a context manager that closes
    # the connection on exit, which would break the long-lived compiled graph.
    checkpointer = RedisSaver(redis_url=redis_url)
    try:
        checkpointer.setup()
    except Exception as exc:
        # redisvl raises if the search indices already exist (non-idempotent).
        # Safe to ignore — indices from a prior run are still valid.
        if "already exists" not in str(exc).lower():
            raise
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    query: str,
    thread_id: str,
    graph=None,
    trace=None,
) -> str:
    """
    Invoke the graph for one user turn.

    Args:
        query:     The user's message.
        thread_id: LangGraph thread id — maps to a Chainlit session id.
        graph:     Compiled graph instance (create once, reuse).
        trace:     Optional Langfuse trace object for the current request.

    Returns:
        The final answer string.
    """
    if graph is None:
        graph = create_graph()

    config = {
        "configurable": {
            "thread_id": thread_id,
            "trace":     trace,
        }
    }

    # retriever_outputs / extractor_results: None triggers _reset_or_add to clear
    # the values persisted from the previous turn before workers accumulate into them.
    # messages: [] leaves history unchanged; terminal node appends [Human, AI] pair.
    input_state = {
        "query":             query,
        "route":             None,
        "sub_queries":       [],
        "retriever_outputs": None,
        "extractor_results": None,
        "final_answer":      None,
        "messages":          [],
    }

    result = graph.invoke(input_state, config=config)
    return result["final_answer"]
