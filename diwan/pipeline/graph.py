"""
LangGraph pipeline graph for Diwan.

Flow:
  START → router_node
    مالي_استرجاع  → planner_node → Send × N → retrieve_and_extract_node → synthesizer_node → END
    اجتماعي / عام / مالي_مباشر   → direct_response_node → END
    خارج_النطاق                  → redirect_node → END

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
    return {"route": result["route"]}


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


_DIRECT_RESPONSE_PROMPTS: dict[str, str] = {
    "اجتماعي": """\
أنت مساعد ودود لنظام أسئلة وأجوبة حول الوثائق المالية والمحاسبية.
الرسالة الحالية اجتماعية أو غير رسمية. رد بشكل طبيعي ومختصر باللغة العربية.
لا تتطوع بمعلومات مالية لم يُطلب منك.""",

    "عام": """\
أنت مساعد متخصص في نظام أسئلة وأجوبة يغطي وثيقتين:
- معايير المحاسبة المصرية 2020: معايير تنظيمية، مواد، تعريفات، قواعد القياس والإفصاح.
- القوائم المالية المنفردة لبنك CIB: الميزانية العمومية، قائمة الدخل، الإيضاحات، الإفصاحات.

أجب على سؤال المستخدم عن النظام باختصار ووضوح باللغة العربية.
إذا كان السؤال عما يغطيه النظام، اذكر الوثيقتين والمواضيع الرئيسية التي تتناولها.""",

    "مالي_مباشر": """\
أنت مساعد مالي. الإجابة على سؤال المستخدم موجودة بالفعل في سياق المحادثة السابقة.
استخرجها من تاريخ المحادثة وأجب مباشرةً باللغة العربية دون البحث في وثائق إضافية.
لا تخترع معلومات غير موجودة في السياق.""",
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


_REDIRECT_MSG = (
    "عذراً، هذا السؤال خارج نطاق ما يغطيه النظام. "
    "أنا متخصص في الإجابة عن أسئلة تتعلق بمعايير المحاسبة المصرية 2020 "
    "والقوائم المالية المنفردة لبنك CIB. "
    "هل يمكنني مساعدتك في موضوع من هذه الوثائق؟"
)


def redirect_node(state: GraphState, config: RunnableConfig) -> dict:
    return {
        "final_answer": _REDIRECT_MSG,
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=_REDIRECT_MSG)],
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_after_router(state: GraphState) -> str:
    r = state["route"]
    if r == "مالي_استرجاع":
        return "planner"
    if r == "خارج_النطاق":
        return "redirect"
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

    g.add_node("router",              router_node)
    g.add_node("planner",             planner_node)
    g.add_node("retrieve_and_extract", retrieve_and_extract_node)
    g.add_node("synthesizer",         synthesizer_node)
    g.add_node("direct_response",     direct_response_node)
    g.add_node("redirect",            redirect_node)

    g.add_edge(START, "router")

    g.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "planner":         "planner",
            "direct_response": "direct_response",
            "redirect":        "redirect",
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
    g.add_edge("redirect",             END)

    return g


def create_graph(redis_url: str = REDIS_URL):
    """Build and compile the graph with a Redis checkpointer."""
    g = _build_graph()
    checkpointer = RedisSaver.from_conn_string(redis_url)
    checkpointer.setup()
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
