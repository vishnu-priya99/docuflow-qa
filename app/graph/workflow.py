"""Builds the LangGraph question-answering workflow described in the spec:

START -> load_session -> router --SEMANTIC--> semantic_retrieval ---------------\
                              |                                                  \
                              |--STRUCTURED--> schema_selection                    \
                              |                    -> sql_generation                -> answer_generation -> END
                              |                         -> sql_validation          /
                              |                              -> sql_execution     /
                              |--HYBRID------> hybrid_discovery                  /
                                                   -> semantic_retrieval -------/
                                                        -> schema_selection
                                                             -> sql_generation -> ... -> sql_execution

The router only classifies; it never produces the final answer.

semantic_retrieval_node is shared by SEMANTIC and HYBRID (not just
SEMANTIC) - a HYBRID question needs both the document evidence it
gathers and the structured/SQL evidence gathered downstream, merged
before answer_generation (see sql_execution_node's own merge logic).
Routed there via hybrid_discovery first so HYBRID's sheet-discovery step
still runs before schema_selection needs it. semantic_retrieval then
branches: HYBRID continues on to schema_selection, plain SEMANTIC goes
straight to answer_generation, same as before.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.deps import GraphDeps
from app.graph.state import GraphState


def _route_from_classification(state: GraphState) -> str:
    return {
        "SEMANTIC": "semantic_retrieval",
        "STRUCTURED": "schema_selection",
        "HYBRID": "hybrid_discovery",
    }.get(state.get("question_type", "SEMANTIC"), "semantic_retrieval")


def _route_after_semantic_retrieval(state: GraphState) -> str:
    # HYBRID still needs its structured/SQL half after gathering document
    # evidence here; plain SEMANTIC is done once retrieval completes.
    return "schema_selection" if state.get("question_type") == "HYBRID" else "answer_generation"


@lru_cache
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("load_session", nodes.load_session_node)
    graph.add_node("router", nodes.router_node)
    graph.add_node("semantic_retrieval", nodes.semantic_retrieval_node)
    graph.add_node("hybrid_discovery", nodes.hybrid_discovery_node)
    graph.add_node("schema_selection", nodes.schema_selection_node)
    graph.add_node("sql_generation", nodes.sql_generation_node)
    graph.add_node("sql_validation", nodes.sql_validation_node)
    graph.add_node("sql_execution", nodes.sql_execution_node)
    graph.add_node("answer_generation", nodes.answer_generation_node)

    graph.add_edge(START, "load_session")
    graph.add_edge("load_session", "router")
    graph.add_conditional_edges(
        "router",
        _route_from_classification,
        {
            "semantic_retrieval": "semantic_retrieval",
            "schema_selection": "schema_selection",
            "hybrid_discovery": "hybrid_discovery",
        },
    )
    graph.add_conditional_edges(
        "semantic_retrieval",
        _route_after_semantic_retrieval,
        {
            "schema_selection": "schema_selection",
            "answer_generation": "answer_generation",
        },
    )
    graph.add_edge("hybrid_discovery", "semantic_retrieval")
    graph.add_edge("schema_selection", "sql_generation")
    graph.add_edge("sql_generation", "sql_validation")
    graph.add_edge("sql_validation", "sql_execution")
    graph.add_edge("sql_execution", "answer_generation")
    graph.add_edge("answer_generation", END)

    return graph.compile()


async def run_workflow(*, question: str, user_id: str, session_id: str, deps: GraphDeps) -> GraphState:
    compiled = build_graph()
    initial_state: GraphState = {
        "question": question,
        "user_id": user_id,
        "session_id": session_id,
    }
    result = await compiled.ainvoke(initial_state, config={"configurable": {"deps": deps}})
    return result
