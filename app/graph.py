# This file contains the core logic of your LangGraph application.
# You will define the nodes, edges, and the graph itself in this file.

from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
import logging
import functools

# Import configuration
try:
    from .config import (
        QDRANT_MEMORY_LIMIT
    )
    from .schema import QuestionGenerationState, GraphState
    from .pdf_processing import pdf_to_images_node
    from .embedding import embed_images_node
    from .qdrant import add_to_qdrant_node, retrieve_relevant_images_node
    from .generation import generation_node
except ImportError:
    QDRANT_MEMORY_LIMIT = ":memory:"
    class QuestionGenerationState(dict): pass
    class GraphState(dict): pass
    def pdf_to_images_node(state): return state
    def embed_images_node(state): return state
    def add_to_qdrant_node(state, client): return state
    def retrieve_relevant_images_node(state, client): return state
    def generation_node(state): return state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_question_generation_graph():
    """
    Builds and returns the question generation graph.
    """
    workflow = StateGraph(QuestionGenerationState)

    qdrant_client_instance = QdrantClient(QDRANT_MEMORY_LIMIT)

    workflow.add_node("pdf_to_images", pdf_to_images_node)
    workflow.add_node("embed_images", embed_images_node)
    workflow.add_node("add_to_qdrant", functools.partial(add_to_qdrant_node, client=qdrant_client_instance))
    workflow.add_node("retrieve_relevant_images", functools.partial(retrieve_relevant_images_node, client=qdrant_client_instance))
    workflow.add_node("generate_questions", generation_node)

    workflow.set_entry_point("pdf_to_images")
    workflow.add_edge("pdf_to_images", "embed_images")
    workflow.add_edge("embed_images", "add_to_qdrant")
    workflow.add_edge("add_to_qdrant", "retrieve_relevant_images")
    workflow.add_edge("retrieve_relevant_images", "generate_questions")
    workflow.add_edge("generate_questions", END)

    app = workflow.compile()
    return app


def start_node(state: GraphState):
    """
    A simple node that processes the initial input.
    """
    print("---START NODE---")
    processed = state['input'] + " (processed)"
    return {"processed_input": processed}


def middle_node(state: GraphState):
    """
    A simple middle node that adds a marker.
    """
    print("---MIDDLE NODE---")
    return {"processed_input": " [middle step]"}


def should_continue(state: GraphState):
    """
    A conditional edge that decides the next step.
    """
    print("---CHECKING CONDITION---")
    if len(state['processed_input']) > 20:
        return "end"
    else:
        return "continue"


def get_graph():
    """
    Builds and returns the LangGraph graph.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("start", start_node)
    workflow.add_node("middle", middle_node)

    workflow.set_entry_point("start")
    
    workflow.add_conditional_edges(
        "start",
        should_continue,
        {
            "continue": "middle",
            "end": END,
        },
    )
    
    workflow.add_edge("middle", END)

    app = workflow.compile()
    return app