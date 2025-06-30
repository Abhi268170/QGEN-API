# This file defines the Pydantic models for your API request and response bodies.
# This helps with data validation and documentation.

from pydantic import BaseModel
from typing import List, TypedDict, Annotated, Optional, Dict, Any
import operator

class GraphInput(BaseModel):
    """
    The input schema for the graph.
    """
    input: str

class QuestionGenerationInput(BaseModel):
    """
    The input schema for generating questions from a PDF.
    """
    topic: str
    question_type: str
    num_questions: int
    difficulty: str
    section_ordering: List[str]
    section_marks: List[int]

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        input: The initial input string.
        processed_input: The input after being processed by the start_node.
    """
    input: str
    processed_input: Annotated[str, operator.add]

class QuestionGenerationState(TypedDict):
    """
    Represents the state of the question generation graph.
    """
    pdf_file: bytes
    topic: str
    question_type: str
    num_questions: int
    difficulty: str
    section_ordering: List[str]
    section_marks: List[int]
    images: List[bytes]
    embeddings: List[List[float]]
    collection_name: Optional[str]
    relevant_images: List[bytes]
    error: Optional[str]
    document_images: Dict[str, bytes]
    questions: List[Dict[str, Any]]
    question_paper: Optional[Dict[str, Any]]