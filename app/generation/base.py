# base.py
"""Base classes and constants for question generation."""

from typing import Dict, List, Optional
from dataclasses import dataclass

# Visual keywords to filter out
VISUAL_KEYWORDS = [
    "figure", "diagram", "image", "picture", "illustration",
    "photo", "shown", "depicted", "displayed", "visible",
    "see", "look at", "observe the", "in the", "as shown",
    "as illustrated", "the above", "the below", "referring to",
    "fig.", "fig ", "figure ", "diagram "
]

# Question type configurations with proper structures
QUESTION_TYPE_CONFIG = {
    "mcq": {
        "base_marks": {"easy": 1, "medium": 2, "hard": 3},
        "has_options": True,
        "num_options": 4,
        "multiple_answers": False,
        "structure": {
            "question": str,
            "options": List[str],  # Exactly 4 options
            "correct_answer": str  # Single answer (A, B, C, or D)
        }
    },
    "msq": {
        "base_marks": {"easy": 2, "medium": 3, "hard": 4},
        "has_options": True,
        "num_options": [4, 5],  # Can have 4 or 5 options
        "multiple_answers": True,
        "structure": {
            "question": str,
            "options": List[str],  # 4-5 options
            "correct_answer": List[str]  # Multiple answers (e.g., ["A", "C"])
        }
    },
    "short_answer": {
        "base_marks": {"easy": 2, "medium": 4, "hard": 6},
        "has_options": False,
        "multiple_answers": False,
        "structure": {
            "question": str,
            "correct_answer": str  # No options, just the answer text
        }
    },
    "yes_no": {
        "base_marks": {"easy": 1, "medium": 1, "hard": 2},
        "has_options": True,
        "num_options": 2,
        "multiple_answers": False,
        "structure": {
            "question": str,  # This is actually a statement
            "options": List[str],  # Always ["Yes", "No"]
            "correct_answer": str  # Either "Yes" or "No"
        }
    }
}

@dataclass
class SectionInfo:
    """Information about a question paper section."""
    name: str
    marks: int
    
@dataclass
class GenerationConfig:
    """Configuration for question generation."""
    base_url: str
    model: str
    api_key: str
    batch_size: int = 10
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9

def get_question_structure(question_type: str) -> Dict:
    """Get the expected structure for a question type."""
    config = QUESTION_TYPE_CONFIG.get(question_type.lower())
    if not config:
        raise ValueError(f"Unknown question type: {question_type}")
    
    base_structure = {
        "type": question_type,
        "marks": 0,  # Will be calculated based on difficulty
        "difficulty": ""  # Will be set during generation
    }
    
    # Add type-specific fields
    if question_type.lower() == "mcq":
        base_structure.update({
            "question": "",
            "options": ["A. ", "B. ", "C. ", "D. "],
            "correct_answer": ""  # Single letter
        })
    elif question_type.lower() == "msq":
        base_structure.update({
            "question": "",
            "options": [],  # Will have 4-5 options
            "correct_answer": []  # List of letters
        })
    elif question_type.lower() == "short_answer":
        base_structure.update({
            "question": "",
            "correct_answer": ""  # Text answer, no options
        })
    elif question_type.lower() == "yes_no":
        base_structure.update({
            "question": "",  # Statement to evaluate
            "options": ["Yes", "No"],
            "correct_answer": ""  # Either "Yes" or "No"
        })
    
    return base_structure