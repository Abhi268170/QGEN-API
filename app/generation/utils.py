"""Utility functions for question generation."""

import base64
import json
import logging
from typing import Dict, Optional, List
from datetime import datetime
from .base import QUESTION_TYPE_CONFIG

logger = logging.getLogger(__name__)

def encode_image(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def calculate_marks_per_question(
    question_type: str, 
    difficulty: str, 
    section_marks: Optional[int] = None, 
    num_questions_in_section: int = 1
) -> int:
    """Calculate marks for a question based on type and difficulty."""
    config = QUESTION_TYPE_CONFIG.get(question_type.lower(), QUESTION_TYPE_CONFIG["mcq"])
    base_marks = config["base_marks"]
    
    marks = base_marks.get(difficulty.lower(), 2)
    
    # If section marks are specified, ensure we don't exceed them
    if section_marks:
        max_marks_per_question = section_marks // num_questions_in_section
        marks = min(marks, max_marks_per_question)
    
    return marks

def parse_llm_response(content: str) -> Dict:
    """Parse the LLM response and extract questions."""
    # Handle potential markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    # Clean up content
    content = content.strip()
    if not content:
        logger.error("Empty response content")
        return {"questions": []}
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Content: {content[:500]}...")
        return {"questions": []}

def create_question_paper(
    questions: List[Dict], 
    topic: str,
    section_ordering: Optional[List[str]] = None,
    section_marks: Optional[List[int]] = None
) -> Dict:
    """Create a structured question paper."""
    
    question_paper = {
        "title": f"Question Paper - {topic}",
        "date": datetime.now().isoformat(),
        "total_questions": len(questions),
        "total_marks": sum(q.get("marks", 0) for q in questions),
        "instructions": [
            "Answer all questions.",
            "Read each question carefully before answering.",
            "Marks are indicated for each question."
        ]
    }
    
    if section_ordering:
        # Organize by sections
        sections = []
        for section_name in section_ordering:
            section_questions = [q for q in questions if q.get("section") == section_name]
            if section_questions:
                sections.append({
                    "name": section_name,
                    "questions": section_questions,
                    "total_marks": sum(q.get("marks", 0) for q in section_questions)
                })
        question_paper["sections"] = sections
    else:
        question_paper["questions"] = questions
    
    return question_paper

def prepare_image_content(images: List[bytes]) -> List[Dict]:
    """Prepare image content for API request."""
    image_content = []
    
    for i, image_bytes in enumerate(images):
        if not image_bytes:
            logger.warning(f"Skipping empty image {i}")
            continue
            
        image_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image_bytes)}"
            }
        })
    
    return image_content