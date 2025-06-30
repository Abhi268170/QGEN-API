"""Question validation and cleaning utilities."""

import logging
from typing import List, Dict
from .base import VISUAL_KEYWORDS

logger = logging.getLogger(__name__)

class QuestionValidator:
    """Validates and cleans generated questions."""
    
    def __init__(self):
        self.visual_keywords = VISUAL_KEYWORDS
    
    def validate_and_clean_questions(self, questions: List[Dict]) -> List[Dict]:
        """Validate questions and remove any that contain visual references."""
        cleaned_questions = []
        
        for q in questions:
            if self._is_valid_question(q):
                cleaned_questions.append(q)
        
        return cleaned_questions
    
    def _is_valid_question(self, question: Dict) -> bool:
        """Check if a single question is valid."""
        question_text = question.get("question", "").lower()
        
        # Check if question contains visual references
        if self._contains_visual_reference(question_text):
            logger.warning(f"Removing question with visual reference: {question.get('question', '')[:50]}...")
            return False
        
        # Check options for MCQ/MSQ
        if "options" in question:
            for opt in question["options"]:
                if self._contains_visual_reference(opt.lower()):
                    logger.warning(f"Removing question with visual references in options: {question.get('question', '')[:50]}...")
                    return False
        
        # Validate answer for short answer questions
        if "correct_answer" in question and isinstance(question["correct_answer"], str):
            if self._contains_visual_reference(question["correct_answer"].lower()):
                logger.warning(f"Removing question with visual references in answer")
                return False
        
        # Validate question structure
        if not self._has_valid_structure(question):
            logger.warning(f"Removing question with invalid structure")
            return False
        
        return True
    
    def _contains_visual_reference(self, text: str) -> bool:
        """Check if text contains any visual reference keywords."""
        return any(keyword in text for keyword in self.visual_keywords)
    
    def _has_valid_structure(self, question: Dict) -> bool:
        """Check if question has required fields based on type."""
        required_fields = ["question", "correct_answer", "type"]
        
        # Check basic required fields
        for field in required_fields:
            if field not in question:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate question type specific fields
        q_type = question.get("type", "").lower()
        
        if q_type in ["mcq", "msq"]:
            if "options" not in question:
                logger.warning("MCQ/MSQ missing options")
                return False
            if not isinstance(question["options"], list) or len(question["options"]) != 4:
                logger.warning("MCQ/MSQ must have exactly 4 options")
                return False
        
        if q_type == "msq":
            if not isinstance(question["correct_answer"], list):
                logger.warning("MSQ correct_answer must be a list")
                return False
        
        if q_type == "yes_no":
            if question["correct_answer"] not in ["Yes", "No"]:
                logger.warning("Yes/No question must have 'Yes' or 'No' as answer")
                return False
        
        return True