# mark_assignment.py
"""
Mark assignment module for question generation.
Handles mark allocation both with and without section constraints.
"""

import logging
from typing import Dict, List, Any, Optional
from math import ceil, floor

# Import configuration
try:
    from .config import QUESTION_TYPE_CONFIG
    from .schema import QuestionGenerationState
except ImportError:
    # Fallback configuration
    QUESTION_TYPE_CONFIG = {
        "mcq": {"base_marks": {"easy": 1, "medium": 2, "hard": 3}},
        "msq": {"base_marks": {"easy": 2, "medium": 3, "hard": 4}},
        "short_answer": {"base_marks": {"easy": 2, "medium": 4, "hard": 6}},
        "yes_no": {"base_marks": {"easy": 1, "medium": 1, "hard": 2}}
    }
    class QuestionGenerationState(dict): pass

logger = logging.getLogger(__name__)

class MarkAssignmentCalculator:
    """Handles mark calculation and assignment for questions."""
    
    def __init__(self):
        self.question_type_config = QUESTION_TYPE_CONFIG
    
    def get_base_marks(self, question_type: str, difficulty: str) -> int:
        """Get base marks for a question type and difficulty."""
        config = self.question_type_config.get(question_type.lower(), self.question_type_config["mcq"])
        return config["base_marks"].get(difficulty.lower(), 2)
    
    def calculate_complexity_multiplier(self, question: Dict) -> float:
        """Calculate complexity multiplier based on question content."""
        multiplier = 1.0
        question_text = question.get("question", "")
        
        # Length-based complexity
        if len(question_text) > 150:
            multiplier += 0.3
        elif len(question_text) > 100:
            multiplier += 0.2
        elif len(question_text) > 50:
            multiplier += 0.1
        
        # Content complexity indicators
        complexity_indicators = [
            "analyze", "evaluate", "compare", "contrast", "explain", "justify",
            "synthesize", "predict", "calculate", "derive", "prove", "demonstrate",
            "critically", "relationship", "impact", "consequence", "implication"
        ]
        
        text_lower = question_text.lower()
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in text_lower)
        
        if complexity_count >= 3:
            multiplier += 0.4
        elif complexity_count >= 2:
            multiplier += 0.2
        elif complexity_count >= 1:
            multiplier += 0.1
        
        # Multi-step questions (for MCQ/MSQ with complex options)
        if question.get("type") in ["mcq", "msq"]:
            options = question.get("options", [])
            if any(len(str(opt)) > 50 for opt in options):
                multiplier += 0.2
        
        # Cap the multiplier to reasonable bounds
        return min(max(multiplier, 0.5), 2.0)
    
    def assign_marks_with_sections(
        self, 
        questions: List[Dict], 
        section_ordering: List[str], 
        section_marks: List[int]
    ) -> List[Dict]:
        """Assign marks when section constraints are present."""
        logger.info("Assigning marks with section constraints")
        
        # Group questions by section
        section_questions = {}
        for question in questions:
            section = question.get("section", "default")
            if section not in section_questions:
                section_questions[section] = []
            section_questions[section].append(question)
        
        # Assign marks for each section
        for i, (section_name, target_marks) in enumerate(zip(section_ordering, section_marks)):
            if section_name not in section_questions:
                logger.warning(f"No questions found for section: {section_name}")
                continue
            
            section_q_list = section_questions[section_name]
            self._distribute_section_marks(section_q_list, target_marks, section_name)
        
        return questions
    
    def assign_marks_without_sections(self, questions: List[Dict]) -> List[Dict]:
        """Assign marks based on difficulty, complexity, and question type."""
        logger.info("Assigning marks based on question characteristics")
        
        for question in questions:
            question_type = question.get("type", "mcq")
            difficulty = question.get("difficulty", "medium")
            
            # Get base marks
            base_marks = self.get_base_marks(question_type, difficulty)
            
            # Calculate complexity multiplier
            complexity_multiplier = self.calculate_complexity_multiplier(question)
            
            # Calculate final marks
            calculated_marks = base_marks * complexity_multiplier
            
            # Round to reasonable integer values
            final_marks = max(1, round(calculated_marks))
            
            # Apply reasonable caps based on question type
            max_marks = self._get_max_marks_for_type(question_type, difficulty)
            final_marks = min(final_marks, max_marks)
            
            question["marks"] = final_marks
            
            logger.debug(f"Question: {question.get('question', '')[:50]}... "
                        f"Base: {base_marks}, Multiplier: {complexity_multiplier:.2f}, "
                        f"Final: {final_marks}")
        
        return questions
    
    def _distribute_section_marks(
        self, 
        section_questions: List[Dict], 
        target_marks: int, 
        section_name: str
    ) -> None:
        """Distribute marks across questions in a section to match target."""
        if not section_questions:
            return
        
        num_questions = len(section_questions)
        logger.info(f"Distributing {target_marks} marks across {num_questions} questions in section '{section_name}'")
        
        # Calculate base distribution
        base_marks_per_question = target_marks // num_questions
        remainder = target_marks % num_questions
        
        # Get relative weights based on question characteristics
        weights = []
        for question in section_questions:
            question_type = question.get("type", "mcq")
            difficulty = question.get("difficulty", "medium")
            
            base_marks = self.get_base_marks(question_type, difficulty)
            complexity = self.calculate_complexity_multiplier(question)
            weight = base_marks * complexity
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / num_questions] * num_questions
        
        # Distribute marks based on weights
        allocated_marks = []
        total_allocated = 0
        
        for i, weight in enumerate(normalized_weights):
            if i < len(normalized_weights) - 1:
                # For all but the last question, calculate proportional marks
                marks = max(1, round(target_marks * weight))
                allocated_marks.append(marks)
                total_allocated += marks
            else:
                # Last question gets remaining marks
                remaining = target_marks - total_allocated
                marks = max(1, remaining)
                allocated_marks.append(marks)
        
        # Ensure we don't exceed target marks
        total_final = sum(allocated_marks)
        if total_final != target_marks:
            # Adjust the allocation
            difference = total_final - target_marks
            
            if difference > 0:
                # Reduce marks from highest allocated questions
                for i in range(len(allocated_marks)):
                    if difference <= 0:
                        break
                    if allocated_marks[i] > 1:
                        reduction = min(difference, allocated_marks[i] - 1)
                        allocated_marks[i] -= reduction
                        difference -= reduction
            else:
                # Add marks to questions that can handle it
                difference = abs(difference)
                for i in range(len(allocated_marks)):
                    if difference <= 0:
                        break
                    max_for_type = self._get_max_marks_for_type(
                        section_questions[i].get("type", "mcq"),
                        section_questions[i].get("difficulty", "medium")
                    )
                    if allocated_marks[i] < max_for_type:
                        addition = min(difference, max_for_type - allocated_marks[i])
                        allocated_marks[i] += addition
                        difference -= addition
        
        # Assign calculated marks to questions
        for question, marks in zip(section_questions, allocated_marks):
            question["marks"] = marks
        
        final_total = sum(allocated_marks)
        logger.info(f"Section '{section_name}': Target {target_marks}, Assigned {final_total}")
        
        if final_total != target_marks:
            logger.warning(f"Section '{section_name}': Could not exactly match target marks. "
                          f"Target: {target_marks}, Actual: {final_total}")
    
    def _get_max_marks_for_type(self, question_type: str, difficulty: str) -> int:
        """Get maximum reasonable marks for a question type and difficulty."""
        max_marks_map = {
            "mcq": {"easy": 3, "medium": 5, "hard": 8},
            "msq": {"easy": 4, "medium": 6, "hard": 10},
            "short_answer": {"easy": 5, "medium": 10, "hard": 15},
            "yes_no": {"easy": 2, "medium": 3, "hard": 4}
        }
        
        type_map = max_marks_map.get(question_type.lower(), max_marks_map["mcq"])
        return type_map.get(difficulty.lower(), 5)


def assign_marks_node(state: QuestionGenerationState) -> Dict[str, Any]:
    """
    LangGraph node for assigning marks to generated questions.
    Handles both section-based and open mark assignment.
    """
    print("---ASSIGNING MARKS---")
    
    if state.get('error') or not state.get('questions'):
        logger.warning("Skipping mark assignment due to previous error or no questions.")
        return {"questions": state.get('questions', []), "total_marks": 0}
    
    questions = state.get('questions', [])
    section_ordering = state.get('section_ordering', [])
    section_marks = state.get('section_marks', [])
    
    try:
        calculator = MarkAssignmentCalculator()
        
        if section_ordering and section_marks:
            # Section-based mark assignment
            questions = calculator.assign_marks_with_sections(
                questions, section_ordering, section_marks
            )
            logger.info("Completed section-based mark assignment")
        else:
            # Open mark assignment based on question characteristics
            questions = calculator.assign_marks_without_sections(questions)
            logger.info("Completed characteristic-based mark assignment")
        
        # Calculate total marks
        total_marks = sum(q.get("marks", 0) for q in questions)
        
        # Log summary
        logger.info(f"Mark assignment summary:")
        logger.info(f"  Total questions: {len(questions)}")
        logger.info(f"  Total marks: {total_marks}")
        
        if section_ordering and section_marks:
            # Verify section totals
            for section_name, expected_marks in zip(section_ordering, section_marks):
                section_questions = [q for q in questions if q.get("section") == section_name]
                actual_marks = sum(q.get("marks", 0) for q in section_questions)
                logger.info(f"  Section '{section_name}': {actual_marks}/{expected_marks} marks")
        
        return {
            "questions": questions,
            "total_marks": total_marks
        }
        
    except Exception as e:
        logger.error(f"Error in mark assignment: {e}")
        return {
            "questions": questions,
            "total_marks": sum(q.get("marks", 0) for q in questions),
            "error": f"Mark assignment error: {str(e)}"
        }