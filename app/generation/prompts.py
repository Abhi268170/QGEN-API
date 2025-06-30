# prompts.py
"""Prompt templates and examples for question generation."""

from typing import Dict, Optional

class PromptBuilder:
    """Builds prompts for question generation."""
    
    @staticmethod
    def get_question_type_instructions(question_type: str) -> str:
        """Get specific instructions based on question type."""
        instructions = {
            "mcq": """Generate Multiple Choice Questions (MCQ) with:
- A clear, conceptual question testing understanding
- Exactly 4 answer options labeled A, B, C, D
- Only ONE correct answer
- Plausible distractors that test conceptual understanding
- No visual references or figure numbers""",
            
            "msq": """Generate Multiple Selection Questions (MSQ) with:
- A clear, conceptual question that indicates multiple answers may be correct
- Exactly 4 or 5 answer options labeled A, B, C, D (and E if 5 options)
- One or more correct answers as a list
- Clear indication of all correct options
- Focus on concept relationships, not visual elements""",
            
            "short_answer": """Generate Short Answer Questions with:
- A clear, specific question about concepts or principles
- An answer that ranges from a few words to a few sentences
- NO OPTIONS - just question and answer
- Focus on testing comprehension of academic concepts
- Questions must be answerable without any visual aid""",
            
            "yes_no": """Generate Yes/No Questions with:
- A clear statement or question to evaluate
- MUST include options array with exactly ["Yes", "No"]
- The correct answer must be either "Yes" or "No"
- Test understanding of principles, not visual recognition"""
        }
        return instructions.get(question_type.lower(), instructions["mcq"])
    
    @staticmethod
    def get_question_examples(question_type: str) -> str:
        """Get example questions for each type showing proper format without visual references."""
        examples = {
            "mcq": """
Example MCQ transformations:
BAD: "Which figure shows how a convex lens corrects myopia?"
GOOD: "How does a convex lens help correct myopia?"

Example well-formed MCQ:
{
  "question": "What is the primary function of chloroplasts in plant cells?",
  "options": [
    "A. To store genetic material",
    "B. To conduct photosynthesis",
    "C. To provide structural support",
    "D. To regulate water balance"
  ],
  "correct_answer": "B",
  "marks": 2,
  "type": "mcq",
  "difficulty": "medium"
}""",
            
            "msq": """
Example MSQ transformations:
BAD: "Select all processes shown in the cellular respiration diagram"
GOOD: "Which of the following are stages of cellular respiration?"

Example well-formed MSQ:
{
  "question": "Which of the following are properties of acids? (Select all that apply)",
  "options": [
    "A. Turn blue litmus paper red",
    "B. Have pH values less than 7",
    "C. React with metals to produce hydrogen gas",
    "D. Feel slippery to touch",
    "E. Taste bitter"
  ],
  "correct_answer": ["A", "B", "C"],
  "marks": 3,
  "type": "msq",
  "difficulty": "medium"
}""",
            
            "short_answer": """
Example Short Answer transformations:
BAD: "Describe what is happening in the figure showing mitosis"
GOOD: "Describe the main stages of mitosis in order"

Example well-formed Short Answer:
{
  "question": "Explain how the greenhouse effect contributes to global warming",
  "correct_answer": "The greenhouse effect occurs when gases like CO2 and methane trap heat in Earth's atmosphere. These gases allow sunlight to enter but prevent heat from escaping, causing the planet's temperature to rise.",
  "marks": 4,
  "type": "short_answer",
  "difficulty": "medium"
}

NOTE: Short answer questions have NO options field.""",
            
            "yes_no": """
Example Yes/No transformations:
BAD: "Does the diagram show parallel circuits?"
GOOD: "Do parallel circuits have multiple paths for electric current to flow?"

Example well-formed Yes/No:
{
  "question": "Is water considered a renewable resource?",
  "options": ["Yes", "No"],
  "correct_answer": "Yes",
  "marks": 1,
  "type": "yes_no",
  "difficulty": "easy"
}

IMPORTANT: Yes/No questions MUST include the options field with ["Yes", "No"]"""
        }
        return examples.get(question_type.lower(), examples["mcq"])
    
    @staticmethod
    def get_difficulty_guidelines(difficulty: str) -> str:
        """Get guidelines based on difficulty level."""
        guidelines = {
            "easy": "Focus on basic concepts, definitions, and simple recall. Questions should test fundamental understanding.",
            "medium": "Include application of concepts, analysis, and moderate complexity. Questions should require understanding relationships between concepts.",
            "hard": "Create challenging questions requiring synthesis, evaluation, and deep understanding. Include complex scenarios and critical thinking."
        }
        return guidelines.get(difficulty.lower(), guidelines["medium"])
    
    @staticmethod
    def build_system_prompt(
        topic: str,
        question_type: str,
        difficulty: str,
        batch_size: int,
        section_info: Optional[Dict] = None,
        existing_questions: Optional[list] = None
    ) -> str:
        """Build the complete system prompt."""
        
        prompt_builder = PromptBuilder()
        type_instructions = prompt_builder.get_question_type_instructions(question_type)
        difficulty_guidelines = prompt_builder.get_difficulty_guidelines(difficulty)
        example_section = prompt_builder.get_question_examples(question_type)
        
        # Handle section-specific instructions
        section_prompt = ""
        if section_info:
            section_prompt = f"\nSection: {section_info['name']}\nTotal marks for section: {section_info['marks']}\n"
        
        # Create deduplication context
        dedup_prompt = ""
        if existing_questions:
            dedup_prompt = f"\n\nAvoid creating questions similar to these already generated ones:\n{chr(10).join(existing_questions[:5])}\n"
        
        # Add structure reminder based on question type
        structure_reminder = ""
        if question_type.lower() == "yes_no":
            structure_reminder = "\nREMEMBER: Yes/No questions MUST have an 'options' field with exactly [\"Yes\", \"No\"]"
        elif question_type.lower() == "short_answer":
            structure_reminder = "\nREMEMBER: Short answer questions must NOT have an 'options' field"
        elif question_type.lower() == "msq":
            structure_reminder = "\nREMEMBER: MSQ must have 4-5 options and correct_answer must be a list"
        
        return f"""You are an expert educator creating academic questions based on educational content.

TASK: Generate {batch_size} {question_type} questions about {topic}.

{type_instructions}

Difficulty: {difficulty}
{difficulty_guidelines}
{section_prompt}

CRITICAL RULES:
1. Extract CONCEPTS from the images, not visual elements
2. NEVER reference figures, diagrams, images, or any visual elements
3. Create SELF-CONTAINED questions that test understanding
4. Questions must be answerable without seeing any images
5. Focus on academic principles and concepts
6. Each question must be unique
{structure_reminder}
{dedup_prompt}

{example_section}

Return ONLY a valid JSON object with this exact structure:
{{
  "questions": [
    {{
      "question": "your question here",{' "options": [...],  // Include for mcq, msq, yes_no. Exclude for short_answer' if question_type.lower() != 'short_answer' else ''}
      "correct_answer": {"'...'" if question_type.lower() != 'msq' else "[...]"},
      "marks": {batch_size},
      "type": "{question_type}",
      "difficulty": "{difficulty}"
    }}
  ]
}}"""