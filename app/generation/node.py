"""LangGraph node for question generation."""

import logging
from typing import Dict, Any
from .base import GenerationConfig
from .generator import QuestionGenerator

# Import configuration
try:
    from ..config import QWEN_VL_BASE_URL, QWEN_VL_MODEL, QWEN_VL_API_KEY
    from ..schema import QuestionGenerationState
except ImportError:
    QWEN_VL_BASE_URL = "http://192.168.1.55:1234/v1"
    QWEN_VL_MODEL = "qwen/qwen2.5-vl-7b"
    QWEN_VL_API_KEY = "test-api-key"
    class QuestionGenerationState(dict): pass

logger = logging.getLogger(__name__)

async def generation_node(state: QuestionGenerationState) -> Dict[str, Any]:
    """
    Generates questions based on the retrieved images and user parameters.
    Uses the local Qwen2.5-VL-7B model.
    """
    print("---GENERATING QUESTIONS---")

    if state.get('error') or not state.get('relevant_images'):
        logger.warning("Skipping generation due to previous error or no relevant images.")
        return {"questions": [], "question_paper": None}

    try:
        # Create configuration
        config = GenerationConfig(
            base_url=QWEN_VL_BASE_URL,
            model=QWEN_VL_MODEL,
            api_key=QWEN_VL_API_KEY
        )
        
        # Create generator instance
        generator = QuestionGenerator(config)
        
        # Generate questions
        result = await generator.generate_questions(
            images=state['relevant_images'],
            topic=state['topic'],
            question_type=state['question_type'],
            num_questions=state['num_questions'],
            difficulty=state['difficulty'],
            section_ordering=state.get('section_ordering'),
            section_marks=state.get('section_marks')
        )
        
        logger.info(f"Successfully generated {len(result['questions'])} questions.")
        return result

    except Exception as e:
        logger.error(f"Unexpected error during question generation: {e}")
        return {"questions": [], "question_paper": None, "error": f"Generation failed: {str(e)}"}