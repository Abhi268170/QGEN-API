
import httpx
import logging
from typing import List, Dict, Any

# Import configuration
try:
    from .config import (
        GENERATION_SERVICE_URL, GENERATION_TIMEOUT
    )
    from .schema import QuestionGenerationState
except ImportError:
    GENERATION_SERVICE_URL = "http://192.168.1.55:8001/generate/questions"
    GENERATION_TIMEOUT = 120.0
    class QuestionGenerationState(dict): pass

logger = logging.getLogger(__name__)

async def generation_node(state: QuestionGenerationState) -> Dict[str, Any]:
    """
    Generates questions based on the retrieved images and user parameters.
    """
    print("---GENERATING QUESTIONS---")

    if state.get('error') or not state.get('relevant_images'):
        logger.warning("Skipping generation due to previous error or no relevant images.")
        return {"questions": [], "question_paper": None}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(GENERATION_TIMEOUT)) as client:
            
            files = [('images', (f'image_{i}.jpg', image_bytes, 'image/jpeg')) 
                     for i, image_bytes in enumerate(state['relevant_images'])]
            
            data = {
                "topic": state['topic'],
                "question_type": state['question_type'],
                "num_questions": state['num_questions'],
                "difficulty": state['difficulty'],
                "section_ordering": state['section_ordering'],
                "section_marks": state['section_marks']
            }

            logger.info(f"Sending {len(files)} images to generation service.")
            
            response = await client.post(
                GENERATION_SERVICE_URL,
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()

            logger.info("Successfully generated questions.")
            return {
                "questions": result.get("questions", []),
                "question_paper": result.get("question_paper")
            }

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during question generation: {e.response.status_code} - {e.response.text}")
        return {"questions": [], "question_paper": None, "error": f"Generation failed: {e.response.text}"}
    except Exception as e:
        logger.error(f"Unexpected error during question generation: {e}")
        return {"questions": [], "question_paper": None, "error": f"Generation failed: {str(e)}"}
