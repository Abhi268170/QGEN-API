"""Main question generator class."""

import logging
from typing import List, Dict, Any, Optional
from .base import GenerationConfig, SectionInfo
from .prompts import PromptBuilder
from .validator import QuestionValidator
from .api_client import QwenVLClient
from .utils import (
    calculate_marks_per_question, 
    parse_llm_response,
    create_question_paper,
    prepare_image_content
)

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """Handles question generation using Qwen2.5-VL-7B model."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.validator = QuestionValidator()
        self.prompt_builder = PromptBuilder()
        self.api_client = QwenVLClient(config)
        
    async def generate_questions(
        self, 
        images: List[bytes], 
        topic: str, 
        question_type: str,
        num_questions: int, 
        difficulty: str,
        section_ordering: Optional[List[str]] = None,
        section_marks: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Generate questions with proper batching and section handling."""
        
        all_questions = []
        existing_questions = []
        
        if section_ordering and section_marks:
            # Handle section-based generation
            all_questions = await self._generate_with_sections(
                images, topic, question_type, num_questions, 
                difficulty, section_ordering, section_marks, 
                existing_questions
            )
        else:
            # Generate without sections
            all_questions = await self._generate_without_sections(
                images, topic, question_type, num_questions, 
                difficulty, existing_questions
            )
        
        # Note: Mark assignment is now handled by a separate node
        # Questions will have temporary/placeholder marks that will be updated later
        
        logger.info(f"Total questions generated after validation: {len(all_questions)}")
        
        return {
            "questions": all_questions
        }
    
    async def _generate_with_sections(
        self, 
        images: List[bytes], 
        topic: str, 
        question_type: str,
        num_questions: int, 
        difficulty: str,
        section_ordering: List[str],
        section_marks: List[int],
        existing_questions: List[str]
    ) -> List[Dict]:
        """Generate questions organized by sections."""
        all_questions = []
        questions_per_section = num_questions // len(section_ordering)
        remainder = num_questions % len(section_ordering)
        
        for i, (section_name, marks) in enumerate(zip(section_ordering, section_marks)):
            section_questions_needed = questions_per_section + (1 if i < remainder else 0)
            
            if section_questions_needed == 0:
                continue
            
            section_info = SectionInfo(name=section_name, marks=marks)
            section_questions = await self._generate_for_section(
                images, topic, question_type, difficulty,
                section_questions_needed, section_info, existing_questions
            )
            
            all_questions.extend(section_questions)
        
        return all_questions
    
    async def _generate_without_sections(
        self, 
        images: List[bytes], 
        topic: str, 
        question_type: str,
        num_questions: int, 
        difficulty: str,
        existing_questions: List[str]
    ) -> List[Dict]:
        """Generate questions without section organization."""
        all_questions = []
        questions_generated = 0
        retry_count = 0
        
        while questions_generated < num_questions and retry_count < self.config.max_retries:
            batch_size = min(self.config.batch_size, num_questions - questions_generated)
            
            batch = await self._generate_batch(
                images, topic, question_type, difficulty,
                batch_size, existing_questions
            )
            
            if len(batch) == 0:
                retry_count += 1
                logger.warning(f"No valid questions generated, retry {retry_count}/{self.config.max_retries}")
                continue
            
            for q in batch:
                existing_questions.append(q["question"])
            
            all_questions.extend(batch)
            questions_generated += len(batch)
            
            if len(batch) < batch_size:
                logger.info(f"Generated {len(batch)} questions (requested {batch_size})")
        
        return all_questions
    
    async def _generate_for_section(
        self,
        images: List[bytes],
        topic: str,
        question_type: str,
        difficulty: str,
        questions_needed: int,
        section_info: SectionInfo,
        existing_questions: List[str]
    ) -> List[Dict]:
        """Generate questions for a specific section."""
        section_questions = []
        questions_generated = 0
        retry_count = 0
        
        while questions_generated < questions_needed and retry_count < self.config.max_retries:
            batch_size = min(self.config.batch_size, questions_needed - questions_generated)
            
            batch = await self._generate_batch(
                images, topic, question_type, difficulty,
                batch_size, existing_questions,
                {"name": section_info.name, "marks": section_info.marks}
            )
            
            if len(batch) == 0:
                retry_count += 1
                logger.warning(f"No valid questions for section {section_info.name}, retry {retry_count}/{self.config.max_retries}")
                continue
            
            # Add section info to questions
            for q in batch:
                q["section"] = section_info.name
                existing_questions.append(q["question"])
            
            section_questions.extend(batch)
            questions_generated += len(batch)
            
            if len(batch) < batch_size:
                logger.info(f"Generated {len(batch)} questions for section {section_info.name}")
        
        return section_questions
    
    async def _generate_batch(
        self,
        images: List[bytes],
        topic: str,
        question_type: str,
        difficulty: str,
        batch_size: int,
        existing_questions: List[str],
        section_info: Optional[Dict] = None
    ) -> List[Dict]:
        """Generate a batch of questions using the VL model."""
        
        # Build the prompt
        system_prompt = self.prompt_builder.build_system_prompt(
            topic, question_type, difficulty, batch_size,
            section_info, existing_questions
        )
        
        # Prepare message content
        user_content = [
            {
                "type": "text",
                "text": system_prompt + f"\n\nNow analyze these educational materials and generate {batch_size} {question_type} questions about {topic}:"
            }
        ]
        
        # Add images
        user_content.extend(prepare_image_content(images))
        
        messages = [{"role": "user", "content": user_content}]
        
        # Make API call
        result = await self.api_client.generate_completion(messages)
        if not result:
            return []
        
        # Extract and parse content
        content = self.api_client.extract_content_from_response(result)
        questions_data = parse_llm_response(content)
        questions = questions_data.get("questions", [])
        
        # Validate and clean questions
        questions = self.validator.validate_and_clean_questions(questions)
        
        # Add marks if not present
        for q in questions:
            if "marks" not in q:
                q["marks"] = 0  # Placeholder - will be calculated by mark assignment node
            
            # Ensure required fields are present
            if "type" not in q:
                q["type"] = question_type
            if "difficulty" not in q:
                q["difficulty"] = difficulty
        
        logger.info(f"Successfully generated {len(questions)} valid questions")
        return questions