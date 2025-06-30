"""
Question generation module for the LangGraph API.

This module has been modularized into separate components:
- generation/base.py: Base classes and constants
- generation/prompts.py: Prompt templates and examples
- generation/validator.py: Question validation logic
- generation/utils.py: Utility functions
- generation/api_client.py: API client for Qwen2.5-VL
- generation/generator.py: Main generator class
- generation/node.py: LangGraph node implementation
"""

# Re-export the main components for backward compatibility
from generation import (
    GenerationConfig,
    QuestionGenerator,
    generation_node
)

# For backward compatibility, expose the generation_node at module level
__all__ = ['generation_node', 'QuestionGenerator', 'GenerationConfig']