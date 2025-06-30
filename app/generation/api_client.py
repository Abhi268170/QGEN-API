"""API client for interacting with the Qwen2.5-VL model."""

import httpx
import logging
import json
from typing import List, Dict, Optional
from .base import GenerationConfig
from .utils import parse_llm_response

logger = logging.getLogger(__name__)

class QwenVLClient:
    """Client for interacting with Qwen2.5-VL API."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        
    async def generate_completion(
        self, 
        messages: List[Dict],
        timeout: float = 300.0
    ) -> Optional[Dict]:
        """Send a completion request to the API."""
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            try:
                request_data = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
                
                # Log request details
                logger.info(f"Sending request to {self.config.base_url}/chat/completions")
                logger.debug(f"Model: {self.config.model}")
                
                headers = {
                    "Content-Type": "application/json"
                }
                
                # Add authorization if API key is provided
                if self.config.api_key and self.config.api_key != "test-api-key":
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                
                response = await client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=headers,
                    json=request_data
                )
                
                # Handle error responses
                if response.status_code != 200:
                    logger.error(f"API returned status code: {response.status_code}")
                    logger.error(f"Response text: {response.text}")
                    
                    try:
                        error_data = response.json()
                        logger.error(f"Error details: {json.dumps(error_data, indent=2)}")
                    except:
                        pass
                    
                    response.raise_for_status()
                
                return response.json()
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e}")
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text[:500]}")
                return None
                
            except httpx.TimeoutException:
                logger.error(f"Request timeout after {timeout} seconds")
                return None
                
            except Exception as e:
                logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                return None
    
    def extract_content_from_response(self, result: Dict) -> str:
        """Extract content from API response."""
        content = ""
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0].get("message", {}).get("content", "")
        elif "content" in result:
            content = result["content"]
        else:
            logger.error(f"Unexpected response format: {json.dumps(result, indent=2)}")
        
        return content