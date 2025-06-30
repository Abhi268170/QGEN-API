import httpx
import asyncio
import logging
from typing import List, Dict, Any

# Import configuration
try:
    from .config import (
        EMBEDDING_SERVICE_BASE_URL, EMBEDDING_TIMEOUT, EMBEDDING_RETRY_DELAY
    )
    from .schema import QuestionGenerationState
except ImportError:
    EMBEDDING_SERVICE_BASE_URL = "http://192.168.1.55:8000/embed"
    EMBEDDING_TIMEOUT = 60.0
    EMBEDDING_RETRY_DELAY = 5.0
    class QuestionGenerationState(dict): pass

logger = logging.getLogger(__name__)

async def embed_text_async(text: str) -> List[float]:
    """
    Embeds text using the colpali API.
    Returns flattened embedding to match app.py behavior.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(EMBEDDING_TIMEOUT)) as client:
            logger.info(f"Sending text embedding request for: '{text}'")
            
            response = await client.post(
                f"{EMBEDDING_SERVICE_BASE_URL}/text",
                data={'text': text}
            )
            response.raise_for_status()
            result = response.json()
            
            embedding = result.get("embedding", [])
            logger.info(f"Received text embedding with shape: {len(embedding) if embedding else 0}")
            
            if embedding and isinstance(embedding[0], list):
                embedding = embedding[0]
            
            return embedding
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error embedding text '{text}': {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        logger.error(f"Failed to embed text '{text}': {type(e).__name__}: {e}")
        return []

async def embed_images_node(state: QuestionGenerationState) -> Dict[str, Any]:
    """
    Embeds the images using the colpali API with async httpx.
    Sends images one at a time to avoid GPU memory issues.
    Modified to match app.py approach.
    """
    print("---EMBEDDING IMAGES---")
    
    if not state.get('images'):
        logger.warning("No images to embed")
        return {"embeddings": [], "error": "No images available for embedding"}

    all_embeddings = []
    failed_images = []
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(EMBEDDING_TIMEOUT)) as client:
        for i, image_bytes in enumerate(state['images']):
            try:
                logger.info(f"Sending image {i+1}/{len(state['images'])} to embedding service")
                
                files = {'file': (f'image_{i}.jpg', image_bytes, 'image/jpeg')}
                
                response = await client.post(
                    f"{EMBEDDING_SERVICE_BASE_URL}/image",
                    files=files
                )
                response.raise_for_status()
                result = response.json()
                embedding = result.get("embedding", [])
                
                if embedding:
                    if isinstance(embedding[0], list):
                        embedding = embedding[0]
                    all_embeddings.append(embedding)
                    logger.info(f"Successfully embedded image {i+1}")
                else:
                    logger.warning(f"No embeddings returned for image {i+1}")
                    failed_images.append(i)
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"Failed to embed image {i+1}: HTTP {e.response.status_code}")
                failed_images.append(i)
                if "CUDA out of memory" in e.response.text:
                    logger.warning(f"GPU out of memory. Waiting {EMBEDDING_RETRY_DELAY} seconds before continuing...")
                    await asyncio.sleep(EMBEDDING_RETRY_DELAY)
                    
            except httpx.TimeoutException:
                logger.error(f"Timeout while embedding image {i+1}")
                failed_images.append(i)
                
            except Exception as e:
                logger.error(f"Unexpected error embedding image {i+1}: {e}")
                failed_images.append(i)
    
    if failed_images:
        error_msg = f"Failed to embed {len(failed_images)} out of {len(state['images'])} images: {failed_images}"
        logger.warning(error_msg)
        if not all_embeddings:
            return {"embeddings": [], "error": error_msg}
        else:
            return {"embeddings": all_embeddings, "error": f"Partial embedding success. {error_msg}"}
    
    logger.info(f"Successfully embedded all {len(state['images'])} images")
    return {"embeddings": all_embeddings, "error": None}
