from qdrant_client import QdrantClient, models
import uuid
import logging
from typing import List, Dict, Any

# Import configuration
try:
    from .config import QDRANT_MEMORY_LIMIT
    from .schema import QuestionGenerationState
    from .embedding import embed_text_async
except ImportError:
    QDRANT_MEMORY_LIMIT = ":memory:"
    class QuestionGenerationState(dict): pass
    async def embed_text_async(text: str) -> List[float]: return []

logger = logging.getLogger(__name__)

def add_to_qdrant_node(state: QuestionGenerationState, client: QdrantClient) -> Dict[str, Any]:
    """
    Adds the embeddings to a Qdrant collection.
    Fixed to use proper vector dimensions and UUID-based IDs like app.py.
    """
    print("---ADDING TO QDRANT---")
    
    if state.get('error'):
        logger.warning(f"Skipping Qdrant due to previous error: {state['error']}")
        return {"collection_name": None}
        
    if not state.get('embeddings') or not state['embeddings']:
        logger.warning("No embeddings were generated. Skipping Qdrant node.")
        return {"collection_name": None}
    
    try:
        client = QdrantClient(QDRANT_MEMORY_LIMIT)
        collection_name = f"{state['topic']}_{state['difficulty']}".replace(" ", "_").lower()
        
        vector_size = 128
        logger.info(f"Using fixed embedding vector size: {vector_size}")
        
        try:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Could not check/delete existing collection: {e}")
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "multi-vector": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            }
        )
        logger.info(f"Created collection '{collection_name}' with {vector_size}D vectors")
        
        points = []
        document_images = {}
        
        for i, embedding in enumerate(state['embeddings']):
            point_id = str(uuid.uuid4())
            page_number = i + 1
            
            if i < len(state['images']):
                document_images[point_id] = state['images'][i]
            
            point = models.PointStruct(
                id=point_id,
                vector={"multi-vector": embedding},
                payload={"page_number": page_number}
            )
            points.append(point)
        
        chunk_size = 1
        points_added = 0
        
        for i in range(0, len(points), chunk_size):
            chunk = points[i:i+chunk_size]
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=chunk,
                    wait=True,
                )
                points_added += len(chunk)
                logger.info(f"Upserted chunk {i//chunk_size + 1}/{(len(points) + chunk_size - 1)//chunk_size}")
            except Exception as e:
                logger.error(f"Failed to upsert chunk starting at index {i}: {e}")
        
        logger.info(f"Successfully added {points_added}/{len(points)} points to Qdrant collection '{collection_name}'")
        logger.info(f"Stored {len(document_images)} images in document_images mapping")
        
        return {
            "collection_name": collection_name,
            "document_images": document_images
        }
        
    except Exception as e:
        logger.error(f"Failed to add to Qdrant: {e}")
        return {"collection_name": None, "error": f"Qdrant error: {str(e)}", "document_images": {}}

async def retrieve_relevant_images_node(state: QuestionGenerationState, client: QdrantClient) -> Dict[str, Any]:
    """
    Retrieves relevant images from Qdrant based on the topic.
    Fixed to work with UUID-based point IDs and proper vector search.
    """
    print("---RETRIEVING RELEVANT IMAGES---")
    
    if state.get('error') or not state.get('collection_name'):
        logger.warning(f"Skipping retrieval due to previous error or missing collection name.")
        return {"relevant_images": []}
        
    try:
        topic_embedding = await embed_text_async(state['topic'])
        if not topic_embedding:
            logger.error("Failed to embed topic, cannot retrieve images.")
            return {"relevant_images": [], "error": "Failed to embed topic for retrieval."}
        
        logger.info(f"Topic '{state['topic']}' embedded successfully.")
        logger.info(f"Topic embedding shape: {len(topic_embedding)}")
        
        client = QdrantClient(QDRANT_MEMORY_LIMIT)
        logger.info(f"Searching Qdrant collection '{state['collection_name']}' for topic '{state['topic']}'")
        search_results = client.search(
            collection_name=state['collection_name'],
            query_vector=("multi-vector", topic_embedding),
            limit=state.get('num_questions', 5), 
            with_payload=True
        )
        logger.info(f"Search results: {len(search_results)} points found")
        
        relevant_images = []
        document_images = state.get('document_images', {})        
        if search_results:
            for result in search_results:
                point_id = str(result.id)
                image_bytes = document_images.get(point_id)
                
                if image_bytes:
                    relevant_images.append(image_bytes)
                    logger.info(f"Retrieved image for point {point_id}, page {result.payload.get('page_number', 'unknown')}")
                else:
                    logger.warning(f"No image found for point ID {point_id}")
                    logger.info(f"Available image IDs: {list(document_images.keys())[:5]}...")
                    
            logger.info(f"Retrieved {len(relevant_images)} relevant images from Qdrant.")
        else:
            logger.warning("No relevant images found in Qdrant for the given topic.")
            
        return {"relevant_images": relevant_images}

    except Exception as e:
        logger.error(f"Failed to retrieve relevant images: {e}")
        return {"relevant_images": [], "error": f"Image retrieval failed: {str(e)}"}
