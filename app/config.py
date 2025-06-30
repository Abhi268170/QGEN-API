# Configuration settings for the LangGraph API

# PDF to Image conversion settings
PDF_DPI = 150  # DPI for PDF to image conversion (lower = smaller images, faster processing)
MAX_IMAGE_DIMENSION = 2048  # Maximum width or height for images
IMAGE_QUALITY = 95  # JPEG quality (1-100)

# Embedding service settings
EMBEDDING_SERVICE_URL = "http://192.168.1.55:8000/embed/batch_images"
EMBEDDING_SERVICE_BASE_URL = "http://192.168.1.55:8000/embed"  # Base URL for other endpoints
EMBEDDING_TIMEOUT = 60.0  # Timeout per image in seconds
EMBEDDING_RETRY_DELAY = 5.0  # Delay between retries on GPU OOM errors

# Qdrant settings
QDRANT_MEMORY_LIMIT = "http://localhost:6333"
# Processing settings
BATCH_IMAGES = False  # If True, send all images at once; if False, send one at a time
MAX_CONCURRENT_EMBEDDINGS = 1  # Number of concurrent embedding requests (if BATCH_IMAGES is False)

# Generation service settings
GENERATION_SERVICE_URL = "http://192.168.1.55:8001/generate/questions"
GENERATION_TIMEOUT = 120.0  # Timeout for the generation request in seconds