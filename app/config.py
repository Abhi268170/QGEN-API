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

# Generation service settings - Updated for Qwen2.5-VL-7B
QWEN_VL_BASE_URL = "http://192.168.1.55:1234/v1"
QWEN_VL_MODEL = "qwen/qwen2.5-vl-7b"
QWEN_VL_API_KEY = "test-api-key"
GENERATION_TIMEOUT = 300.0  # Increased timeout for LLM generation
GENERATION_BATCH_SIZE = 10  # Maximum questions per batch to avoid context limits

# Question type mappings
SUPPORTED_QUESTION_TYPES = ["mcq", "msq", "short_answer", "yes_no"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Logging configuration for debugging
import logging
logging.basicConfig(level=logging.INFO)

# Add this to your config.py if it's not already there
QUESTION_TYPE_CONFIG = {
    "mcq": {
        "base_marks": {"easy": 1, "medium": 2, "hard": 3},
        "has_options": True,
        "num_options": 4,
        "multiple_answers": False
    },
    "msq": {
        "base_marks": {"easy": 2, "medium": 3, "hard": 4},
        "has_options": True,
        "num_options": [4, 5],
        "multiple_answers": True
    },
    "short_answer": {
        "base_marks": {"easy": 2, "medium": 4, "hard": 6},
        "has_options": False,
        "multiple_answers": False
    },
    "yes_no": {
        "base_marks": {"easy": 1, "medium": 1, "hard": 2},
        "has_options": True,
        "num_options": 2,
        "multiple_answers": False
    }
}