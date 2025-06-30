from pdf2image import convert_from_bytes
from PIL import Image
import io
import logging
from typing import List, Dict, Any

# Import configuration
try:
    from .config import (
        PDF_DPI, MAX_IMAGE_DIMENSION, IMAGE_QUALITY
    )
    from .schema import QuestionGenerationState
except ImportError:
    PDF_DPI = 150
    MAX_IMAGE_DIMENSION = 2048
    IMAGE_QUALITY = 95
    class QuestionGenerationState(dict): pass

logger = logging.getLogger(__name__)

def pdf_to_images_node(state: QuestionGenerationState) -> Dict[str, Any]:
    """
    Converts each page of the PDF to an image.
    Optionally resizes images to reduce memory usage.
    """
    print("---CONVERTING PDF TO IMAGES---")
    try:
        images = convert_from_bytes(state['pdf_file'], dpi=PDF_DPI)
        image_bytes: List[bytes] = []
        
        for i, image in enumerate(images):
            if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
                ratio = min(MAX_IMAGE_DIMENSION / image.width, MAX_IMAGE_DIMENSION / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image {i+1} from {image.width}x{image.height} to {new_size[0]}x{new_size[1]}")
            
            img_byte_arr = io.BytesIO()
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3] if len(image.split()) == 4 else None)
                image = rgb_image
            
            image.save(img_byte_arr, format='JPEG', quality=IMAGE_QUALITY, optimize=True)
            image_bytes.append(img_byte_arr.getvalue())
            
        logger.info(f"Successfully converted PDF to {len(image_bytes)} images")
        total_size = sum(len(img) for img in image_bytes)
        logger.info(f"Total image data size: {total_size / 1024 / 1024:.2f} MB")
        
        return {"images": image_bytes, "document_images": {}, "error": None}
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        return {"images": [], "error": f"PDF conversion failed: {str(e)}", "document_images": {}}
