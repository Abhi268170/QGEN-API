# This file defines the FastAPI application and the API endpoints.

from fastapi import FastAPI, File, UploadFile, Form, HTTPException  # The main framework for building the API.
from fastapi.responses import JSONResponse
from .graph import get_graph, get_question_generation_graph  # Import the graph builder from the graph module.
from .schema import GraphInput, QuestionGenerationInput  # Import the input schema.
from typing import List, Dict, Any
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangGraph API",
    description="An API for running a LangGraph application with question generation from PDFs.",
    version="0.1.0",
)

def convert_bytes_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert all bytes objects in a dictionary to base64 strings.
    This prevents JSON serialization errors.
    """
    if isinstance(data, dict):
        return {k: convert_bytes_in_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_bytes_in_dict(item) for item in data]
    elif isinstance(data, bytes):
        return base64.b64encode(data).decode('utf-8')
    else:
        return data

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "LangGraph API",
        "endpoints": {
            "/run": "Run the basic LangGraph workflow",
            "/generate-questions": "Generate questions from a PDF document",
            "/docs": "API documentation"
        }
    }

@app.post("/run")
async def run_graph(graph_input: GraphInput):
    """
    Runs the LangGraph application with the given input.
    """
    try:
        graph = get_graph()
        # The `ainvoke` method runs the graph asynchronously.
        result = await graph.ainvoke({"input": graph_input.input})
        return result
    except Exception as e:
        logger.error(f"Error running graph: {e}")
        raise HTTPException(status_code=500, detail=f"Error running graph: {str(e)}")

@app.post("/generate-questions")
async def generate_questions(
    pdf_file: UploadFile = File(...),
    topic: str = Form(...),
    question_type: str = Form(...),
    num_questions: int = Form(...),
    difficulty: str = Form(...),
    section_ordering: List[str] = Form(...),
    section_marks: List[int] = Form(...),
):
    """
    Generates questions from a PDF document.
    
    Parameters:
    - pdf_file: The PDF file to process
    - topic: The topic for question generation
    - question_type: Type of questions to generate
    - num_questions: Number of questions to generate
    - difficulty: Difficulty level (easy, medium, hard)
    - section_ordering: Order of sections in the question paper
    - section_marks: Marks allocated for each section
    """
    try:
        # Validate file type
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read PDF content
        pdf_content = await pdf_file.read()
        if not pdf_content:
            raise HTTPException(status_code=400, detail="PDF file is empty")
        
        logger.info(f"Processing PDF: {pdf_file.filename} ({len(pdf_content)} bytes)")
        
        # Validate other inputs
        if len(section_ordering) != len(section_marks):
            raise HTTPException(
                status_code=400, 
                detail="section_ordering and section_marks must have the same length"
            )
        
        # Run the graph
        graph = get_question_generation_graph()
        
        # Log the input parameters
        logger.info(f"Graph input parameters:")
        logger.info(f"  - topic: '{topic}'")
        logger.info(f"  - question_type: '{question_type}'")
        logger.info(f"  - difficulty: '{difficulty}'")
        logger.info(f"  - num_questions: {num_questions}")
        
        result = await graph.ainvoke({
            "pdf_file": pdf_content,
            "topic": topic,
            "question_type": question_type,
            "num_questions": num_questions,
            "difficulty": difficulty,
            "section_ordering": section_ordering,
            "section_marks": section_marks,
        })
        
        # Convert all bytes in the result to base64 strings to avoid JSON serialization issues
        result = convert_bytes_in_dict(result)
        
        # Add metadata to response
        response_data = {
            "status": "success" if not result.get("error") else "partial_success",
            "message": "Question generation completed" if not result.get("error") else f"Completed with errors: {result.get('error')}",
            "data": result,
            "metadata": {
                "pdf_filename": pdf_file.filename,
                "pdf_size_bytes": len(pdf_content),
                "num_images": len(result.get("images", [])),
                "num_embeddings": len(result.get("embeddings", [])),
                "num_relevant_images": len(result.get("relevant_images", [])),
                "collection_name": result.get("collection_name")
            }
        }
        
        return JSONResponse(
            content=response_data,
            status_code=200 if not result.get("error") else 207  # 207 for partial success
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_questions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "LangGraph API"}

@app.post("/test-text-embedding")
async def test_text_embedding(text: str = Form(...)):
    """Test endpoint to verify text embedding is working."""
    try:
        from .embedding import embed_text_async
        embedding = await embed_text_async(text)
        
        return {
            "text": text,
            "embedding_received": bool(embedding),
            "embedding_shape": len(embedding) if embedding else 0,
            "embedding_sample": embedding[:3] if embedding else None  # First 3 values
        }
    except Exception as e:
        logger.error(f"Test embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))