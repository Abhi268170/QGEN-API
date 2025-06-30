# This file defines the FastAPI application and the API endpoints.

from fastapi import FastAPI, File, UploadFile, Form, HTTPException  # The main framework for building the API.
from fastapi.responses import JSONResponse
from .graph import get_graph, get_question_generation_graph  # Import the graph builder from the graph module.
from .schema import GraphInput, QuestionGenerationInput  # Import the input schema.
from .config import SUPPORTED_QUESTION_TYPES, DIFFICULTY_LEVELS
from typing import List, Dict, Any, Optional
import base64
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangGraph API",
    description="An API for running a LangGraph application with question generation from PDFs using Qwen2.5-VL-7B.",
    version="0.2.0",
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

def parse_form_lists(value: str) -> List:
    """Parse form data that might be JSON arrays or comma-separated values."""
    if not value:
        return []
    
    # Try parsing as JSON first
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Fall back to comma-separated values
    return [v.strip() for v in value.split(',') if v.strip()]

def validate_section_marks_constraint(
    questions: List[Dict], 
    section_ordering: List[str], 
    section_marks: List[int]
) -> Dict[str, Any]:
    """Validate that section marks match exactly with assigned marks."""
    validation_result = {"valid": True, "issues": []}
    
    for section_name, expected_marks in zip(section_ordering, section_marks):
        section_questions = [q for q in questions if q.get("section") == section_name]
        actual_marks = sum(q.get("marks", 0) for q in section_questions)
        
        if actual_marks != expected_marks:
            validation_result["valid"] = False
            validation_result["issues"].append({
                "section": section_name,
                "expected_marks": expected_marks,
                "actual_marks": actual_marks,
                "difference": actual_marks - expected_marks
            })
    
    return validation_result

def _generate_response_message(
    result: Dict, 
    mark_validation: Dict, 
    mark_assignment_mode: str
) -> str:
    """Generate appropriate response message based on results."""
    base_message = "Question generation completed"
    
    issues = []
    
    if result.get("error"):
        issues.append(f"Generation errors: {result.get('error')}")
    
    if not mark_validation["valid"]:
        issues.append("Mark assignment constraints not fully met")
    
    if issues:
        return f"{base_message} with issues: {'; '.join(issues)}"
    
    if mark_assignment_mode == "section_constrained":
        return f"{base_message} with exact section mark allocation"
    else:
        return f"{base_message} with adaptive mark assignment"

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "LangGraph API with Qwen2.5-VL-7B",
        "endpoints": {
            "/run": "Run the basic LangGraph workflow",
            "/generate-questions": "Generate questions from a PDF document using Qwen2.5-VL-7B",
            "/docs": "API documentation",
            "/health": "Health check endpoint",
            "/supported-options": "Get supported question types and difficulty levels"
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
    section_ordering: Optional[str] = Form(None),
    section_marks: Optional[str] = Form(None),
):
    """
    Generates questions from a PDF document using Qwen2.5-VL-7B with advanced mark assignment.
    
    Parameters:
    - pdf_file: The PDF file to process
    - topic: The topic for question generation (context within images)
    - question_type: Type of questions (mcq, msq, short_answer, yes_no)
    - num_questions: Number of questions to generate (1-100)
    - difficulty: Difficulty level (easy, medium, hard)
    - section_ordering: JSON array or comma-separated list of section names (optional)
    - section_marks: JSON array or comma-separated list of marks per section (optional)
    
    Mark Assignment Behavior:
    - When section_marks is provided: Total marks for each section will EXACTLY match the specified marks
    - When section_marks is not provided: Marks assigned based on difficulty, complexity, and question type
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
        
        # Validate question type
        if question_type.lower() not in SUPPORTED_QUESTION_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid question_type. Supported types: {', '.join(SUPPORTED_QUESTION_TYPES)}"
            )
        
        # Validate difficulty
        if difficulty.lower() not in DIFFICULTY_LEVELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid difficulty. Supported levels: {', '.join(DIFFICULTY_LEVELS)}"
            )
        
        # Validate number of questions
        if num_questions < 1 or num_questions > 100:
            raise HTTPException(
                status_code=400,
                detail="num_questions must be between 1 and 100"
            )
        
        # Parse section data
        section_ordering_list = []
        section_marks_list = []
        
        if section_ordering:
            section_ordering_list = parse_form_lists(section_ordering)
            
        if section_marks:
            section_marks_str_list = parse_form_lists(section_marks)
            try:
                section_marks_list = [int(m) for m in section_marks_str_list]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="section_marks must contain only integers"
                )
        
        # Validate section data consistency
        if section_ordering_list and section_marks_list:
            if len(section_ordering_list) != len(section_marks_list):
                raise HTTPException(
                    status_code=400, 
                    detail="section_ordering and section_marks must have the same length"
                )
            
            # Validate total marks (should be reasonable for the number of questions)
            total_section_marks = sum(section_marks_list)
            if total_section_marks < num_questions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Total section marks ({total_section_marks}) must be at least equal to number of questions ({num_questions})"
                )
        
        # Determine mark assignment mode
        mark_assignment_mode = "section_constrained" if section_marks_list else "adaptive"
        
        # Run the graph (now includes mark assignment node)
        graph = get_question_generation_graph()
        
        # Log the input parameters
        logger.info(f"Graph input parameters:")
        logger.info(f"  - topic: '{topic}'")
        logger.info(f"  - question_type: '{question_type}'")
        logger.info(f"  - difficulty: '{difficulty}'")
        logger.info(f"  - num_questions: {num_questions}")
        logger.info(f"  - mark_assignment_mode: {mark_assignment_mode}")
        if section_ordering_list:
            logger.info(f"  - section_ordering: {section_ordering_list}")
            logger.info(f"  - section_marks: {section_marks_list}")
        
        result = await graph.ainvoke({
            "pdf_file": pdf_content,
            "topic": topic,
            "question_type": question_type.lower(),
            "num_questions": num_questions,
            "difficulty": difficulty.lower(),
            "section_ordering": section_ordering_list,
            "section_marks": section_marks_list,
        })
        
        # Validate mark assignment if section marks were specified
        mark_validation = {"valid": True, "issues": []}
        if section_marks_list:
            mark_validation = validate_section_marks_constraint(
                result.get("questions", []), section_ordering_list, section_marks_list
            )
        
        # Convert all bytes in the result to base64 strings to avoid JSON serialization issues
        result = convert_bytes_in_dict(result)
        
        # Calculate mark statistics
        questions = result.get("questions", [])
        total_marks = result.get("total_marks", sum(q.get("marks", 0) for q in questions))
        
        # Prepare section-wise statistics
        section_stats = []
        if section_ordering_list:
            for section_name in section_ordering_list:
                section_questions = [q for q in questions if q.get("section") == section_name]
                section_stats.append({
                    "section_name": section_name,
                    "questions_count": len(section_questions),
                    "total_marks": sum(q.get("marks", 0) for q in section_questions),
                    "expected_marks": section_marks_list[section_ordering_list.index(section_name)] if section_marks_list else None,
                    "marks_per_question": [q.get("marks", 0) for q in section_questions]
                })
        
        # Add metadata to response
        response_data = {
            "status": "success" if not result.get("error") and mark_validation["valid"] else "partial_success",
            "message": _generate_response_message(result, mark_validation, mark_assignment_mode),
            "data": {
                "questions": questions,
                "question_paper": result.get("question_paper"),
                "total_questions": len(questions),
                "total_marks": total_marks
            },
            "mark_assignment": {
                "mode": mark_assignment_mode,
                "validation": mark_validation,
                "section_stats": section_stats if section_stats else None,
                "average_marks_per_question": round(total_marks / len(questions), 2) if questions else 0
            },
            "metadata": {
                "pdf_filename": pdf_file.filename,
                "pdf_size_bytes": len(pdf_content),
                "num_images": len(result.get("images", [])),
                "num_embeddings": len(result.get("embeddings", [])),
                "num_relevant_images": len(result.get("relevant_images", [])),
                "collection_name": result.get("collection_name"),
                "model_used": "qwen/qwen2.5-vl-7b"
            }
        }
        
        # Determine HTTP status code
        status_code = 200
        if result.get("error") or not mark_validation["valid"]:
            status_code = 207  # 207 for partial success
        
        return JSONResponse(
            content=response_data,
            status_code=status_code
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
    return {"status": "healthy", "service": "LangGraph API", "model": "qwen/qwen2.5-vl-7b"}

@app.get("/supported-options")
async def get_supported_options():
    """Get supported question types and difficulty levels."""
    return {
        "question_types": SUPPORTED_QUESTION_TYPES,
        "difficulty_levels": DIFFICULTY_LEVELS,
        "max_questions": 100,
        "batch_size": 10
    }

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