
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.generation import generation_node
from app.schema import QuestionGenerationState

@pytest.mark.asyncio
async def test_generation_node_success():
    """ 
    Tests the generation_node with a successful API call.
    """
    initial_state = {
        "relevant_images": [b"image1_bytes", b"image2_bytes"],
        "topic": "testing",
        "question_type": "mcq",
        "num_questions": 2,
        "difficulty": "easy",
        "section_ordering": ["A"],
        "section_marks": [10],
        "error": None
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "questions": [{"question_text": "What is 1+1?"}],
        "question_paper": {"title": "Test Paper"}
    }
    
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client) as mock_async_client:
        result = await generation_node(initial_state)

        assert "error" not in result
        assert result["questions"] == [{"question_text": "What is 1+1?"}]
        assert result["question_paper"] == {"title": "Test Paper"}
        mock_async_client.assert_called_once()

@pytest.mark.asyncio
async def test_generation_node_api_error():
    """
    Tests the generation_node with a failed API call.
    """
    initial_state = {
        "relevant_images": [b"image1_bytes"],
        "topic": "testing",
        "question_type": "mcq",
        "num_questions": 1,
        "difficulty": "hard",
        "section_ordering": ["A"],
        "section_marks": [5],
        "error": None
    }

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.HTTPStatusError(
        message="Server error", request=MagicMock(), response=mock_response
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await generation_node(initial_state)

        assert result["error"] is not None
        assert "Generation failed" in result["error"]
        assert result["questions"] == []
        assert result["question_paper"] is None

@pytest.mark.asyncio
async def test_generation_node_no_images():
    """
    Tests that the node skips generation if there are no relevant images.
    """
    initial_state = {
        "relevant_images": [],
        "topic": "testing",
        "question_type": "mcq",
        "num_questions": 5,
        "difficulty": "medium",
        "section_ordering": ["A"],
        "section_marks": [10],
        "error": None
    }

    result = await generation_node(initial_state)

    assert result["questions"] == []
    assert result["question_paper"] is None
