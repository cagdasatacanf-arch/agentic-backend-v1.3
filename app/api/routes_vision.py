"""
Vision API Routes

Endpoints for Phase 5: Vision & Multimodal Integration

Features:
- Image analysis with GPT-4o Vision
- OCR (text extraction)
- Chart/graph data extraction
- Image comparison
- Visual question answering

Usage:
    POST /api/v1/vision/analyze        # Analyze image
    POST /api/v1/vision/ocr            # Extract text
    POST /api/v1/vision/chart          # Extract chart data
    POST /api/v1/vision/compare        # Compare images
    POST /api/v1/vision/question       # Visual Q&A
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import base64

from app.services.agents.vision_agent import VisionSpecialist

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/vision", tags=["vision"])


# ============================================================================
# Request/Response Models
# ============================================================================

class VisionAnalyzeRequest(BaseModel):
    """Request model for image analysis"""
    image_base64: str = Field(..., description="Base64-encoded image")
    query: Optional[str] = Field(None, description="Specific question about the image")
    detail: str = Field(default="auto", regex="^(auto|low|high)$")
    session_id: Optional[str] = None


class VisionAnalyzeResponse(BaseModel):
    """Response model for image analysis"""
    description: str
    objects: List[str]
    text_content: Optional[str]
    confidence: float
    analysis: dict
    agent_type: str


class OCRRequest(BaseModel):
    """Request model for OCR"""
    image_base64: str = Field(..., description="Base64-encoded image")
    language: str = Field(default="en", description="Expected language code")
    session_id: Optional[str] = None


class OCRResponse(BaseModel):
    """Response model for OCR"""
    text: str
    char_count: int
    has_text: bool
    language: str


class ChartExtractRequest(BaseModel):
    """Request model for chart extraction"""
    image_base64: str = Field(..., description="Base64-encoded chart/graph image")
    session_id: Optional[str] = None


class ChartExtractResponse(BaseModel):
    """Response model for chart extraction"""
    chart_type: str
    title: Optional[str]
    x_axis_label: Optional[str]
    y_axis_label: Optional[str]
    data_points: List[dict]
    insights: List[str]
    num_data_points: int


class CompareImagesRequest(BaseModel):
    """Request model for image comparison"""
    images_base64: List[str] = Field(..., min_items=2, max_items=4, description="2-4 base64-encoded images")
    comparison_type: str = Field(default="differences", regex="^(differences|similarities|both)$")
    session_id: Optional[str] = None


class CompareImagesResponse(BaseModel):
    """Response model for image comparison"""
    comparison_type: str
    num_images: int
    analysis: str


class VisualQuestionRequest(BaseModel):
    """Request model for visual question answering"""
    image_base64: str = Field(..., description="Base64-encoded image")
    question: str = Field(..., min_length=1, description="Question about the image")
    session_id: Optional[str] = None


class VisualQuestionResponse(BaseModel):
    """Response model for visual question answering"""
    question: str
    answer: str


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/analyze", response_model=VisionAnalyzeResponse)
async def analyze_image(request: VisionAnalyzeRequest):
    """
    Analyze an image with GPT-4o Vision.

    Provides:
    - Detailed description of the image
    - List of detected objects
    - Any visible text (OCR)
    - Confidence score
    - Optional answer to specific query

    Example:
        POST /api/v1/vision/analyze
        {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
            "query": "What objects are in this image?",
            "detail": "high"
        }

        Response:
        {
            "description": "A modern office workspace...",
            "objects": ["desk", "laptop", "coffee mug"],
            "text_content": "Welcome",
            "confidence": 0.95,
            "analysis": {...},
            "agent_type": "vision"
        }
    """
    try:
        logger.info(f"Vision analyze request (query={request.query is not None})")

        agent = VisionSpecialist()

        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)

        # Analyze
        result = await agent.analyze(
            image_data=image_data,
            query=request.query,
            session_id=request.session_id,
            detail=request.detail
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

        response = VisionAnalyzeResponse(
            description=result["description"],
            objects=result["objects"],
            text_content=result.get("text_content"),
            confidence=result["confidence"],
            analysis=result["analysis"],
            agent_type=result["agent_type"]
        )

        logger.info(f"Vision analyze complete: {len(result['objects'])} objects")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vision analyze failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr", response_model=OCRResponse)
async def extract_text_ocr(request: OCRRequest):
    """
    Extract text from image using OCR.

    Powered by GPT-4o Vision for high-quality text extraction.

    Example:
        POST /api/v1/vision/ocr
        {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
            "language": "en"
        }

        Response:
        {
            "text": "Extracted text content...",
            "char_count": 156,
            "has_text": true,
            "language": "en"
        }
    """
    try:
        logger.info(f"OCR request (language={request.language})")

        agent = VisionSpecialist()

        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)

        # Extract text
        result = await agent.extract_text(
            image_data=image_data,
            session_id=request.session_id,
            language=request.language
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "OCR failed"))

        response = OCRResponse(
            text=result["text"],
            char_count=result["char_count"],
            has_text=result["has_text"],
            language=result["language"]
        )

        logger.info(f"OCR complete: {result['char_count']} characters")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chart", response_model=ChartExtractResponse)
async def extract_chart_data(request: ChartExtractRequest):
    """
    Extract structured data from charts/graphs.

    Supports:
    - Bar charts
    - Line graphs
    - Pie charts
    - Scatter plots
    - Histograms
    - And more

    Example:
        POST /api/v1/vision/chart
        {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
        }

        Response:
        {
            "chart_type": "bar",
            "title": "Sales by Quarter",
            "x_axis_label": "Quarter",
            "y_axis_label": "Revenue ($M)",
            "data_points": [
                {"label": "Q1", "value": 2.5},
                {"label": "Q2", "value": 3.2}
            ],
            "insights": ["Revenue increased 28% Q1 to Q2"],
            "num_data_points": 4
        }
    """
    try:
        logger.info("Chart extraction request")

        agent = VisionSpecialist()

        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)

        # Extract chart data
        result = await agent.extract_chart(
            image_data=image_data,
            session_id=request.session_id
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Chart extraction failed"))

        response = ChartExtractResponse(
            chart_type=result["chart_type"],
            title=result.get("title"),
            x_axis_label=result.get("x_axis_label"),
            y_axis_label=result.get("y_axis_label"),
            data_points=result["data_points"],
            insights=result["insights"],
            num_data_points=result["num_data_points"]
        )

        logger.info(f"Chart extraction complete: {result['chart_type']} with {result['num_data_points']} points")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=CompareImagesResponse)
async def compare_images(request: CompareImagesRequest):
    """
    Compare 2-4 images and identify differences or similarities.

    Use cases:
    - Before/after comparisons
    - Version comparison
    - Quality control
    - Change detection

    Example:
        POST /api/v1/vision/compare
        {
            "images_base64": ["img1_base64...", "img2_base64..."],
            "comparison_type": "differences"
        }

        Response:
        {
            "comparison_type": "differences",
            "num_images": 2,
            "analysis": "The main differences are..."
        }
    """
    try:
        logger.info(f"Image comparison request ({len(request.images_base64)} images, {request.comparison_type})")

        agent = VisionSpecialist()

        # Decode all images
        images = [base64.b64decode(img) for img in request.images_base64]

        # Compare
        result = await agent.compare(
            images=images,
            session_id=request.session_id,
            comparison_type=request.comparison_type
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Comparison failed"))

        response = CompareImagesResponse(
            comparison_type=result["comparison_type"],
            num_images=result["num_images"],
            analysis=result["analysis"]
        )

        logger.info(f"Image comparison complete: {len(result['analysis'])} chars")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/question", response_model=VisualQuestionResponse)
async def answer_visual_question(request: VisualQuestionRequest):
    """
    Answer a specific question about an image.

    Use cases:
    - "How many people are in this image?"
    - "What color is the car?"
    - "Is this product defective?"
    - "What text is visible on the sign?"

    Example:
        POST /api/v1/vision/question
        {
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
            "question": "How many people are in this image?"
        }

        Response:
        {
            "question": "How many people are in this image?",
            "answer": "There are 3 people visible in the image."
        }
    """
    try:
        logger.info(f"Visual Q&A request: {request.question[:60]}...")

        agent = VisionSpecialist()

        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)

        # Answer question
        result = await agent.answer_question(
            image_data=image_data,
            question=request.question,
            session_id=request.session_id
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Visual Q&A failed"))

        response = VisualQuestionResponse(
            question=result["question"],
            answer=result["answer"]
        )

        logger.info(f"Visual Q&A complete: {len(result['answer'])} chars")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visual Q&A failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    detail: str = Form("auto")
):
    """
    Upload an image file and analyze it.

    Accepts: JPG, PNG, GIF, BMP, WEBP

    Example using curl:
        curl -X POST "http://localhost:8000/api/v1/vision/upload-analyze" \
          -F "file=@image.jpg" \
          -F "query=What's in this image?"
    """
    try:
        logger.info(f"File upload & analyze: {file.filename}")

        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {allowed_types}"
            )

        # Read file
        image_data = await file.read()

        # Analyze
        agent = VisionSpecialist()
        result = await agent.analyze(
            image_data=image_data,
            query=query,
            detail=detail
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

        logger.info(f"File analyze complete: {len(result['objects'])} objects")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File analyze failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for vision system.

    Returns:
        Status of vision capabilities
    """
    try:
        return {
            "status": "healthy",
            "model": "gpt-4o",
            "capabilities": [
                "image_analysis",
                "ocr",
                "chart_extraction",
                "image_comparison",
                "visual_qa"
            ],
            "system": "vision_multimodal"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
