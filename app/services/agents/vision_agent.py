"""
Vision Specialist Agent

Specialized agent for visual understanding and image analysis:
- Image description and object detection
- OCR (text extraction from images)
- Chart/graph data extraction
- Visual question answering
- Image comparison

Based on research:
- GPT-4o Vision capabilities
- Multimodal reasoning patterns
- Structured visual understanding
"""

from typing import Dict, List, Optional, Union
import logging
import time

from app.services.vision_analyzer import get_vision_analyzer, VisionAnalyzer
from app.services.interaction_logger import log_interaction
from app.services.output_quality import OutputQualityScorer

logger = logging.getLogger(__name__)


class VisionSpecialist:
    """
    Specialized agent for visual understanding.

    Features:
    - General image analysis
    - Object detection and counting
    - OCR (text extraction)
    - Chart/graph data extraction
    - Image comparison
    - Visual question answering

    Usage:
        agent = VisionSpecialist()

        # Analyze image
        result = await agent.analyze(
            image_data=image_bytes,
            query="What's in this image?"
        )

        # Extract text
        result = await agent.extract_text(image_bytes)

        # Extract chart data
        result = await agent.extract_chart(chart_image)
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize vision specialist.

        Args:
            model: Vision model to use (gpt-4o or gpt-4o-mini)
        """
        self.model = model
        self.analyzer = get_vision_analyzer()

        logger.info(f"VisionSpecialist initialized with {model}")

    async def analyze(
        self,
        image_data: Union[bytes, str],
        query: Optional[str] = None,
        session_id: Optional[str] = None,
        detail: str = "auto"
    ) -> Dict:
        """
        Analyze an image with optional query.

        Args:
            image_data: Image as bytes or base64 string
            query: Optional specific question about the image
            session_id: Optional session ID for logging
            detail: Vision detail level ("auto", "low", "high")

        Returns:
            {
                "description": "...",
                "objects": [...],
                "text_content": "...",
                "confidence": 0.95,
                "analysis": {...},
                "agent_type": "vision"
            }
        """
        logger.info(f"Vision analysis: query={query is not None}")
        start_time = time.perf_counter()

        error_occurred = False
        error_message = None

        try:
            # Analyze image
            analysis_result = await self.analyzer.analyze_image(
                image_data=image_data,
                query=query,
                detail=detail
            )

            result = {
                "description": analysis_result.description,
                "objects": analysis_result.objects,
                "text_content": analysis_result.text_content,
                "confidence": analysis_result.confidence,
                "analysis": analysis_result.analysis,
                "agent_type": "vision",
                "success": True
            }

            logger.info(f"Vision analysis complete: {len(result['objects'])} objects found")

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}", exc_info=True)
            error_occurred = True
            error_message = str(e)
            result = {
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "agent_type": "vision",
                "success": False
            }

        finally:
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log interaction (Phase 4: Self-Improvement)
            try:
                # Quality scoring
                quality_scores = None
                if not error_occurred and result.get("success"):
                    quality_scorer = OutputQualityScorer()
                    answer_str = f"Description: {result.get('description', '')}\nObjects: {', '.join(result.get('objects', []))}"
                    quality_scores = await quality_scorer.score_answer(
                        question=query or "Analyze this image",
                        answer=answer_str
                    )

                # Log to training data
                log_interaction(
                    query=query or "Analyze image",
                    answer=result.get("description", ""),
                    agent_type="vision",
                    quality_scores=quality_scores,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    tools_used=["gpt4o_vision"],
                    error_occurred=error_occurred,
                    error_message=error_message
                )

            except Exception as log_error:
                # Don't fail the request if logging fails
                logger.warning(f"Failed to log vision interaction: {log_error}")

        return result

    async def extract_text(
        self,
        image_data: Union[bytes, str],
        session_id: Optional[str] = None,
        language: str = "en"
    ) -> Dict:
        """
        Extract text from image (OCR).

        Args:
            image_data: Image as bytes or base64 string
            session_id: Optional session ID for logging
            language: Expected language code

        Returns:
            {
                "text": "...",
                "char_count": 123,
                "agent_type": "vision"
            }
        """
        logger.info(f"OCR text extraction (language={language})")
        start_time = time.perf_counter()

        error_occurred = False
        error_message = None

        try:
            # Extract text
            text = await self.analyzer.extract_text(
                image_data=image_data,
                language=language
            )

            result = {
                "text": text,
                "char_count": len(text),
                "has_text": bool(text),
                "language": language,
                "agent_type": "vision",
                "success": True
            }

            logger.info(f"OCR complete: {len(text)} characters extracted")

        except Exception as e:
            logger.error(f"OCR failed: {e}", exc_info=True)
            error_occurred = True
            error_message = str(e)
            result = {
                "text": "",
                "error": str(e),
                "agent_type": "vision",
                "success": False
            }

        finally:
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log interaction
            try:
                quality_scores = None
                if not error_occurred and result.get("success") and result.get("has_text"):
                    quality_scorer = OutputQualityScorer()
                    quality_scores = await quality_scorer.score_answer(
                        question="Extract text from image",
                        answer=result.get("text", "")[:500]  # First 500 chars
                    )

                log_interaction(
                    query=f"Extract text (language={language})",
                    answer=result.get("text", ""),
                    agent_type="vision",
                    quality_scores=quality_scores,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    tools_used=["gpt4o_vision_ocr"],
                    error_occurred=error_occurred,
                    error_message=error_message
                )

            except Exception as log_error:
                logger.warning(f"Failed to log OCR interaction: {log_error}")

        return result

    async def extract_chart(
        self,
        image_data: Union[bytes, str],
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Extract structured data from chart/graph.

        Args:
            image_data: Image containing chart/graph
            session_id: Optional session ID for logging

        Returns:
            {
                "chart_type": "bar",
                "title": "...",
                "data_points": [...],
                "insights": [...],
                "agent_type": "vision"
            }
        """
        logger.info("Chart/graph data extraction")
        start_time = time.perf_counter()

        error_occurred = False
        error_message = None

        try:
            # Extract chart data
            chart_data = await self.analyzer.extract_chart_data(image_data)

            result = {
                "chart_type": chart_data.chart_type,
                "title": chart_data.title,
                "x_axis_label": chart_data.x_axis_label,
                "y_axis_label": chart_data.y_axis_label,
                "data_points": chart_data.data_points,
                "insights": chart_data.insights,
                "num_data_points": len(chart_data.data_points),
                "agent_type": "vision",
                "success": True
            }

            logger.info(f"Chart extraction complete: {chart_data.chart_type} with {len(chart_data.data_points)} points")

        except Exception as e:
            logger.error(f"Chart extraction failed: {e}", exc_info=True)
            error_occurred = True
            error_message = str(e)
            result = {
                "error": str(e),
                "agent_type": "vision",
                "success": False
            }

        finally:
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log interaction
            try:
                quality_scores = None
                if not error_occurred and result.get("success"):
                    quality_scorer = OutputQualityScorer()
                    answer_str = f"Chart type: {result.get('chart_type')}\nData points: {result.get('num_data_points')}\nInsights: {result.get('insights')}"
                    quality_scores = await quality_scorer.score_answer(
                        question="Extract chart data",
                        answer=answer_str
                    )

                log_interaction(
                    query="Extract chart/graph data",
                    answer=str(result.get("data_points", [])),
                    agent_type="vision",
                    quality_scores=quality_scores,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    tools_used=["gpt4o_vision_chart"],
                    error_occurred=error_occurred,
                    error_message=error_message
                )

            except Exception as log_error:
                logger.warning(f"Failed to log chart extraction interaction: {log_error}")

        return result

    async def compare(
        self,
        images: List[Union[bytes, str]],
        session_id: Optional[str] = None,
        comparison_type: str = "differences"
    ) -> Dict:
        """
        Compare multiple images.

        Args:
            images: List of 2-4 images
            session_id: Optional session ID for logging
            comparison_type: "differences", "similarities", or "both"

        Returns:
            {
                "comparison_type": "...",
                "num_images": 2,
                "analysis": "...",
                "agent_type": "vision"
            }
        """
        logger.info(f"Comparing {len(images)} images ({comparison_type})")
        start_time = time.perf_counter()

        error_occurred = False
        error_message = None

        try:
            # Compare images
            comparison = await self.analyzer.compare_images(
                images=images,
                comparison_type=comparison_type
            )

            result = {
                **comparison,
                "agent_type": "vision",
                "success": True
            }

            logger.info(f"Image comparison complete: {len(result['analysis'])} chars")

        except Exception as e:
            logger.error(f"Image comparison failed: {e}", exc_info=True)
            error_occurred = True
            error_message = str(e)
            result = {
                "error": str(e),
                "agent_type": "vision",
                "success": False
            }

        finally:
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log interaction
            try:
                quality_scores = None
                if not error_occurred and result.get("success"):
                    quality_scorer = OutputQualityScorer()
                    quality_scores = await quality_scorer.score_answer(
                        question=f"Compare {len(images)} images ({comparison_type})",
                        answer=result.get("analysis", "")
                    )

                log_interaction(
                    query=f"Compare {len(images)} images",
                    answer=result.get("analysis", ""),
                    agent_type="vision",
                    quality_scores=quality_scores,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    tools_used=["gpt4o_vision_compare"],
                    error_occurred=error_occurred,
                    error_message=error_message
                )

            except Exception as log_error:
                logger.warning(f"Failed to log comparison interaction: {log_error}")

        return result

    async def answer_question(
        self,
        image_data: Union[bytes, str],
        question: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Answer a specific question about an image.

        Args:
            image_data: Image to analyze
            question: Question to answer
            session_id: Optional session ID for logging

        Returns:
            {
                "question": "...",
                "answer": "...",
                "agent_type": "vision"
            }
        """
        logger.info(f"Visual Q&A: {question[:60]}...")
        start_time = time.perf_counter()

        error_occurred = False
        error_message = None

        try:
            # Answer question
            answer = await self.analyzer.answer_visual_question(
                image_data=image_data,
                question=question
            )

            result = {
                "question": question,
                "answer": answer,
                "agent_type": "vision",
                "success": True
            }

            logger.info(f"Visual Q&A complete: {len(answer)} chars")

        except Exception as e:
            logger.error(f"Visual Q&A failed: {e}", exc_info=True)
            error_occurred = True
            error_message = str(e)
            result = {
                "question": question,
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "agent_type": "vision",
                "success": False
            }

        finally:
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log interaction
            try:
                quality_scores = None
                if not error_occurred and result.get("success"):
                    quality_scorer = OutputQualityScorer()
                    quality_scores = await quality_scorer.score_answer(
                        question=question,
                        answer=result.get("answer", "")
                    )

                log_interaction(
                    query=question,
                    answer=result.get("answer", ""),
                    agent_type="vision",
                    quality_scores=quality_scores,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    tools_used=["gpt4o_vision_qa"],
                    error_occurred=error_occurred,
                    error_message=error_message
                )

            except Exception as log_error:
                logger.warning(f"Failed to log visual Q&A interaction: {log_error}")

        return result
