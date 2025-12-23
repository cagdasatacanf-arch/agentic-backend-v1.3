"""
Vision Analyzer - Multi-modal Analysis using GPT-4o Vision

Features:
- Image analysis with natural language queries
- Chart/graph data extraction
- OCR (Optical Character Recognition)
- Image comparison and reasoning
- Structured output parsing

Based on:
- GPT-4o Vision capabilities
- Multimodal reasoning patterns
- Production-grade image processing

Usage:
    analyzer = VisionAnalyzer()
    result = await analyzer.analyze_image(
        image_data=image_bytes,
        query="What's in this image?"
    )
"""

from typing import Dict, List, Optional, Union
import logging
import base64
from io import BytesIO
from dataclasses import dataclass
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VisionAnalysisResult:
    """Result from vision analysis"""
    description: str
    objects: List[str]
    text_content: Optional[str]
    confidence: float
    analysis: Dict
    raw_response: str


@dataclass
class ChartData:
    """Extracted chart/graph data"""
    chart_type: str  # bar, line, pie, scatter, etc.
    title: Optional[str]
    x_axis_label: Optional[str]
    y_axis_label: Optional[str]
    data_points: List[Dict]
    insights: List[str]
    raw_data: Optional[str]


class VisionAnalyzer:
    """
    Multi-modal image analysis using GPT-4o Vision.

    Capabilities:
    - General image understanding
    - Object detection and counting
    - Text extraction (OCR)
    - Chart/graph data extraction
    - Image comparison
    - Visual reasoning

    Usage:
        analyzer = VisionAnalyzer()

        # Analyze image
        result = await analyzer.analyze_image(
            image_data=image_bytes,
            query="Describe this image in detail"
        )

        # Extract text (OCR)
        text = await analyzer.extract_text(image_bytes)

        # Extract chart data
        chart_data = await analyzer.extract_chart_data(image_bytes)

        # Compare images
        comparison = await analyzer.compare_images([img1, img2])
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 1000,
        temperature: float = 0.0
    ):
        """
        Initialize vision analyzer.

        Args:
            model: Vision model to use (gpt-4o or gpt-4o-mini)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0 = deterministic)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.llm = ChatOpenAI(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=settings.openai_api_key
        )

        logger.info(f"VisionAnalyzer initialized with {model}")

    async def analyze_image(
        self,
        image_data: Union[bytes, str],
        query: Optional[str] = None,
        detail: str = "auto"
    ) -> VisionAnalysisResult:
        """
        Analyze an image with optional query.

        Args:
            image_data: Image as bytes or base64 string
            query: Optional specific question about the image
            detail: Vision detail level ("auto", "low", "high")

        Returns:
            VisionAnalysisResult with description, objects, text, etc.

        Example:
            result = await analyzer.analyze_image(
                image_data=image_bytes,
                query="What objects are in this image?",
                detail="high"
            )
            print(result.description)
            print(result.objects)
        """
        logger.info(f"Analyzing image (detail={detail}, query={query is not None})")

        # Prepare image data
        image_url = self._prepare_image_data(image_data)

        # Build prompt
        if query:
            prompt = f"""{query}

Additionally, provide:
1. A detailed description
2. List of objects/entities present
3. Any text visible in the image
4. Overall confidence in your analysis

Format your response as:
DESCRIPTION: [detailed description]
OBJECTS: [object1, object2, ...]
TEXT: [any visible text]
CONFIDENCE: [0.0-1.0]
ANALYSIS: [your detailed answer to the query]"""
        else:
            prompt = """Analyze this image comprehensively.

Provide:
1. A detailed description of what you see
2. List all objects and entities present
3. Extract any visible text
4. Note colors, composition, and context
5. Rate your confidence in this analysis (0.0-1.0)

Format your response as:
DESCRIPTION: [detailed description]
OBJECTS: [object1, object2, ...]
TEXT: [any visible text]
CONFIDENCE: [0.0-1.0]
ANALYSIS: [comprehensive analysis]"""

        try:
            # Create vision message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": detail
                        }
                    }
                ]
            )

            # Get response
            response = await self.llm.ainvoke([message])
            content = response.content

            # Parse structured response
            result = self._parse_analysis_response(content)

            logger.info(f"Image analysis complete: {len(result.objects)} objects found")
            return result

        except Exception as e:
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            raise

    async def extract_text(
        self,
        image_data: Union[bytes, str],
        language: str = "en"
    ) -> str:
        """
        Extract text from image (OCR).

        Args:
            image_data: Image as bytes or base64 string
            language: Expected language code

        Returns:
            Extracted text

        Example:
            text = await analyzer.extract_text(scan_image)
            print(f"Extracted: {text}")
        """
        logger.info(f"Extracting text from image (language={language})")

        image_url = self._prepare_image_data(image_data)

        prompt = f"""Extract ALL text from this image.

Language: {language}

Instructions:
1. Extract text preserving original layout if possible
2. Include formatting (bold, italic, etc.) if evident
3. Note any unclear or ambiguous text
4. If no text is found, respond with "NO_TEXT_FOUND"

Return ONLY the extracted text, nothing else."""

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"}
                    }
                ]
            )

            response = await self.llm.ainvoke([message])
            text = response.content.strip()

            if text == "NO_TEXT_FOUND":
                logger.info("No text found in image")
                return ""

            logger.info(f"Extracted {len(text)} characters of text")
            return text

        except Exception as e:
            logger.error(f"Text extraction failed: {e}", exc_info=True)
            raise

    async def extract_chart_data(
        self,
        image_data: Union[bytes, str]
    ) -> ChartData:
        """
        Extract structured data from charts/graphs.

        Args:
            image_data: Image containing chart/graph

        Returns:
            ChartData with type, labels, data points, insights

        Example:
            chart = await analyzer.extract_chart_data(chart_image)
            print(f"Type: {chart.chart_type}")
            print(f"Data points: {chart.data_points}")
        """
        logger.info("Extracting chart/graph data")

        image_url = self._prepare_image_data(image_data)

        prompt = """Analyze this chart/graph and extract structured data.

Provide:
1. Chart type (bar, line, pie, scatter, histogram, etc.)
2. Chart title (if present)
3. X-axis label and values
4. Y-axis label and values
5. All data points in structured format
6. Key insights from the data

Format your response as JSON:
{
  "chart_type": "...",
  "title": "...",
  "x_axis_label": "...",
  "y_axis_label": "...",
  "data_points": [
    {"label": "...", "value": ...},
    ...
  ],
  "insights": [
    "...",
    "..."
  ]
}

If this is not a chart/graph, respond with: {"error": "Not a chart/graph"}"""

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"}
                    }
                ]
            )

            response = await self.llm.ainvoke([message])
            content = response.content.strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in response")

            data = json.loads(json_match.group(0))

            if "error" in data:
                raise ValueError(data["error"])

            chart_data = ChartData(
                chart_type=data.get("chart_type", "unknown"),
                title=data.get("title"),
                x_axis_label=data.get("x_axis_label"),
                y_axis_label=data.get("y_axis_label"),
                data_points=data.get("data_points", []),
                insights=data.get("insights", []),
                raw_data=content
            )

            logger.info(f"Extracted {chart_data.chart_type} chart with {len(chart_data.data_points)} data points")
            return chart_data

        except Exception as e:
            logger.error(f"Chart extraction failed: {e}", exc_info=True)
            raise

    async def compare_images(
        self,
        images: List[Union[bytes, str]],
        comparison_type: str = "differences"
    ) -> Dict:
        """
        Compare multiple images.

        Args:
            images: List of images (2-4 images)
            comparison_type: "differences", "similarities", or "both"

        Returns:
            Comparison analysis

        Example:
            comparison = await analyzer.compare_images(
                [img1, img2],
                comparison_type="differences"
            )
        """
        if len(images) < 2 or len(images) > 4:
            raise ValueError("Can only compare 2-4 images at a time")

        logger.info(f"Comparing {len(images)} images ({comparison_type})")

        # Prepare all images
        image_urls = [self._prepare_image_data(img) for img in images]

        if comparison_type == "differences":
            prompt = "Compare these images and identify all DIFFERENCES. Be specific about what has changed, been added, or removed."
        elif comparison_type == "similarities":
            prompt = "Compare these images and identify all SIMILARITIES. What do they have in common?"
        else:  # both
            prompt = "Compare these images and identify both SIMILARITIES and DIFFERENCES. Provide a comprehensive comparison."

        try:
            # Build message with multiple images
            content = [{"type": "text", "text": prompt}]
            for url in image_urls:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "high"}
                })

            message = HumanMessage(content=content)

            response = await self.llm.ainvoke([message])
            comparison_text = response.content

            result = {
                "comparison_type": comparison_type,
                "num_images": len(images),
                "analysis": comparison_text,
                "timestamp": "2025-12-22"
            }

            logger.info(f"Image comparison complete: {len(comparison_text)} chars")
            return result

        except Exception as e:
            logger.error(f"Image comparison failed: {e}", exc_info=True)
            raise

    async def answer_visual_question(
        self,
        image_data: Union[bytes, str],
        question: str
    ) -> str:
        """
        Answer a specific question about an image.

        Args:
            image_data: Image to analyze
            question: Question to answer

        Returns:
            Answer to the question

        Example:
            answer = await analyzer.answer_visual_question(
                image_data,
                "How many people are in this image?"
            )
        """
        logger.info(f"Answering visual question: {question[:60]}...")

        image_url = self._prepare_image_data(image_data)

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"}
                    }
                ]
            )

            response = await self.llm.ainvoke([message])
            answer = response.content.strip()

            logger.info(f"Visual question answered: {len(answer)} chars")
            return answer

        except Exception as e:
            logger.error(f"Visual question answering failed: {e}", exc_info=True)
            raise

    def _prepare_image_data(self, image_data: Union[bytes, str]) -> str:
        """
        Prepare image data for API call.

        Args:
            image_data: Image as bytes or base64 string

        Returns:
            Data URL for image
        """
        if isinstance(image_data, bytes):
            # Convert bytes to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_image}"
        elif image_data.startswith('data:image'):
            # Already a data URL
            return image_data
        else:
            # Assume it's base64
            return f"data:image/jpeg;base64,{image_data}"

    def _parse_analysis_response(self, content: str) -> VisionAnalysisResult:
        """Parse structured analysis response"""
        # Extract sections
        description = self._extract_section(content, "DESCRIPTION") or "No description available"
        objects_str = self._extract_section(content, "OBJECTS") or ""
        text_content = self._extract_section(content, "TEXT")
        confidence_str = self._extract_section(content, "CONFIDENCE") or "0.8"
        analysis_text = self._extract_section(content, "ANALYSIS") or description

        # Parse objects list
        objects = []
        if objects_str:
            # Handle both comma-separated and newline-separated lists
            objects = [
                obj.strip().strip('[]"\'')
                for obj in re.split(r'[,\n]', objects_str)
                if obj.strip()
            ]

        # Parse confidence
        try:
            confidence = float(re.search(r'[\d.]+', confidence_str).group())
            confidence = max(0.0, min(1.0, confidence))
        except:
            confidence = 0.8

        return VisionAnalysisResult(
            description=description,
            objects=objects,
            text_content=text_content if text_content != "None" else None,
            confidence=confidence,
            analysis={
                "objects_detected": len(objects),
                "has_text": bool(text_content and text_content != "None"),
                "analysis": analysis_text
            },
            raw_response=content
        )

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a section from formatted response"""
        pattern = f"{section_name}:\\s*(.+?)(?=\\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None


# ============================================================================
# Singleton Instance
# ============================================================================

_analyzer: Optional[VisionAnalyzer] = None


def get_vision_analyzer() -> VisionAnalyzer:
    """Get or create global vision analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = VisionAnalyzer()
    return _analyzer
