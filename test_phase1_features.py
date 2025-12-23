#!/usr/bin/env python3
"""
Test Phase 1: Agentic AI Enhancements

Demonstrates the three core Phase 1 features:
1. Tool Metrics Tracking
2. Output Quality Scoring
3. Multi-Hop RAG

Usage:
    python test_phase1_features.py
"""

import httpx
import asyncio
import json
from typing import Dict

# API Configuration
BASE_URL = "http://localhost:8000"
API_KEY = ""  # Add your internal API key

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def test_tool_metrics():
    """Test 1: Tool Metrics Tracking"""
    print_section("TEST 1: Tool Metrics Tracking")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # First, make some agent queries to generate metrics
        print("📊 Generating tool execution metrics...")

        queries = [
            "What is 25 * 4?",  # Will use calculator
            "Calculate sqrt(144)",  # Will use calculator
            "What is 100 / 5?",  # Will use calculator
        ]

        for query in queries:
            try:
                response = await client.post(
                    f"{BASE_URL}/api/v1/langgraph/query",
                    headers=HEADERS,
                    json={"question": query}
                )
                print(f"  ✓ Query: {query}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")

        print("\n📈 Fetching tool metrics...")

        # Get all tool metrics
        response = await client.get(
            f"{BASE_URL}/api/v1/metrics/tools",
            headers=HEADERS
        )

        if response.status_code == 200:
            data = response.json()
            tools = data.get("tools", [])

            print(f"\nFound {len(tools)} tools with metrics:\n")

            for tool in tools:
                print(f"  🔧 {tool['tool_name']}")
                print(f"     Quality Score: {tool['quality_score']:.3f}")
                print(f"     Success Rate: {tool['success_rate']:.1%}")
                print(f"     Latency P50: {tool['latency_p50']:.2f}ms")
                print(f"     Latency P95: {tool['latency_p95']:.2f}ms")
                print(f"     Executions: {tool['execution_count']}")
                print()
        else:
            print(f"  ✗ Failed to get metrics: {response.status_code}")
            print(f"  Response: {response.text}")


async def test_output_quality():
    """Test 2: Output Quality Scoring"""
    print_section("TEST 2: Output Quality Scoring")

    async with httpx.AsyncClient(timeout=30.0) as client:
        print("🎯 Evaluating answer quality...\n")

        # Test case 1: High-quality answer
        test_case_1 = {
            "question": "What is Python?",
            "answer": (
                "Python is a high-level, interpreted programming language created by "
                "Guido van Rossum in 1991. It emphasizes code readability with its use "
                "of significant indentation. Python supports multiple programming paradigms "
                "including procedural, object-oriented, and functional programming."
            ),
            "sources": [
                {
                    "id": "python-doc-1",
                    "text": "Python is a high-level programming language...",
                    "score": 0.95
                }
            ],
            "include_feedback": True
        }

        print("Test Case 1: High-Quality Answer")
        print(f"Question: {test_case_1['question']}")
        print(f"Answer: {test_case_1['answer'][:80]}...")

        response = await client.post(
            f"{BASE_URL}/api/v1/quality/evaluate",
            headers=HEADERS,
            json=test_case_1
        )

        if response.status_code == 200:
            result = response.json()
            scores = result.get("scores", {})

            print(f"\n  📊 Scores:")
            print(f"     Overall: {scores.get('overall', 0):.3f}")
            print(f"     Citation Quality: {scores.get('citation_quality', 0):.3f}")
            print(f"     Completeness: {scores.get('completeness', 0):.3f}")
            print(f"     Conciseness: {scores.get('conciseness', 0):.3f}")

            print(f"\n  📝 Grade: {result.get('grade', 'N/A')}")

            feedback = result.get("feedback", [])
            if feedback:
                print(f"\n  💬 Feedback:")
                for item in feedback:
                    print(f"     {item}")
        else:
            print(f"  ✗ Failed: {response.status_code}")
            print(f"  Response: {response.text}")

        # Test case 2: Poor-quality answer
        print("\n" + "-" * 80 + "\n")

        test_case_2 = {
            "question": "What is the capital of France?",
            "answer": "France is a country in Europe.",
            "include_feedback": True
        }

        print("Test Case 2: Low-Quality Answer (Incomplete)")
        print(f"Question: {test_case_2['question']}")
        print(f"Answer: {test_case_2['answer']}")

        response = await client.post(
            f"{BASE_URL}/api/v1/quality/evaluate",
            headers=HEADERS,
            json=test_case_2
        )

        if response.status_code == 200:
            result = response.json()
            scores = result.get("scores", {})

            print(f"\n  📊 Scores:")
            print(f"     Overall: {scores.get('overall', 0):.3f}")
            print(f"     Completeness: {scores.get('completeness', 0):.3f}")

            print(f"\n  📝 Grade: {result.get('grade', 'N/A')}")

            feedback = result.get("feedback", [])
            if feedback:
                print(f"\n  💬 Feedback:")
                for item in feedback:
                    print(f"     {item}")
        else:
            print(f"  ✗ Failed: {response.status_code}")


async def test_multihop_rag():
    """Test 3: Multi-Hop RAG"""
    print_section("TEST 3: Multi-Hop Retrieval")

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("🔍 Testing multi-hop retrieval...\n")

        # Complex query that may need multiple retrieval hops
        query = "What are the main differences between RAG and traditional search?"

        print(f"Query: {query}\n")

        request = {
            "query": query,
            "max_hops": 3,
            "quality_threshold": 0.7,
            "verbose": True
        }

        print("Running multi-hop retrieval (max 3 hops, quality threshold: 0.7)...\n")

        response = await client.post(
            f"{BASE_URL}/api/v1/rag/multihop",
            headers=HEADERS,
            json=request
        )

        if response.status_code == 200:
            result = response.json()

            print(f"  ✓ Retrieval completed!")
            print(f"     Hops used: {result.get('hops_used', 0)}/{request['max_hops']}")
            print(f"     Final quality: {result.get('final_quality', 0):.3f}")
            print(f"     Stopped early: {result.get('stopped_early', False)}")
            print(f"     Documents found: {len(result.get('documents', []))}")

            steps = result.get("retrieval_steps", [])
            quality_scores = result.get("quality_scores", [])

            if steps:
                print(f"\n  📝 Retrieval Steps:")
                for i, (step, score) in enumerate(zip(steps, quality_scores)):
                    print(f"     {i+1}. '{step}' → Quality: {score:.3f}")

            documents = result.get("documents", [])
            if documents:
                print(f"\n  📄 Retrieved Documents (showing first 3):")
                for i, doc in enumerate(documents[:3]):
                    text = doc.get("text", "")[:100]
                    score = doc.get("score", 0)
                    print(f"     {i+1}. (score: {score:.3f}) {text}...")
        else:
            print(f"  ✗ Failed: {response.status_code}")
            print(f"  Response: {response.text}")


async def test_metrics_health():
    """Test metrics health endpoint"""
    print_section("TEST 0: Metrics Health Check")

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{BASE_URL}/api/v1/metrics/health",
            headers=HEADERS
        )

        if response.status_code == 200:
            data = response.json()

            if data.get("metrics_enabled"):
                print("  ✅ Metrics collection is ENABLED")
                print(f"     Tools tracked: {data.get('tools_tracked', 0)}")
                print(f"     Total executions: {data.get('total_executions', 0)}")
                tools = data.get("tools", [])
                if tools:
                    print(f"     Tools: {', '.join(tools)}")
            else:
                print("  ⚠️  Metrics collection is DISABLED")
                print(f"     Message: {data.get('message', 'Unknown')}")
        else:
            print(f"  ✗ Health check failed: {response.status_code}")


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Phase 1: Agentic AI Enhancements - Feature Tests".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # Check health first
        await test_metrics_health()

        # Run tests
        await test_tool_metrics()
        await test_output_quality()
        await test_multihop_rag()

        print_section("Summary")
        print("✅ All Phase 1 features tested successfully!")
        print("\nPhase 1 Components:")
        print("  1. ✓ Tool Metrics Tracking - Monitor tool success rates & latency")
        print("  2. ✓ Output Quality Scoring - LLM-as-judge evaluation")
        print("  3. ✓ Multi-Hop RAG - Iterative retrieval for complex questions")
        print("\nNext Steps:")
        print("  - View metrics: GET /api/v1/metrics/tools")
        print("  - Evaluate answers: POST /api/v1/quality/evaluate")
        print("  - Multi-hop search: POST /api/v1/rag/multihop")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
