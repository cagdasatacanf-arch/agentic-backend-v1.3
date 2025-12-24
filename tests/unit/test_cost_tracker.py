"""
Unit Tests for Cost Tracking and Budget Management

Tests the comprehensive cost tracking system including:
- Token usage tracking
- Cost calculation
- Budget limits and alerts
- Usage forecasting
- Cost optimization recommendations
"""

import pytest
from datetime import datetime, timedelta
from app.services.cost_tracker import (
    CostTracker,
    ModelPricing,
    UsageForecast,
    get_cost_tracker
)


@pytest.fixture
async def cost_tracker():
    """Get cost tracker instance"""
    tracker = get_cost_tracker()
    # Clean up any test data
    import redis.asyncio as redis
    redis_client = await tracker._get_redis()
    await redis_client.flushdb()
    return tracker


@pytest.mark.asyncio
async def test_calculate_cost_gpt4o(cost_tracker):
    """Test cost calculation for GPT-4o model"""
    # GPT-4o pricing: $2.50 per 1M input tokens, $10.00 per 1M output tokens
    cost = cost_tracker.calculate_cost(
        model=ModelPricing.GPT4O,
        input_tokens=1000,
        output_tokens=500
    )

    # Expected: (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
    # = 0.0025 + 0.005 = 0.0075
    assert abs(cost - 0.0075) < 0.0001


@pytest.mark.asyncio
async def test_calculate_cost_gpt4o_mini(cost_tracker):
    """Test cost calculation for GPT-4o-mini model"""
    # GPT-4o-mini pricing: $0.150 per 1M input, $0.600 per 1M output
    cost = cost_tracker.calculate_cost(
        model=ModelPricing.GPT4O_MINI,
        input_tokens=10000,
        output_tokens=5000
    )

    # Expected: (10000 / 1_000_000) * 0.150 + (5000 / 1_000_000) * 0.600
    # = 0.0015 + 0.003 = 0.0045
    assert abs(cost - 0.0045) < 0.0001


@pytest.mark.asyncio
async def test_calculate_cost_embeddings(cost_tracker):
    """Test cost calculation for embedding model"""
    # Embeddings pricing: $0.020 per 1M tokens
    cost = cost_tracker.calculate_cost(
        model=ModelPricing.TEXT_EMBEDDING_3_SMALL,
        input_tokens=5000,
        output_tokens=0
    )

    # Expected: (5000 / 1_000_000) * 0.020 = 0.0001
    assert abs(cost - 0.0001) < 0.00001


@pytest.mark.asyncio
async def test_track_usage(cost_tracker):
    """Test tracking usage and cost"""
    cost = await cost_tracker.track_usage(
        model=ModelPricing.GPT4O,
        input_tokens=1000,
        output_tokens=500,
        user_id="test_user",
        endpoint="/api/v1/query",
        session_id="session_123"
    )

    assert cost > 0

    # Verify usage was recorded
    stats = await cost_tracker.get_usage_stats(
        user_id="test_user",
        start_time=datetime.now() - timedelta(hours=1)
    )

    assert stats["total_cost"] > 0
    assert stats["total_requests"] == 1
    assert stats["total_tokens"] == 1500


@pytest.mark.asyncio
async def test_budget_limit_check(cost_tracker):
    """Test budget limit enforcement"""
    user_id = "test_user_budget"

    # Set budget limit
    await cost_tracker.set_budget_limit(
        user_id=user_id,
        limit=0.01,  # $0.01 limit
        period="hour"
    )

    # Track usage that exceeds budget
    for _ in range(10):
        await cost_tracker.track_usage(
            model=ModelPricing.GPT4O,
            input_tokens=1000,
            output_tokens=500,
            user_id=user_id
        )

    # Check if budget exceeded
    is_within_budget, remaining = await cost_tracker.check_budget(
        user_id=user_id,
        period="hour"
    )

    # Should be over budget after 10 requests
    assert is_within_budget is False
    assert remaining <= 0


@pytest.mark.asyncio
async def test_budget_alert(cost_tracker):
    """Test budget alert thresholds"""
    user_id = "test_user_alert"

    # Set budget with alert at 80%
    await cost_tracker.set_budget_limit(
        user_id=user_id,
        limit=0.10,
        period="day",
        alert_threshold=0.80
    )

    # Track usage to 85% of budget
    # 0.10 limit * 0.85 = 0.085
    # Each request costs ~0.0075
    for _ in range(11):  # 11 * 0.0075 = 0.0825
        await cost_tracker.track_usage(
            model=ModelPricing.GPT4O,
            input_tokens=1000,
            output_tokens=500,
            user_id=user_id
        )

    # Check if alert should be triggered
    should_alert, usage_percent = await cost_tracker.should_alert(
        user_id=user_id,
        period="day"
    )

    assert should_alert is True
    assert usage_percent > 80.0


@pytest.mark.asyncio
async def test_usage_forecast(cost_tracker):
    """Test usage forecasting"""
    user_id = "test_user_forecast"

    # Generate historical usage over last 7 days
    now = datetime.now()
    for day in range(7):
        timestamp = now - timedelta(days=day)
        for _ in range(10):  # 10 requests per day
            await cost_tracker.track_usage(
                model=ModelPricing.GPT4O,
                input_tokens=1000,
                output_tokens=500,
                user_id=user_id,
                timestamp=timestamp
            )

    # Get forecast
    forecast = await cost_tracker.forecast_usage(
        user_id=user_id,
        days=7
    )

    assert isinstance(forecast, UsageForecast)
    assert forecast.forecasted_cost > 0
    assert forecast.forecasted_tokens > 0
    assert forecast.confidence_level in ["low", "medium", "high"]


@pytest.mark.asyncio
async def test_cost_optimization_recommendations(cost_tracker):
    """Test cost optimization recommendations"""
    user_id = "test_user_optimize"

    # Track expensive usage pattern (lots of GPT-4o)
    for _ in range(50):
        await cost_tracker.track_usage(
            model=ModelPricing.GPT4O,
            input_tokens=2000,
            output_tokens=1000,
            user_id=user_id
        )

    # Get recommendations
    recommendations = await cost_tracker.get_cost_optimization_recommendations(
        user_id=user_id,
        period="day"
    )

    assert len(recommendations) > 0
    # Should suggest using cheaper model
    assert any("gpt-4o-mini" in rec.lower() for rec in recommendations)


@pytest.mark.asyncio
async def test_get_top_users_by_cost(cost_tracker):
    """Test getting top users by cost"""
    # Create usage for multiple users
    for i in range(5):
        user_id = f"user_{i}"
        # Each user has different usage
        for _ in range((i + 1) * 10):
            await cost_tracker.track_usage(
                model=ModelPricing.GPT4O,
                input_tokens=1000,
                output_tokens=500,
                user_id=user_id
            )

    # Get top 3 users
    top_users = await cost_tracker.get_top_users_by_cost(
        limit=3,
        period="day"
    )

    assert len(top_users) <= 3
    # Should be sorted by cost (descending)
    assert top_users[0]["user_id"] == "user_4"  # Most usage
    assert top_users[0]["total_cost"] > top_users[1]["total_cost"]


@pytest.mark.asyncio
async def test_get_usage_by_endpoint(cost_tracker):
    """Test getting usage breakdown by endpoint"""
    user_id = "test_user_endpoint"

    # Track usage on different endpoints
    await cost_tracker.track_usage(
        model=ModelPricing.GPT4O,
        input_tokens=1000,
        output_tokens=500,
        user_id=user_id,
        endpoint="/api/v1/query"
    )

    await cost_tracker.track_usage(
        model=ModelPricing.GPT4O,
        input_tokens=2000,
        output_tokens=1000,
        user_id=user_id,
        endpoint="/api/v1/chat"
    )

    # Get breakdown
    breakdown = await cost_tracker.get_usage_by_endpoint(
        user_id=user_id,
        period="day"
    )

    assert "/api/v1/query" in breakdown
    assert "/api/v1/chat" in breakdown
    assert breakdown["/api/v1/chat"]["total_cost"] > breakdown["/api/v1/query"]["total_cost"]


@pytest.mark.asyncio
async def test_reset_budget(cost_tracker):
    """Test resetting user budget"""
    user_id = "test_user_reset"

    # Set budget and track some usage
    await cost_tracker.set_budget_limit(user_id, 1.00, "day")
    await cost_tracker.track_usage(
        model=ModelPricing.GPT4O,
        input_tokens=1000,
        output_tokens=500,
        user_id=user_id
    )

    # Reset budget
    await cost_tracker.reset_budget(user_id, "day")

    # Check budget is reset
    stats = await cost_tracker.get_usage_stats(
        user_id=user_id,
        start_time=datetime.now() - timedelta(hours=1)
    )

    # Usage stats should be cleared
    assert stats.get("total_cost", 0) == 0 or stats == {}
