"""
Unit Tests for Role-Based Access Control (RBAC)

Tests the RBAC system including:
- Role management (Admin, User, ReadOnly)
- Permission checking
- API key generation and validation
- Audit logging
- User access patterns
"""

import pytest
from datetime import datetime, timedelta
from app.services.rbac_service import (
    RBACService,
    Role,
    Permission,
    get_rbac_service
)


@pytest.fixture
async def rbac():
    """Get RBAC service instance"""
    service = get_rbac_service()
    # Clean up test data
    import redis.asyncio as redis
    redis_client = await service._get_redis()
    await redis_client.flushdb()
    return service


@pytest.mark.asyncio
async def test_generate_api_key(rbac):
    """Test API key generation"""
    api_key = await rbac.generate_api_key(
        user_id="test_user",
        role=Role.USER
    )

    assert api_key.startswith("sk_live_")
    assert len(api_key) > 32


@pytest.mark.asyncio
async def test_validate_api_key(rbac):
    """Test API key validation"""
    # Generate key
    api_key = await rbac.generate_api_key(
        user_id="test_user",
        role=Role.USER
    )

    # Validate key
    user_id = await rbac.validate_api_key(api_key)

    assert user_id == "test_user"


@pytest.mark.asyncio
async def test_validate_invalid_api_key(rbac):
    """Test validation of invalid API key"""
    user_id = await rbac.validate_api_key("invalid_key")
    assert user_id is None


@pytest.mark.asyncio
async def test_get_user_role(rbac):
    """Test getting user role"""
    user_id = "test_user"

    # Set role
    await rbac.set_user_role(user_id, Role.PRO)

    # Get role
    role = await rbac.get_user_role(user_id)

    assert role == Role.PRO


@pytest.mark.asyncio
async def test_default_user_role(rbac):
    """Test default role for new users"""
    role = await rbac.get_user_role("new_user")
    assert role == Role.USER  # Default role


@pytest.mark.asyncio
async def test_admin_permissions(rbac):
    """Test admin role has all permissions"""
    user_id = "admin_user"
    await rbac.set_user_role(user_id, Role.ADMIN)

    # Admin should have all permissions
    assert await rbac.has_permission(user_id, Permission.READ) is True
    assert await rbac.has_permission(user_id, Permission.WRITE) is True
    assert await rbac.has_permission(user_id, Permission.DELETE) is True
    assert await rbac.has_permission(user_id, Permission.ADMIN) is True


@pytest.mark.asyncio
async def test_user_permissions(rbac):
    """Test regular user permissions"""
    user_id = "regular_user"
    await rbac.set_user_role(user_id, Role.USER)

    # User should have read/write but not delete/admin
    assert await rbac.has_permission(user_id, Permission.READ) is True
    assert await rbac.has_permission(user_id, Permission.WRITE) is True
    assert await rbac.has_permission(user_id, Permission.DELETE) is False
    assert await rbac.has_permission(user_id, Permission.ADMIN) is False


@pytest.mark.asyncio
async def test_readonly_permissions(rbac):
    """Test read-only user permissions"""
    user_id = "readonly_user"
    await rbac.set_user_role(user_id, Role.READONLY)

    # ReadOnly should only have read permission
    assert await rbac.has_permission(user_id, Permission.READ) is True
    assert await rbac.has_permission(user_id, Permission.WRITE) is False
    assert await rbac.has_permission(user_id, Permission.DELETE) is False
    assert await rbac.has_permission(user_id, Permission.ADMIN) is False


@pytest.mark.asyncio
async def test_log_action(rbac):
    """Test audit logging"""
    user_id = "test_user"

    # Log some actions
    await rbac.log_action(
        user_id=user_id,
        action="query",
        resource="/api/v1/query",
        success=True
    )

    await rbac.log_action(
        user_id=user_id,
        action="delete",
        resource="/api/v1/document/123",
        success=False,
        details="Permission denied"
    )

    # Get audit logs
    logs = await rbac.get_audit_logs(
        user_id=user_id,
        limit=10
    )

    assert len(logs) == 2
    assert logs[0]["action"] == "delete"  # Most recent first
    assert logs[0]["success"] is False
    assert logs[1]["action"] == "query"
    assert logs[1]["success"] is True


@pytest.mark.asyncio
async def test_get_audit_logs_with_filters(rbac):
    """Test filtered audit log retrieval"""
    user_id = "test_user"

    # Log actions over time
    for i in range(10):
        await rbac.log_action(
            user_id=user_id,
            action=f"action_{i}",
            resource="/api/v1/test",
            success=i % 2 == 0  # Alternate success/failure
        )

    # Get only successful actions
    logs = await rbac.get_audit_logs(
        user_id=user_id,
        success=True,
        limit=20
    )

    assert len(logs) == 5
    assert all(log["success"] is True for log in logs)


@pytest.mark.asyncio
async def test_get_audit_logs_time_range(rbac):
    """Test audit logs with time range filter"""
    user_id = "test_user"

    # Log action now
    await rbac.log_action(
        user_id=user_id,
        action="recent_action",
        resource="/api/v1/test",
        success=True
    )

    # Get logs from last hour
    logs = await rbac.get_audit_logs(
        user_id=user_id,
        start_time=datetime.now() - timedelta(hours=1),
        limit=100
    )

    assert len(logs) >= 1
    assert any(log["action"] == "recent_action" for log in logs)


@pytest.mark.asyncio
async def test_revoke_api_key(rbac):
    """Test API key revocation"""
    user_id = "test_user"

    # Generate and validate key
    api_key = await rbac.generate_api_key(user_id, Role.USER)
    assert await rbac.validate_api_key(api_key) == user_id

    # Revoke key
    await rbac.revoke_api_key(api_key)

    # Key should no longer be valid
    assert await rbac.validate_api_key(api_key) is None


@pytest.mark.asyncio
async def test_list_user_api_keys(rbac):
    """Test listing all API keys for a user"""
    user_id = "test_user"

    # Generate multiple keys
    key1 = await rbac.generate_api_key(user_id, Role.USER)
    key2 = await rbac.generate_api_key(user_id, Role.USER)

    # List keys
    keys = await rbac.list_user_api_keys(user_id)

    assert len(keys) >= 2
    # Keys should be hashed in storage
    assert all(not k.startswith("sk_live_") for k in keys)


@pytest.mark.asyncio
async def test_check_rate_limit_integration(rbac):
    """Test RBAC integration with rate limiting"""
    user_id = "test_user"

    # Set role
    await rbac.set_user_role(user_id, Role.USER)

    # Check if user's role affects rate limits
    role = await rbac.get_user_role(user_id)
    assert role == Role.USER

    # Different roles should have different rate limits
    # (This would be checked in the rate limiter service)


@pytest.mark.asyncio
async def test_get_user_stats(rbac):
    """Test getting user activity statistics"""
    user_id = "test_user"

    # Generate some activity
    for i in range(15):
        await rbac.log_action(
            user_id=user_id,
            action="query",
            resource=f"/api/v1/test/{i}",
            success=i % 3 != 0  # Some failures
        )

    # Get stats
    stats = await rbac.get_user_stats(
        user_id=user_id,
        period="day"
    )

    assert stats["total_actions"] == 15
    assert stats["successful_actions"] == 10
    assert stats["failed_actions"] == 5
    assert stats["success_rate"] == pytest.approx(66.67, rel=0.1)


@pytest.mark.asyncio
async def test_bulk_role_update(rbac):
    """Test updating roles for multiple users"""
    user_ids = ["user_1", "user_2", "user_3"]

    # Set all to PRO
    for user_id in user_ids:
        await rbac.set_user_role(user_id, Role.PRO)

    # Verify all updated
    for user_id in user_ids:
        role = await rbac.get_user_role(user_id)
        assert role == Role.PRO


@pytest.mark.asyncio
async def test_permission_denied_logging(rbac):
    """Test that permission denials are logged"""
    user_id = "readonly_user"
    await rbac.set_user_role(user_id, Role.READONLY)

    # Attempt write operation (should fail)
    has_permission = await rbac.has_permission(user_id, Permission.WRITE)
    assert has_permission is False

    # Log the denial
    await rbac.log_action(
        user_id=user_id,
        action="write",
        resource="/api/v1/document",
        success=False,
        details="Permission denied: User has READONLY role"
    )

    # Verify logged
    logs = await rbac.get_audit_logs(user_id, success=False, limit=10)
    assert len(logs) >= 1
    assert "Permission denied" in logs[0]["details"]
