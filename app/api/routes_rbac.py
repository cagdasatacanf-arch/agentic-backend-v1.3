"""
RBAC & Security API Routes

Endpoints for role-based access control and security management:
- User management
- Role assignment
- API key generation
- Audit logging
- Security analytics

Usage:
    POST /api/v1/security/users              # Create user
    GET  /api/v1/security/users              # List users
    GET  /api/v1/security/users/{user_id}    # Get user
    PUT  /api/v1/security/users/{user_id}/role  # Update role
    POST /api/v1/security/users/{user_id}/api-key  # Generate API key
    POST /api/v1/security/check-permission   # Check permission
    GET  /api/v1/security/audit              # Get audit logs
    GET  /api/v1/security/stats              # Security statistics
"""

from fastapi import APIRouter, HTTPException, Query, Path, Header, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import logging

from app.services.rbac_service import (
    get_rbac_service,
    Role,
    Permission,
    User,
    AuditEvent
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/security", tags=["security"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateUserRequest(BaseModel):
    """Request to create a user"""
    user_id: str = Field(..., min_length=1, description="Unique user ID")
    role: Role = Field(..., description="User role")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


class UpdateRoleRequest(BaseModel):
    """Request to update user role"""
    role: Role = Field(..., description="New role")


class CheckPermissionRequest(BaseModel):
    """Request to check permission"""
    user_id: str = Field(..., description="User ID")
    permission: Permission = Field(..., description="Permission to check")


class UserResponse(BaseModel):
    """User response"""
    user_id: str
    role: str
    has_api_key: bool
    created_at: datetime
    last_active: Optional[datetime]
    metadata: Dict


class AuditEventResponse(BaseModel):
    """Audit event response"""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    result: str
    details: Optional[Dict]
    ip_address: Optional[str]
    session_id: Optional[str]


class SecurityStatsResponse(BaseModel):
    """Security statistics response"""
    users: Dict
    audit: Dict


# ============================================================================
# Helper Functions
# ============================================================================

def user_to_response(user: User) -> UserResponse:
    """Convert User to UserResponse"""
    return UserResponse(
        user_id=user.user_id,
        role=user.role.value,
        has_api_key=user.api_key_hash is not None,
        created_at=user.created_at,
        last_active=user.last_active,
        metadata=user.metadata
    )


def audit_to_response(event: AuditEvent) -> AuditEventResponse:
    """Convert AuditEvent to AuditEventResponse"""
    return AuditEventResponse(
        timestamp=event.timestamp,
        user_id=event.user_id,
        action=event.action,
        resource=event.resource,
        result=event.result,
        details=event.details,
        ip_address=event.ip_address,
        session_id=event.session_id
    )


# ============================================================================
# Middleware for Auto-Auditing
# ============================================================================

async def get_user_from_header(
    x_user_id: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
) -> Optional[str]:
    """
    Extract user ID from headers.

    Supports both X-User-ID header and API key authentication.
    """
    if x_api_key:
        rbac = get_rbac_service()
        user = await rbac.authenticate_api_key(x_api_key)
        return user.user_id if user else None

    return x_user_id


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/users", response_model=UserResponse)
async def create_user(request: CreateUserRequest):
    """
    Create a new user.

    Requires admin privileges.

    Example:
        POST /api/v1/security/users
        {
            "user_id": "user_123",
            "role": "user",
            "metadata": {
                "email": "user@example.com",
                "name": "John Doe"
            }
        }

        Response:
        {
            "user_id": "user_123",
            "role": "user",
            "has_api_key": false,
            "created_at": "2024-01-20T10:30:00",
            "last_active": null,
            "metadata": {
                "email": "user@example.com",
                "name": "John Doe"
            }
        }
    """
    try:
        logger.info(f"Creating user: {request.user_id} (role={request.role})")

        rbac = get_rbac_service()
        user = await rbac.create_user(
            user_id=request.user_id,
            role=request.role,
            metadata=request.metadata
        )

        # Log audit event
        await rbac.log_audit(
            user_id="system",
            action="create_user",
            resource=f"user:{request.user_id}",
            result="success"
        )

        return user_to_response(user)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    role: Optional[Role] = Query(None, description="Filter by role")
):
    """
    List all users.

    Optionally filter by role.

    Example:
        GET /api/v1/security/users?role=admin

        Response:
        [
            {
                "user_id": "admin_1",
                "role": "admin",
                "has_api_key": true,
                "created_at": "2024-01-15T08:00:00",
                "last_active": "2024-01-20T14:30:00",
                "metadata": {"email": "admin@example.com"}
            }
        ]
    """
    try:
        logger.info(f"Listing users (role_filter={role})")

        rbac = get_rbac_service()
        users = await rbac.list_users(role=role)

        return [user_to_response(u) for u in users]

    except Exception as e:
        logger.error(f"Failed to list users: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str = Path(..., description="User ID")
):
    """
    Get user details.

    Example:
        GET /api/v1/security/users/user_123

        Response:
        {
            "user_id": "user_123",
            "role": "user",
            "has_api_key": true,
            "created_at": "2024-01-18T09:00:00",
            "last_active": "2024-01-20T14:25:00",
            "metadata": {"email": "user@example.com"}
        }
    """
    try:
        logger.info(f"Getting user: {user_id}")

        rbac = get_rbac_service()
        user = await rbac.get_user(user_id)

        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        return user_to_response(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/users/{user_id}/role")
async def update_user_role(
    request: UpdateRoleRequest,
    user_id: str = Path(..., description="User ID")
):
    """
    Update user's role.

    Requires admin privileges.

    Example:
        PUT /api/v1/security/users/user_123/role
        {
            "role": "admin"
        }

        Response:
        {
            "status": "success",
            "user_id": "user_123",
            "old_role": "user",
            "new_role": "admin"
        }
    """
    try:
        logger.info(f"Updating role for {user_id} to {request.role}")

        rbac = get_rbac_service()

        # Get old role
        user = await rbac.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        old_role = user.role

        # Update role
        await rbac.update_user_role(user_id, request.role)

        # Log audit event
        await rbac.log_audit(
            user_id="system",
            action="update_role",
            resource=f"user:{user_id}",
            result="success",
            details={"old_role": old_role.value, "new_role": request.role.value}
        )

        return {
            "status": "success",
            "user_id": user_id,
            "old_role": old_role.value,
            "new_role": request.role.value
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user role: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/api-key")
async def generate_api_key(
    user_id: str = Path(..., description="User ID")
):
    """
    Generate API key for user.

    Returns the API key (only shown once).

    Example:
        POST /api/v1/security/users/user_123/api-key

        Response:
        {
            "api_key": "sk_live_abc123...",
            "user_id": "user_123",
            "note": "Save this key securely. It won't be shown again."
        }
    """
    try:
        logger.info(f"Generating API key for user: {user_id}")

        rbac = get_rbac_service()
        api_key = await rbac.generate_api_key(user_id)

        # Log audit event
        await rbac.log_audit(
            user_id=user_id,
            action="generate_api_key",
            resource=f"user:{user_id}",
            result="success"
        )

        return {
            "api_key": api_key,
            "user_id": user_id,
            "note": "Save this key securely. It won't be shown again."
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate API key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-permission")
async def check_permission(request: CheckPermissionRequest):
    """
    Check if user has permission.

    Example:
        POST /api/v1/security/check-permission
        {
            "user_id": "user_123",
            "permission": "query_agent"
        }

        Response:
        {
            "allowed": true,
            "user_id": "user_123",
            "permission": "query_agent",
            "role": "user"
        }
    """
    try:
        logger.info(f"Checking permission: {request.user_id} -> {request.permission}")

        rbac = get_rbac_service()

        # Get user to include role in response
        user = await rbac.get_user(request.user_id)
        if not user:
            raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")

        # Check permission
        allowed = await rbac.check_permission(request.user_id, request.permission)

        return {
            "allowed": allowed,
            "user_id": request.user_id,
            "permission": request.permission.value,
            "role": user.role.value
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check permission: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit", response_model=List[AuditEventResponse])
async def get_audit_logs(
    user_id: Optional[str] = Query(None, description="Filter by user"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results")
):
    """
    Get audit logs.

    Returns audit trail of all security events.

    Example:
        GET /api/v1/security/audit?user_id=user_123&limit=50

        Response:
        [
            {
                "timestamp": "2024-01-20T14:30:00",
                "user_id": "user_123",
                "action": "query",
                "resource": "math_agent",
                "result": "success",
                "details": {"query": "2+2"},
                "ip_address": "192.168.1.100",
                "session_id": "sess_abc123"
            }
        ]
    """
    try:
        logger.info(f"Getting audit logs (user={user_id}, limit={limit})")

        rbac = get_rbac_service()
        events = await rbac.get_audit_logs(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        return [audit_to_response(e) for e in events]

    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=SecurityStatsResponse)
async def get_security_stats():
    """
    Get security statistics.

    Returns metrics about users, roles, and audit events.

    Example:
        GET /api/v1/security/stats

        Response:
        {
            "users": {
                "by_role": {
                    "admin": 2,
                    "user": 15,
                    "readonly": 3
                },
                "total": 20
            },
            "audit": {
                "total_events": 1250,
                "total_failures": 12,
                "failures_24h": 2
            }
        }
    """
    try:
        logger.info("Getting security statistics")

        rbac = get_rbac_service()
        stats = await rbac.get_security_stats()

        return SecurityStatsResponse(
            users=stats["users"],
            audit=stats["audit"]
        )

    except Exception as e:
        logger.error(f"Failed to get security stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/authenticate")
async def authenticate_api_key(
    x_api_key: str = Header(..., description="API key")
):
    """
    Authenticate with API key.

    Example:
        POST /api/v1/security/authenticate
        Headers:
            X-API-Key: sk_live_abc123...

        Response:
        {
            "authenticated": true,
            "user_id": "user_123",
            "role": "user"
        }
    """
    try:
        logger.info("Authenticating API key")

        rbac = get_rbac_service()
        user = await rbac.authenticate_api_key(x_api_key)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return {
            "authenticated": True,
            "user_id": user.user_id,
            "role": user.role.value
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for security system.

    Returns:
        Status of RBAC capabilities
    """
    try:
        rbac = get_rbac_service()

        # Try a simple operation
        stats = await rbac.get_security_stats()

        return {
            "status": "healthy",
            "rbac": "enabled",
            "features": [
                "role-based access control",
                "api key authentication",
                "audit logging",
                "user management",
                "permission checking"
            ],
            "roles": [r.value for r in Role],
            "total_users": stats["users"]["total"],
            "total_audit_events": stats["audit"]["total_events"]
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/permissions")
async def list_permissions():
    """
    List all available permissions.

    Example:
        GET /api/v1/security/permissions

        Response:
        {
            "permissions": [
                "query_agent",
                "view_costs",
                "manage_budgets",
                ...
            ],
            "roles": {
                "admin": ["query_agent", "view_costs", ...],
                "user": ["query_agent", "view_costs", ...],
                "readonly": ["view_costs", "view_cache", ...]
            }
        }
    """
    from app.services.rbac_service import ROLE_PERMISSIONS

    return {
        "permissions": [p.value for p in Permission],
        "roles": {
            role.value: [p.value for p in perms]
            for role, perms in ROLE_PERMISSIONS.items()
        }
    }
