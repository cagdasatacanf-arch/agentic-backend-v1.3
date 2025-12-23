"""
Role-Based Access Control (RBAC) Service

Comprehensive security and access control system:
- Role-based permissions (admin, user, readonly)
- Resource-level access control
- Audit logging for all operations
- User session management
- API key management

Benefits:
- 🔒 Granular access control
- 📝 Complete audit trail
- 👥 Multi-user support
- 🔑 Secure API key management
- 📊 Security analytics

Usage:
    rbac = get_rbac_service()

    # Check permission
    allowed = await rbac.check_permission(
        user_id="user_123",
        resource="costs",
        action="read"
    )

    # Log audit event
    await rbac.log_audit(
        user_id="user_123",
        action="query",
        resource="agent",
        details={"query": "..."}
    )
"""

from typing import Optional, Dict, List, Set
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
import secrets

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


class Permission(str, Enum):
    """System permissions"""
    # Agent operations
    QUERY_AGENT = "query_agent"

    # Cost operations
    VIEW_COSTS = "view_costs"
    MANAGE_BUDGETS = "manage_budgets"

    # Cache operations
    VIEW_CACHE = "view_cache"
    CLEAR_CACHE = "clear_cache"

    # User management
    VIEW_USERS = "view_users"
    MANAGE_USERS = "manage_users"

    # System operations
    VIEW_METRICS = "view_metrics"
    MANAGE_SYSTEM = "manage_system"

    # Document operations
    UPLOAD_DOCS = "upload_docs"
    DELETE_DOCS = "delete_docs"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        # Admin has all permissions
        Permission.QUERY_AGENT,
        Permission.VIEW_COSTS,
        Permission.MANAGE_BUDGETS,
        Permission.VIEW_CACHE,
        Permission.CLEAR_CACHE,
        Permission.VIEW_USERS,
        Permission.MANAGE_USERS,
        Permission.VIEW_METRICS,
        Permission.MANAGE_SYSTEM,
        Permission.UPLOAD_DOCS,
        Permission.DELETE_DOCS,
    },
    Role.USER: {
        # Regular users
        Permission.QUERY_AGENT,
        Permission.VIEW_COSTS,
        Permission.VIEW_CACHE,
        Permission.VIEW_METRICS,
        Permission.UPLOAD_DOCS,
    },
    Role.READONLY: {
        # Read-only access
        Permission.VIEW_COSTS,
        Permission.VIEW_CACHE,
        Permission.VIEW_METRICS,
    }
}


@dataclass
class User:
    """User model"""
    user_id: str
    role: Role
    api_key_hash: Optional[str]
    created_at: datetime
    last_active: Optional[datetime]
    metadata: Dict


@dataclass
class AuditEvent:
    """Audit log event"""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    result: str  # success, failure, denied
    details: Optional[Dict]
    ip_address: Optional[str]
    session_id: Optional[str]


class RBACService:
    """
    Role-Based Access Control service.

    Features:
    - Role-based permissions
    - User management
    - API key authentication
    - Audit logging
    - Session tracking

    Example:
        rbac = RBACService()

        # Create user
        user = await rbac.create_user(
            user_id="user_123",
            role=Role.USER,
            metadata={"email": "user@example.com"}
        )

        # Check permission
        allowed = await rbac.check_permission(
            user_id="user_123",
            permission=Permission.QUERY_AGENT
        )

        # Log audit event
        await rbac.log_audit(
            user_id="user_123",
            action="query",
            resource="math_agent",
            result="success"
        )
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize RBAC service.

        Args:
            redis_url: Redis connection URL
        """
        redis_host = settings.redis_host if hasattr(settings, 'redis_host') else "localhost"
        redis_port = settings.redis_port if hasattr(settings, 'redis_port') else 6379
        self.redis_url = redis_url or f"redis://{redis_host}:{redis_port}/6"  # DB 6 for RBAC
        self.redis = None

        logger.info("RBACService initialized")

    async def _get_redis(self):
        """Get or create Redis connection"""
        if self.redis is None:
            self.redis = await redis.from_url(
                self.redis_url,
                decode_responses=False
            )
        return self.redis

    # ============================================================================
    # User Management
    # ============================================================================

    async def create_user(
        self,
        user_id: str,
        role: Role,
        metadata: Optional[Dict] = None
    ) -> User:
        """
        Create a new user.

        Args:
            user_id: Unique user identifier
            role: User role
            metadata: Additional user metadata

        Returns:
            Created user

        Example:
            user = await rbac.create_user(
                user_id="user_123",
                role=Role.USER,
                metadata={"email": "user@example.com", "name": "John Doe"}
            )
        """
        try:
            redis_client = await self._get_redis()

            # Check if user exists
            existing = await redis_client.get(f"user:{user_id}")
            if existing:
                raise ValueError(f"User {user_id} already exists")

            user = User(
                user_id=user_id,
                role=role,
                api_key_hash=None,
                created_at=datetime.now(),
                last_active=None,
                metadata=metadata or {}
            )

            # Store user
            user_data = {
                "user_id": user.user_id,
                "role": user.role.value,
                "api_key_hash": user.api_key_hash or "",
                "created_at": user.created_at.isoformat(),
                "last_active": user.last_active.isoformat() if user.last_active else "",
                "metadata": json.dumps(user.metadata)
            }

            await redis_client.hset(
                f"user:{user_id}",
                mapping={k: v.encode() if isinstance(v, str) else v for k, v in user_data.items()}
            )

            # Add to role index
            await redis_client.sadd(f"role:{role.value}", user_id)

            logger.info(f"User created: {user_id} (role={role.value})")

            return user

        except Exception as e:
            logger.error(f"Failed to create user: {e}", exc_info=True)
            raise

    async def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User object or None
        """
        try:
            redis_client = await self._get_redis()

            user_data = await redis_client.hgetall(f"user:{user_id}")
            if not user_data:
                return None

            return User(
                user_id=user_data[b"user_id"].decode(),
                role=Role(user_data[b"role"].decode()),
                api_key_hash=user_data[b"api_key_hash"].decode() or None,
                created_at=datetime.fromisoformat(user_data[b"created_at"].decode()),
                last_active=datetime.fromisoformat(user_data[b"last_active"].decode()) if user_data[b"last_active"] else None,
                metadata=json.loads(user_data[b"metadata"].decode())
            )

        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None

    async def update_user_role(self, user_id: str, new_role: Role):
        """
        Update user's role.

        Args:
            user_id: User ID
            new_role: New role
        """
        try:
            redis_client = await self._get_redis()

            # Get current role
            user = await self.get_user(user_id)
            if not user:
                raise ValueError(f"User {user_id} not found")

            old_role = user.role

            # Update role
            await redis_client.hset(f"user:{user_id}", "role", new_role.value)

            # Update role indexes
            await redis_client.srem(f"role:{old_role.value}", user_id)
            await redis_client.sadd(f"role:{new_role.value}", user_id)

            logger.info(f"User role updated: {user_id} ({old_role.value} -> {new_role.value})")

        except Exception as e:
            logger.error(f"Failed to update user role: {e}", exc_info=True)
            raise

    async def list_users(self, role: Optional[Role] = None) -> List[User]:
        """
        List all users, optionally filtered by role.

        Args:
            role: Optional role filter

        Returns:
            List of users
        """
        try:
            redis_client = await self._get_redis()

            if role:
                user_ids = await redis_client.smembers(f"role:{role.value}")
            else:
                # Get all users across all roles
                user_ids = set()
                for r in Role:
                    ids = await redis_client.smembers(f"role:{r.value}")
                    user_ids.update(ids)

            users = []
            for user_id_bytes in user_ids:
                user_id = user_id_bytes.decode()
                user = await self.get_user(user_id)
                if user:
                    users.append(user)

            return users

        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []

    # ============================================================================
    # API Key Management
    # ============================================================================

    async def generate_api_key(self, user_id: str) -> str:
        """
        Generate API key for user.

        Args:
            user_id: User ID

        Returns:
            Generated API key

        Example:
            api_key = await rbac.generate_api_key("user_123")
            # Returns: "sk_live_abc123..."
        """
        try:
            redis_client = await self._get_redis()

            # Check user exists
            user = await self.get_user(user_id)
            if not user:
                raise ValueError(f"User {user_id} not found")

            # Generate secure API key
            api_key = f"sk_live_{secrets.token_urlsafe(32)}"

            # Hash the key for storage
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Store hash
            await redis_client.hset(f"user:{user_id}", "api_key_hash", key_hash)
            await redis_client.set(f"apikey:{key_hash}", user_id)

            logger.info(f"API key generated for user: {user_id}")

            return api_key

        except Exception as e:
            logger.error(f"Failed to generate API key: {e}", exc_info=True)
            raise

    async def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """
        Authenticate user by API key.

        Args:
            api_key: API key

        Returns:
            User object if valid, None otherwise
        """
        try:
            redis_client = await self._get_redis()

            # Hash the provided key
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Look up user
            user_id_bytes = await redis_client.get(f"apikey:{key_hash}")
            if not user_id_bytes:
                return None

            user_id = user_id_bytes.decode()

            # Update last active
            await redis_client.hset(
                f"user:{user_id}",
                "last_active",
                datetime.now().isoformat()
            )

            return await self.get_user(user_id)

        except Exception as e:
            logger.error(f"API key authentication error: {e}")
            return None

    # ============================================================================
    # Permission Checking
    # ============================================================================

    async def check_permission(
        self,
        user_id: str,
        permission: Permission
    ) -> bool:
        """
        Check if user has permission.

        Args:
            user_id: User ID
            permission: Permission to check

        Returns:
            True if allowed, False otherwise

        Example:
            allowed = await rbac.check_permission(
                user_id="user_123",
                permission=Permission.QUERY_AGENT
            )
        """
        try:
            user = await self.get_user(user_id)
            if not user:
                return False

            return permission in ROLE_PERMISSIONS.get(user.role, set())

        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return False

    async def check_resource_access(
        self,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """
        Check if user can perform action on resource.

        Args:
            user_id: User ID
            resource: Resource name (e.g., "costs", "cache", "agents")
            action: Action (e.g., "read", "write", "delete")

        Returns:
            True if allowed, False otherwise
        """
        # Map resource/action to permission
        permission_map = {
            ("costs", "read"): Permission.VIEW_COSTS,
            ("costs", "write"): Permission.MANAGE_BUDGETS,
            ("cache", "read"): Permission.VIEW_CACHE,
            ("cache", "write"): Permission.CLEAR_CACHE,
            ("agents", "query"): Permission.QUERY_AGENT,
            ("users", "read"): Permission.VIEW_USERS,
            ("users", "write"): Permission.MANAGE_USERS,
            ("system", "read"): Permission.VIEW_METRICS,
            ("system", "write"): Permission.MANAGE_SYSTEM,
        }

        permission = permission_map.get((resource, action))
        if not permission:
            logger.warning(f"Unknown resource/action: {resource}/{action}")
            return False

        return await self.check_permission(user_id, permission)

    # ============================================================================
    # Audit Logging
    # ============================================================================

    async def log_audit(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str = "success",
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Log audit event.

        Args:
            user_id: User who performed action
            action: Action performed
            resource: Resource affected
            result: Result (success, failure, denied)
            details: Additional details
            ip_address: Client IP address
            session_id: Session ID

        Example:
            await rbac.log_audit(
                user_id="user_123",
                action="query",
                resource="math_agent",
                result="success",
                details={"query": "2+2", "response": "4"}
            )
        """
        try:
            redis_client = await self._get_redis()

            event = AuditEvent(
                timestamp=datetime.now(),
                user_id=user_id,
                action=action,
                resource=resource,
                result=result,
                details=details,
                ip_address=ip_address,
                session_id=session_id
            )

            event_json = json.dumps({
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "action": event.action,
                "resource": event.resource,
                "result": event.result,
                "details": event.details,
                "ip_address": event.ip_address,
                "session_id": event.session_id
            })

            # Store in sorted set for time-based queries
            timestamp_score = event.timestamp.timestamp()

            await redis_client.zadd(
                "audit:all",
                {event_json: timestamp_score}
            )

            await redis_client.zadd(
                f"audit:user:{user_id}",
                {event_json: timestamp_score}
            )

            # Track failed/denied attempts
            if result in ["failure", "denied"]:
                await redis_client.zadd(
                    f"audit:failures",
                    {event_json: timestamp_score}
                )

            logger.debug(f"Audit logged: {user_id} {action} {resource} -> {result}")

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}", exc_info=True)

    async def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Get audit logs.

        Args:
            user_id: Filter by user
            start_time: Start time
            end_time: End time
            limit: Maximum results

        Returns:
            List of audit events
        """
        try:
            redis_client = await self._get_redis()

            # Determine time range
            min_score = start_time.timestamp() if start_time else "-inf"
            max_score = end_time.timestamp() if end_time else "+inf"

            # Get logs
            if user_id:
                key = f"audit:user:{user_id}"
            else:
                key = "audit:all"

            records = await redis_client.zrevrangebyscore(
                key,
                max_score,
                min_score,
                start=0,
                num=limit
            )

            # Parse events
            events = []
            for record_bytes in records:
                data = json.loads(record_bytes)
                events.append(AuditEvent(
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    user_id=data["user_id"],
                    action=data["action"],
                    resource=data["resource"],
                    result=data["result"],
                    details=data.get("details"),
                    ip_address=data.get("ip_address"),
                    session_id=data.get("session_id")
                ))

            return events

        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []

    async def get_security_stats(self) -> Dict:
        """
        Get security statistics.

        Returns:
            Security metrics and stats
        """
        try:
            redis_client = await self._get_redis()

            # Count users by role
            role_counts = {}
            for role in Role:
                count = await redis_client.scard(f"role:{role.value}")
                role_counts[role.value] = count

            # Count total audit events
            total_events = await redis_client.zcard("audit:all")
            total_failures = await redis_client.zcard("audit:failures")

            # Get recent failures (last 24h)
            yesterday = datetime.now() - timedelta(days=1)
            recent_failures = await redis_client.zcount(
                "audit:failures",
                yesterday.timestamp(),
                "+inf"
            )

            return {
                "users": {
                    "by_role": role_counts,
                    "total": sum(role_counts.values())
                },
                "audit": {
                    "total_events": total_events,
                    "total_failures": total_failures,
                    "failures_24h": recent_failures
                }
            }

        except Exception as e:
            logger.error(f"Failed to get security stats: {e}")
            return {}


# ============================================================================
# Singleton Instance
# ============================================================================

_rbac_service: Optional[RBACService] = None


def get_rbac_service() -> RBACService:
    """Get or create global RBAC service instance"""
    global _rbac_service
    if _rbac_service is None:
        _rbac_service = RBACService()
    return _rbac_service
