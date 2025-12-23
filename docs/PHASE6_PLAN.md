# Phase 6: Production & Enterprise Features

**Status:** 🚧 In Progress
**Timeline:** 2-3 weeks
**Focus:** Production readiness, scalability, and enterprise capabilities

## Overview

Phase 6 transforms the system from a feature-complete prototype into a production-ready, enterprise-grade platform with:
1. **Streaming Responses** - Real-time SSE streaming for better UX
2. **Cost Optimization** - Token tracking, caching, and budget controls
3. **Advanced Security** - Role-based access control (RBAC), API key management
4. **Production Monitoring** - Complete observability with Prometheus + Grafana
5. **Horizontal Scalability** - Load balancing and distributed execution

## Why Phase 6?

After implementing Phases 1-5, we have:
- ✅ Complete multi-agent system (Math, Code, RAG, Vision)
- ✅ Self-improvement pipeline with RL training
- ✅ Vision and multimodal capabilities
- ✅ Quality scoring and interaction logging

**What's still needed for production:**
- ❌ Real-time streaming responses
- ❌ Cost tracking and optimization
- ❌ Enterprise security (RBAC, audit logs)
- ❌ Production monitoring dashboard
- ❌ Horizontal scaling support
- ❌ Performance optimization

## Phase 6 Components

### 1. Streaming Responses (SSE)

**Goal:** Provide real-time streaming for better user experience.

**Problem:**
- Current: Users wait for complete response (5-30 seconds)
- Better: Stream tokens as they're generated (perceived latency < 1s)

**Implementation:**
```python
# app/services/streaming_service.py
from fastapi.responses import StreamingResponse
from typing import AsyncIterator

class StreamingAgent:
    """Stream agent responses in real-time"""

    async def stream_response(
        self,
        query: str,
        agent_type: str = "general"
    ) -> AsyncIterator[str]:
        """
        Stream response chunks as they're generated.

        Yields SSE events:
        - data: {"type": "token", "content": "..."}
        - data: {"type": "tool_call", "tool": "search", "status": "running"}
        - data: {"type": "done", "total_tokens": 1234}
        """
        # Route to appropriate agent
        agent = self._get_agent(agent_type)

        # Stream tokens
        async for chunk in agent.stream(query):
            yield f"data: {json.dumps(chunk)}\n\n"

        # Send completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

# API endpoint
@router.post("/stream")
async def stream_query(request: QueryRequest):
    """Stream response in real-time"""
    return StreamingResponse(
        streaming_service.stream_response(
            query=request.query,
            agent_type=request.agent_type
        ),
        media_type="text/event-stream"
    )
```

**Benefits:**
- ⚡ Perceived latency reduced by 80%
- 🎯 Better UX for long responses
- 📊 Real-time tool execution visibility
- 💰 Can cancel expensive operations early

---

### 2. Cost Optimization & Tracking

**Goal:** Track, optimize, and control API costs.

**Implementation:**
```python
# app/services/cost_tracker.py
class CostTracker:
    """Track and optimize API costs"""

    def __init__(self):
        self.rates = {
            "gpt-4o": {"input": 0.005, "output": 0.015},  # per 1K tokens
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o-vision": {"input": 0.01, "output": 0.03}
        }

    async def track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: Optional[str] = None
    ):
        """Track token usage and calculate cost"""
        cost = (
            (input_tokens / 1000) * self.rates[model]["input"] +
            (output_tokens / 1000) * self.rates[model]["output"]
        )

        # Store in Redis
        await self.redis.zincrby(
            f"costs:user:{user_id}",
            cost,
            datetime.now().isoformat()
        )

        return cost

    async def get_usage_stats(
        self,
        user_id: str,
        period: str = "today"
    ) -> Dict:
        """Get cost statistics"""
        return {
            "total_cost": ...,
            "total_tokens": ...,
            "requests_count": ...,
            "cost_by_model": {...},
            "cost_by_agent": {...}
        }

# Budget controls
class BudgetController:
    """Enforce budget limits"""

    async def check_budget(
        self,
        user_id: str,
        estimated_cost: float
    ) -> bool:
        """Check if user has budget available"""
        current_spend = await self.get_current_spend(user_id)
        budget_limit = await self.get_budget_limit(user_id)

        return (current_spend + estimated_cost) <= budget_limit

# Cost optimization strategies
class CostOptimizer:
    """Optimize costs automatically"""

    async def optimize_request(
        self,
        query: str,
        context: Dict
    ) -> Dict:
        """
        Choose optimal model and parameters.

        Strategies:
        1. Use gpt-4o-mini for simple queries
        2. Cache frequent queries
        3. Reduce context window when possible
        4. Use streaming to allow early cancellation
        """
        # Classify query complexity
        complexity = await self.classify_complexity(query)

        if complexity == "simple":
            return {"model": "gpt-4o-mini", "max_tokens": 500}
        elif complexity == "medium":
            return {"model": "gpt-4o", "max_tokens": 1000}
        else:
            return {"model": "gpt-4o", "max_tokens": 2000}
```

**API Endpoints:**
```python
GET  /api/v1/costs/stats          # Get cost statistics
GET  /api/v1/costs/usage/{user}   # Get user usage
POST /api/v1/costs/budget         # Set budget limit
GET  /api/v1/costs/forecast       # Cost forecast
```

**Expected Impact:**
- 💰 30-50% cost reduction through optimization
- 📊 Complete cost visibility
- 🎯 Budget controls prevent overages
- 📈 Usage analytics for optimization

---

### 3. Advanced Security (RBAC + Audit)

**Goal:** Enterprise-grade security with role-based access control.

**Implementation:**
```python
# app/services/auth_service.py
from enum import Enum

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_USER = "api_user"

class Permission(str, Enum):
    # Agent permissions
    USE_MATH_AGENT = "use_math_agent"
    USE_CODE_AGENT = "use_code_agent"
    USE_RAG_AGENT = "use_rag_agent"
    USE_VISION_AGENT = "use_vision_agent"

    # Admin permissions
    VIEW_METRICS = "view_metrics"
    MANAGE_USERS = "manage_users"
    EXPORT_DATA = "export_data"
    MANAGE_TRAINING = "manage_training"

    # Data permissions
    UPLOAD_DOCUMENTS = "upload_documents"
    DELETE_DOCUMENTS = "delete_documents"

class RBACService:
    """Role-Based Access Control"""

    def __init__(self):
        self.role_permissions = {
            Role.ADMIN: [p for p in Permission],  # All permissions
            Role.USER: [
                Permission.USE_MATH_AGENT,
                Permission.USE_CODE_AGENT,
                Permission.USE_RAG_AGENT,
                Permission.USE_VISION_AGENT,
                Permission.UPLOAD_DOCUMENTS
            ],
            Role.VIEWER: [
                Permission.USE_RAG_AGENT,
                Permission.VIEW_METRICS
            ],
            Role.API_USER: [
                Permission.USE_MATH_AGENT,
                Permission.USE_CODE_AGENT,
                Permission.USE_RAG_AGENT
            ]
        }

    async def check_permission(
        self,
        user_id: str,
        permission: Permission
    ) -> bool:
        """Check if user has permission"""
        user = await self.get_user(user_id)
        return permission in self.role_permissions[user.role]

    async def require_permission(
        self,
        user_id: str,
        permission: Permission
    ):
        """Require permission (raises exception if denied)"""
        if not await self.check_permission(user_id, permission):
            raise PermissionDenied(f"Missing permission: {permission}")

# Audit logging
class AuditLogger:
    """Log all security-relevant events"""

    async def log_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        status: str,
        metadata: Optional[Dict] = None
    ):
        """Log audit event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "status": status,
            "ip_address": ...,
            "user_agent": ...,
            "metadata": metadata
        }

        # Store in dedicated audit log
        await self.store_audit_event(event)

# API Key management
class APIKeyService:
    """Manage API keys"""

    async def create_key(
        self,
        user_id: str,
        name: str,
        permissions: List[Permission],
        expires_at: Optional[datetime] = None
    ) -> str:
        """Create new API key"""
        key = self.generate_key()
        await self.store_key(key, user_id, permissions, expires_at)
        return key

    async def validate_key(self, key: str) -> Optional[Dict]:
        """Validate API key and return metadata"""
        # Check if key exists and not expired
        # Return user_id, permissions, rate_limits
        pass
```

**Middleware:**
```python
# app/middleware/auth.py
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Authenticate and authorize requests"""
    # Extract API key or JWT token
    auth_header = request.headers.get("Authorization")

    if auth_header:
        user = await auth_service.authenticate(auth_header)
        request.state.user = user

    response = await call_next(request)
    return response

# Dependency for routes
async def require_permission(permission: Permission):
    """Dependency to require specific permission"""
    async def check(request: Request):
        user = request.state.user
        await rbac_service.require_permission(user.id, permission)
        return user
    return check

# Usage in routes
@router.post("/vision/analyze")
async def analyze_image(
    request: VisionRequest,
    user: User = Depends(require_permission(Permission.USE_VISION_AGENT))
):
    """Vision analysis (requires permission)"""
    await audit_logger.log_event(
        user_id=user.id,
        action="vision_analyze",
        resource="vision_agent",
        status="success"
    )
    # ... process request
```

**Expected Impact:**
- 🔒 Enterprise-grade security
- 👥 Multi-tenant support
- 📝 Complete audit trail
- 🎯 Fine-grained permissions
- 🔑 API key management

---

### 4. Production Monitoring (Prometheus + Grafana)

**Goal:** Complete observability for production systems.

**Implementation:**
```python
# app/services/metrics_exporter.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Agent metrics
agent_requests_total = Counter(
    'agent_requests_total',
    'Total agent requests',
    ['agent_type', 'status']
)

agent_quality_score = Histogram(
    'agent_quality_score',
    'Agent output quality score',
    ['agent_type'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

# Cost metrics
token_usage_total = Counter(
    'token_usage_total',
    'Total tokens used',
    ['model', 'type']  # type: input/output
)

api_cost_dollars = Counter(
    'api_cost_dollars_total',
    'Total API cost in dollars',
    ['model', 'agent_type']
)

# System metrics
active_sessions = Gauge(
    'active_sessions',
    'Number of active user sessions'
)

# Tool metrics
tool_calls_total = Counter(
    'tool_calls_total',
    'Total tool calls',
    ['tool_name', 'status']
)

tool_duration_seconds = Histogram(
    'tool_duration_seconds',
    'Tool execution duration',
    ['tool_name']
)
```

**Grafana Dashboards:**

**Dashboard 1: System Overview**
- Total requests (24h, 7d, 30d)
- P50/P95/P99 latency
- Error rate
- Active sessions
- Cost per day

**Dashboard 2: Agent Performance**
- Requests by agent type (pie chart)
- Quality scores over time (line chart)
- Agent success rates (bar chart)
- Latency by agent (heatmap)

**Dashboard 3: Cost Analysis**
- Token usage trends (area chart)
- Cost by model (stacked bar)
- Cost by agent type (pie chart)
- Daily/weekly/monthly costs (line chart)
- Cost per request (gauge)

**Dashboard 4: Tool Execution**
- Tool call frequency (bar chart)
- Tool success rates (table)
- Tool latency distribution (histogram)
- Tool errors (log panel)

**Dashboard 5: Quality Metrics**
- Average quality score (gauge)
- Quality distribution (histogram)
- Quality by agent type (multi-line chart)
- Low-quality alerts (alert list)

**docker-compose.yml:**
```yaml
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus

  # Add metrics endpoint
  api:
    environment:
      - PROMETHEUS_ENABLED=true
      - PROMETHEUS_PORT=8001

volumes:
  prometheus_data:
  grafana_data:
```

**Expected Impact:**
- 📊 Real-time system visibility
- 🔍 Quick incident detection
- 📈 Performance optimization insights
- 💰 Cost tracking and alerts
- 📉 Quality monitoring

---

### 5. Horizontal Scalability

**Goal:** Scale horizontally for high traffic.

**Implementation:**
```python
# app/services/load_balancer.py
class AgentLoadBalancer:
    """Distribute agent requests across workers"""

    async def route_request(
        self,
        query: str,
        agent_type: str
    ) -> Dict:
        """Route to least-loaded worker"""
        # Get available workers
        workers = await self.get_healthy_workers(agent_type)

        # Select least-loaded
        worker = min(workers, key=lambda w: w.current_load)

        # Execute on worker
        return await worker.execute(query)

# Redis-based coordination
class DistributedCoordinator:
    """Coordinate across multiple instances"""

    async def acquire_lock(
        self,
        resource: str,
        timeout: int = 30
    ) -> bool:
        """Distributed lock for resource"""
        # Use Redis SETNX
        pass

    async def get_next_task(self) -> Optional[Task]:
        """Get next task from distributed queue"""
        # Use Redis RPOPLPUSH
        pass

# Session affinity
class SessionManager:
    """Maintain session state across instances"""

    async def get_session(self, session_id: str) -> Session:
        """Get session from Redis"""
        pass

    async def update_session(
        self,
        session_id: str,
        data: Dict
    ):
        """Update session in Redis"""
        pass
```

**Load Balancer Config (nginx):**
```nginx
upstream api_backend {
    least_conn;
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}

server {
    listen 80;

    location / {
        proxy_pass http://api_backend;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Session affinity
        ip_hash;
    }
}
```

**Expected Impact:**
- 🚀 Handle 10x more traffic
- 📈 Horizontal scaling
- 🎯 High availability
- ⚡ Zero-downtime deployments

---

## Implementation Plan

### Week 1: Streaming & Cost
- **Days 1-2**: Streaming service implementation
- **Days 3-4**: Cost tracking and optimization
- **Day 5**: API endpoints and testing

### Week 2: Security & Monitoring
- **Days 1-2**: RBAC and auth service
- **Days 3-4**: Prometheus + Grafana setup
- **Day 5**: Dashboards and alerts

### Week 3: Scale & Polish
- **Days 1-2**: Load balancing and scaling
- **Days 3-4**: End-to-end testing
- **Day 5**: Documentation and deployment

## Success Criteria

**Streaming:**
- ✅ Token-by-token streaming works
- ✅ Perceived latency < 1 second
- ✅ Tool execution visible in real-time

**Cost:**
- ✅ All costs tracked and visible
- ✅ Budget controls working
- ✅ 30%+ cost reduction through optimization

**Security:**
- ✅ RBAC enforced on all endpoints
- ✅ Complete audit trail
- ✅ API key management working

**Monitoring:**
- ✅ All metrics in Grafana
- ✅ Alerts configured
- ✅ Cost tracking visible

**Scale:**
- ✅ Can handle 1000 req/sec
- ✅ Zero-downtime deployments
- ✅ Session persistence works

## Technologies

**New Dependencies:**
```txt
# Streaming
sse-starlette==1.6.5

# Monitoring
prometheus-client==0.19.0
grafana-client==3.5.0

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Scale
redis-py-cluster==2.1.3
```

**Infrastructure:**
- Nginx for load balancing
- Prometheus for metrics
- Grafana for visualization
- Redis for distributed coordination

## Next Steps

Ready to start Phase 6 implementation!

**Recommended order:**
1. **Streaming** - Immediate UX improvement
2. **Cost Tracking** - Financial visibility
3. **Monitoring** - Production observability
4. **Security** - Enterprise readiness
5. **Scaling** - Handle growth

Which component should I start with?
