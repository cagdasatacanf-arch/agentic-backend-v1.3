import logging
import os
from logging.config import dictConfig
from typing import Dict

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import settings



# --- OpenTelemetry Setup ---
resource = Resource(attributes={
    "service.name": "agentic-backend",
    "service.version": "1.0.0"
})

trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Export to Jaeger (via OTLP gRPC endpoint)
otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

from app.services.logging import setup_logging, correlation_id, get_logger
logger = get_logger()

# Initialize structured logging
USE_JSON_LOGS = os.getenv("LOG_FORMAT", "text").lower() == "json"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
setup_logging(json_logs=USE_JSON_LOGS, level=LOG_LEVEL)

app = FastAPI(title="Agentic AI Prototype", openapi_url="/api/v1/openapi.json")

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """
    Middleware that extracts/generates a correlation ID and attaches it to the logging context.
    """
    import uuid
    header_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    token = correlation_id.set(header_id)
    
    # Bind to request logger
    req_logger = get_logger()
    req_logger.info(f"START {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = header_id
        req_logger.info(f"DONE  {request.method} {request.url.path} - {response.status_code}")
        return response
    finally:
        correlation_id.reset(token)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = settings.internal_api_key or ""
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Dict | None = None
    correlation_id: str


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Dict | None = None,
):
    import uuid
    # Use structured logger if available
    try:
        from app.services.logging import get_logger
        log = get_logger()
    except ImportError:
        log = logger

    cid = correlation_id.get() or str(uuid.uuid4())
    log.error(f"{code} {message} details={details} cid={cid}")
    raise HTTPException(
        status_code=status_code,
        detail=ErrorBody(
            code=code, message=message, details=details, correlation_id=cid
        ).model_dump(),
    )


async def api_key_auth(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        error_response(
            status.HTTP_401_UNAUTHORIZED,
            "UNAUTHENTICATED",
            "Authentication is required for this endpoint.",
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request body validation failed.",
                "details": exc.errors(),
                "correlation_id": "auto",
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "BAD_REQUEST",
                "message": str(exc.detail),
                "details": {},
                "correlation_id": "auto",
            }
        },
    )


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/v1/metadata")
def metadata():
    return {
        "models": {
            "chat": settings.openai_chat_model,
            "embedding": settings.openai_embedding_model,
        },
        "vector_store": "qdrant",
        "max_context_tokens": 16000,
    }


# Hide docs on prod if desired
if ENVIRONMENT == "prod":
    app.docs_url = None
    app.redoc_url = None
    app.openapi_url = None


# Routers
from app.api.routes_query import router as query_router  # noqa: E402
from app.api.routes_docs import router as docs_router  # noqa: E402
from app.api.routes_query_langgraph import router as langgraph_router  # noqa: E402
from app.api.routes_metrics import router as metrics_router  # noqa: E402
from app.api.routes_training import router as training_router  # noqa: E402
from app.api.routes_vision import router as vision_router  # noqa: E402
from app.api.routes_streaming import router as streaming_router  # noqa: E402
from app.api.routes_cache import router as cache_router  # noqa: E402
from app.api.routes_cost import router as cost_router  # noqa: E402
from app.api.routes_rbac import router as rbac_router  # noqa: E402
from app.api.routes_monitoring import router as monitoring_router  # noqa: E402
from app.api.routes_ab_testing import router as ab_testing_router  # noqa: E402

# Base API v1 prefix for standard routes
app.include_router(query_router, prefix="/api/v1")
app.include_router(docs_router, prefix="/api/v1")

# Specialized routes with unique prefixes
app.include_router(langgraph_router, prefix="/api/v1/langgraph")
app.include_router(metrics_router, prefix="/api/v1/metrics")
app.include_router(training_router, prefix="/api/v1/training")
app.include_router(vision_router, prefix="/api/v1/vision")
app.include_router(streaming_router, prefix="/api/v1/streaming")
app.include_router(cache_router, prefix="/api/v1/cache")
app.include_router(cost_router, prefix="/api/v1/cost")
app.include_router(rbac_router, prefix="/api/v1/rbac")
app.include_router(monitoring_router, prefix="/api/v1/monitoring")
app.include_router(ab_testing_router, prefix="/api/v1/ab-testing")
