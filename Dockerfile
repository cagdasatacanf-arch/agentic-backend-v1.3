# syntax=docker/dockerfile:1

# Use a slim Python base image for both builder and final stages
FROM python:3.11-slim AS base

# Builder stage: install dependencies into a virtual environment
FROM base AS builder
WORKDIR /app

# Install system dependencies required for pip packages (if any)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file(s) first for better caching
COPY --link requirements.txt ./

# Create virtual environment and install dependencies
RUN python -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r requirements.txt

# Copy the rest of the application code
COPY --link app ./app

# Final stage: minimal image with only runtime dependencies
FROM base AS final
WORKDIR /app

# Create a non-root user and switch to it
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser
USER appuser

# Copy the virtual environment from builder
COPY --from=builder /app/.venv .venv
# Copy application code
COPY --from=builder /app/app ./app

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port your app runs on (update if different)
EXPOSE 8000

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default command to run the app (update if different)
CMD ["python", "-m", "app.main"]
