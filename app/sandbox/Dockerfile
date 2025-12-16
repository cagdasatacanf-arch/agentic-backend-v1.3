FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install fastapi uvicorn

# Copy code
COPY main.py .

# Run as non-root user for extra safety
RUN useradd -m sandboxuser
USER sandboxuser

# Expose port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
