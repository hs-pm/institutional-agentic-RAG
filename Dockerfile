FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install curl for healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    MAX_WORKERS=1 \
    LOG_LEVEL=info

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Create entrypoint script
COPY <<EOF /app/entrypoint.sh
#!/bin/sh
# Trap SIGTERM and forward it to the Python process
trap 'kill -TERM $PID' TERM INT

# Start the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000 --workers $MAX_WORKERS --log-level $LOG_LEVEL &
PID=$!

# Wait for the process to finish or be killed
wait $PID
EOF

RUN chmod +x /app/entrypoint.sh

# Run the entrypoint script
CMD ["/app/entrypoint.sh"] 