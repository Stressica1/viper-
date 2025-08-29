# Enhanced Market Scanner Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements/enhanced_scanner.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/enhanced_market_scanner.py .
COPY scripts/github_mcp_task_tracker.py .
COPY scripts/validate_github_mcp.py .
COPY config/system_requirements.json config/

# Create logs directory
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "enhanced_market_scanner.py"]
