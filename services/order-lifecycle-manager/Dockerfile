FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY ./services/order-lifecycle-manager/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared infrastructure
COPY ./infrastructure/shared ./shared

# Copy application code
COPY ./services/order-lifecycle-manager/main.py .

# Create logs directory
RUN mkdir -p /app/logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8013

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8013/health || exit 1

# Run the application
CMD ["python", "main.py"]
