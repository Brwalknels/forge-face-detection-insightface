FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libheif-dev \
    libgl1 \
    libglib2.0-0 \
    execstack \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
# InsightFace models will be downloaded on first run
RUN pip install --no-cache-dir -r requirements.txt

# Clear executable stack flag from ONNX Runtime shared library
# This fixes the "cannot enable executable stack" error on systems with strict security
RUN execstack -c /usr/local/lib/python3.11/site-packages/onnxruntime/capi/*.so || true

# Copy application code
COPY app/ ./app/

# Create directory for InsightFace models (will be cached)
RUN mkdir -p /root/.insightface

# Create user for security (non-root)
RUN useradd -m -u 1000 facedetect && \
    chown -R facedetect:facedetect /app && \
    chown -R facedetect:facedetect /root/.insightface

# Switch to non-root user
USER facedetect

# Expose port (5001 to differentiate from dlib service on 5000)
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/health').raise_for_status()" || exit 1

# Fix ONNX Runtime executable stack issue on systems with strict security policies
ENV ONNXRUNTIME_ALLOW_LOADING_LIB_WITH_EXEC_STACK=1

# Run with gunicorn for production
# --workers: Number of worker processes (1 for now due to model memory)
# --timeout: Worker timeout in seconds (InsightFace can take time)
# --bind: Host and port to bind to
# --log-level: Logging level
CMD ["gunicorn", \
     "--workers", "1", \
     "--timeout", "180", \
     "--bind", "0.0.0.0:5001", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app.main:app"]
