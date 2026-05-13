FROM python:3.11-slim

WORKDIR /app

# Install system deps for sentence-transformers and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (layer caching)
COPY task_a/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (NOT data — mounted as volume)
COPY shared/ ./shared/
COPY task_a/  ./task_a/

ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

EXPOSE 8001
CMD ["uvicorn", "task_a.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]