FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (layer caching)
COPY task_b/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source code (NOT data ” mounted as volume)
COPY shared/ ./shared/
COPY task_b/  ./task_b/

ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

EXPOSE 8002
CMD ["uvicorn", "task_b.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]