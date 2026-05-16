# PersonaRAG — single consolidated service (Task A + Task B)
# Build context is the repo root.
FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / chromadb native bits
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first — layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ONNX model at build time. _download_model_if_not_exists() only
# fires inside __call__(), so we must actually embed a dummy string here — just
# constructing ONNXMiniLM_L6_V2() does nothing. The model lands at
# /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/ and is reused at runtime.
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()(['warmup'])"

# Application code
COPY shared/ ./shared/
COPY task_a/ ./task_a/
COPY task_b/ ./task_b/
COPY app/    ./app/

# Bake only runtime-needed processed data into the image.
# reviews.parquet (2.6 GB) and users.parquet are build-only — never needed at runtime.
COPY data/processed/persona_store.parquet ./data/processed/persona_store.parquet
COPY data/processed/ui_metadata.json      ./data/processed/ui_metadata.json
COPY data/processed/chroma/               ./data/processed/chroma/

ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO
# Cloud Run injects PORT; default to 8080 for local runs.
ENV PORT=8080

EXPOSE 8080

# NOTE ON DATA:
#   For LOCAL docker-compose runs, data/ is bind-mounted (see docker-compose.yml)
#   so the image stays small.
#   For CLOUD RUN, either:
#     (a) bake processed data in with `COPY data/processed ./data/processed`, or
#     (b) pull it from a GCS bucket at startup.
#   Keep the image lean by NOT copying raw Yelp JSON ever.

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
