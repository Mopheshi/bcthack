FROM python:3.11-slim

WORKDIR /app

COPY task_b/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY task_b/ ./task_b/

ENV PYTHONPATH=/app

EXPOSE 8002
CMD ["uvicorn", "task_b.main:app", "--host", "0.0.0.0", "--port", "8002"]
