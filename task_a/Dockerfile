FROM python:3.11-slim

WORKDIR /app

COPY task_a/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY task_a/ ./task_a/

ENV PYTHONPATH=/app

EXPOSE 8001
CMD ["uvicorn", "task_a.main:app", "--host", "0.0.0.0", "--port", "8001"]
