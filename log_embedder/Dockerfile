FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY db ./db
COPY sift_common ./sift_common
COPY log_embedder ./log_embedder

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
RUN pip install --no-cache-dir ".[log_embedder]"

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "log_embedder.worker"]
