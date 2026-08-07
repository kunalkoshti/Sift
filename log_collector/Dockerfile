FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY sift_common ./sift_common
COPY log_generator ./log_generator
COPY log_collector ./log_collector

RUN pip install --no-cache-dir ".[log_collector]"

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "log_collector.app:app", "--host", "0.0.0.0", "--port", "8000"]
