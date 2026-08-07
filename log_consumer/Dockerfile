FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY db ./db
COPY sift_common ./sift_common
COPY log_generator ./log_generator
COPY log_consumer ./log_consumer

RUN pip install --no-cache-dir ".[log_consumer]"

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "log_consumer.consumer"]
