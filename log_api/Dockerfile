FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY sift_common ./sift_common
COPY log_api ./log_api

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
RUN pip install --no-cache-dir ".[log_api]"

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "log_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
