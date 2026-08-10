# Sift

Sift is a log-ingestion and retrieval-augmented question-answering pipeline.

> **Status:** This project is actively under development and is not complete.
> The current ingestion, chunking, embedding, retrieval, and QA flow is working,
> but production hardening, evaluation, live incremental embedding, and advanced
> retrieval improvements are still pending.

```text
log generator → log collector → Redis stream → log consumer → Postgres
                                                               ↓
                                                    chunking + embeddings
                                                               ↓
                                                        log QA API → Groq
```

## Current design

- `log_generator` creates deterministic noise and incident scenarios.
- `log_collector` validates incoming records and writes them to Redis.
- `log_consumer` reads Redis and stores records in Postgres `raw_logs`.
- `log_embedder` is a one-shot job that rebuilds `chunks` using BGE embeddings.
- Records are partitioned by `trace_id` before windowing, so incident chunks do
  not mix unrelated traces. Records without a trace ID remain noise.
- `log_api` performs dense pgvector retrieval, expands the top result to all
  chunks sharing its trace ID, and sends the chronological context to Groq.
- Groq is the default LLM provider. Ollama remains available through the
  optional `local-llm` Compose profile.

## Chunking, embeddings, and retrieval

- Raw logs are first partitioned by `trace_id`. Each non-null trace is one
  partition; all null trace IDs form the noise partition.
- Each partition is grouped into fixed 60-second UTC windows. Empty windows are
  skipped. Windows over 25 records are split into sequential sub-chunks using
  `(timestamp, id)` order.
- Chunk content uses one newline-separated line per record:
  `[HH:MM:SS] LEVEL service: message`. Each chunk stores its services, trace
  IDs, source raw-log IDs, and time bounds in Postgres.
- `log_embedder` uses `BAAI/bge-small-en-v1.5`, producing normalized 384-dimensional
  embeddings in batches. It currently truncates and rebuilds the derived
  `chunks` table each time it runs.
- Retrieval embeds the question with the same model and normalization, performs
  pgvector cosine-distance top-k search, and then expands the top result to all
  chunks sharing its trace ID. Expanded chunks are ordered chronologically.
  If the top result is noise, the original top-k results are used. Close
  similarity ties between multiple plausible incidents are a known limitation.

## Setup

Create the local configuration:

```bash
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`. Never commit `.env`.

Start ingestion:

```bash
docker compose up -d redis postgres log-collector log-consumer
```

Generate logs. Run the command once per scenario when creating a larger corpus:

```bash
log_generator/.venv/bin/python log_generator/send_to_collector.py \
  --collector-url http://localhost:8000 \
  --scenario payment-timeout-v1 \
  --seed 101 \
  --noise-count 250 \
  --batch-size 50 \
  --output data/logs/payment-timeout-v1.jsonl \
  --ground-truth data/ground_truth/payment-timeout-v1.json
```

After the consumer has stored the records, rebuild chunks:

```bash
docker compose run --rm --build log-embedder
```

Start the QA API:

```bash
docker compose up -d --build log-api
```

Check health and ask a question:

```bash
curl http://localhost:8001/health
```

```bash
curl -X POST http://localhost:8001/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What caused the payment failures?"}'
```

The response contains the answer and the retrieved chunks used as context.

## Optional local Ollama

To use local inference instead of Groq, set these values in `.env`:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:1b
```

Then run:

```bash
docker compose --profile local-llm up -d ollama
docker compose --profile local-llm run --rm ollama-model-init
docker compose --profile local-llm up -d log-api
```
