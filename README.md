# Sift

Sift is a log-ingestion and RAG system for investigating application incidents from structured logs.

> **Status:** actively under development. The batch pipeline works end to end; live incremental embedding and additional production hardening remain future work.

## Architecture

```text
log_generator → log_collector → Redis → log_consumer → Postgres
                                                        │
                                           log_embedder → chunks
                                                        │
                                           log_api → Groq/Ollama
```

- `log_generator` creates seeded noise and incident scenarios.
- `log_collector` validates records and writes them to Redis.
- `log_consumer` stores records in Postgres `raw_logs`.
- `log_embedder` creates and embeds derived chunks.
- `log_api` performs retrieval and answers questions through `POST /ask`.
- `evaluation` runs RAGAS and behavior checks against the QA API.

## Design

Records are partitioned by `trace_id`; records without a trace ID form the noise partition. Each partition is grouped into fixed 60-second UTC windows and capped at 25 records. Oversized windows are split chronologically. Chunk content uses one newline-separated log per record:

```text
[HH:MM:SS] LEVEL service: message
```

Chunks use normalized `BAAI/bge-small-en-v1.5` embeddings with 384 dimensions.

The API supports:

- Dense pgvector cosine retrieval.
- PostgreSQL full-text retrieval using `tsvector`.
- Weighted reciprocal rank fusion. Defaults: `RRF_K=60`, dense weight `1.0`, lexical weight `1.0`.
- Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Chronological expansion of the top result's trace when it has a trace ID.

Close-similarity ties between unrelated incidents are not fully resolved yet.

## Setup

Create local configuration and install dependencies:

```bash
cp .env.example .env
# Set GROQ_API_KEY and EVAL_API_KEY in .env

python3 -m venv log_generator/.venv
log_generator/.venv/bin/pip install -e '.[dev,log_generator,log_collector,log_consumer,log_embedder,log_api,evaluation]'
```

Start ingestion services:

```bash
docker compose up -d redis postgres log-collector log-consumer
```

Generate and send logs:

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

Build chunks and start the API:

```bash
docker compose run --rm --build log-embedder
docker compose up -d --build log-api
```

Query the API:

```bash
curl -X POST http://localhost:8001/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What caused the payment failures?"}'
```

## Evaluation

The harness evaluates 20 questions covering root cause, ambiguity, unrelated services, out-of-scope questions, and weak similarity. It records the answer, retrieved chunks, latency, behavior classification, and these RAGAS metrics:

- Faithfulness
- Answer relevancy
- Context precision
- Context recall

Context precision and recall are calculated for the 11 questions with reference answers. The other 9 values are intentionally `NULL`.

### Dense versus hybrid retrieval + reranking

| Metric | Stage 1: Dense | Stage 2: Hybrid + reranking | Change |
|---|---:|---:|---:|
| Faithfulness | 0.809 | 0.759 | -0.050 |
| Answer relevancy | 0.843 | 0.860 | +0.017 |
| Context precision | 0.705 | 0.788 | +0.083 |
| Context recall | 0.818 | 0.909 | +0.091 |
| Behavior match | 80.0% | 85.0% | +5.0 pp |
| Average latency | 839 ms | 2,352 ms | +1,513 ms |

Hybrid retrieval improved precision, recall, answer relevancy, and behavior matching because lexical search captures exact log terms while dense search captures semantic matches; RRF and cross-encoder reranking then prioritize stronger candidates. Faithfulness declined slightly because more relevant context can still contain multiple plausible incidents, and the answer model may combine evidence or overstate a causal chain. Latency increased as expected because hybrid retrieval adds a second database search, fusion, and cross-encoder inference.

Run the evaluation with the configured Cerebras evaluator:

```bash
RAGAS_MAX_TOKENS=4096 \
log_generator/.venv/bin/python -m evaluation.run_eval \
  --api-url http://localhost:8001 \
  --questions evaluation/questions.json \
  --stage stage_2_hybrid_cerebras_gpt_oss \
  --delay-seconds 120 \
  --metric-delay-seconds 120
```

The evaluator uses `EVAL_*` settings independently from the model used by `/ask`. The evaluation database table is `eval_runs`.

## Tests

```bash
log_generator/.venv/bin/python -m pytest \
  evaluation log_collector log_consumer log_embedder log_generator log_api -q
```

Known limitations are close-similarity ambiguity, rebuild-only embedding, and provider rate limits during long evaluations.
