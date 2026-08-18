# Sift

Sift is a log-ingestion and retrieval-augmented question-answering system for investigating application incidents from structured logs.

> **Status:** actively under development. The end-to-end batch pipeline is working: logs can be generated, validated, queued, stored, chunked, embedded, retrieved, and queried through the QA API. Live incremental embedding, production hardening, advanced retrieval, and durable evaluation checkpoints are still unfinished.

## Architecture

```text
log_generator
      │ POST batches
      ▼
log_collector ──► Redis stream ──► log_consumer ──► Postgres raw_logs
                                                        │
                                                        ▼
                                             log_embedder rebuilds chunks
                                                        │ writes
                                                        ▼
                                                   Postgres chunks
                                                        ▲ reads
                                                        │
                                         log_api dense retrieval + QA
                                                        │
                                                        ▼
                                              Groq or optional Ollama
```

## Components

- `log_generator` creates deterministic noise and scripted incident scenarios. Batch and live modes use the same event iterator. A seed makes generated output reproducible, and scenario ground truth is stored separately from the JSONL stream.
- `sift_common.schema` contains the shared `LogRecord` contract used to validate records across services.
- `log_collector` exposes `POST /ingest`, accepts one record or a JSON array of records, validates them, and writes them to the Redis stream.
- `log_consumer` reads the Redis stream through a consumer group and inserts records into `raw_logs`. The Redis entry ID is unique, so redelivery is safe. Successful records are acknowledged with `XACK`; stream entries remain available for replay and debugging.
- `log_embedder` reads `raw_logs`, partitions records by trace ID, creates time-windowed chunks, generates BGE embeddings, and rebuilds the derived `chunks` table.
- `log_api` performs dense pgvector retrieval, expands an incident result to its trace, and sends chronological context to the configured language model.
- `evaluation` contains the question set, RAGAS scoring, behavior classification, and the evaluation harness.

## Data and chunking design

The canonical log fields are `schema_version`, `timestamp`, `level`, `service`, `host`, `message`, `trace_id`, and `metadata`.

Chunking works as follows:

1. Records are partitioned by `trace_id`. Every non-null trace ID gets its own partition. All records with `NULL` trace IDs form one noise partition.
2. Each partition is grouped into fixed 60-second UTC windows. Empty windows are skipped.
3. A window is capped at 25 records. Larger windows are split into sequential sub-chunks ordered by `(timestamp, id)`.
4. Each chunk stores its time bounds, services, trace IDs, source `raw_logs` IDs, readable content, and a 384-dimensional vector.
5. Chunk content contains one newline-separated line per log:

   ```text
   [HH:MM:SS] LEVEL service: message
   ```

6. `log_embedder` uses `BAAI/bge-small-en-v1.5` with normalized embeddings and batch insertion. Because chunks are derived data, the worker truncates and rebuilds `chunks` on each run.

This partitioning prevents unrelated incidents with overlapping timestamps from being placed in the same chunk. Noise is still embedded so that the corpus resembles a real operational log stream.

## Retrieval and QA

The API embeds a question using the same BGE model and normalization settings as the embedder, then performs a pgvector cosine-distance top-k search.

- If the top-ranked chunk has a non-empty trace ID, the API retrieves all chunks for that trace and orders them chronologically.
- If the top-ranked chunk is noise, the original top-k results are used.
- The answer prompt includes chunk IDs, time windows, services, trace IDs, and log content.
- The model is instructed to use only the supplied context, distinguish facts from uncertainty, and avoid inventing events.

The current retrieval strategy is a dense-only baseline. It does not yet use hybrid search, reranking, query expansion, or a robust solution for close-similarity ties between multiple plausible incidents.

## Model configuration

Groq is the default provider. The current model is `openai/gpt-oss-20b`, which replaced the deprecated `llama-3.1-8b-instant`. The API, behavior classifier, and RAGAS judge use low reasoning effort to keep responses focused and within the available output budget.

Groq model references:

- [Model deprecation guidance](https://console.groq.com/docs/deprecations)
- [GPT-OSS 20B model](https://console.groq.com/docs/model/openai/gpt-oss-20b)
- [Groq rate limits](https://console.groq.com/docs/rate-limits)

Create local configuration from the example file:

```bash
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`. The file is ignored by Git and must never be committed.

Important settings include:

```env
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RAGAS_MAX_TOKENS=2048
```

Ollama remains available through the `local-llm` Compose profile for machines that can support local inference:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:1b
```

## Setup and batch pipeline

Create a virtual environment and install the project extras you need:

```bash
python3 -m venv log_generator/.venv
log_generator/.venv/bin/pip install -e '.[dev,log_generator,log_collector,log_consumer,log_embedder,log_api]'
```

Install evaluation dependencies separately when running the evaluation harness:

```bash
log_generator/.venv/bin/pip install -e '.[evaluation]'
```

Start the ingestion services:

```bash
docker compose up -d redis postgres log-collector log-consumer
```

Generate and send a corpus. Repeat with different scenarios and seeds to create a larger dataset:

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

The collector accepts batches, while the consumer continuously drains Redis into Postgres. Check the ingestion path:

```bash
curl http://localhost:8000/health
docker compose exec redis redis-cli XLEN sift-logs
docker compose exec postgres psql -U sift -d sift -c \
  "SELECT count(*) FROM raw_logs;"
```

Rebuild the derived chunks and embeddings:

```bash
docker compose run --rm --build log-embedder
```

Start and query the QA API:

```bash
docker compose up -d --build log-api
curl http://localhost:8001/health
```

```bash
curl -X POST http://localhost:8001/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What caused the payment failures?"}'
```

The response contains both the generated answer and the retrieved chunks used as context.

If Postgres is using an existing volume, the initialization script will not be replayed automatically. Apply schema changes explicitly:

```bash
docker compose exec -T postgres psql -U sift -d sift < db/schema.sql
```

## Evaluation harness

The reusable harness is in `evaluation/`. It evaluates the current QA API against 20 structured questions covering:

- 10 root-cause questions
- 3 ambiguity questions
- 1 unrelated-service question
- 1 cross-service attribution question
- 2 out-of-scope questions
- 3 weak-similarity questions

Questions tied to a scenario resolve their reference answer from the scenario's ground-truth object. Ground truth is not copied into the question file.

For each question, the harness:

1. Calls `POST /ask` and records the answer, retrieved chunk IDs, and latency.
2. Runs an independent behavior classifier with four allowed labels: `answer_with_evidence`, `abstain`, `flag_ambiguity`, and `report_no_correlated_incident`.
3. Compares the classified behavior with `expected_behavior` and stores `behavior_match`.
4. Runs RAGAS metrics independently from the behavior classifier:
   - `faithfulness`
   - `answer_relevancy`
   - `context_precision`
   - `context_recall`
5. Stores one evaluation row per question in `eval_runs`.

Faithfulness and answer relevancy are evaluated for all questions. Context precision and context recall are evaluated only for the 11 questions with reference answers. The other 9 questions still retrieve context; their context metrics are `NULL` because there is no ground-truth evidence set against which to calculate those metrics.

Run the complete evaluation with provider-rate-limit delays:

```bash
RAGAS_MAX_TOKENS=2048 \
log_generator/.venv/bin/python -m evaluation.run_eval \
  --api-url http://localhost:8001 \
  --questions evaluation/questions.json \
  --stage stage_0_ragas_complete \
  --delay-seconds 120 \
  --metric-delay-seconds 120 \
  2>&1 | tee /tmp/ragas-complete.log
```

The harness stores rows after the question loop completes. An interrupted run can therefore lose its in-memory rows; durable per-question checkpointing is a future improvement.

Verify completion and missing values:

```bash
grep -n -E \
  "stored 20 evaluation rows|Traceback|RAGAS metric .* failed|429|HTTPStatusError|ERROR" \
  /tmp/ragas-complete.log
```

```bash
docker compose exec postgres psql -U sift -d sift -c "
WITH latest AS (
  SELECT run_id
  FROM eval_runs
  WHERE stage_name = 'stage_0_ragas_complete'
  GROUP BY run_id
  ORDER BY max(run_timestamp) DESC
  LIMIT 1
)
SELECT
  count(*) AS total_questions,
  count(*) FILTER (WHERE faithfulness IS NULL) AS missing_faithfulness,
  count(*) FILTER (WHERE answer_relevancy IS NULL) AS missing_answer_relevancy,
  count(*) FILTER (WHERE context_precision IS NULL) AS missing_context_precision,
  count(*) FILTER (WHERE context_recall IS NULL) AS missing_context_recall,
  round(avg(behavior_match::int)::numeric, 3) AS behavior_match_rate
FROM eval_runs
WHERE run_id = (SELECT run_id FROM latest);
"
```

For a diagnostic run containing only two questions:

```bash
RAGAS_MAX_TOKENS=2048 \
log_generator/.venv/bin/python -m evaluation.run_eval \
  --api-url http://localhost:8001 \
  --questions evaluation/diagnostic_questions.json \
  --stage diagnostic_2048 \
  --delay-seconds 90 \
  --metric-delay-seconds 90
```

## Latest evaluation findings

The latest complete 20-question run was stored under:

```text
stage_0_ragas_complete
run_id: 5c5a4e52-44f0-45e5-897f-6da3870ca2c4
```

> **Baseline caveat:** This recorded run was produced before the migration to
> GPT-OSS 20B. Treat these values as the previous-model baseline. Run the
> evaluation again before comparing results for the current model.

Results:

| Metric | Score | Coverage |
|---|---:|---:|
| Faithfulness | 0.794 | 20 questions |
| Answer relevancy | 0.720 | 20 questions |
| Context precision | 0.992 | 11 reference-backed questions |
| Context recall | 0.955 | 11 reference-backed questions |
| Behavior match rate | 0.850 | 17 of 20 questions |

No RAGAS failures, rate-limit errors, or missing faithfulness/answer-relevancy values occurred in that run. The nine `NULL` context precision/recall values were intentional for questions without references.

The results show that retrieval is strong, while answer generation still needs work. The behavior mismatches were:

- `midnight-downstream-ambiguity`: the answer selected one plausible incident instead of clearly flagging ambiguity.
- `customer-churn-weak-similarity`: the answer responded to a question without sufficient supporting evidence instead of abstaining.
- `checkout-customer-impact-weak-similarity`: the answer generalized from one affected order to a broader customer segment.

These are baseline findings for improving ambiguity detection, abstention, and unsupported-inference control. RAGAS values are model-based evaluation signals, not absolute truth.

## Testing

Run the non-network unit tests:

```bash
log_generator/.venv/bin/python -m pytest \
  evaluation log_collector log_consumer log_embedder log_generator -q
```

The current service tests use injected fakes where possible. The live pipeline still requires Redis, Postgres, the embedding model cache, and a configured LLM provider.

## Known limitations

- The default retrieval path is dense-only top-k search.
- Close-similarity ties between unrelated incidents are not resolved robustly.
- The embedder is a rebuild job, not a continuously running incremental worker.
- The evaluation harness currently batches database writes until the run completes.
- Groq quotas can limit long evaluation runs; use the delay options or a different provider.
- Ollama provides a local alternative but may exceed the memory available on smaller machines.
