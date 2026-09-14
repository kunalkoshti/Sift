# Sift

Sift is a structured-log ingestion and RAG system for investigating application incidents.

> **Status:** actively under development. The batch pipeline works end to end. Live incremental embedding and production hardening are not implemented yet.

## Architecture

~~~text
log_generator -> log_collector -> Redis -> log_consumer -> PostgreSQL.raw_logs
                                                               |
                                                log_embedder -> PostgreSQL.chunks
                                                               |
                                                   log_api -> LLM answer

evaluation -> log_api /ask -> RAGAS + behavior checks -> PostgreSQL.eval_runs
~~~

log_api reads the chunks table directly; it does not receive records from log_embedder at runtime.

Services:

- log_generator creates seeded background noise and scripted incidents.
- log_collector validates single records or batches and writes them to Redis.
- log_consumer persists deduplicated records in raw_logs.
- log_embedder builds time-windowed chunks and embeddings.
- log_api retrieves evidence and answers questions through POST /ask.
- evaluation measures answer, retrieval, and behavior quality.

## Core design

Logs follow one contract:

~~~text
schema_version, timestamp, level, service, host, message, trace_id, metadata
~~~

Records are partitioned by trace ID before chunking. Each non-null trace ID forms an incident partition; null trace IDs form the background-noise partition. Each partition is grouped into fixed 60-second UTC windows and capped at 25 records. Oversized windows are split chronologically using sub_index.

Chunk content contains one newline-separated record per line:

~~~text
[HH:MM:SS] LEVEL service: message
~~~

Newlines preserve event boundaries. Trace IDs and raw log IDs remain in database columns for exact traceability rather than being included in embedded prose.

Embeddings use normalized BAAI/bge-small-en-v1.5 vectors with 384 dimensions. Chunks are stored in PostgreSQL with pgvector and a generated PostgreSQL full-text index.

## Retrieval

The default mode is hybrid:

1. Parse supported service and time expressions.
2. Apply service/time filters to the initial SQL search.
3. Run dense pgvector search and PostgreSQL full-text search.
4. Fuse results with weighted reciprocal rank fusion.
5. Rerank candidates with cross-encoder/ms-marco-MiniLM-L-6-v2.
6. Discover trace IDs from the larger candidate pool.
7. Expand selected traces without the initial service/time boundaries.
8. Sort context chronologically and send it to the LLM.

Default retrieval settings:

~~~text
RETRIEVAL_TOP_K=5
TRACE_CANDIDATE_K=40
RRF_K=60
DENSE_RRF_WEIGHT=1.0
LEXICAL_RRF_WEIGHT=1.0
MAX_TRACE_CHUNKS=50
~~~

The RRF weights are score contributions, not percentages.

Severity is still stored and shown in context, but it is not used as a hard retrieval filter. Hard severity filtering previously removed causal precursor events.

Unknown services, unsupported time expressions, and empty retrievals are handled deterministically. If multiple incident traces have similarly strong scores, the API expands them separately and adds an ambiguity note.

## Setup

Create local configuration:

~~~bash
cp .env.example .env
# Set GROQ_API_KEY and EVAL_API_KEY in .env as required.
~~~

Install the project and development dependencies:

~~~bash
python3 -m venv log_generator/.venv
log_generator/.venv/bin/pip install -e '.[dev,log_generator,log_collector,log_consumer,log_embedder,log_api,evaluation]'
~~~

Start the ingestion services:

~~~bash
docker compose up -d redis postgres log-collector log-consumer
~~~

Generate and send an incident corpus:

~~~bash
log_generator/.venv/bin/python log_generator/send_to_collector.py \
  --collector-url http://localhost:8000 \
  --scenario payment-timeout-v1 \
  --seed 101 \
  --noise-count 250 \
  --batch-size 50 \
  --output data/logs/payment-timeout-v1.jsonl \
  --ground-truth data/ground_truth/payment-timeout-v1.json
~~~

Rebuild chunks and start the API:

~~~bash
docker compose run --rm --build log-embedder
docker compose up -d --build log-api
~~~

Check health and ask a question:

~~~bash
curl -s http://localhost:8001/health | jq

curl -s -X POST http://localhost:8001/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What caused the payment failures?"}' | jq
~~~

Ollama is optional and runs only with the local-llm Compose profile. Remote providers can be used to avoid loading a local generation model into device memory.

## Evaluation

The current filter-retrieval evaluation uses the frozen 32-question file:

~~~text
evaluation/questions_stage4.json
~~~

Questions cover root cause, alternate phrasing, ambiguity, unrelated services, out-of-scope requests, weak similarity, service/time filtering, multi-service retrieval, unknown services, unsupported time expressions, and no-result behavior.

The harness stores one row per question in eval_runs and records:

- generated answer;
- retrieved chunk IDs;
- latency;
- faithfulness;
- answer relevancy;
- context precision;
- context recall;
- classified behavior;
- behavior match;
- classifier failure state.

Context precision and recall are skipped when there is no reference answer or no retrieved context. The last-week-no-results fallback question is excluded from aggregate context precision/recall reporting because its result depends on the fallback retrieval note rather than ordinary in-range retrieval.

Run the current evaluation with the provider configured in .env:

~~~bash
RAGAS_MAX_TOKENS=4096 \
log_generator/.venv/bin/python -m evaluation.run_eval \
  --api-url http://localhost:8001 \
  --questions evaluation/questions_stage4.json \
  --stage stage_4_filter_coverage_32_final \
  --delay-seconds 60 \
  --metric-delay-seconds 60
~~~

Use a new stage name for every distinct run. Compare results only when the question file, data corpus, model configuration, and scoring policy are the same.

## Latest recorded evaluation

The final 32-question filter evaluation stored all 32 rows.

Reportable aggregate results:

| Metric | Result | Valid values |
|---|---:|---:|
| Faithfulness | 0.7361 | 29/32 |
| Answer relevancy | 0.7467 | 32/32 |
| Context precision | 0.8182 | 18/18 |
| Context recall | 0.9167 | 18/18 |
| Behavior match | 90.63% | 29/32 |

The run had no provider rate-limit failures and no classifier failures. Three faithfulness values were intentionally unavailable because those questions retrieved no chunks.

Three ambiguity questions were classified as answer_with_evidence instead of flag_ambiguity:

- midnight-downstream-ambiguity
- payment-vs-lock-ambiguity
- postgres-causal-expansion

The scores indicate strong retrieval coverage on the reportable reference-backed questions, but answer grounding and ambiguity handling still need improvement.

Earlier evaluation tables are intentionally not included here because they used different question sets and are not directly comparable.

## Tests

~~~bash
log_generator/.venv/bin/python -m pytest \
  evaluation log_collector log_consumer log_embedder log_generator log_api -q

docker compose config --quiet
~~~

## Known limitations

- Broad service-only queries can select one relevant trace while missing another related trace.
- Broad temporal queries can include background noise and only partial incident coverage.
- Similar incidents are not always disambiguated correctly.
- When three or more traces are found, only the top two are expanded.
- Trace expansion is capped to protect context size.
- Severity is stored but not an exact retrieval filter.
- Embeddings are rebuilt in batch rather than updated continuously.
- External LLM provider limits can interrupt evaluation metrics.
- The Docker Compose setup is for local development, not high availability.
- Authentication, authorization, multi-tenancy, alerting, and production scaling are not implemented.

## Future work

Potential next improvements are:

- improve multi-trace selection for broad service queries;
- improve temporal retrieval coverage while controlling noise;
- add query expansion or HyDE and evaluate each independently;
- add bounded retries and explicit error storage for evaluation metrics;
- replace truncate-and-rebuild embedding with incremental processing;
- add operational metrics, authentication, retention, and deployment hardening.
