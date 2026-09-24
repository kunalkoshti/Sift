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

Gate 1 self-RAG confidence handling is controlled by `SELFRAG_CONFIDENCE_GATE_MODE`.
In `shadow` mode it logs the best dense cosine similarity and proposed decision
without changing answers. In `enforce` mode, weak dense evidence with no correlated
trace returns a deterministic abstention before the LLM is called. The initial
default threshold is provisional and must be calibrated from score distributions;
RRF and reranker scores are not used as confidence thresholds.
Recognized service-filter queries that return chunks bypass this gate, including
noise-only results, so they retain service-specific no-incident handling. Queries
that use the temporal full-corpus fallback also bypass the gate because their
retrieval result has a different interpretation. A trace in the top-k candidates
counts as strong evidence; a trace found deeper in the pre-expansion pool counts
only when its dense similarity reaches the configured threshold.

Gate 2 performs post-generation evidence verification. It asks the configured
API model to check whether factual answer claims are supported by specific
retrieved chunks, timestamps, and services. It is controlled by:

~~~text
SELFRAG_EVIDENCE_GATE_MODE=shadow
SELFRAG_EVIDENCE_MODEL=
SELFRAG_EVIDENCE_MAX_UNSUPPORTED_CLAIMS=0
SELFRAG_EVIDENCE_MAX_RETRIES=1
~~~

In shadow mode, Gate 2 logs verification results without changing answers. In
enforce mode, an unsupported answer gets at most one evidence-focused rewrite;
if it still cannot be verified, the API returns a deterministic cautious
response. Empty retrieval bypasses Gate 2 because it already uses deterministic
abstention. The verifier references are then checked deterministically against
the actual retrieved chunk IDs, services, and timestamps. A factual answer with
missing or invalid references is treated as unsupported. Gate 2 uses the
log-api LLM configuration, while RAGAS and behavior classification continue
using the separate `EVAL_*` configuration.

Gate 3 performs a deterministic scope and answerability check before retrieval.
It blocks only high-confidence out-of-scope categories such as weather,
financial metrics, customer demographics, and aggregate business statistics.
Ambiguous questions and normal operational log questions continue through the
regular retrieval, Gate 1, and Gate 2 flow.

~~~text
SELFRAG_SCOPE_GATE_MODE=shadow
~~~

In shadow mode, the scope decision is logged without changing the response. In
enforce mode, clearly out-of-scope questions receive a deterministic response
without an embedding, database retrieval, or LLM call. This gate is
high-precision by design to avoid blocking valid operational questions such as
rate-limit incidents.

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
- classifier failure state;
- fallback disclosure for time-range fallback answers.

Faithfulness is skipped when no context was retrieved. Context precision and recall are skipped when there is no reference answer or no retrieved context. The last-week-no-results fallback question is excluded from aggregate context precision/recall reporting because its result depends on the fallback retrieval note rather than ordinary in-range retrieval.
Behavior classification receives the API retrieval note, and fallback disclosure is recorded separately from the four behavior labels.

Run the current evaluation with the provider configured in .env:

~~~bash
RAGAS_MAX_TOKENS=4096 \
log_generator/.venv/bin/python -m evaluation.run_eval \
  --api-url http://localhost:8001 \
  --questions evaluation/questions_stage4.json \
  --stage gate1_final_baseline_clean \
  --delay-seconds 60 \
  --metric-delay-seconds 60
~~~

Use a new stage name for every distinct run. Compare results only when the question file, data corpus, model configuration, and scoring policy are the same.

## Latest recorded evaluation

The final Self-RAG evaluation used the frozen 32-question set under
`gate2_evidence_strict_recheck`.

Aggregate results:

| Metric | Result | Valid values |
|---|---:|---:|
| Faithfulness | 0.784 | 25/32 |
| Answer relevancy | 0.737 | 32/32 |
| Context precision | 0.883 | 21/32 |
| Context recall | 0.905 | 21/32 |
| Behavior match | 96.88% | 31/32 |

The run stored all 32 rows with no provider, API, or classifier failures. Seven
questions had deterministic empty retrieval, so their faithfulness, context
precision, and context recall values were intentionally NULL. Context precision
and recall were also excluded for questions where those metrics are not
meaningful, such as no-incident and fallback cases. The evidence gate ran for
25 questions and completed three successful corrective retries.

The only behavior mismatch was `midnight-downstream-ambiguity`, where the model
answered from a single retrieved trace instead of explicitly flagging ambiguity.
This remains a documented retrieval limitation. The recorded run also contained
one false-negative `fallback_disclosed` flag caused by wording variation; the
answer itself disclosed that it used the retrieved evidence outside the requested
time window, and the checker now recognizes that wording. Earlier evaluation
tables are not included because they used different question sets or scoring
policies.

For comparison, the earlier Gate 1 run used the same question set:

| Metric | Gate 1 | Final Self-RAG | Valid values |
|---|---:|---:|---:|
| Faithfulness | 0.785 | 0.784 | 25/32 |
| Answer relevancy | 0.768 | 0.737 | 32/32 |
| Context precision | 0.902 | 0.883 | 21/32 |
| Context recall | 0.893 | 0.905 | 21/32 |
| Behavior match | 96.88% | 96.88% | 32/32 |

The final Self-RAG run recovered faithfulness from the preceding evidence-gate
run's 0.754 to 0.784 and improved context recall from 0.889 to 0.905. It did
not materially exceed the Gate 1 faithfulness result. Self-RAG therefore adds
deterministic scope handling, evidence validation, bounded correction, and
better observability, but not a large aggregate RAGAS improvement.

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
- When only one trace is retrieved, the answer may describe that trace confidently
  without proving that competing incidents are absent from the full corpus.
- When three or more traces are found, only the top two are expanded.
- Trace expansion is capped to protect context size.
- Severity is stored but not an exact retrieval filter.
- Embeddings are rebuilt in batch rather than updated continuously.
- External LLM provider limits can interrupt evaluation metrics.
- The evidence verifier can still accept plausible causal explanations when
  the logs show sequence but do not explicitly prove causation.
- Evidence-reference validation confirms that a citation points to a real
  chunk, timestamp, and service; it does not independently prove the semantic
  truth of the claim.
- Self-RAG retries add latency and token usage, and verifier outages currently
  fail open by retaining the original answer.
- The evaluation set is frozen for comparison but is not a held-out test set;
  further tuning against it risks overfitting.
- The Docker Compose setup is for local development, not high availability.
- Authentication, authorization, multi-tenancy, alerting, and production scaling are not implemented.

## Future work

Potential next improvements are:

- improve multi-trace selection for broad service queries;
- improve temporal retrieval coverage while controlling noise;
- add query expansion or HyDE and evaluate each independently;
- replace truncate-and-rebuild embedding with incremental processing;
- add operational metrics, authentication, retention, and deployment hardening.
