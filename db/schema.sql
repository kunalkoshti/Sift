CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS raw_logs (
  id BIGSERIAL PRIMARY KEY,
  schema_version TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  level TEXT NOT NULL,
  service TEXT NOT NULL,
  host TEXT NOT NULL,
  message TEXT NOT NULL,
  trace_id TEXT,
  metadata JSONB NOT NULL,
  redis_entry_id TEXT NOT NULL UNIQUE,
  received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chunk_type TEXT NOT NULL DEFAULT 'window',
  window_start TIMESTAMPTZ NOT NULL,
  window_end TIMESTAMPTZ NOT NULL,
  sub_index INTEGER NOT NULL DEFAULT 0,
  services TEXT[] NOT NULL,
  trace_ids TEXT[] NOT NULL DEFAULT '{}',
  content TEXT NOT NULL,
  embedding VECTOR(384),
  fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  raw_log_ids BIGINT[] NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_fts_idx
  ON chunks USING GIN(fts);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_window_idx
  ON chunks (window_start, window_end);

CREATE TABLE IF NOT EXISTS eval_runs (
  id BIGSERIAL PRIMARY KEY,
  run_id UUID NOT NULL,
  run_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
  stage_name TEXT NOT NULL,
  question_id TEXT NOT NULL,
  question_text TEXT NOT NULL,
  category TEXT NOT NULL,
  generated_answer TEXT NOT NULL,
  retrieved_chunk_ids UUID[] NOT NULL DEFAULT '{}',
  faithfulness DOUBLE PRECISION,
  context_precision DOUBLE PRECISION,
  context_recall DOUBLE PRECISION,
  answer_relevancy DOUBLE PRECISION,
  classified_behavior TEXT NOT NULL,
  behavior_match BOOLEAN NOT NULL,
  latency_ms DOUBLE PRECISION NOT NULL
);

-- Existing Postgres volumes already have eval_runs, so add behavior-classifier
-- fields independently as an idempotent schema migration.
ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS classified_behavior TEXT;

ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS behavior_match BOOLEAN;

-- The first evaluation draft used abstention_correct. Preserve any historical
-- values, but allow new classifier-based inserts to omit that legacy column.

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = current_schema()
      AND table_name = 'eval_runs'
      AND column_name = 'abstention_correct'
  ) THEN
    ALTER TABLE eval_runs
      ALTER COLUMN abstention_correct DROP NOT NULL;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS eval_runs_run_idx
  ON eval_runs (run_id);

CREATE INDEX IF NOT EXISTS eval_runs_stage_question_idx
  ON eval_runs (stage_name, question_id);
