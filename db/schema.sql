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
