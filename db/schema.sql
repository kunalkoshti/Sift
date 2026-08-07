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
