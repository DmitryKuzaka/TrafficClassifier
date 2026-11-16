CREATE TABLE IF NOT EXISTS import_runs
(
    import_id FixedString(64),
    attempt_id UUID,
    recorded_at DateTime64(6, 'UTC') DEFAULT now64(6),
    source_sha256 FixedString(64),
    source_name String,
    selection String,
    status Enum8('started' = 1, 'completed' = 2, 'failed' = 3),
    flow_count UInt64 DEFAULT 0,
    packet_count UInt64 DEFAULT 0,
    total_bytes UInt64 DEFAULT 0
)
ENGINE = MergeTree
ORDER BY (import_id, recorded_at, attempt_id, status)
