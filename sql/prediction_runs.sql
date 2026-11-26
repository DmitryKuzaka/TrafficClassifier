CREATE TABLE IF NOT EXISTS prediction_runs
(
    run_id FixedString(64),
    import_id FixedString(64),
    model_id String,
    n UInt8,
    max_flows Nullable(UInt64),
    flow_count UInt64,
    status Enum8('started' = 1, 'completed' = 2),
    recorded_at DateTime64(6, 'UTC') DEFAULT now64(6)
)
ENGINE = MergeTree
ORDER BY (run_id, status, recorded_at)
