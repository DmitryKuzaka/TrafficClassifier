CREATE TABLE IF NOT EXISTS predictions
(
    run_id FixedString(64),
    import_id FixedString(64),
    flow_id FixedString(64),
    model_id String,
    n UInt8,
    observed_packets UInt8,
    predicted_class LowCardinality(String),
    score_video Float32,
    score_voice Float32,
    score_file_transfer Float32,
    CONSTRAINT valid_prefix CHECK n IN (8, 16, 32, 64) AND observed_packets BETWEEN 1 AND n,
    CONSTRAINT valid_class CHECK predicted_class IN ('video', 'voice', 'file_transfer')
)
ENGINE = MergeTree
PARTITION BY run_id
ORDER BY (run_id, flow_id)
