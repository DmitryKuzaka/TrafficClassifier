CREATE TABLE IF NOT EXISTS flows
(
    import_id FixedString(64),
    flow_id FixedString(64),
    source_row UInt64,
    capture_name LowCardinality(String),
    capture_group LowCardinality(String),
    traffic_class LowCardinality(String),
    first_packet_at DateTime64(6, 'UTC'),
    packet_count UInt64,
    total_bytes UInt64,
    arrival_offsets_us Array(Int64),
    packet_sizes_bytes Array(UInt32),
    directions Array(UInt8),
    CONSTRAINT nonempty_flow CHECK packet_count > 0,
    CONSTRAINT matching_packet_arrays CHECK
        length(arrival_offsets_us) = packet_count AND
        length(packet_sizes_bytes) = packet_count AND
        length(directions) = packet_count
)
ENGINE = MergeTree
PARTITION BY import_id
ORDER BY (import_id, traffic_class, capture_group, flow_id)
