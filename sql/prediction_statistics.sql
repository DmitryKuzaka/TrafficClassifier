SELECT
    toStartOfHour(f.first_packet_at, 'UTC') AS hour,
    p.predicted_class AS predicted,
    f.traffic_class AS truth,
    count() AS flows,
    sum(f.packet_count) AS packets,
    sum(p.observed_packets) AS observed_packets,
    sum(f.total_bytes) AS bytes
FROM predictions AS p
INNER JOIN flows AS f ON p.import_id = f.import_id AND p.flow_id = f.flow_id
WHERE p.run_id = {run_id:String}
GROUP BY hour, predicted, truth
ORDER BY hour, predicted, truth
