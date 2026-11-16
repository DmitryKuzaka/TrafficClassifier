SELECT
    traffic_class,
    min(packet_count) AS min_packets,
    quantileExact(0.5)(packet_count) AS median_packets,
    quantileExact(0.9)(packet_count) AS p90_packets,
    max(packet_count) AS max_packets,
    countIf(packet_count >= 8) / count() AS share_at_least_8,
    countIf(packet_count >= 16) / count() AS share_at_least_16,
    countIf(packet_count >= 32) / count() AS share_at_least_32,
    countIf(packet_count >= 64) / count() AS share_at_least_64
FROM flows
WHERE import_id = {import_id:String}
GROUP BY traffic_class
ORDER BY traffic_class
