SELECT
    traffic_class,
    count() AS flows,
    sum(packet_count) AS packets,
    sum(total_bytes) AS bytes
FROM flows
WHERE import_id = {import_id:String}
GROUP BY traffic_class
ORDER BY traffic_class
