SELECT
    toStartOfHour(first_packet_at, 'UTC') AS start_hour_utc,
    traffic_class,
    count() AS flows_started,
    sum(total_bytes) AS bytes_of_those_flows
FROM flows
WHERE import_id = {import_id:String}
GROUP BY start_hour_utc, traffic_class
ORDER BY start_hour_utc, traffic_class
