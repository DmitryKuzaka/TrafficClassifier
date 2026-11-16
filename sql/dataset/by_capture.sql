SELECT
    traffic_class,
    capture_group,
    capture_name,
    count() AS flows
FROM flows
WHERE import_id = {import_id:String}
GROUP BY traffic_class, capture_group, capture_name
ORDER BY traffic_class, flows DESC, capture_name
