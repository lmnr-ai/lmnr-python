-- 120,000 events spread over the last 29 days, ids as the ingestion API assigns them.
INSERT INTO usage_events
SELECT 'acct_4kT9qLm2VxR8', concat('evt_', lower(hex(sipHash128('tallgrass', number)))),
       now() - toIntervalSecond(number * 21), 'api.request'
FROM numbers(120000);
-- A retry storm about three days ago: 4,000 of that day's events recorded ten more times each.
INSERT INTO usage_events
SELECT 'acct_4kT9qLm2VxR8', concat('evt_', lower(hex(sipHash128('tallgrass', number)))),
       now() - toIntervalSecond(number * 21), 'api.request'
FROM numbers(12343, 4000) ARRAY JOIN range(10) AS retry;
-- Another account's traffic.
INSERT INTO usage_events
SELECT 'acct_Z81fWc0dNe3H', concat('evt_', lower(hex(sipHash128('harborline', number)))),
       now() - toIntervalSecond(number * 150), 'api.request'
FROM numbers(15000);
