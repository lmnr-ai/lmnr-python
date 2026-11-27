-- 160,000 distinct events over the last 30 days: a busy month, no duplicates.
INSERT INTO usage_events
SELECT 'acct_4kT9qLm2VxR8', concat('evt_', lower(hex(sipHash128('tallgrass', number)))),
       now() - toIntervalSecond(number * 16), 'api.request'
FROM numbers(160000);
-- Another account's traffic.
INSERT INTO usage_events
SELECT 'acct_Z81fWc0dNe3H', concat('evt_', lower(hex(sipHash128('harborline', number)))),
       now() - toIntervalSecond(number * 150), 'api.request'
FROM numbers(15000);
