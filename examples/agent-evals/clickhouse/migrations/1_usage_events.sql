CREATE TABLE IF NOT EXISTS usage_events
(
    account_id String,
    event_id String,
    ts DateTime,
    kind LowCardinality(String)
)
ENGINE = MergeTree
ORDER BY (account_id, ts);
