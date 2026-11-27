CREATE TABLE IF NOT EXISTS events (id UInt64, kind String) ENGINE = MergeTree ORDER BY id;
