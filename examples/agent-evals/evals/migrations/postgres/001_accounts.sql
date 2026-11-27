CREATE TABLE accounts (
    id text PRIMARY KEY,
    email text NOT NULL UNIQUE,
    name text NOT NULL,
    plan text NOT NULL,
    included_events integer NOT NULL,
    overage_cents_per_1000 integer NOT NULL
);
