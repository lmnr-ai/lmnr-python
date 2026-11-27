CREATE TABLE credits (
    id serial PRIMARY KEY,
    account_id text NOT NULL REFERENCES accounts (id),
    amount_cents integer NOT NULL CHECK (amount_cents > 0),
    reason text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
