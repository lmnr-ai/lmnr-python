#!/bin/bash
# Graded by Harbor's own verifier (grade_with_test_sh), in a fresh container
# (environment_mode = "separate") that holds only what the run left behind:
#   /logs/artifacts/state/postgres/credits.json  Postgres(snapshot=["credits"])
#   /srv/app/out/reply.txt                       the declared artifact
# The image is Node's, so the checks are a few lines of Node.
node - <<'JS'
const fs = require("node:fs");
const credits = JSON.parse(fs.readFileSync("/logs/artifacts/state/postgres/credits.json", "utf8"));
const reply = fs.readFileSync("/srv/app/out/reply.txt", "utf8");
const cents = credits.reduce((sum, c) => sum + c.amount_cents, 0);
console.log(`credits: ${cents} cents; reply: ${reply}`);
const rewards = {
  credit: Number(cents === 8000),
  reply_states_amount: Number(/\$\s?80(\.00)?\b/.test(reply)),
};
fs.writeFileSync("/logs/verifier/reward.json", JSON.stringify(rewards));
JS
