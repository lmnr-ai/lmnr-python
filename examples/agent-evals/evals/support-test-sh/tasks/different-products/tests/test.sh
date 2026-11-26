#!/bin/bash
# The two charges are different products, not a duplicate: nothing is owed.
# Runs in the agent's container after it finished (grade_with_test_sh).
refunds=$(python -c "import sqlite3; print(sqlite3.connect('/srv/app/app.db').execute('SELECT COUNT(*) FROM refunds').fetchone()[0])")
echo "refunds: $refunds"
if [ "$refunds" = "0" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
