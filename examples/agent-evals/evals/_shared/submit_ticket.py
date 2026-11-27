# Runs in the container: queues the case's ticket with the ticket service and
# prints its id. The runner uploads it for the one command and deletes it after.
import json
import sys
import urllib.request

with open(sys.argv[1]) as f:
    ticket = json.load(f)["ticket"]
request = urllib.request.Request(
    "http://127.0.0.1:8080/tickets",
    data=json.dumps({"ticket": ticket}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=10) as response:
    print(json.load(response)["id"])
