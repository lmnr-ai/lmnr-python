"""`lmnr import langfuse`: copy a Langfuse project into a Laminar project.

Traces are read from the Langfuse public API (v2 observations, v3 scores, datasets)
and written to Laminar's OTLP/HTTP+JSON endpoint with their original trace/span ids
and timestamps, so the same trace can be found in both systems. Source token counts
and costs are sent explicitly, so Laminar does not re-price spans the source already
priced.
"""

from argparse import Namespace
from datetime import datetime, timezone
from typing import Any

import asyncio
import hashlib
import re
import sys

import httpx
import orjson

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import describe_response

LOG = get_default_logger(__name__, verbose=False)

OBSERVATION_FIELDS = "core,basic,time,io,metadata,model,usage,trace_context"
ASSOCIATION = "lmnr.association.properties"
LLM_TYPES = {"GENERATION", "EMBEDDING"}
HEX32 = re.compile(r"^[0-9a-fA-F]{32}$")
HEX16 = re.compile(r"^[0-9a-fA-F]{16}$")
DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"
# OTel scope/resource attributes Langfuse folds into observation metadata (SDK name,
# public key); they describe Langfuse's ingestion, not the user's trace.
LANGFUSE_INTERNAL_METADATA_PREFIXES = ("scope.", "resourceAttributes")
DEFAULT_TRACE_BATCH_SIZE = 20
MAX_REQUEST_BYTES = 4 * 1024 * 1024
CONCURRENCY = 8


# ---------------------------------------------------------------------------
# Pure mapping: Langfuse observation -> OTLP JSON span
# ---------------------------------------------------------------------------


def to_trace_id(raw: str) -> str:
    """32 lowercase hex chars; ids that are not get hashed deterministically."""
    compact = raw.replace("-", "")
    if HEX32.match(compact):
        return compact.lower()
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def to_span_id(raw: str) -> str:
    """16 lowercase hex chars; ids that are not get hashed deterministically."""
    if HEX16.match(raw):
        return raw.lower()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def trace_id_to_uuid(trace_id_hex: str) -> str:
    h = trace_id_hex
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def span_type(observation_type: str | None) -> str:
    if observation_type in LLM_TYPES:
        return "LLM"
    if observation_type == "TOOL":
        return "TOOL"
    return "DEFAULT"


def to_unix_nanos(value: str | None) -> str | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Integer arithmetic keeps microseconds exact; float seconds would round them.
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    return str(
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


def as_json_string(value: Any) -> str:
    """Laminar expects span input/output as JSON; Langfuse v2 returns raw strings."""
    if isinstance(value, str):
        try:
            orjson.loads(value)
            return value
        except orjson.JSONDecodeError:
            return orjson.dumps(value).decode()
    return orjson.dumps(value).decode()


def otlp_value(value: Any) -> dict:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [otlp_value(v) for v in value]}}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"stringValue": orjson.dumps(value).decode()}


def number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_mapping(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = orjson.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except orjson.JSONDecodeError:
            return {}
    return {}


def score_value(score: dict) -> Any:
    value = score.get("value")
    if score.get("dataType") in ("NUMERIC", "BOOLEAN"):
        n = number(value)
        return int(n) if n.is_integer() else n
    return value if value is not None else score.get("stringValue")


def resolve_parent(observation: dict, trace_observation_ids: set[str]) -> str | None:
    """The parent id, or None for a root.

    Langfuse root observations can point at a parent id that is not an observation of
    the trace (e.g. a remote OTel parent derived from the trace id). Laminar would keep
    that dangling pointer and pick some other span as the trace's root, so it is
    dropped here.
    """
    parent = observation.get("parentObservationId")
    return parent if parent and parent in trace_observation_ids else None


def observation_to_span(
    observation: dict,
    *,
    parent_id: str | None,
    trace_scores: list[dict] | None = None,
    observation_scores: list[dict] | None = None,
) -> dict:
    """Map one Langfuse v2 observation to an OTLP JSON span with Laminar attributes."""
    attributes: dict[str, Any] = {
        "lmnr.span.type": span_type(observation.get("type")),
        "langfuse.observation.type": observation.get("type") or "",
    }
    raw_trace_id = observation["traceId"]
    raw_span_id = observation["id"]
    trace_id = to_trace_id(raw_trace_id)
    span_id = to_span_id(raw_span_id)
    if trace_id != raw_trace_id.replace("-", "").lower():
        attributes["langfuse.trace.id"] = raw_trace_id
    if span_id != raw_span_id.lower():
        attributes["langfuse.observation.id"] = raw_span_id

    if observation.get("input") is not None:
        attributes["lmnr.span.input"] = as_json_string(observation["input"])
    if observation.get("output") is not None:
        attributes["lmnr.span.output"] = as_json_string(observation["output"])

    if observation.get("model"):
        attributes["gen_ai.request.model"] = observation["model"]
        attributes["gen_ai.response.model"] = observation["model"]

    # Laminar's input_tokens is the TOTAL including cached tokens (it subtracts the
    # cache parts itself), which is what Langfuse's inputUsage holds;
    # usageDetails["input"] excludes them.
    usage = parse_mapping(observation.get("usageDetails"))
    input_tokens = int(number(observation.get("inputUsage")))
    output_tokens = int(number(observation.get("outputUsage")))
    cache_read = int(sum(number(v) for k, v in usage.items() if "cache_read" in k))
    cache_write = int(
        sum(
            number(v)
            for k, v in usage.items()
            if "cache_creation" in k or "cache_write" in k
        )
    )
    for key, tokens in (
        ("gen_ai.usage.input_tokens", input_tokens),
        ("gen_ai.usage.output_tokens", output_tokens),
        ("gen_ai.usage.cache_read_input_tokens", cache_read),
        ("gen_ai.usage.cache_creation_input_tokens", cache_write),
    ):
        if tokens > 0:
            attributes[key] = tokens

    # Any explicit cost > 0 makes Laminar keep the source price instead of its own.
    for key, cost in (
        ("gen_ai.usage.input_cost", number(observation.get("inputCost"))),
        ("gen_ai.usage.output_cost", number(observation.get("outputCost"))),
        ("gen_ai.usage.cost", number(observation.get("totalCost"))),
    ):
        if cost > 0:
            attributes[key] = cost

    if observation.get("sessionId"):
        attributes[f"{ASSOCIATION}.session_id"] = observation["sessionId"]
    if observation.get("userId"):
        attributes[f"{ASSOCIATION}.user_id"] = observation["userId"]
    if observation.get("tags"):
        attributes[f"{ASSOCIATION}.tags"] = list(observation["tags"])

    metadata = parse_mapping(observation.get("metadata"))
    is_root = parent_id is None
    if is_root:
        if observation.get("name"):
            attributes["trace.name"] = observation["name"]
        trace_metadata = dict(metadata)
        for key in ("environment", "release"):
            if observation.get(key) and observation[key] != "default":
                trace_metadata[key] = observation[key]
        for score in trace_scores or []:
            trace_metadata[f"score.{score['name']}"] = score_value(score)
        for key, value in trace_metadata.items():
            if key.startswith(LANGFUSE_INTERNAL_METADATA_PREFIXES):
                continue
            scalar = isinstance(value, (str, int, float, bool))
            attributes[f"{ASSOCIATION}.metadata.{key}"] = (
                value if scalar else orjson.dumps(value).decode()
            )
    else:
        metadata = {
            k: v
            for k, v in metadata.items()
            if not k.startswith(LANGFUSE_INTERNAL_METADATA_PREFIXES)
        }
        if metadata:
            attributes["langfuse.metadata"] = orjson.dumps(metadata).decode()

    for score in observation_scores or []:
        attributes[f"langfuse.score.{score['name']}"] = score_value(score)

    level = observation.get("level")
    if level and level != "DEFAULT":
        attributes["langfuse.level"] = level

    span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": observation.get("name") or observation.get("type") or "span",
        "kind": 1,
        "startTimeUnixNano": to_unix_nanos(observation.get("startTime")),
        "endTimeUnixNano": to_unix_nanos(
            observation.get("endTime") or observation.get("startTime")
        ),
        "attributes": [
            {"key": k, "value": otlp_value(v)} for k, v in attributes.items()
        ],
    }
    if parent_id is not None:
        span["parentSpanId"] = to_span_id(parent_id)
    if level == "ERROR":
        span["status"] = {"code": 2, "message": observation.get("statusMessage") or ""}
    return span


def export_request(spans: list[dict]) -> dict:
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "langfuse-import"},
                        }
                    ]
                },
                "scopeSpans": [
                    {"scope": {"name": "lmnr.cli.import_langfuse"}, "spans": spans}
                ],
            }
        ]
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


class LangfuseReader:
    def __init__(self, client: httpx.AsyncClient, time_window: dict[str, str]):
        self.client = client
        self.time_window = time_window

    async def _get(self, path: str, params: dict) -> dict:
        for attempt in range(5):
            response = await self.client.get(path, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                await asyncio.sleep(2**attempt)
                continue
            if response.status_code == 404 and path.endswith("/v2/observations"):
                raise SystemExit(
                    "This Langfuse server does not serve "
                    + "GET /api/public/v2/observations. The importer needs Langfuse v4 "
                    + "(or v3 with the v4 API preview enabled)."
                )
            response.raise_for_status()
            return response.json()
        response.raise_for_status()
        return response.json()

    async def trace_ids(self) -> list[str]:
        """Trace ids whose root observation starts inside the time window."""
        ids: list[str] = []
        cursor = None
        while True:
            params = {
                "limit": 1000,
                "fields": "core",
                "isRootObservation": "true",
                **self.time_window,
            }
            if cursor:
                params["cursor"] = cursor
            page = await self._get("/api/public/v2/observations", params)
            ids.extend(o["traceId"] for o in page["data"] if o.get("traceId"))
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor:
                return list(dict.fromkeys(ids))

    async def observations(self, trace_id: str) -> list[dict]:
        out: list[dict] = []
        cursor = None
        while True:
            params = {"limit": 1000, "fields": OBSERVATION_FIELDS, "traceId": trace_id}
            if cursor:
                params["cursor"] = cursor
            page = await self._get("/api/public/v2/observations", params)
            out.extend(page["data"])
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor:
                return out

    async def scores(self) -> list[dict]:
        out: list[dict] = []
        cursor = None
        while True:
            params = {"limit": 100, "fields": "core,details,subject"}
            if cursor:
                params["cursor"] = cursor
            page = await self._get("/api/public/v3/scores", params)
            out.extend(page["data"])
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor:
                return out

    async def datasets(self) -> list[dict]:
        return await self._paged("/api/public/v2/datasets", {})

    async def dataset_items(self, dataset_name: str) -> list[dict]:
        return await self._paged(
            "/api/public/dataset-items", {"datasetName": dataset_name}
        )

    async def _paged(self, path: str, params: dict) -> list[dict]:
        out: list[dict] = []
        page_number = 1
        while True:
            page = await self._get(path, {**params, "page": page_number, "limit": 100})
            out.extend(page["data"])
            meta = page.get("meta") or {}
            if page_number >= int(meta.get("totalPages") or 1):
                return out
            page_number += 1


class LaminarWriter:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def existing_trace_ids(self, trace_ids_hex: list[str]) -> set[str]:
        if not trace_ids_hex:
            return set()
        values = ", ".join(f"'{trace_id_to_uuid(t)}'" for t in trace_ids_hex)
        response = await self.client.post(
            "/v1/sql/query",
            json={
                "query": "SELECT DISTINCT trace_id FROM spans "
                + f"WHERE trace_id IN ({values})"
            },
        )
        response.raise_for_status()
        return {row["trace_id"].replace("-", "") for row in response.json()["data"]}

    async def send_spans(self, spans: list[dict]) -> None:
        body = orjson.dumps(export_request(spans))
        response = await self.client.post(
            "/v1/traces", content=body, headers={"Content-Type": "application/json"}
        )
        if response.status_code >= 300:
            raise RuntimeError(
                f"Laminar rejected {len(spans)} spans: {describe_response(response)}"
            )

    async def existing_dataset_names(self) -> set[str]:
        response = await self.client.get("/v1/datasets")
        if response.status_code >= 300:
            return set()
        payload = response.json()
        items = payload if isinstance(payload, list) else payload.get("items", [])
        return {d.get("name") for d in items}

    async def push_datapoints(self, dataset_name: str, datapoints: list[dict]) -> None:
        response = await self.client.post(
            "/v1/datasets/datapoints",
            json={
                "name": dataset_name,
                "datapoints": datapoints,
                "createDataset": True,
            },
        )
        if response.status_code >= 300:
            raise RuntimeError(
                f"Laminar rejected dataset '{dataset_name}': "
                + describe_response(response)
            )


def index_scores(
    scores: list[dict],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    by_trace: dict[str, list[dict]] = {}
    by_observation: dict[str, list[dict]] = {}
    for score in scores:
        subject = score.get("subject") or {}
        kind, target = subject.get("kind"), subject.get("id")
        if not target:
            target, kind = score.get("traceId"), "trace"
        if kind == "trace" and target:
            by_trace.setdefault(target, []).append(score)
        elif kind == "observation" and target:
            by_observation.setdefault(target, []).append(score)
    return by_trace, by_observation


def dataset_item_to_datapoint(item: dict) -> dict:
    metadata = parse_mapping(item.get("metadata"))
    metadata["langfuse.dataset_item_id"] = item.get("id")
    datapoint: dict[str, Any] = {"data": item.get("input"), "metadata": metadata}
    if item.get("expectedOutput") is not None:
        datapoint["target"] = item["expectedOutput"]
    return datapoint


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def _laminar_base_url(args: Namespace) -> str:
    base = args.base_url.rstrip("/")
    return f"{base}:{args.port}" if args.port else base


async def _import_traces(
    args: Namespace, reader: LangfuseReader, writer: LaminarWriter, summary: dict
) -> None:
    trace_ids = await reader.trace_ids()
    summary["traces_found"] = len(trace_ids)
    scores_by_trace, scores_by_observation = index_scores(await reader.scores())
    summary["scores_found"] = sum(len(v) for v in scores_by_trace.values()) + sum(
        len(v) for v in scores_by_observation.values()
    )
    if args.dry_run:
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def fetch(trace_id: str) -> list[dict]:
        async with semaphore:
            return await reader.observations(trace_id)

    for start in range(0, len(trace_ids), args.batch_size):
        batch = trace_ids[start : start + args.batch_size]
        already = await writer.existing_trace_ids([to_trace_id(t) for t in batch])
        todo = [t for t in batch if to_trace_id(t) not in already]
        summary["traces_skipped"] += len(batch) - len(todo)
        groups = await asyncio.gather(*(fetch(t) for t in todo))
        pending: list[dict] = []
        pending_bytes = 0
        for trace_id, observations in zip(todo, groups):
            ids = {o["id"] for o in observations}
            spans = []
            for o in observations:
                parent_id = resolve_parent(o, ids)
                spans.append(
                    observation_to_span(
                        o,
                        parent_id=parent_id,
                        trace_scores=scores_by_trace.get(trace_id)
                        if parent_id is None
                        else None,
                        observation_scores=scores_by_observation.get(o["id"]),
                    )
                )
            for o in observations:
                summary["spans"][span_type(o.get("type"))] = (
                    summary["spans"].get(span_type(o.get("type")), 0) + 1
                )
                if (
                    span_type(o.get("type")) == "LLM"
                    and number(o.get("totalCost")) <= 0
                ):
                    summary["llm_spans_priced_by_laminar"] += 1
            size = len(orjson.dumps(spans))
            # One trace never splits across requests, so a crash can only lose whole
            # traces, which the existing-trace check then re-imports on the next run.
            if pending and pending_bytes + size > MAX_REQUEST_BYTES:
                await writer.send_spans(pending)
                pending, pending_bytes = [], 0
            pending.extend(spans)
            pending_bytes += size
            summary["traces_imported"] += 1
        if pending:
            await writer.send_spans(pending)
        LOG.info(
            f"Traces: {min(start + args.batch_size, len(trace_ids))}/{len(trace_ids)}"
        )


async def _import_datasets(
    args: Namespace, reader: LangfuseReader, writer: LaminarWriter, summary: dict
) -> None:
    datasets = await reader.datasets()
    existing = set() if args.dry_run else await writer.existing_dataset_names()
    for dataset in datasets:
        name = dataset["name"]
        items = await reader.dataset_items(name)
        summary["datasets_found"] += 1
        summary["dataset_items_found"] += len(items)
        if args.dry_run:
            continue
        if name in existing:
            summary["datasets_skipped"] += 1
            continue
        datapoints = [
            dataset_item_to_datapoint(i) for i in items if i.get("input") is not None
        ]
        for start in range(0, len(datapoints), 100):
            await writer.push_datapoints(name, datapoints[start : start + 100])
        summary["datasets_imported"] += 1
        summary["dataset_items_imported"] += len(datapoints)


async def handle_import_langfuse(args: Namespace) -> None:
    if not args.langfuse_public_key or not args.langfuse_secret_key:
        LOG.error(
            "Langfuse keys are required: --langfuse-public-key/--langfuse-secret-key "
            + "or LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY"
        )
        sys.exit(1)
    if not args.dry_run and not args.project_api_key:
        LOG.error(
            "Laminar project API key is required: --project-api-key "
            + "or LMNR_PROJECT_API_KEY"
        )
        sys.exit(1)

    time_window = {}
    if args.from_time:
        time_window["fromStartTime"] = args.from_time
    # Pin the upper bound so a project still receiving traces cannot shift pages.
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    time_window["toStartTime"] = args.to_time or now

    summary: dict[str, Any] = {
        "traces_found": 0,
        "traces_imported": 0,
        "traces_skipped": 0,
        "spans": {},
        "scores_found": 0,
        "llm_spans_priced_by_laminar": 0,
        "datasets_found": 0,
        "dataset_items_found": 0,
        "datasets_imported": 0,
        "dataset_items_imported": 0,
        "datasets_skipped": 0,
    }

    async with (
        httpx.AsyncClient(
            base_url=args.langfuse_host.rstrip("/"),
            auth=(args.langfuse_public_key, args.langfuse_secret_key),
            timeout=120,
        ) as langfuse,
        httpx.AsyncClient(
            base_url=_laminar_base_url(args),
            headers={"Authorization": f"Bearer {args.project_api_key}"},
            timeout=120,
        ) as laminar,
    ):
        reader = LangfuseReader(langfuse, time_window)
        writer = LaminarWriter(laminar)
        if not args.skip_traces:
            await _import_traces(args, reader, writer, summary)
        if not args.skip_datasets:
            await _import_datasets(args, reader, writer, summary)

    _print_summary(summary, args.dry_run)


def _print_summary(summary: dict, dry_run: bool) -> None:
    print("Dry run, nothing written:" if dry_run else "Import finished:")
    print(
        f"  traces: {summary['traces_found']} found, "
        f"{summary['traces_imported']} imported, "
        f"{summary['traces_skipped']} already in Laminar"
    )
    if summary["spans"]:
        by_type = ", ".join(f"{k} {v}" for k, v in sorted(summary["spans"].items()))
        print(f"  spans by type: {by_type}")
    print(
        f"  scores: {summary['scores_found']} found "
        "(trace scores -> trace metadata 'score.<name>', "
        "observation scores -> span attribute 'langfuse.score.<name>')"
    )
    if summary["llm_spans_priced_by_laminar"]:
        print(
            f"  {summary['llm_spans_priced_by_laminar']} LLM spans had no source cost; "
            "Laminar priced them from its own model table"
        )
    print(
        f"  datasets: {summary['datasets_found']} found "
        f"({summary['dataset_items_found']} items), "
        f"{summary['datasets_imported']} imported "
        f"({summary['dataset_items_imported']} items), "
        f"{summary['datasets_skipped']} already in Laminar"
    )
