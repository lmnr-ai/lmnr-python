"""Write lmnr-atif trajectory files (`spec/lmnr-atif/lmnr-atif.md`).

Pure file handling on top of `builder.build_trajectory`: where each file goes,
writing it atomically, and storing inline media next to it. Collecting the
spans is the tracing layer's job (`opentelemetry_lib/tracing/trajectory.py`).
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any

from lmnr.sdk.trajectory.builder import BuildOptions, SpanRecord, build_trajectory

# Placeholders `LMNR_TRAJECTORY_PATH` may contain.
TRACE_ID = "{trace_id}"
PID = "{pid}"

# ATIF's media convention: an `images/` directory next to the trajectory file.
IMAGES_DIR = "images"
_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif", "image/webp": "webp"}


def render_path(template: str, trace_id: str | None = None, pid: int | None = None) -> Path:
    """The file path for a template, with `~` expanded and placeholders filled."""
    path = template.replace(PID, str(os.getpid() if pid is None else pid))
    if trace_id is not None:
        path = path.replace(TRACE_ID, trace_id)
    return Path(path).expanduser()


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON so a reader never sees a half-written file, even if the
    process is killed mid-write: a temporary file in the same directory is
    renamed over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def media_sink(directory: Path) -> Callable[[str, str], str | None]:
    """A `MediaSink` that stores images under `<directory>/images/`, named by
    content hash, and returns their path relative to `directory`."""

    def store(data: str, media_type: str) -> str | None:
        extension = _EXTENSIONS.get(media_type)
        if extension is None:
            return None
        try:
            content = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            return None
        name = f"{hashlib.sha256(content).hexdigest()[:32]}.{extension}"
        target = directory / IMAGES_DIR / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{name}.{os.getpid()}.tmp")
            _ = tmp.write_bytes(content)
            os.replace(tmp, target)
        return f"{IMAGES_DIR}/{name}"

    return store


def write_trajectories(
    records: Iterable[SpanRecord],
    template: str,
    options: BuildOptions,
    open_trace_ids: Collection[str] = (),
) -> list[Path]:
    """Build and write the trajectory files for `records`.

    With `{trace_id}` in the template, each trace gets its own file; otherwise
    every trace goes into one. A file is a snapshot (`complete: false`) when a
    trace it covers is in `open_trace_ids`, so a finished trace's own file stays
    complete while another trace runs. Returns the paths written (none when
    there were no LLM or tool calls).
    """
    groups: dict[str | None, list[SpanRecord]] = {}
    per_trace = TRACE_ID in template
    for record in records:
        groups.setdefault(record.trace_id if per_trace else None, []).append(record)

    written: list[Path] = []
    for trace_id, group in groups.items():
        path = render_path(template, trace_id)
        is_open = trace_id in open_trace_ids if per_trace else bool(open_trace_ids)
        trajectory = build_trajectory(
            group,
            dataclasses.replace(
                options,
                media=media_sink(path.parent),
                complete=options.complete and not is_open,
            ),
        )
        if trajectory is None:
            continue
        write_json_atomic(path, trajectory)
        written.append(path)
    return written
