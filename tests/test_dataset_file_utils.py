from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lmnr.sdk.datasets.file_utils import load_from_paths


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r", "\u0085", "\u2028", "\u2029", ""])
def test_csv_load_preserves_field_content(tmp_path: Path, separator: str) -> None:
    path = tmp_path / "dataset.csv"
    expected = [
        {"data": f'Question, with "quotes":{separator}你好', "target": f"first{separator}second"},
        {"data": "single line", "target": ""},
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["data", "target"])
        writer.writeheader()
        writer.writerows(expected)

    assert load_from_paths([path]) == expected
