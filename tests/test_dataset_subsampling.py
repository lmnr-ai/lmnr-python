import json
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from lmnr import Laminar
from lmnr.sdk.datasets import EvaluationDataset, LaminarDataset
from lmnr.sdk.datasets.seeded_perm import seeded_perm
from lmnr.sdk.evaluations import evaluate
from lmnr.sdk.types import Datapoint


FIXTURE = (
    Path(__file__).parent / "data" / "dataset" / "seeded_perm_cases.json"
)


def make_dp(i: int) -> Datapoint:
    return Datapoint(
        id=uuid.uuid4(),
        data=i,
        target=i,
        createdAt=datetime.now(),
    )


class InMemoryDataset(EvaluationDataset):
    """Tiny in-memory dataset implementing the base (size + by-index) contract."""

    def __init__(self, datapoints):
        self._dp = list(datapoints)

    def __len__(self) -> int:
        return len(self._dp)

    def __getitem__(self, idx) -> Datapoint:
        return self._dp[idx]


def values(ds: EvaluationDataset) -> list:
    return [ds[i].data for i in range(len(ds))]


# --- take -------------------------------------------------------------------


def test_take_first_n():
    ds = InMemoryDataset([make_dp(i) for i in range(5)])
    assert values(ds.take(3)) == [0, 1, 2]


def test_take_exceeds_size_returns_all():
    ds = InMemoryDataset([make_dp(i) for i in range(3)])
    assert values(ds.take(10)) == [0, 1, 2]


def test_take_returns_new_dataset_original_unchanged():
    ds = InMemoryDataset([make_dp(i) for i in range(5)])
    taken = ds.take(2)
    assert len(taken) == 2
    assert len(ds) == 5  # original unchanged
    assert taken is not ds


# --- select -----------------------------------------------------------------


def test_select_order():
    ds = InMemoryDataset([make_dp(i) for i in range(5)])
    assert values(ds.select([4, 0, 2])) == [4, 0, 2]


def test_select_out_of_range_raises():
    ds = InMemoryDataset([make_dp(i) for i in range(3)])
    with pytest.raises(IndexError) as exc:
        len(ds.select([0, 5]))
    assert "5" in str(exc.value) and "3" in str(exc.value)


def test_select_negative_raises():
    ds = InMemoryDataset([make_dp(i) for i in range(3)])
    with pytest.raises(IndexError):
        len(ds.select([-1]))


# --- filter -----------------------------------------------------------------


def test_filter_predicate():
    ds = InMemoryDataset([make_dp(i) for i in range(6)])
    even = ds.filter(lambda dp: dp.data % 2 == 0)
    assert values(even) == [0, 2, 4]


def test_filter_then_take():
    ds = InMemoryDataset([make_dp(i) for i in range(10)])
    assert values(ds.filter(lambda dp: dp.data % 2 == 1).take(2)) == [1, 3]


def test_filter_scans_once_and_caches():
    ds = InMemoryDataset([make_dp(i) for i in range(6)])
    calls = {"n": 0}

    def pred(dp):
        calls["n"] += 1
        return True

    filtered = ds.filter(pred)
    # Multiple accesses must not re-scan.
    _ = len(filtered)
    _ = filtered[0]
    _ = values(filtered)
    assert calls["n"] == 6  # exactly one scan of the base


# --- shuffle ----------------------------------------------------------------


def test_shuffle_deterministic_same_seed():
    ds = InMemoryDataset([make_dp(i) for i in range(10)])
    assert values(ds.shuffle(seed=42)) == values(ds.shuffle(seed=42))


def test_shuffle_matches_seeded_perm():
    ds = InMemoryDataset([make_dp(i) for i in range(10)])
    assert values(ds.shuffle(seed=42)) == seeded_perm(10, 42)


def test_shuffle_different_seed():
    ds = InMemoryDataset([make_dp(i) for i in range(10)])
    assert values(ds.shuffle(seed=1)) != values(ds.shuffle(seed=2))


def test_shuffle_original_unchanged():
    ds = InMemoryDataset([make_dp(i) for i in range(5)])
    _ = ds.shuffle(seed=1)
    assert values(ds) == [0, 1, 2, 3, 4]


# --- chain composition (order matters) --------------------------------------


def test_shuffle_then_take_vs_take_then_shuffle():
    ds = InMemoryDataset([make_dp(i) for i in range(10)])
    # shuffle then take = a random N of the whole set
    random_n = values(ds.shuffle(seed=7).take(3))
    assert random_n == seeded_perm(10, 7)[:3]
    # take then shuffle = shuffle of the first N
    shuffle_of_first = values(ds.take(3).shuffle(seed=7))
    first3 = [0, 1, 2]
    assert shuffle_of_first == [first3[i] for i in seeded_perm(3, 7)]
    # the two differ in general
    assert random_n != shuffle_of_first


def test_chain_take_then_select_indexes_into_subset():
    ds = InMemoryDataset([make_dp(i) for i in range(10)])
    # take(5) -> [0..4]; select([4,0]) indexes into that subset
    assert values(ds.take(5).select([4, 0])) == [4, 0]


def test_filter_then_shuffle_then_take():
    ds = InMemoryDataset([make_dp(i) for i in range(20)])
    evens = list(range(0, 20, 2))  # [0,2,...,18], 10 elements
    chained = ds.filter(lambda dp: dp.data % 2 == 0).shuffle(seed=3).take(4)
    expected = [evens[i] for i in seeded_perm(10, 3)[:4]]
    assert values(chained) == expected


# --- base hooks: source_dataset + set_client forwarding ---------------------


def test_source_dataset_none_for_in_memory():
    ds = InMemoryDataset([make_dp(i) for i in range(3)])
    assert ds.source_dataset() is None
    assert ds.shuffle(seed=1).take(2).source_dataset() is None


def test_source_dataset_through_chain():
    base = LaminarDataset(id=uuid.uuid4())
    chained = base.shuffle(seed=1).take(2).select([0])
    assert chained.source_dataset() is base


def test_set_client_forwards_through_chain():
    base = LaminarDataset(id=uuid.uuid4())
    chained = base.shuffle(seed=1).take(2)
    client = MagicMock()
    chained.set_client(client)
    assert base.client is client


# --- page-cached random access (LaminarDataset) -----------------------------


def _paged_laminar_dataset(total: int, fetch_size: int):
    ds = LaminarDataset(name="d", fetch_size=fetch_size)
    all_items = [make_dp(i) for i in range(total)]
    client = MagicMock()

    def pull(name=None, id=None, offset=0, limit=fetch_size):
        resp = MagicMock()
        resp.items = all_items[offset : offset + limit]
        resp.total_count = total
        return resp

    client.datasets.pull.side_effect = pull
    ds.set_client(client)
    return ds, client, all_items


def test_random_access_returns_correct_datapoint():
    ds, _, all_items = _paged_laminar_dataset(total=5, fetch_size=2)
    assert ds[4].data == all_items[4].data
    assert ds[0].data == all_items[0].data


def test_len_backfilled_from_first_page():
    ds, client, _ = _paged_laminar_dataset(total=5, fetch_size=2)
    assert len(ds) == 5
    assert client.datasets.pull.call_count == 1


def test_page_fetched_at_most_once():
    ds, client, _ = _paged_laminar_dataset(total=5, fetch_size=2)
    _ = ds[0]  # len -> page 0
    _ = ds[1]  # page 0 (cached)
    _ = ds[4]  # page 2
    # re-access everything, no new fetches
    _ = ds[0]
    _ = ds[1]
    _ = ds[4]
    assert client.datasets.pull.call_count == 2  # page 0 + page 2


def test_out_of_range_raises():
    ds, _, _ = _paged_laminar_dataset(total=5, fetch_size=2)
    with pytest.raises(IndexError):
        _ = ds[5]


def test_shuffle_over_paged_dataset():
    ds, _, all_items = _paged_laminar_dataset(total=5, fetch_size=2)
    perm = seeded_perm(5, 1)
    assert values(ds.shuffle(seed=1)) == [all_items[i].data for i in perm]


# --- parity fixture ---------------------------------------------------------


def test_seeded_perm_matches_fixture():
    cases = json.loads(FIXTURE.read_text())
    assert len(cases) >= 4
    for case in cases:
        assert seeded_perm(case["n"], case["seed"]) == case["permutation"]


def test_seeded_perm_edge_cases():
    assert seeded_perm(0, 0) == []
    assert seeded_perm(1, 123) == [0]


# --- eval integration: dataset-link survives chaining -----------------------


@pytest.mark.asyncio
@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_eval_over_chained_dataset_has_dataset_link(
    mock_init,
    mock_save_datapoints,
    mock_pull,
    span_exporter: InMemorySpanExporter,
):
    dataset_id = uuid.uuid4()
    all_items = [make_dp(i) for i in range(5)]

    eval_resp = MagicMock()
    eval_resp.id = "00000000-0000-0000-0000-000000000000"
    eval_resp.projectId = "mock-project-id"
    mock_init.return_value = eval_resp

    def pull(name=None, id=None, offset=0, limit=25):
        resp = MagicMock()
        resp.items = all_items[offset : offset + limit]
        resp.total_count = len(all_items)
        return resp

    mock_pull.side_effect = pull

    data = LaminarDataset(id=dataset_id).shuffle(seed=1).take(2)

    await evaluate(
        data=data,
        executor=lambda d: d,
        evaluators={"test": lambda output, target: 1},
        project_api_key="test",
    )
    Laminar.flush()

    # Collect every datapoint passed to save_datapoints across all calls.
    seen_links = []
    processed = 0
    for call in mock_save_datapoints.call_args_list:
        datapoints = call.args[1]
        for dp in datapoints:
            processed += 1
            if getattr(dp, "dataset_link", None) is not None:
                seen_links.append(dp.dataset_link)

    assert seen_links, "expected dataset_link on chained-dataset result datapoints"
    for link in seen_links:
        assert link.dataset_id == dataset_id
    # shuffle(seed=1).take(2) over 5 datapoints -> 2 processed datapoints,
    # each saved twice (partial + final).
    assert processed == 4
