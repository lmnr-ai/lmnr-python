import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from logging import Logger
from pathlib import Path
from typing import Any, cast

from typing_extensions import TypedDict, override

from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.datasets.file_utils import load_from_paths
from lmnr.sdk.datasets.seeded_perm import seeded_perm
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import Datapoint

DEFAULT_FETCH_SIZE = 25
LOG = get_default_logger(__name__, verbose=False)


class _ById(TypedDict):
    id: uuid.UUID


class _ByName(TypedDict):
    name: str | None


# A dataset is looked up by exactly one of id or name; this union lets
# `**identifier` type-check against the `id`/`name` keyword params below.
DatasetIdentifier = _ById | _ByName


def _require_dataset_integer(value: object, message: str) -> None:
    """Raise ``TypeError`` unless ``value`` is a plain ``int``.

    ``bool`` is a subclass of ``int``, but ``True`` / ``False`` are never a
    meaningful count or index, so they are rejected too instead of being
    silently read as 1 / 0.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(message)


class EvaluationDataset(ABC):
    @abstractmethod
    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Datapoint:
        pass

    def slice(self, start: int, end: int):
        return [self[i] for i in range(max(start, 0), min(end, len(self)))]

    def __iter__(self) -> Iterator[Datapoint]:
        """Iterate the dataset one datapoint at a time.

        Evals consume datasets through this, never by materializing them, so a
        subclass that can stream (see ``LaminarDataset``) overrides it to keep
        peak memory bounded. The default walks ``__getitem__`` so existing
        custom datasets, which only implement the size + by-index contract,
        keep working unchanged.
        """
        for i in range(len(self)):
            yield self[i]

    # --- base hooks forwarded by wrappers -------------------------------------

    def source_dataset(self) -> "LaminarDataset | None":
        """Return the underlying remote ``LaminarDataset`` if any, else ``None``.

        Used by the eval integration to rebuild dataset-links through any depth of
        chaining without branching on the concrete dataset type.
        """
        return None

    # --- chainable, immutable subsampling -------------------------------------

    def take(self, n: int) -> "EvaluationDataset":
        """Return a new dataset with the first ``n`` datapoints (or all if ``n``
        exceeds the size).

        Raises ``TypeError`` at resolve time when ``n`` is not an integer (no
        silent coercion)."""

        def resolver() -> list[int]:
            _require_dataset_integer(n, f"take count {n} is not an integer")
            return list(range(min(max(n, 0), len(self))))

        return _SubsetDataset(self, resolver=resolver)

    def select(self, indices: list[int]) -> "EvaluationDataset":
        """Return a new dataset with exactly ``indices``, in that order.

        Raises ``TypeError`` at resolve time on any non-integer index, and
        ``IndexError`` on any out-of-range index (no clamping, no negative
        indices)."""
        requested = list(indices)

        def resolver() -> list[int]:
            size = len(self)
            for i in requested:
                _require_dataset_integer(i, f"select index {i} is not an integer")
                if i < 0 or i >= size:
                    raise IndexError(
                        f"select index {i} is out of range for dataset of size " + f"{size}"
                    )
            return requested

        return _SubsetDataset(self, resolver=resolver)

    def shuffle(self, seed: int = 0) -> "EvaluationDataset":
        """Return a new dataset in a reproducible random order.

        The permutation is a pure function of ``(size, seed)`` — the same seed
        always yields the same order."""

        def resolver() -> list[int]:
            return seeded_perm(len(self), seed)

        return _SubsetDataset(self, resolver=resolver)


class _SubsetDataset(EvaluationDataset):
    """An immutable view over a base dataset defined by a list of indices.

    Each subsampling op returns one of these. Indices are resolved lazily (once)
    and cached; element access delegates down the chain. Because a wrapper only
    ever reasons about indices into its *immediate* base, chains of any order
    compose with no special cases."""

    def __init__(
        self,
        base: EvaluationDataset,
        resolver: Callable[[], list[int]],
    ):
        self._base: EvaluationDataset = base
        self._resolver: Callable[[], list[int]] = resolver
        self._indices: list[int] | None = None

    def _resolve(self) -> list[int]:
        if self._indices is None:
            self._indices = self._resolver()
        return self._indices

    @override
    def __len__(self) -> int:
        return len(self._resolve())

    @override
    def __getitem__(self, idx: int) -> Datapoint:
        indices = self._resolve()
        return self._base[indices[idx]]

    @override
    def source_dataset(self) -> "LaminarDataset | None":
        return self._base.source_dataset()


class LaminarDataset(EvaluationDataset):
    client: LaminarClient
    id: uuid.UUID | None = None

    def __init__(
        self,
        name: str | None = None,
        id: uuid.UUID | None = None,
        fetch_size: int = DEFAULT_FETCH_SIZE,
    ):
        self.name: str | None = name
        self.id = id
        if name is None and id is None:
            raise ValueError("Either name or id must be provided")
        if name is not None and id is not None:
            raise ValueError("Only one of name or id must be provided")
        self._len: int | None = None
        # page index -> list of datapoints on that page. Each page is fetched at
        # most once, so arbitrary-index (random) access is correct and cheap.
        self._pages: dict[int, list[Datapoint]] = {}
        self._fetch_size: int = fetch_size
        self._logger: Logger = get_default_logger(self.__class__.__name__)

    def _pull_page(self, page_index: int) -> list[Datapoint]:
        """Fetch one page from the API WITHOUT caching it.

        ``_len`` is still learned from the response, since it costs nothing and
        is what lets `__iter__` know when to stop.
        """
        offset = page_index * self._fetch_size
        self._logger.debug(
            f"dataset name: {self.name}, id: {self.id}. Fetching page "
            + f"{page_index} (offset {offset}, limit {self._fetch_size})"
        )
        if self.id is not None:
            resp = self.client.datasets.pull(
                id=self.id,
                offset=offset,
                limit=self._fetch_size,
            )
        else:
            resp = self.client.datasets.pull(
                name=self.name,
                offset=offset,
                limit=self._fetch_size,
            )
        if self._len is None:
            self._len = resp["total_count"]
        return resp["items"]

    def _fetch_page(self, page_index: int) -> list[Datapoint]:
        page = self._pull_page(page_index)
        self._pages[page_index] = page
        return page

    @override
    def __len__(self) -> int:
        if self._len is None:
            _fetch_result = self._fetch_page(0)
        # _fetch_page always sets _len from the response's (non-optional)
        # total_count, so it is resolved to an int by this point.
        assert self._len is not None
        return self._len

    @override
    def __getitem__(self, idx: int) -> Datapoint:
        size = len(self)
        if idx < 0 or idx >= size:
            raise IndexError(f"dataset index {idx} is out of range for dataset of size {size}")
        page_index = idx // self._fetch_size
        if page_index not in self._pages:
            _fetch_result = self._fetch_page(page_index)
        page = self._pages[page_index]
        offset_in_page = idx - page_index * self._fetch_size
        if offset_in_page >= len(page):
            raise IndexError(f"dataset index {idx} is out of range for dataset of size {size}")
        return page[offset_in_page]

    @override
    def __iter__(self) -> Iterator[Datapoint]:
        """Stream the dataset page by page, retaining at most one page.

        Pages already cached by `__len__` / `__getitem__` are served from
        memory; every other page is pulled, yielded and then dropped, so a full
        eval run over a huge dataset never holds more than `fetch_size`
        datapoints at once. `__getitem__` / `__len__` keep their
        accumulate-and-cache behavior for random access.
        """
        page_index = 0
        yielded = 0
        while self._len is None or yielded < self._len:
            page = self._pages.get(page_index)
            if page is None:
                page = self._pull_page(page_index)
            # A short page before `_len` datapoints have been seen means the
            # dataset shrank (or the server disagrees with its own
            # total_count); stop rather than spin on empty pages.
            if not page:
                return
            yield from page
            yielded += len(page)
            page_index += 1

    def set_client(self, client: LaminarClient):
        self.client = client

    @override
    def source_dataset(self) -> "LaminarDataset | None":
        return self

    def push(self, paths: str | list[str], recursive: bool = False):
        path_strs = [paths] if isinstance(paths, str) else paths
        path_objs = [Path(path) for path in path_strs]
        data = cast(list[Datapoint], load_from_paths(path_objs, recursive))
        if len(data) == 0:
            LOG.warning("No data to push. Skipping")
            return
        identifier: DatasetIdentifier = (
            {"id": self.id} if self.id is not None else {"name": self.name}
        )
        if self.id is not None:
            _push_result = self.client.datasets.push(data, id=self.id)
        else:
            _push_result = self.client.datasets.push(data, name=self.name)
        LOG.info(f"Successfully pushed {len(data)} datapoints to dataset [{identifier}]")
