from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import uuid

from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.datasets.file_utils import load_from_paths
from lmnr.sdk.datasets.seeded_perm import seeded_perm
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import Datapoint

DEFAULT_FETCH_SIZE = 25
LOG = get_default_logger(__name__, verbose=False)


class EvaluationDataset(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Datapoint:
        pass

    def slice(self, start: int, end: int):
        return [self[i] for i in range(max(start, 0), min(end, len(self)))]

    # --- base hooks forwarded by wrappers -------------------------------------

    def set_client(self, client: LaminarClient):
        """Inject the client into the underlying source (no-op by default).

        In-memory datasets need no client; ``LaminarDataset`` overrides this and
        subsampling wrappers forward it down to their base.
        """

    def source_dataset(self) -> "LaminarDataset | None":
        """Return the underlying remote ``LaminarDataset`` if any, else ``None``.

        Used by the eval integration to rebuild dataset-links through any depth of
        chaining without branching on the concrete dataset type.
        """
        return None

    # --- chainable, immutable subsampling -------------------------------------

    def take(self, n: int) -> "EvaluationDataset":
        """Return a new dataset with the first ``n`` datapoints (or all if ``n``
        exceeds the size)."""

        def resolver() -> list[int]:
            return list(range(min(max(n, 0), len(self))))

        return _SubsetDataset(self, resolver=resolver)

    def select(self, indices: list[int]) -> "EvaluationDataset":
        """Return a new dataset with exactly ``indices``, in that order.

        Raises ``IndexError`` at resolve time on any out-of-range index (no
        clamping, no negative indices)."""
        requested = list(indices)

        def resolver() -> list[int]:
            size = len(self)
            for i in requested:
                if i < 0 or i >= size:
                    raise IndexError(
                        f"select index {i} is out of range for dataset of size "
                        f"{size}"
                    )
            return requested

        return _SubsetDataset(self, resolver=resolver)

    def filter(
        self, predicate: Callable[[Datapoint], bool]
    ) -> "EvaluationDataset":
        """Return a new dataset keeping only datapoints where ``predicate`` is
        truthy, order preserved.

        Scans the whole dataset once (in pages, reusing the page-cached fetch),
        holding only surviving indices in memory. Emits a one-time debug log that
        it materializes the full dataset. The predicate is synchronous."""

        def resolver() -> list[int]:
            LOG.debug(
                "filter() scans the full dataset once to materialize surviving "
                "indices"
            )
            return [i for i in range(len(self)) if predicate(self[i])]

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
        self._base = base
        self._resolver = resolver
        self._indices: list[int] | None = None

    def _resolve(self) -> list[int]:
        if self._indices is None:
            self._indices = self._resolver()
        return self._indices

    def __len__(self) -> int:
        return len(self._resolve())

    def __getitem__(self, idx) -> Datapoint:
        indices = self._resolve()
        return self._base[indices[idx]]

    def set_client(self, client: LaminarClient):
        self._base.set_client(client)

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
        self.name = name
        self.id = id
        if name is None and id is None:
            raise ValueError("Either name or id must be provided")
        if name is not None and id is not None:
            raise ValueError("Only one of name or id must be provided")
        self._len: int | None = None
        # page index -> list of datapoints on that page. Each page is fetched at
        # most once, so arbitrary-index (random) access is correct and cheap.
        self._pages: dict[int, list[Datapoint]] = {}
        self._fetch_size = fetch_size
        self._logger = get_default_logger(self.__class__.__name__)

    def _fetch_page(self, page_index: int) -> list[Datapoint]:
        offset = page_index * self._fetch_size
        self._logger.debug(
            f"dataset name: {self.name}, id: {self.id}. Fetching page "
            f"{page_index} (offset {offset}, limit {self._fetch_size})"
        )
        identifier = {"id": self.id} if self.id is not None else {"name": self.name}
        resp = self.client.datasets.pull(
            **identifier,
            offset=offset,
            limit=self._fetch_size,
        )
        self._pages[page_index] = resp.items
        if self._len is None:
            self._len = resp.total_count
        return resp.items

    def __len__(self) -> int:
        if self._len is None:
            self._fetch_page(0)
        return self._len

    def __getitem__(self, idx) -> Datapoint:
        size = len(self)
        if idx < 0 or idx >= size:
            raise IndexError(
                f"dataset index {idx} is out of range for dataset of size {size}"
            )
        page_index = idx // self._fetch_size
        if page_index not in self._pages:
            self._fetch_page(page_index)
        page = self._pages[page_index]
        offset_in_page = idx - page_index * self._fetch_size
        if offset_in_page >= len(page):
            raise IndexError(
                f"dataset index {idx} is out of range for dataset of size {size}"
            )
        return page[offset_in_page]

    def set_client(self, client: LaminarClient):
        self.client = client

    def source_dataset(self) -> "LaminarDataset | None":
        return self

    def push(self, paths: str | list[str], recursive: bool = False):
        paths = [paths] if isinstance(paths, str) else paths
        paths = [Path(path) for path in paths]
        data = load_from_paths(paths, recursive)
        if len(data) == 0:
            LOG.warning("No data to push. Skipping")
            return
        identifier = {"id": self.id} if self.id is not None else {"name": self.name}
        self.client.datasets.push(data, **identifier)
        LOG.info(
            f"Successfully pushed {len(data)} datapoints to dataset [{identifier}]"
        )
