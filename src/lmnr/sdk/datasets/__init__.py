from abc import ABC, abstractmethod
from pathlib import Path

import uuid

from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.datasets.file_utils import load_from_paths
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
        self._len = None
        self._fetched_items = []
        self._offset = 0
        self._fetch_size = fetch_size
        self._logger = get_default_logger(self.__class__.__name__)

    def _fetch_batch(self):
        self._logger.debug(
            f"dataset name: {self.name}, id: {self.id}. Fetching batch from {self._offset} to "
            + f"{self._offset + self._fetch_size}"
        )
        identifier = {"id": self.id} if self.id is not None else {"name": self.name}
        resp = self.client.datasets.pull(
            **identifier,
            offset=self._offset,
            limit=self._fetch_size,
        )
        self._fetched_items += resp.items
        self._offset = len(self._fetched_items)
        if self._len is None:
            self._len = resp.total_count

    def __len__(self) -> int:
        if self._len is None:
            self._fetch_batch()
        return self._len

    def __getitem__(self, idx) -> Datapoint:
        if idx >= len(self._fetched_items):
            self._fetch_batch()
        return self._fetched_items[idx]

    def set_client(self, client: LaminarClient):
        self.client = client

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
