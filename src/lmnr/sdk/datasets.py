from abc import ABC, abstractmethod
import logging

from .log import get_default_logger
from .laminar import Laminar as L
from .types import (
    Datapoint,
)

DEFAULT_FETCH_SIZE = 25


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
    def __init__(self, name: str, fetch_size: int = DEFAULT_FETCH_SIZE):
        self.name = name
        self._len = None
        self._fetched_items = []
        self._offset = 0
        self._fetch_size = fetch_size
        self._logger = get_default_logger(self.__class__.__name__)

    def _fetch_batch(self):
        self._logger.debug(
            f"dataset {self.name}. Fetching batch from {self._offset} to "
            + f"{self._offset + self._fetch_size}"
        )
        resp = L.get_datapoints(self.name, self._offset, self._fetch_size)
        self._fetched_items += resp.items
        self._offset = len(self._fetched_items)
        if self._len is None:
            self._len = resp.totalCount

    def __len__(self) -> int:
        if self._len is None:
            self._fetch_batch()
        return self._len

    def __getitem__(self, idx) -> Datapoint:
        if idx >= len(self._fetched_items):
            self._fetch_batch()
        return self._fetched_items[idx]
