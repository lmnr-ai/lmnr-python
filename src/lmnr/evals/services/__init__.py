"""Services: dependencies `EvalRunner.services` installs into the image and
starts before `setup` in every trial."""

from lmnr.evals.services.base import CustomService, Service
from lmnr.evals.services.clickhouse import ClickHouse
from lmnr.evals.services.postgres import Postgres
from lmnr.evals.services.valkey import Valkey

__all__ = ["ClickHouse", "CustomService", "Postgres", "Service", "Valkey"]
