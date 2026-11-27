"""Mirrors: stand-ins for SaaS APIs, declared in `EvalRunner.services`.

from lmnr.evals import case_fixture
from lmnr.evals.mirrors import Rystic

services = [Rystic("slack", version="1.4", seed=case_fixture("slack.json"),
                   env={"SLACK_API_URL": "{url}/api/"})]
"""

from lmnr.evals.mirrors.base import Mirror
from lmnr.evals.mirrors.rystic import Rystic

__all__ = ["Mirror", "Rystic"]
