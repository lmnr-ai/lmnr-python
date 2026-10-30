"""lmnr-atif: Laminar's extension to ATIF trajectories (`spec/lmnr-atif/`)."""

from lmnr.sdk.trajectory.reader import TrajectoryExtension, is_compatible
from lmnr.sdk.trajectory.spec import ATIF_SCHEMA_VERSION, NAMESPACE, SPEC_VERSION

__all__ = [
    "ATIF_SCHEMA_VERSION",
    "NAMESPACE",
    "SPEC_VERSION",
    "TrajectoryExtension",
    "is_compatible",
]
