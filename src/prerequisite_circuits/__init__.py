"""Measure intervention effectiveness over a learning trajectory."""

from importlib.metadata import PackageNotFoundError, version

from .monitor import Criterion, Observation, Trajectory

__all__ = ["Criterion", "Observation", "Trajectory"]
try:
    __version__ = version("prerequisite-circuits")
except PackageNotFoundError:
    __version__ = "0+uninstalled"
