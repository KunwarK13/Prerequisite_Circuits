"""Replay actual tensor batches; check bytes rather than relying on a seed."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from torch import Tensor

from .state import tensor_digest


@dataclass(frozen=True)
class Batch:
    inputs: Tensor
    targets: Tensor
    scope: Tensor | None = None

    def clone(self) -> Batch:
        return Batch(
            self.inputs.clone(),
            self.targets.clone(),
            None if self.scope is None else self.scope.clone(),
        )

    def digest(self) -> str:
        tensors = {"inputs": self.inputs, "targets": self.targets}
        if self.scope is not None:
            tensors["scope"] = self.scope
        return tensor_digest(tensors)


@dataclass(frozen=True)
class ReplayTape:
    """A fresh iterator factory with a fixed number of updates.

    The runner compares actual batch digests between branches and rejects any
    mismatch before the mismatched update. Factory code should use local RNGs.
    """

    factory: Callable[[], Iterator[Batch]]
    updates: int

    def __post_init__(self) -> None:
        if isinstance(self.updates, bool) or not isinstance(self.updates, int) or self.updates < 1:
            raise ValueError("updates must be a positive integer")

    @classmethod
    def from_batches(cls, batches: list[Batch]) -> ReplayTape:
        stored = [b.clone() for b in batches]
        return cls(lambda: (b.clone() for b in stored), len(stored))
