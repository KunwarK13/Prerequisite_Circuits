"""Explicit, finite-horizon manipulation checks; no automatic causal verdicts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _probability(name: str, value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")


@dataclass(frozen=True)
class Criterion:
    """Accuracy-based probe criterion chosen before examining the trajectory.

    Both conditions must hold at an observation: low accuracy with the intervention
    active, and a sufficient drop relative to the same state's available computation.
    A probe is evidence about its measured behavior, not every implementation of it.
    """

    maximum_suppressed: float = 0.2
    minimum_drop: float = 0.3
    acquisition: float = 0.5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is None:
                raise ValueError(f"{name} is required")
            _probability(name, value)

    def satisfied(self, observation: Observation) -> bool | None:
        a, s = observation.prerequisite_available, observation.prerequisite_suppressed
        if a is None or s is None:
            return None
        return s <= self.maximum_suppressed and a - s >= self.minimum_drop


@dataclass(frozen=True)
class Observation:
    step: int
    target_available: float
    prerequisite_available: float | None = None
    prerequisite_suppressed: float | None = None
    target_suppressed: float | None = None
    background: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        for name, value in asdict(self).items():
            if name != "step":
                _probability(name, value)
        if self.target_available is None:
            raise ValueError("target_available is required")


@dataclass
class Trajectory:
    """Append observations from an existing training loop or import saved evidence."""

    run_id: str
    criterion: Criterion = field(default_factory=Criterion)
    metadata: dict[str, Any] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        existing, self.observations = self.observations, []
        for observation in existing:
            self.append(observation)
        json.dumps(self.metadata, allow_nan=False)

    def append(self, observation: Observation) -> None:
        if self.observations and observation.step <= self.observations[-1].step:
            raise ValueError("observations must have strictly increasing update counts")
        self.observations.append(observation)

    def summary(self) -> dict[str, Any]:
        checks = [(o.step, self.criterion.satisfied(o)) for o in self.observations]
        measured = [(step, ok) for step, ok in checks if ok is not None]
        first_loss = None
        previously_effective = False
        for step, ok in measured:
            if previously_effective and not ok:
                first_loss = step
                break
            previously_effective |= bool(ok)
        if not measured:
            status = "unmeasured"
        elif first_loss is not None:
            status = "effectiveness_lost_at_observed_step"
        elif all(ok for _, ok in measured):
            status = "effective_at_observed_steps"
        elif any(ok for _, ok in measured):
            status = "initially_ineffective_then_effective"
        else:
            status = "criterion_not_met"
        first_acquisition = next(
            (o.step for o in self.observations if o.target_available >= self.criterion.acquisition),
            None,
        )
        return {
            "run_id": self.run_id,
            "status": status,
            "observed_through": self.observations[-1].step if self.observations else None,
            "first_observed_acquisition": first_acquisition,
            "first_observed_loss_of_effectiveness": first_loss,
            "measured_checks": len(measured),
            "missing_checks": len(checks) - len(measured),
            "criterion": asdict(self.criterion),
            "training_intervention": self.metadata.get("training_intervention", "unspecified"),
            "diagnostic_intervention": self.metadata.get("diagnostic_intervention", "unspecified"),
            "interpretation": (
                "These checks describe the supplied probe at observed updates. They do not "
                "establish continuous suppression, causal mediation, or impossibility of learning."
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "criterion": asdict(self.criterion),
            "metadata": self.metadata,
            "observations": [asdict(o) for o in self.observations],
            "summary": self.summary(),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        descriptor, name = tempfile.mkstemp(prefix=".trajectory-", dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Trajectory:
        """Validate a record and recompute its summary from the observations."""
        if not isinstance(value, dict):
            raise ValueError("A trajectory must be a JSON object")
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported trajectory schema")
        return cls(
            value["run_id"],
            Criterion(**value["criterion"]),
            value.get("metadata", {}),
            [Observation(**o) for o in value["observations"]],
        )
