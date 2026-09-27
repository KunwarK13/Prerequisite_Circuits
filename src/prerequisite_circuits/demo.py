"""Illustrative fixed-feature experiment, not an additional paper result."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .interventions import Channels
from .monitor import Criterion, Observation
from .state import preserve_rng
from .tapes import Batch, ReplayTape
from .training import Branch, run_branches


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Identity()
        self.readout = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self, inputs):
        return self.readout(self.features(inputs)).squeeze(-1)


def run():
    """Use a fixed informative feature, an independent control, and common evaluation."""
    with preserve_rng():
        generator = torch.Generator().manual_seed(17)
        inputs = torch.randn(128, 2, generator=generator)
        targets = (inputs[:, 0] > 0).float()
        probe = torch.tensor([[a, b] for a in (-1.0, 1.0) for b in (-1.0, 1.0)])
        labels = (probe[:, 0] > 0).float()
        tape = ReplayTape.from_batches([Batch(inputs, targets) for _ in range(60)])

        @contextmanager
        def altered(model, batch, channel, supply=False, wrong=False):
            donor = batch.inputs.flip(-1) if wrong else batch.inputs
            with Channels(model.features, [channel], location="output").apply(
                replacement=donor if supply else None
            ):
                yield

        def observe(model, step):
            def accuracy(logits):
                return float(((logits > 0).float() == labels).float().mean())

            available = accuracy(model(probe))
            # The probe reads the declared feature computation directly.
            prerequisite = accuracy(model.features(probe)[:, 0])
            with Channels(model.features, [0], location="output").apply():
                suppressed_target = accuracy(model(probe))
                suppressed_prerequisite = accuracy(model.features(probe)[:, 0])
            return Observation(
                step, available, prerequisite, suppressed_prerequisite, suppressed_target
            )

        result = run_branches(
            FeatureModel(),
            tape,
            [
                Branch("available"),
                Branch("suppressed", lambda m, b: altered(m, b, 0)),
                Branch("control", lambda m, b: altered(m, b, 1)),
                Branch("supplied", lambda m, b: altered(m, b, 0, supply=True)),
                Branch("wrong_supply", lambda m, b: altered(m, b, 0, supply=True, wrong=True)),
            ],
            optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.15),
            loss=lambda m, b: F.binary_cross_entropy_with_logits(m(b.inputs), b.targets),
            observe=observe,
            criterion=Criterion(0.5, 0.4, 0.95),
            eval_every=10,
        )
        for trajectory in result.values():
            trajectory.metadata["example"] = "Illustrative fixed-feature model; not paper evidence"
            trajectory.metadata["probe"] = "Balanced sign readout of the fixed first input feature"
            trajectory.metadata["training_intervention"] = trajectory.run_id
            trajectory.metadata["diagnostic_intervention"] = "Suppress first feature in every arm"
        return result
