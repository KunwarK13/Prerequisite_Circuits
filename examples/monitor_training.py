"""Attach the monitor to a user-owned optimizer loop on a tiny CPU model."""

import torch
from torch import nn

from prerequisite_circuits import Criterion, Observation, Trajectory
from prerequisite_circuits.interventions import Channels
from prerequisite_circuits.state import evaluation

torch.manual_seed(7)
model = nn.Sequential(nn.Identity(), nn.Linear(2, 1, bias=False))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
probe = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
target = (probe[:, 0] > 0).float()
mask = Channels(model[0], [0], location="output")
monitor = Trajectory(
    "custom-loop",
    Criterion(0.5, 0.4, 0.95),
    {
        "example": "Illustrative CPU feature model, not paper evidence",
        "training_intervention": "Suppress first feature",
        "diagnostic_intervention": "Suppress first feature",
        "evaluation": "Training intervention removed",
    },
)


def accuracy(logits):
    return float(((logits.squeeze(-1) > 0).float() == target).float().mean())


for step in range(21):
    if step % 5 == 0:
        with evaluation(model):
            available_target = accuracy(model(probe))
            available_probe = accuracy(model[0](probe)[:, 0])
            with mask.apply():
                suppressed_target = accuracy(model(probe))
                suppressed_probe = accuracy(model[0](probe)[:, 0])
        monitor.append(
            Observation(
                step, available_target, available_probe, suppressed_probe, suppressed_target
            )
        )
    if step == 20:
        break
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with mask.apply():
        loss = nn.functional.binary_cross_entropy_with_logits(model(probe).squeeze(-1), target)
        loss.backward()
    optimizer.step()

monitor.save("outputs/custom-loop.json")
print(monitor.summary())
