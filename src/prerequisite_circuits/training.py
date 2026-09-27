"""Small matched-branch runner; existing training loops can use Trajectory directly."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import numpy as np
import torch
from torch import Tensor, nn

from .monitor import Criterion, Observation, Trajectory
from .state import evaluation, preserve_rng, tensor_digest
from .tapes import Batch, ReplayTape


@dataclass(frozen=True)
class Branch:
    name: str
    intervention: Callable[[nn.Module, Batch], ContextManager] = lambda model, batch: nullcontext()


def run_branches(
    model: nn.Module,
    tape: ReplayTape,
    branches: Sequence[Branch],
    *,
    optimizer: Callable[[nn.Module], torch.optim.Optimizer],
    loss: Callable[[nn.Module, Batch], Tensor],
    observe: Callable[[nn.Module, int], Observation],
    criterion: Criterion = Criterion(),
    eval_every: int = 10,
    seed: int = 0,
    clip_norm: float | None = None,
) -> dict[str, Trajectory]:
    """Run one copy at a time with fresh, matched optimizers and the same tape.

    The caller owns device placement and any shared parameter freezing. This
    runner intentionally supports ordinary single-device FP32 training only;
    distributed/mixed-precision studies should attach the monitor to their loop.
    No optimizer state, scheduler, or automatic mixed precision is inherited.
    """
    if not branches or len({b.name for b in branches}) != len(branches):
        raise ValueError("Supply uniquely named branches")
    if isinstance(eval_every, bool) or not isinstance(eval_every, int) or eval_every < 1:
        raise ValueError("Evaluation interval must be a positive integer")
    if clip_norm is not None and (not math.isfinite(clip_norm) or clip_norm <= 0):
        raise ValueError("Clipping norm must be finite and positive")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("The reference runner supports float32 parameters only")
    reference = tensor_digest(model.state_dict())
    batch_hashes: list[str] = []
    results: dict[str, Trajectory] = {}
    optimizer_signature = None
    with preserve_rng():
        for arm in branches:
            current = copy.deepcopy(model)
            opt = optimizer(current)
            if opt.state:
                raise ValueError("This runner requires a fresh optimizer in every branch")
            names = {id(parameter): name for name, parameter in current.named_parameters()}
            groups = []
            seen = set()
            for group in opt.param_groups:
                members = []
                for parameter in group["params"]:
                    if id(parameter) not in names or id(parameter) in seen:
                        raise ValueError("Optimizer must own unique parameters from this branch")
                    seen.add(id(parameter))
                    members.append(names[id(parameter)])
                groups.append(
                    {
                        **{k: v for k, v in group.items() if k != "params"},
                        "parameter_names": members,
                    }
                )
            if tensor_digest(current.state_dict()) != reference:
                raise ValueError("Optimizer factory changed the initial model state")
            signature = {
                "class": f"{type(opt).__module__}.{type(opt).__qualname__}",
                "groups": groups,
            }
            if optimizer_signature is not None and signature != optimizer_signature:
                raise ValueError("Optimizer settings differ between branches")
            optimizer_signature = signature
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            trajectory = Trajectory(
                arm.name,
                criterion,
                {
                    "initial_state_sha256": reference,
                    "optimizer": signature,
                    "optimizer_state": "fresh",
                    "seed": seed,
                    "updates": tape.updates,
                    "eval_every": eval_every,
                    "evaluation": "caller-specified; training intervention removed",
                },
            )

            def measure(evaluated: nn.Module, step: int) -> None:
                with evaluation(evaluated):
                    reading = observe(evaluated, step)
                if reading.step != step:
                    raise ValueError("Observer returned the wrong update count")
                trajectory.append(reading)

            measure(current, 0)
            iterator = iter(tape.factory())
            digest = hashlib.sha256()
            for step in range(1, tape.updates + 1):
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise ValueError("Training tape ended before its declared length") from exc
                identity = batch.digest()
                if not results:
                    batch_hashes.append(identity)
                elif identity != batch_hashes[step - 1]:
                    raise ValueError(f"Training tape differs in branch {arm.name} at update {step}")
                digest.update(bytes.fromhex(identity))
                current.train()
                opt.zero_grad(set_to_none=True)
                with arm.intervention(current, batch):
                    objective = loss(current, batch)
                    if objective.ndim or not torch.isfinite(objective):
                        raise ValueError("Loss must be a finite scalar")
                    objective.backward()
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        current.parameters(), clip_norm, error_if_nonfinite=True
                    )
                elif any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in current.parameters()
                ):
                    raise FloatingPointError("Non-finite gradient")
                opt.step()
                if step % eval_every == 0 or step == tape.updates:
                    measure(current, step)
            trajectory.metadata["tape_sha256"] = digest.hexdigest()
            trajectory.metadata["endpoint_sha256"] = tensor_digest(current.state_dict())
            results[arm.name] = trajectory
            del current, opt
    return results
