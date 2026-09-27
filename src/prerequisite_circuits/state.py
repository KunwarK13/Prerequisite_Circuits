"""Observation contexts that preserve randomness, buffers, and module modes."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor, nn


def tensor_digest(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def preserve_rng() -> Iterator[None]:
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextmanager
def evaluation(model: nn.Module) -> Iterator[None]:
    """Restore buffers/modes even on failure; callbacks must not edit parameters."""
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    versions = [(parameter, parameter._version) for parameter in model.parameters()]
    try:
        with preserve_rng(), torch.no_grad():
            model.eval()
            yield
            if any(p._version != version for p, version in versions):
                raise RuntimeError("An evaluation callback modified model parameters")
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for module, mode in modes:
            module.training = mode
