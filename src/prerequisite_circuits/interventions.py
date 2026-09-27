"""Scoped channel interventions with explicit donor and gradient semantics."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager

import torch
from torch import Tensor, nn


class Channels:
    """Replace selected last-axis channels on a module input or output.

    Replacement tensors describe the full activation and are detached by default.
    Scope is a Boolean tensor matching the leading activation dimensions, or a
    one-dimensional row mask. Hooks exist only inside ``apply``.
    """

    def __init__(self, module: nn.Module, channels: Sequence[int], *, location: str = "input"):
        if location not in {"input", "output"}:
            raise ValueError("location must be 'input' or 'output'")
        selected = tuple(channels)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("channels must be nonempty and unique")
        if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in selected):
            raise ValueError("channels must be nonnegative integers")
        self.module, self.channels, self.location = module, selected, location
        self._active = False

    @contextmanager
    def apply(
        self,
        *,
        scope: Tensor | None = None,
        replacement: Tensor | None = None,
        detach_replacement: bool = True,
    ) -> Iterator[None]:
        if self._active:
            raise RuntimeError("The same intervention cannot be nested")
        if scope is not None and scope.dtype != torch.bool:
            raise ValueError("scope must have Boolean dtype")

        def replace(value: Tensor) -> Tensor:
            if not isinstance(value, Tensor) or value.ndim < 2:
                raise TypeError("Expected a tensor with batch and channel dimensions")
            if max(self.channels) >= value.shape[-1]:
                raise ValueError("A selected channel is outside the activation width")
            mask = torch.zeros(value.shape[-1], device=value.device, dtype=torch.bool)
            mask[list(self.channels)] = True
            if scope is not None:
                rows = scope.to(value.device)
                if rows.shape == (value.shape[0],):
                    rows = rows.reshape(value.shape[0], *([1] * (value.ndim - 1)))
                elif rows.shape == value.shape[:-1]:
                    rows = rows.unsqueeze(-1)
                else:
                    raise ValueError("scope does not match activation rows or token positions")
                mask = mask & rows
            if replacement is None:
                donor = torch.zeros((), device=value.device, dtype=value.dtype)
            else:
                if replacement.shape != value.shape:
                    raise ValueError("replacement must have the full activation shape")
                donor = replacement.detach() if detach_replacement else replacement
                donor = donor.to(device=value.device, dtype=value.dtype)
            return torch.where(mask, donor, value)

        if self.location == "input":
            handle = self.module.register_forward_pre_hook(
                lambda module, args: (replace(args[0]), *args[1:])
            )
        else:
            handle = self.module.register_forward_hook(lambda module, args, output: replace(output))
        self._active = True
        try:
            yield
        finally:
            handle.remove()
            self._active = False


class NeoXHeads:
    """Intervene on GPT-NeoX head contributions immediately before output projection.

    This removes complete selected head contributions. It does not isolate only
    the induction computation performed by those heads.
    """

    def __init__(self, model: nn.Module, heads: Sequence[tuple[int, int]]):
        pairs = tuple(heads)
        if not pairs or len(set(pairs)) != len(pairs):
            raise ValueError("heads must be nonempty and unique")
        count, hidden = model.config.num_attention_heads, model.config.hidden_size
        if hidden % count:
            raise ValueError("Hidden width is not divisible by head count")
        width = hidden // count
        grouped: dict[int, list[int]] = {}
        for layer, head in pairs:
            if any(isinstance(i, bool) or not isinstance(i, int) for i in (layer, head)):
                raise ValueError("Layer and head indices must be integers")
            if not 0 <= layer < len(model.gpt_neox.layers) or not 0 <= head < count:
                raise ValueError("Head index outside model")
            grouped.setdefault(layer, []).extend(range(head * width, (head + 1) * width))
        self.interventions = {
            layer: Channels(model.gpt_neox.layers[layer].attention.dense, indices)
            for layer, indices in grouped.items()
        }

    @contextmanager
    def apply(
        self,
        *,
        scope: Tensor | None = None,
        replacements: Mapping[int, Tensor] | None = None,
    ) -> Iterator[None]:
        if replacements is not None and set(replacements) != set(self.interventions):
            raise ValueError("Supply exactly one donor activation for each intervened layer")
        with ExitStack() as stack:
            for layer, intervention in self.interventions.items():
                stack.enter_context(
                    intervention.apply(
                        scope=scope,
                        replacement=None if replacements is None else replacements[layer],
                    )
                )
            yield


@contextmanager
def capture(module: nn.Module, *, location: str = "input") -> Iterator[list[Tensor]]:
    """Capture detached copies, one per invocation; no hooks survive the context."""
    values: list[Tensor] = []
    if location == "input":
        handle = module.register_forward_pre_hook(
            lambda m, args: values.append(args[0].detach().clone())
        )
    elif location == "output":
        handle = module.register_forward_hook(
            lambda m, args, out: values.append(out.detach().clone())
        )
    else:
        raise ValueError("location must be 'input' or 'output'")
    try:
        yield values
    finally:
        handle.remove()
