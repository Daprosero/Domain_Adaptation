"""MIL-CREDA: reference implementation of the managed proposal.

One module per mathematical object. Every module declares `__provenance__`
binding it to the revision it was written against.

The numeric backend is PyTorch. That is a change of backend, not of method:
every equation is the same equation, but each term of Eq. (39) now carries a
gradient and can be placed on a device, which is the difference between an
objective that is evaluated and one that is trained.

This file holds the only thing here that is not a mathematical object — how a
caller's input becomes a tensor — precisely so that no module of mathematics has
to spend lines on plumbing.

Double precision is deliberate. The invariants of Section 2 are checked at
tolerances near 1e-12, and single precision cannot resolve them: a bound that
holds only to 1e-6 is not the bound the proposal states. A caller who trains in
single precision passes float32 tensors and gets them back respected.
"""

from __future__ import annotations

import torch

#: The dtype used for anything this package has to materialize itself.
DTYPE = torch.float64


def as_tensor(value: object, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Accept whatever the caller has; return a tensor without disturbing one.

    A tensor is returned untouched. Re-casting it would silently detach it from
    the graph that produced it, or pull a device-resident term back to the host,
    and either one turns a trainable implementation back into a calculator.
    Anything else — a list, an array, a Python float — becomes a tensor.
    """
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=DTYPE if dtype is None else dtype)


def as_matrix(value: object) -> torch.Tensor:
    """`as_tensor`, then at least two dimensions: one row per sample."""
    tensor = as_tensor(value)
    return tensor.reshape(1, -1) if tensor.dim() < 2 else tensor


def as_index(value: object, *, device: torch.device | None = None) -> torch.Tensor:
    """An index set or a label vector as a flat `long` tensor.

    `device` moves it next to the matrix it will index: advanced indexing
    requires both to live in the same place, and the caller's index set is
    usually built on the host even when the kernel matrices are not.
    """
    if isinstance(value, torch.Tensor):
        index = value.reshape(-1).long()
    else:
        index = torch.as_tensor(value, dtype=torch.long).reshape(-1)
    return index if device is None else index.to(device)
