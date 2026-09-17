#!/usr/bin/env python3
"""Reproduce `config.KERNEL_SIGMA` by measurement, rather than by trusting a
comment describing how it was obtained.

`config.KERNEL_SIGMA` is Decision 1's one bandwidth (Eq. 14), measured once
by the median heuristic prior work already applies per batch
(`CREDA.models.CREDALoss._compute_sigma`): `sqrt(median(d) + 1e-6)` over the
off-diagonal pairwise squared Euclidean distances `d` of embeddings
`h = F_theta(x)`. The constant in `config.py` carried that description in
prose and nothing that would let a later session run it again; this is that
script, so the number has a script rather than only a paragraph.

**Exactly what is measured, in order:**

1. `torch.manual_seed(config.SEEDS[0])` — the global generator, seeded once,
   before anything else touches it. It matters only in principle: the
   encoder is `FeatureExtractor(backbone="resnet18", pretrained=True)`, and
   loading ImageNet weights overwrites whatever `nn.init` would have drawn,
   so the embeddings below do not actually depend on this seed. It is set
   anyway because `harness.run_one` always does, and this measurement claims
   to run in the regime training actually uses.
2. `bags.build("M", config.DATA_CACHE, config.SEEDS[0], 0.0)` — domain M
   (MNIST), seed 0, the clean pilot material (`noise=0.0`, `config.NOISE`'s
   own default). Its `train_idx` role is the 64 `TRAIN_BAGS`.
3. `wiring.build(<any arm>, ...)` — which arm does not matter. `Arm.__init__`
   constructs `self.encoder` first, before the head or any arm-specific
   attention parameters, so no randomness arm-specific plumbing might draw
   is ever consumed before the encoder exists, and the encoder's own weights
   come from the pretrained checkpoint rather than from the generator.
4. The model stays in **training mode** (`nn.Module`'s default after
   construction) — BatchNorm normalizes with each chunk's own batch
   statistics rather than accumulated running ones, which is the regime
   `harness.run_one` actually trains in and is NOT the regime an earlier,
   superseded measurement used (`eval()`, accumulated running statistics;
   see `config.py`'s history for that earlier number).
5. The 64 training bags — `INSTANCES_PER_BAG = 30` each, 1920 images total —
   walked in `train_idx`'s own sequential order (construction order, `0..63`;
   never the class-stratified shuffle `balanced_batches` draws), in
   `BAGS_PER_STEP = 10`-bag / 300-image chunks — the exact chunking
   `harness.accuracy` already uses for the same dataset shape. Each chunk is
   passed through `model.instance_embeddings`, under `torch.no_grad()`
   (nothing here trains; only BatchNorm's *mode* matters, not gradient
   tracking) and every chunk's embeddings are concatenated once measurement
   is done.
6. Every tensor stays in its natural dtype throughout — float32, the dtype
   `bags.build` already hands back and the dtype every campaign trains in
   unless a caller explicitly passes float64. This is "the regime training
   actually uses", stated literally: **not** `MIL_CREDA_Benchmark.DTYPE`,
   which governs only that package's own internal tensor construction and
   is unrelated to what a campaign trains with by default.
7. `sqrt(median(off-diagonal squared distances) + 1e-6)`, the same formula
   `CREDALoss._compute_sigma` applies, computed directly over the 1920
   embeddings against themselves rather than through the two-argument
   `_compute_sigma(x, y)` — that method concatenates `x` and `y` before
   measuring, which is the right shape for a source/target class-conditional
   pair inside the loss and the wrong shape here: concatenating this
   material with itself would put every embedding beside a zero-distance
   duplicate of itself, off the diagonal, and bias the median down.

**What this script does NOT claim.** Running it today measures
36.135013580322266 (float32) — 7.2e-7 away from the constant
`config.py` carries (36.135014304860874). Repeated on this machine the
figure is stable to every digit shown; a residual gap at that size is
consistent with floating-point summation order inside a multi-threaded BLAS
matrix multiply, which is not guaranteed bit-identical run to run, machine to
machine, or across a torch/torchvision point release even when every input
is unchanged. Whether that is the whole explanation or the original
measurement used a build of this stack with a bitwise-different GEMM
implementation is not established here — that would need running this exact
script under the original environment, which this session does not have
access to. What this script establishes is that the described procedure
reproduces the constant to eight significant figures, not that it reproduces
it exactly, and it prints the measured number rather than asserting either
way.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import torch  # noqa: E402

from MIL_CREDA_Benchmark import bags, config, wiring  # noqa: E402


def measure() -> float:
    """The bandwidth, measured by the procedure this module's docstring states."""
    torch.manual_seed(config.SEEDS[0])
    material = bags.build("M", config.DATA_CACHE, config.SEEDS[0], 0.0)
    train_ds, _, _ = bags.roles(material)

    pool = wiring.Pool(images=material.images, members=material.members,
                       labels=material.labels)
    arm_id = next(iter(config.ARMS_BY_ID))
    model = wiring.build(arm_id, config.CLASSES, pool, pool)
    model.train()

    chunks = []
    with torch.no_grad():
        for start in range(0, len(train_ds), config.BAGS_PER_STEP):
            items = [train_ds[i] for i in
                     range(start, min(start + config.BAGS_PER_STEP, len(train_ds)))]
            x = torch.stack([item[0] for item in items])
            h = model.instance_embeddings(x)
            chunks.append(h.reshape(-1, h.shape[-1]))

    H = torch.cat(chunks, dim=0)
    x_norm = (H ** 2).sum(dim=1).view(-1, 1)
    y_norm = (H ** 2).sum(dim=1).view(1, -1)
    dist_sq = torch.clamp(x_norm + y_norm - 2.0 * (H @ H.T), min=0.0)
    non_diag = dist_sq[~torch.eye(dist_sq.shape[0], dtype=bool)]
    return torch.sqrt(torch.median(non_diag) + 1e-6).item()


def main() -> int:
    measured = measure()
    print(f"measured KERNEL_SIGMA: {measured}")
    print(f"config.KERNEL_SIGMA:   {config.KERNEL_SIGMA}")
    print(f"difference:            {measured - config.KERNEL_SIGMA:+.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
