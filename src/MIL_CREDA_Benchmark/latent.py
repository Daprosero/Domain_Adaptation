"""The rows an arm actually aligns, read from its trained encoders.

`represent` is the one latent reader the toyset calls: it embeds a role of a
bag set through the encoder of the declared domain and returns one row per bag
(bag unit) or per instance (instance unit), with the class of each. The digit
report's geometry, separability, correspondence and checkpoint readers were
retired with the image benchmark.
"""

from __future__ import annotations

import torch

from MIL_CREDA_Benchmark import bags, config, wiring


@torch.no_grad()
def represent(model, bagset: bags.BagSet, positions: torch.Tensor,
              device: torch.device, domain: str,
              unit: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The rows this arm actually aligns, and the class of each.

    A bag-unit arm aligns bag representations, one row per subject. An
    instance-unit arm aligns instances, one row each, carrying the class of the
    bag they came from. Each is **measured** in its own unit and the two are never
    put in the same number.

    `unit` overrides that for drawing only. Every arm encodes instances, so
    forcing the instance unit gives a space every arm has and every panel of a
    grid the same number of points; forcing it for a measurement would put two
    families in one number, which is what the default exists to prevent.

    `domain` states which encoder embeds `bagset` -- `"source"` for
    `model.instance_embeddings` (theta_s), `"target"` for
    `model.target_embeddings` (theta_t). Required, with no default anywhere in
    the call chain (spec `per-domain-encoders` -- "No embedding without a
    declared domain"): a caller that cannot state which domain it holds is
    refused rather than defaulted to either encoder, because a silent default
    would return a plausible number instead of an error.
    """
    if domain not in wiring.SIDES:
        raise ValueError(f"unknown domain {domain!r}; known: {wiring.SIDES}")
    embed = model.instance_embeddings if domain == "source" else model.target_embeddings

    rows, labels = [], []
    for start in range(0, len(positions), config.BAGS_PER_STEP):
        chunk = positions[start:start + config.BAGS_PER_STEP]
        instances = bagset.images[bagset.members[chunk]].to(device)
        embeddings = embed(instances)
        classes = bagset.labels[chunk].to(device)
        if (unit or model.spec["unit"]) == "bag":
            Z, _ = model.bag_representations(embeddings, config.KERNEL_SIGMA)
            rows.append(Z)
            labels.append(classes)
        else:
            rows.append(embeddings.reshape(-1, embeddings.shape[-1]))
            labels.append(classes.repeat_interleave(embeddings.shape[1]))
    return torch.cat(rows).float().cpu(), torch.cat(labels).cpu()
