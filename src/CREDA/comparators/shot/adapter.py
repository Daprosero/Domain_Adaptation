"""SHOT adapter (Liang, Hu, Feng, ICML 2020) over the vendored tim-learn/SHOT code.

Per-batch term (`__call__`): the information-maximisation loss of
object/image_target.py L199-206 on the target logits `g_t`, plus -- when
`pseudo_labels` is given -- the self-supervised pseudo-label cross-entropy of
L190-197 scaled by `cls_par`. The sum is exactly upstream's `classifier_loss`
before backward (L190-207). SHOT is source-free: `f_s`, `g_s`, `y_s`, `f_t` are
accepted for the common signature and ignored.

NOT a per-batch loss: `obtain_label` (L242-296) is a per-epoch procedure that
needs a full pass over the target set (feature-space clustering seeded by the
predictions). Upstream calls it every `interval_iter` iterations (L174-179); the
caller owns that schedule and passes its output as `pseudo_labels`.

Pseudo-label indexing. `obtain_label` returns one label per target sample (the
whole dataset, length N). Upstream reads the batch's labels as
`mem_label[tar_idx]` (L190-191), where `tar_idx` are the batch's dataset indices
from the target loader. The caller passes both: `pseudo_labels=<obtain_label
output>, tar_idx=<batch indices>`. Without `tar_idx`, `pseudo_labels` is taken
as already batch-aligned and must have exactly B entries; any other length
raises ValueError instead of silently applying the first B labels to every
batch. `tar_idx` must have B entries.

Upstream defaults (argparse, L314-326): cls_par=0.3, ent_par=1.0, epsilon=1e-5,
gent=True, ent=True, threshold=0, distance='cosine'.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import torch

from .upstream_image_target import im_loss_fn, obtain_label, pseudo_label_loss


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        'third-party baseline: SHOT (Liang et al., ICML 2020), Eq. 3 and Eq. '
        '7 through the vendored tim-learn/SHOT@f7d555a code; it implements no '
        'equation of the CREDA research concept (see CITATIONS.md)'
    ),
    "invariants": [
        "shot_obtain_label_runs_as_a_per_epoch_procedure",
        "whole_dataset_labels_with_tar_idx_equal_pre_indexed_labels",
        "whole_dataset_labels_without_tar_idx_raise",
        "default_call_without_pseudo_labels_is_unchanged",
    ],
}


class SHOT:
    def __init__(self, feature_dim: int, num_classes: int, cls_par: float = 0.3, ent_par: float = 1.0,
                 epsilon: float = 1e-5, gent: bool = True, ent: bool = True, threshold: int = 0,
                 distance: str = "cosine") -> None:
        self.args = SimpleNamespace(cls_par=cls_par, ent_par=ent_par, epsilon=epsilon, gent=gent, ent=ent,
                                    threshold=threshold, distance=distance, class_num=num_classes,
                                    dset="", out_file=io.StringIO())

    def __call__(self, f_s, f_t, g_s, g_t, y_s, pseudo_labels=None, tar_idx=None, **extra):
        loss = im_loss_fn(g_t, self.args)
        if pseudo_labels is not None:
            batch = g_t.size(0)
            if tar_idx is None:
                if pseudo_labels.size(0) != batch:
                    raise ValueError(
                        f"SHOT: pseudo_labels has {pseudo_labels.size(0)} entries but the batch has {batch}; "
                        "pass the whole-dataset labels from obtain_label together with the batch's dataset "
                        "indices as tar_idx (upstream mem_label[tar_idx]), or pass batch-aligned labels.")
                idx = torch.arange(batch, device=pseudo_labels.device)
            else:
                idx = torch.as_tensor(tar_idx, device=pseudo_labels.device).long()
                if idx.dim() != 1 or idx.size(0) != batch:
                    raise ValueError(
                        f"SHOT: tar_idx must be a 1-d index of length {batch} (the batch size), got shape "
                        f"{tuple(idx.shape)}.")
            # iter_num/interval_iter only gate the VISDA-C warm-up (L194), which dset="" disables.
            loss = pseudo_label_loss(g_t, pseudo_labels.to(g_t.device), idx.to(g_t.device), 1, 0, self.args) + loss
        return loss

    def obtain_label(self, loader, net_f, net_b, net_c) -> torch.Tensor:
        """Upstream obtain_label verbatim. `loader` yields (inputs, labels, ...); labels
        only feed the accuracy line upstream writes to `args.out_file`."""
        return torch.from_numpy(obtain_label(loader, net_f, net_b, net_c, self.args))
