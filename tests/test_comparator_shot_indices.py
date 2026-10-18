"""SHOT adapter: whole-dataset pseudo-labels are indexed by the batch's dataset indices.

`obtain_label` returns one label per target sample (the whole dataset). Upstream
image_target.py reads them as `mem_label[tar_idx]`; the adapter must do the same
instead of silently applying the first B labels to every batch.
"""
import pytest
import torch

from CREDA.comparators.shot.adapter import SHOT

N, B, C, D = 40, 8, 5, 16


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    f_s = torch.randn(B, D, generator=g)
    f_t = torch.randn(B, D, generator=g)
    g_s = torch.randn(B, C, generator=g)
    g_t = torch.randn(B, C, generator=g)
    y_s = torch.randint(0, C, (B,), generator=g)
    return f_s, f_t, g_s, g_t, y_s


def _dataset_labels(seed=1):
    return torch.randint(0, C, (N,), generator=torch.Generator().manual_seed(seed))


def test_whole_dataset_labels_with_tar_idx_equal_pre_indexed_labels():
    method = SHOT(feature_dim=D, num_classes=C)
    batch = _batch()
    mem_label = _dataset_labels()
    tar_idx = torch.tensor([33, 2, 17, 39, 8, 21, 5, 30])

    via_idx = method(*batch, pseudo_labels=mem_label, tar_idx=tar_idx)
    pre_indexed = method(*batch, pseudo_labels=mem_label[tar_idx])

    assert torch.equal(via_idx, pre_indexed)
    # And it is NOT what the first B labels would give (the old silent behaviour).
    first_b = method(*batch, pseudo_labels=mem_label[:B])
    assert not torch.equal(via_idx, first_b)


def test_whole_dataset_labels_without_tar_idx_raise():
    method = SHOT(feature_dim=D, num_classes=C)
    with pytest.raises(ValueError, match="tar_idx"):
        method(*_batch(), pseudo_labels=_dataset_labels())


def test_tar_idx_of_wrong_length_raises():
    method = SHOT(feature_dim=D, num_classes=C)
    with pytest.raises(ValueError, match="tar_idx"):
        method(*_batch(), pseudo_labels=_dataset_labels(), tar_idx=torch.arange(B - 1))


def test_default_call_without_pseudo_labels_is_unchanged():
    # Previous behaviour: IM loss only (upstream L199-206); computed independently here.
    method = SHOT(feature_dim=D, num_classes=C)
    batch = _batch()
    g_t = batch[3]
    p = torch.softmax(g_t, dim=1)
    ent = torch.mean(-(p * torch.log(p + 1e-5)).sum(dim=1))
    pm = p.mean(dim=0)
    gent = torch.sum(-pm * torch.log(pm + 1e-5))
    expected = (ent - gent) * 1.0

    assert torch.allclose(method(*batch), expected)
