"""A diverged phase 2 (non-finite class scores) fails as FloatingPointError, which the
search scores 0 and the seed loop records as one failed seed."""
import pytest
import torch

from MIL_CREDA.confidence import pseudolabel


def test_pseudolabel_refuses_non_finite_scores_and_keeps_the_tie_break():
    with pytest.raises(FloatingPointError):
        pseudolabel(torch.tensor([float("nan"), 0.3]))
    assert pseudolabel(torch.tensor([0.5, 0.5])) == 0
    assert pseudolabel(torch.tensor([0.2, 0.8])) == 1
