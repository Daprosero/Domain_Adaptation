"""Plan v63 record helpers: window pointers, attention and correspondence."""

import numpy as np
import torch

from MIL_CREDA_EEG import v63_records as R
from MIL_CREDA_EEG import v63_setup as V


class _Raw:
    subject, condition, channels = "S1", "Robots", tuple(V.SHARED) + ("X",)

    def marks(self):
        n = 130
        return {"window_index": np.arange(n) + 5, "window_start_s": np.arange(n) * 4.0,
                "contact_bad": np.zeros((n, 13), bool) | (np.arange(13) == 12)}


def test_the_window_pointers_are_the_windows_the_model_read():
    info = R.window_info(_Raw(), V.SHARED)
    assert info["index"] == list(range(0, 130, 3))[:60]
    assert info["storedIndex"][:2] == [5, 8]
    assert not any(info["contactBad"])            # channel X is not a transfer channel
    assert any(R.window_info(_Raw(), None)["contactBad"])


def test_the_attention_weights_of_a_bag_sum_to_one():
    w = torch.softmax(torch.randn(7), 0)
    assert abs(float(w.sum()) - 1) < 1e-6
    assert R.selected_indices(10) == list(range(10))


def test_the_correspondence_of_a_target_bag_is_a_distribution_over_the_source_bags():
    cross = torch.tensor([0.9, 0.2, 0.5, 0.4], dtype=torch.float64)
    labels = torch.tensor([0, 1, 0, 1])
    pi = V._corr(cross, labels, torch.tensor([0.3, 0.7], dtype=torch.float64), V.row_tau(cross))
    assert pi.shape == (4,) and abs(float(pi.sum()) - 1) < 1e-6 and bool((pi >= 0).all())
