"""CDAN+E multilinear map selection (paper Eq. 8) and RandomLayer device/state fixes.

Long et al., NeurIPS 2018, Eq. 8: use the randomized map T_(.) (Eq. 6) when
d_f * d_g > 4096, else the full map T_(x). With 256 features and 31 classes
(7936 > 4096) the paper uses the randomized map with d = 1024.
"""
from __future__ import annotations

import math
import unittest

import torch

from CREDA.comparators import build
from CREDA.comparators.cdan.adapter import CDANE
from CREDA.comparators.cdan.upstream_network import RandomLayer

RANDOM_PARAMS_256_31 = 2_100_225
FULL_PARAMS_256_31 = 9_178_113


def _trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class DefaultFollowsEq8(unittest.TestCase):
    def test_default_at_256_31_uses_random_map(self):
        m = build("cdan_e", 256, 31)
        self.assertIsInstance(m.random_layer, RandomLayer)
        self.assertEqual(m.random_layer.output_dim, 1024)
        self.assertEqual(m.ad_net.ad_layer1.in_features, 1024)
        self.assertEqual(_trainable(m), RANDOM_PARAMS_256_31)

    def test_small_product_uses_full_map(self):
        # 64 * 64 = 4096 is not > 4096: full map.
        m = CDANE(64, 64)
        self.assertIsNone(m.random_layer)
        self.assertEqual(m.ad_net.ad_layer1.in_features, 64 * 64)
        m2 = CDANE(16, 10)
        self.assertIsNone(m2.random_layer)

    def test_threshold_boundary(self):
        self.assertIsNone(CDANE(64, 64).random_layer)  # 4096
        self.assertIsNotNone(CDANE(65, 64).random_layer)  # 4160

    def test_explicit_false_keeps_full_map(self):
        m = build("cdan_e", 256, 31, use_random=False)
        self.assertIsNone(m.random_layer)
        self.assertEqual(_trainable(m), FULL_PARAMS_256_31)

    def test_explicit_true_forces_random_on_small(self):
        m = CDANE(16, 10, use_random=True)
        self.assertIsInstance(m.random_layer, RandomLayer)

    def test_default_forward_is_finite_scalar(self):
        torch.manual_seed(0)
        m = build("cdan_e", 256, 31)
        f = torch.randn(8, 256).requires_grad_()
        g = torch.randn(8, 31).requires_grad_()
        out = m(f, f.clone(), g, g.clone(), torch.zeros(8, dtype=torch.long))
        self.assertEqual(out.dim(), 0)
        self.assertTrue(torch.isfinite(out))


class RandomLayerIsModuleState(unittest.TestCase):
    def test_matrices_are_buffers_in_state_dict_not_parameters(self):
        layer = RandomLayer([256, 31], 1024)
        sd = layer.state_dict()
        mats = [k for k in sd if "random_matrix" in k]
        self.assertEqual(len(mats), 2)
        self.assertEqual(list(layer.parameters()), [])
        self.assertEqual(len(list(layer.buffers())), 2)
        for t in layer.random_matrix:
            self.assertFalse(t.requires_grad)

    def test_adapter_state_dict_holds_random_matrices(self):
        m = build("cdan_e", 256, 31)
        self.assertTrue(any("random_layer" in k and "random_matrix" in k for k in m.state_dict()))

    def test_matrices_follow_dtype_change(self):
        m = build("cdan_e", 256, 31).to(torch.float64)
        for t in m.random_layer.random_matrix:
            self.assertEqual(t.dtype, torch.float64)
        f = torch.randn(4, 256, dtype=torch.float64).requires_grad_()
        g = torch.randn(4, 31, dtype=torch.float64).requires_grad_()
        # The random map itself runs in float64 (the full CDAN loss stays float32-only
        # upstream: its domain target is built with .float(), loss.py L31).
        out = m.random_layer([f, torch.softmax(g, dim=1)])
        self.assertEqual(out.dtype, torch.float64)
        self.assertEqual(tuple(out.shape), (4, 1024))

    def test_matrices_follow_meta_device(self):
        layer = RandomLayer([8, 4], 16).to(torch.device("meta"))
        for t in layer.random_matrix:
            self.assertEqual(t.device.type, "meta")

    def test_state_dict_round_trip(self):
        a = RandomLayer([8, 4], 16)
        b = RandomLayer([8, 4], 16)
        b.load_state_dict(a.state_dict())
        for x, y in zip(a.random_matrix, b.random_matrix):
            self.assertTrue(torch.equal(x, y))

    def test_math_matches_eq6(self):
        torch.manual_seed(1)
        layer = RandomLayer([8, 4], 16)
        f, g = torch.randn(3, 8), torch.randn(3, 4)
        rf, rg = layer.random_matrix
        expected = (f @ rf) / math.pow(16.0, 1.0 / 2) * (g @ rg)
        self.assertTrue(torch.allclose(layer([f, g]), expected))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_mps_forward(self):
        dev = torch.device("mps")
        m = build("cdan_e", 256, 31).to(dev)
        for t in m.random_layer.random_matrix:
            self.assertEqual(t.device.type, "mps")
        f = torch.randn(4, 256, device=dev).requires_grad_()
        g = torch.randn(4, 31, device=dev).requires_grad_()
        out = m(f, f.clone(), g, g.clone(), torch.zeros(4, dtype=torch.long, device=dev))
        self.assertTrue(torch.isfinite(out.cpu()))


if __name__ == "__main__":
    unittest.main()
