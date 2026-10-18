"""A weight vector the caller supplies, so a control can exist at all.

`CREDALoss` computes `target_weights` INSIDE `forward` from `g_t`, and accepted
no vector. For a BINARY weight that was not a limitation -- selecting the rows
and multiplying are the same operation, which is how the oracle and the gate
arms run without touching this module. For a CONTINUOUS weight it is: a
shuffled weight needs the same VALUES on different points, and selection cannot
express that.

That control is what turns the comparison into a test of MECHANISM rather than
of outcome. If the declared weight and the same weights randomly reassigned
reach the same place, the weight selects nothing and only the distribution of
weights matters -- and every null measured so far would be compatible with
that, since none of them could tell the two apart.

Opt-in and default-identical, the same discipline as `bounds_form` and
`balance_mass`: with `target_weights=None` the loss computes its own, and the
published behaviour is bit-for-bit what it was.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA.models import CREDALoss


def _lote(n=96, d=8, C=3, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    f_s = torch.randn(n, d, generator=g)
    f_t = torch.randn(n, d, generator=g)
    y_s = torch.randint(0, C, (n,), generator=g)
    g_t = torch.softmax(torch.randn(n, C, generator=g) * 2.0, 1)
    return f_s, f_t, y_s, g_t


class LosPesosInyectadosSonOptIn(unittest.TestCase):

    def setUp(self):
        self.cr = CREDALoss(use_entropy_weighting=True, balance_mix_blocks=True,
                            balance_mass='trace', normalize_by_bounds=True,
                            bounds_form='entropy-cosh', weight_cross_block=True)
        self.lote = _lote()

    def test_none_is_bit_for_bit_what_the_loss_already_did(self):
        a = self.cr(*self.lote)
        b = self.cr(*self.lote, target_weights=None)
        self.assertEqual(float(a), float(b))

    def test_passing_the_loss_its_own_weights_changes_nothing(self):
        """The injected path and the internal one have to be the SAME
        arithmetic, or a control run through it measures the path and not the
        weights."""
        f_s, f_t, y_s, g_t = self.lote
        w = 1.0 - (-torch.log((g_t ** 2).sum(1) + 1e-6)) / torch.log(
            torch.tensor(float(g_t.shape[1])))
        a = self.cr(*self.lote)
        b = self.cr(*self.lote, target_weights=w)
        self.assertAlmostEqual(float(a), float(b), places=6)

    def test_a_different_vector_reaches_a_different_value(self):
        f_s, f_t, y_s, g_t = self.lote
        w = torch.rand(len(f_t), generator=torch.Generator().manual_seed(7))
        self.assertNotAlmostEqual(float(self.cr(*self.lote)),
                                  float(self.cr(*self.lote, target_weights=w)),
                                  places=5)

    def test_it_is_refused_when_the_weighting_is_off(self):
        """A caller handing weights to an arm that ignores them believes it is
        running something it is not."""
        cr = CREDALoss(use_entropy_weighting=False)
        w = torch.rand(len(self.lote[1]))
        with self.assertRaises(ValueError) as ctx:
            cr(*self.lote, target_weights=w)
        self.assertIn('use_entropy_weighting', str(ctx.exception))

    def test_a_wrong_length_is_refused_rather_than_broadcast(self):
        """Silent broadcasting would weight the wrong points and look fine."""
        w = torch.rand(len(self.lote[1]) - 1)
        with self.assertRaises(ValueError):
            self.cr(*self.lote, target_weights=w)


if __name__ == '__main__':
    unittest.main()
