"""One bandwidth, calibrated on the source, frozen for the whole of phase 2.

The scope `sigma_from` used to be a searchable hyperparameter, and all three
families picked `batch-pooled` -- the scope whose own docstring records the
defect it carries: the pooled median reads the TARGET's features, which are
exactly what the optimizer is moving, so the kernel's unit of measurement moves
with the thing it measures. Measured there and not argued: displace one class's
target features and that class's score FALLS, 0.505 at a displacement of 10 to
0.437 at 30 to 0.397 at 100. A worse-aligned class reports a smaller
discrepancy, which is the sign inverted.

So the bandwidth is no longer estimated inside the loss at all. After phase 1,
with the source encoder frozen, the tutorial notebook reads
`sigma_de_la_fuente` -- the median pairwise distance over the WHOLE source
train split in the representation space -- and hands that one number to
`CREDALoss` as `sigma`, which short-circuits every scope. Two consequences
this file exists to pin:

* it is a function of the SOURCE alone, so nothing done to the target can
  move it; and
* it is one number for every batch and every class, so the readings of the term
  become comparable ACROSS STEPS, which under a per-batch median they are not.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA import toy_experiment as T
from CREDA.models import CREDALoss


def _representacion(fam, pos='batchnorm', semilla=100):
    """The source train split through a seeded, frozen encoder."""
    D = T.datos_semilla(fam, semilla)
    torch.manual_seed(semilla)
    enc = T.MLP(D['Xs_tr'].shape[1], pos).eval()
    with torch.no_grad():
        return D, enc(D['Xs_tr'])


class SigmaEsUnaConstanteDeLaFuente(unittest.TestCase):

    def test_the_bandwidth_is_the_median_over_the_whole_source_train(self):
        """Exactly the loss's own heuristic, not a second statistic that
        resembles it, over the whole split and returned as a float."""
        for fam in T.FAMILIAS:
            D, H = _representacion(fam)
            self.assertEqual(H.shape[0], D['Xs_tr'].shape[0])
            sigma = T.sigma_de_la_fuente(H)
            self.assertIsInstance(sigma, float)
            self.assertGreater(sigma, 0.0)
            esperado = float(CREDALoss()._median_pairwise(H))
            self.assertEqual(sigma, esperado, fam['nombre'])
            # The whole train and not a batch of it: a 32-point subset reads
            # another median.
            self.assertNotEqual(sigma, T.sigma_de_la_fuente(H[:32]))

    def test_moving_the_target_train_cannot_move_the_bandwidth(self):
        """Replace the target train split wholesale and the bandwidth comes
        back byte-identical, because it never reads it."""
        fam = T.FAMILIAS[0]
        D, H = _representacion(fam)
        D2 = dict(D)
        D2['Xt_tr'] = D['Xt_tr'] * 7.0 + 13.0
        self.assertFalse(torch.equal(D2['Xt_tr'], D['Xt_tr']))
        self.assertTrue(torch.equal(D2['Xs_tr'], D['Xs_tr']))
        torch.manual_seed(100)
        enc = T.MLP(D2['Xs_tr'].shape[1], 'batchnorm').eval()
        with torch.no_grad():
            H2 = enc(D2['Xs_tr'])
        self.assertEqual(T.sigma_de_la_fuente(H), T.sigma_de_la_fuente(H2))

    def test_the_batch_size_cannot_move_the_bandwidth(self):
        """One sigma for every batch: the estimate reads the whole split once,
        so the order the batches would cut it in changes nothing."""
        _, H = _representacion(T.FAMILIAS[1])
        orden = torch.randperm(H.shape[0], generator=torch.Generator().manual_seed(3))
        vistos = {round(T.sigma_de_la_fuente(H), 10),
                  round(T.sigma_de_la_fuente(H[orden]), 10)}
        self.assertEqual(len(vistos), 1, vistos)

    def test_the_scope_key_is_gone_from_the_defaults(self):
        """A stale key that changes nothing is how this comes back unnoticed."""
        self.assertNotIn('SIGMA_FROM', T.HP_2)


if __name__ == '__main__':
    unittest.main()
