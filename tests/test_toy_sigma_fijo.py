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
with the source encoder frozen, `corrida` reads the median pairwise distance
over the WHOLE source train split in the representation space and hands that
one number to `CREDALoss` as `sigma`, which short-circuits every scope. Two
consequences this file exists to pin:

* it is a function of the SOURCE alone, so nothing injected into the target --
  intruders, relocations, contamination -- can move it, which is what makes a
  weighting experiment measurable at all; and
* it is one number for every batch and every class, so the readings of the term
  become comparable ACROSS STEPS, which under a per-batch median they are not.

The cost is declared rather than hidden: a frozen sigma is calibrated in phase
1's space while `enc_t` moves, so if the target expands far enough the
exponential saturates and the term stops producing gradient. Nothing in the
search compensates for scale -- `DELTA` is the lambda ramp, not a bandwidth --
so `corrida` reports the drift instead, and the last test here pins that it
does.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA import toy_experiment as T
from CREDA.models import CREDALoss


def _una(fam, pos='batchnorm', semilla=100, **kw):
    D = T.datos_semilla(fam, semilla)
    return D, T.corrida(fam, D, pos, semilla=semilla, mover_fuente=False,
                        cuadros=T.CTRL, **kw)


class SigmaEsUnaConstanteDeLaFuente(unittest.TestCase):

    def test_the_run_reports_the_one_bandwidth_it_used(self):
        """Not a promise in a flag: the number the kernels actually saw."""
        _, (_, _, _, _, info) = _una(T.FAMILIAS[0])
        self.assertIn('sigma', info)
        self.assertIsInstance(info['sigma'], float)
        self.assertGreater(info['sigma'], 0.0)

    def test_the_bandwidth_is_the_median_over_the_whole_source_train(self):
        """Exactly the loss's own heuristic, not a second statistic that
        resembles it. Recomputing it here from the phase-1 representation has
        to land on the same float."""
        D, (_, _, _, _, info) = _una(T.FAMILIAS[0])
        self.assertIn('Hs_fase1', info)
        H = torch.as_tensor(info['Hs_fase1'])
        self.assertEqual(H.shape[0], D['Xs_tr'].shape[0])
        esperado = float(CREDALoss()._median_pairwise(H))
        self.assertAlmostEqual(info['sigma'], esperado, places=6)

    def test_moving_the_target_train_cannot_move_the_bandwidth(self):
        """The whole point. Replace the target train split wholesale and the
        bandwidth comes back byte-identical, because it never read it."""
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        _, _, _, _, a = T.corrida(fam, D, 'batchnorm', semilla=100,
                                  mover_fuente=False, cuadros=T.CTRL)
        D2 = dict(D)
        D2['Xt_tr'] = D['Xt_tr'] * 7.0 + 13.0
        _, _, _, _, b = T.corrida(fam, D2, 'batchnorm', semilla=100,
                                  mover_fuente=False, cuadros=T.CTRL)
        self.assertEqual(a['sigma'], b['sigma'])

    def test_the_batch_size_cannot_move_the_bandwidth(self):
        """One sigma for every batch: the estimate is not taken per batch, so
        changing the batch size changes the trajectory and not the unit."""
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        vistos = set()
        for lote in (32, 256):
            _, _, _, _, info = T.corrida(fam, D, 'batchnorm', semilla=100,
                                         mover_fuente=False, cuadros=T.CTRL,
                                         hp2={'LOTE_2': lote})
            vistos.add(round(info['sigma'], 10))
        self.assertEqual(len(vistos), 1, vistos)

    def test_the_scope_key_is_gone_from_the_defaults_and_the_search(self):
        """A stale key that changes nothing is how this comes back unnoticed."""
        self.assertNotIn('SIGMA_FROM', T.HP_2)
        fuente = (Path(__file__).resolve().parents[1]
                  / 'src' / 'CREDA' / 'toy_experiment.py').read_text()
        self.assertNotIn("suggest_categorical(\n                'SIGMA_FROM'",
                         fuente)
        self.assertNotIn("'SIGMA_FROM',", fuente.split('def objetivo')[-1])

    def test_a_stale_scope_key_is_refused_rather_than_ignored(self):
        """Passing the retired key is a caller who believes it still decides
        something. Refuse, naming it."""
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        with self.assertRaises(ValueError) as ctx:
            T.corrida(fam, D, 'batchnorm', semilla=100, mover_fuente=False,
                      cuadros=T.CTRL, hp2={'SIGMA_FROM': 'batch-pooled'})
        self.assertIn('SIGMA_FROM', str(ctx.exception))


class LaDerivaSeReportaEnLugarDeAdaptarse(unittest.TestCase):

    def test_the_run_reports_the_scale_drift_it_paid_for(self):
        """A frozen bandwidth in a moving space can saturate the kernel. That
        is not a reason to adapt it; it is a number that has to come back from
        every run, or the stall is invisible."""
        _, (_, _, _, _, info) = _una(T.FAMILIAS[0])
        for campo in ('sigma_deriva', 'k_t_saturada'):
            self.assertIn(campo, info)
        self.assertGreater(info['sigma_deriva'], 0.0)
        self.assertGreaterEqual(info['k_t_saturada'], 0.0)
        self.assertLessEqual(info['k_t_saturada'], 1.0)


if __name__ == '__main__':
    unittest.main()


class ElRuteoSeRegistraComoInstrumento(unittest.TestCase):
    """The routing indicator, in every arm, and never in a decision.

    It used to be computed only inside the ceiling arm, so "does `w` rank the
    misrouted points?" could only be answered by comparing TWO runs whose
    trajectories differ -- which does not answer it. It reads `y_t`, so it has
    exactly the standing of `acc_te`: a number the record carries and no
    criterion may read.
    """

    def test_every_arm_records_the_routing_indicator_not_only_the_ceiling(self):
        for peso in ('apagado', 'declarado', 'oraculo'):
            _, (_, cu, _, _, info) = _una(T.FAMILIAS[0], peso=peso)
            self.assertIn('rut_cu', info, peso)
            self.assertEqual(len(info['rut_cu']), len(cu), peso)
            v = np.asarray(info['rut_cu'][0])
            self.assertEqual(set(np.unique(v)) - {0.0, 1.0}, set(), peso)
            self.assertEqual(len(v), len(np.asarray(info['w_cu'][0])), peso)

    def test_the_routing_indicator_never_reaches_a_decision(self):
        """It reads `y_t`. The off arm has to be bit-identical with it present,
        or it is not an instrument -- it is a leak."""
        import inspect
        fuente = inspect.getsource(T.corrida)
        cuerpo = fuente.split('rut_cu.append')[-1]
        for prohibido in ('rut_cu[', 'rut_cu)', 'rut_cu,'):
            if prohibido == 'rut_cu,':
                continue
            self.assertNotIn(prohibido, cuerpo.split('info = dict')[0])
        _, (_, _, _, _, a) = _una(T.FAMILIAS[0], peso='apagado')
        _, (_, _, _, _, b) = _una(T.FAMILIAS[0], peso='apagado')
        self.assertEqual(a['acc_te'], b['acc_te'])
        self.assertTrue(np.array_equal(np.asarray(a['w_usado_cu'][0]),
                                       np.ones(len(a['w_usado_cu'][0]))))
