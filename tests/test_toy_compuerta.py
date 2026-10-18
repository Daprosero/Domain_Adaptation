"""The gate: the declared signal, applied as {0,1} instead of as a nudge.

Measured on clean data with the frozen bandwidth, the declared weight RANKS the
misrouted points well in two of three families -- AUC 0.820 in toroide, 0.713
in rollo -- and at chance in lunas (0.491, with a Delta-w of +0.001). Yet the
declared arm captures 12% of a +0.0788 ceiling in rollo and goes BACKWARDS in
toroide (-0.0334, p=0.0078, winning 1 seed of 10) against a +0.0110 ceiling,
and the two weights leave nearly the same total mass in the target block
(0.797 against 0.760, 0.796 against 0.761). So the failure is neither the
signal nor the amount removed. It is the SHAPE of the removal: the declared
weight discounts a misrouted point from 0.906 to 0.735 -- it keeps 73% of a
point that should be gone -- while the oracle takes it from 1.0 to 0.0.

The hypothesis this file is the instrument for: a continuous weight RESCALES
every row and so deforms the geometry the entropy measures, where the oracle's
{0,1} REMOVES rows. Cleaning and deforming are not the same operation.

`pesos_compuerta` tests it label-free: threshold the declared weight at its own
within-batch quantile. It reads no `y_t`, so unlike the oracle it IS a method
and the search may select it.

Its prediction is asymmetric ON PURPOSE, and that asymmetry is what makes the
result refutable rather than a fishing trip: the gate must recover part of the
ceiling where the ranking is informative, and it must NOT help in lunas, where
the ranking is chance. A gate that helps in lunas falsifies the hypothesis --
whatever it bought there, it was not selectivity.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA import toy_experiment as T


class LaCompuertaEsUnMetodo(unittest.TestCase):

    def test_the_gate_is_a_declared_weight_and_not_a_ceiling(self):
        """It reads no labels, so it belongs among the searchable weights --
        which is the whole difference from the oracle it imitates."""
        self.assertIn(T.PESO_COMPUERTA, T.PESOS)
        self.assertIn(T.PESO_COMPUERTA, T.PESOS_BUSCABLES)
        self.assertNotIn(T.PESO_COMPUERTA, T.PESOS_TECHO)
        self.assertNotIn(T.PALABRA_TECHO, T.PESO_COMPUERTA)

    def test_the_gate_takes_only_logits_and_a_quantile(self):
        """No `y` in the signature: a label it cannot receive is a leak it
        cannot have."""
        import inspect
        par = list(inspect.signature(T.pesos_compuerta).parameters)
        self.assertEqual(par, ['logits', 'q'])

    def test_the_gate_output_is_exactly_zero_or_one(self):
        g = torch.randn(64, T.CLASES) * 3.0
        w = T.pesos_compuerta(g, 0.25)
        self.assertEqual(set(np.unique(w)) - {0.0, 1.0}, set())

    def test_the_gate_closes_on_the_declared_quantile_fraction(self):
        """`q` is what fraction it shuts, and it has to be that fraction --
        a gate whose rate is not its parameter cannot be swept."""
        g = torch.randn(400, T.CLASES) * 3.0
        for q in (0.10, 0.25, 0.50):
            w = T.pesos_compuerta(g, q)
            self.assertAlmostEqual(float((w == 0.0).mean()), q, delta=0.02, msg=q)

    def test_the_gate_shuts_on_the_LEAST_confident_points(self):
        """It has to keep the declared signal's ORDER. Shutting the confident
        ones would be the same arithmetic with the sign flipped, and it would
        pass every test above."""
        g = torch.randn(200, T.CLASES) * 3.0
        wd = T.pesos_declarados(g)
        w = T.pesos_compuerta(g, 0.30)
        self.assertLess(wd[w == 0.0].max(), wd[w == 1.0].min())

    def test_a_zero_quantile_is_the_off_arm_exactly(self):
        """The ladder has to contain its own floor, or `q` has no origin."""
        g = torch.randn(128, T.CLASES) * 3.0
        self.assertTrue(np.array_equal(T.pesos_compuerta(g, 0.0),
                                       np.ones(128, dtype=np.float32)))

    def test_the_run_records_the_gate_it_used(self):
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        *_, info = T.corrida(fam, D, 'batchnorm', semilla=100,
                             mover_fuente=False, cuadros=T.CTRL,
                             peso=T.PESO_COMPUERTA, hp2={'Q_COMPUERTA': 0.25})
        self.assertEqual(info['peso'], T.PESO_COMPUERTA)
        self.assertEqual(info['hp2']['Q_COMPUERTA'], 0.25)
        u = np.asarray(info['w_usado_cu'][0])
        self.assertEqual(set(np.unique(u)) - {0.0, 1.0}, set())
        # y el declarado se sigue registrando al lado, sin umbralar
        d = np.asarray(info['w_cu'][0])
        self.assertGreater(len(set(np.round(d, 6))), 2)


if __name__ == '__main__':
    unittest.main()
