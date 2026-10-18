"""How precise does a selection have to be before removing points pays?

The gate was falsified by its own pre-declared prediction: with the declared
signal thresholded at four rates it LOST in all four in toroide -- the family
whose ranking is best, AUC 0.820 -- monotonically in the rate, with p from
0.0039 to 0.0137 and winning one seed of ten in every cut. So the shape of the
discount was not the problem, and "rescaling deforms where {0,1} cleans" is
dead.

What the monotonicity leaves is an asymmetry worth measuring rather than
asserting: the oracle removes with precision 1.0 BY CONSTRUCTION and wins
(+0.0110 in toroide, +0.0788 in rollo), while a gate removing at roughly half
precision loses badly. If deleting a well-routed point costs much more than
keeping a misrouted one, then the question is not which label-free signal to
build -- it is what precision any of them would have to reach.

`pesos_oraculo_ruidoso` answers it by corrupting the oracle to a DECLARED
precision at the oracle's own removal count, so precision is the only thing
that varies between rungs. Count and precision would otherwise confound: a
less precise selection that also removes fewer points is two changes.

It reads `y_t`, so it is a CEILING and never a method -- it exists to bound
what any method could buy, and its break-even is a statement about the
formulation rather than about our estimator. If break-even precision sits at
0.95, no label-free signal reaches it, and that is a result about the
contribution.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA import toy_experiment as T


def _caso(n=300, mal=0.25, semilla=0):
    g = torch.randn(n, T.CLASES, generator=torch.Generator().manual_seed(semilla))
    y = g.argmax(1).clone()
    rng = np.random.default_rng(semilla)
    rotos = rng.choice(n, int(round(mal * n)), replace=False)
    y[rotos] = (y[rotos] + 1) % T.CLASES          # mal ruteados por construccion
    return g, y, set(rotos.tolist())


class LaPrecisionEsElEjeYNadaMas(unittest.TestCase):

    def test_it_is_a_ceiling_and_says_so_where_the_module_says_so(self):
        """The ceiling word lives in the ARM name and in a record key, not in
        the weight name -- `PESO_ORACULO` is a bare 'oraculo'. This asserted
        the weight name first, which imposed a convention the module does not
        have; what has to be unreadable as a method is what gets PRINTED."""
        self.assertIn(T.PALABRA_TECHO, T.BRAZO_TECHO_RUIDOSO.lower())
        self.assertIn(T.PESO_ORACULO_RUIDOSO, T.PESOS)
        self.assertIn(T.PESO_ORACULO_RUIDOSO, T.PESOS_TECHO)
        self.assertNotIn(T.PESO_ORACULO_RUIDOSO, T.PESOS_BUSCABLES)

    def test_a_precision_of_one_is_the_oracle_bit_for_bit(self):
        """The ladder has to contain its own top, or the rungs have no origin
        and a drop cannot be attributed to precision."""
        g, y, _ = _caso()
        a = T.pesos_oraculo(g, y)
        b = T.pesos_oraculo_ruidoso(g, y, 1.0, 0)
        self.assertTrue(np.array_equal(a, b))

    def test_it_removes_exactly_as_many_points_as_the_oracle(self):
        """Precision is the axis. A rung that also changed the count would be
        two changes and its reading would belong to neither."""
        g, y, _ = _caso()
        k = int((T.pesos_oraculo(g, y) == 0.0).sum())
        for pr in (1.0, 0.9, 0.7, 0.5):
            w = T.pesos_oraculo_ruidoso(g, y, pr, 0)
            self.assertEqual(int((w == 0.0).sum()), k, pr)

    def test_the_declared_precision_is_the_precision_it_achieves(self):
        """What fraction of what it removes is genuinely misrouted."""
        g, y, rotos = _caso(n=600)
        for pr in (1.0, 0.9, 0.8, 0.6):
            w = T.pesos_oraculo_ruidoso(g, y, pr, 0)
            quitados = set(np.flatnonzero(w == 0.0).tolist())
            real = len(quitados & rotos) / len(quitados)
            self.assertAlmostEqual(real, pr, delta=0.02, msg=pr)

    def test_it_is_deterministic_given_the_same_draw(self):
        g, y, _ = _caso()
        a = T.pesos_oraculo_ruidoso(g, y, 0.8, 7)
        b = T.pesos_oraculo_ruidoso(g, y, 0.8, 7)
        c = T.pesos_oraculo_ruidoso(g, y, 0.8, 8)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_a_run_with_it_records_the_precision_it_used(self):
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        *_, info = T.corrida(fam, D, 'batchnorm', semilla=100,
                             mover_fuente=False, cuadros=T.CTRL,
                             peso=T.PESO_ORACULO_RUIDOSO,
                             hp2={'PRECISION_TECHO': 0.80})
        self.assertEqual(info['peso'], T.PESO_ORACULO_RUIDOSO)
        self.assertEqual(info['hp2']['PRECISION_TECHO'], 0.80)
        u = np.asarray(info['w_usado_cu'][0])
        self.assertEqual(set(np.unique(u)) - {0.0, 1.0}, set())

    def test_the_search_refuses_it_by_name(self):
        """A ceiling the search could pick stops being a ceiling."""
        with self.assertRaises(AssertionError):
            T.buscar_hp2(T.FAMILIAS[0], T.HP_1, trials=1, semillas=[0],
                         peso=T.PESO_ORACULO_RUIDOSO, al_paso=lambda s: None)


if __name__ == '__main__':
    unittest.main()
