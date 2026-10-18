"""Intruders injected IN THE EMBEDDING, fresh every step.

Three earlier constructions injected through the INPUT and none of them gave a
weighting anything to remove, and the reason was measured rather than guessed:
the damage entered through quantities computed BEFORE any per-point weight
exists. Two of those channels are now shut, and only one of them by this file:

* the encoder's BatchNorm statistics -- shut by injecting AFTER `enc_t`, which
  is exactly what the operator proposed and what this file does; and
* the pooled bandwidth -- shut already, because sigma is now one constant
  calibrated on the source train and frozen (`sigma_de_la_fuente`), so nothing
  injected into the target can move it. Under the old per-batch pooled median
  an intruder moved the unit of measurement itself.

What is left is the weighted class block, which is the one channel `w`
multiplies. So this is the first construction where a weighting either works or
the contribution is dead in this regime.

**The prediction, declared here before the measurement exists, because it is
uncomfortable and already evidenced:** the head is affine per region, so a
latent pushed far along a ray freezes the ReLU pattern, grows the logits
linearly and SATURATES the softmax -- measured at `w` rising to 0.998 in the
OOD sweep. So `w` is expected to KEEP the intruders, possibly weighting them
ABOVE the clean points. If that happens the declared weighting cannot help
here, and the oracle that zeroes them by construction bounds what any weighting
could have bought. If instead `w` discounts them, the affine-saturation reading
is wrong in this regime and that is the surprise worth having.

**What this construction is NOT, said because it changes what is measured:** an
embedding written in directly is a CONSTANT, so unlike `intrusar` it has no
gradient path back to the encoder. It can still poison the term -- it occupies a
row of the target block and, under the trace balance of r28, adds mass to
`tr(K~_t) = sum w_j^2` without contributing similarity, which dilutes the block
for every other point. That is reachable by `w`. But it is a different
intervention from one that flows through the encoder, and the two should not be
read as the same experiment.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA import toy_experiment as T


class LaInyeccionViveEnElEmbedding(unittest.TestCase):

    def test_it_replaces_rather_than_adds(self):
        """The block size and the class composition stay fixed, so the only
        thing that changed is WHERE some points are."""
        f = torch.randn(128, 16)
        g, mal = T.intrusar_el_embedding(f, 0.25, 4.0, 0)
        self.assertEqual(g.shape, f.shape)
        self.assertEqual(int(mal.sum()), 32)

    def test_the_intruders_land_materially_farther_than_the_clean_rows(self):
        """Without latent distance there is no damage, and nothing measured
        afterwards means anything. This is the licence for the rest."""
        f = torch.randn(400, 16)
        g, mal = T.intrusar_el_embedding(f, 0.20, 4.0, 0)
        mu = f.mean(0)
        d_lim = (g[~mal] - mu).norm(dim=1).mean()
        d_int = (g[mal] - mu).norm(dim=1).mean()
        self.assertGreater(float(d_int), 2.0 * float(d_lim))

    def test_the_clean_rows_come_back_untouched(self):
        f = torch.randn(128, 16)
        g, mal = T.intrusar_el_embedding(f, 0.25, 4.0, 0)
        self.assertTrue(torch.equal(g[~mal], f[~mal]))

    def test_a_zero_rate_is_a_no_op(self):
        f = torch.randn(64, 16)
        g, mal = T.intrusar_el_embedding(f, 0.0, 4.0, 0)
        self.assertTrue(torch.equal(g, f))
        self.assertEqual(int(mal.sum()), 0)

    def test_it_is_deterministic_and_refreshes_with_the_draw(self):
        """Fresh every step is the operator's design: a fixed set could be
        routed around. So the draw has to move, and reproducibly."""
        f = torch.randn(200, 16)
        a, ma = T.intrusar_el_embedding(f, 0.2, 4.0, 5)
        b, mb = T.intrusar_el_embedding(f, 0.2, 4.0, 5)
        c, mc = T.intrusar_el_embedding(f, 0.2, 4.0, 6)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(ma, mc))

    def test_the_directions_are_antipodal_pairs(self):
        """A single fixed direction is a feature rescaling wearing an
        intrusion's clothes: it would move the mean. Pairs cancel it."""
        f = torch.zeros(200, 16)
        g, mal = T.intrusar_el_embedding(f, 1.0, 4.0, 0)
        self.assertLess(float(g.mean(0).norm()), 1e-4)


class LosTresBrazosYSuTecho(unittest.TestCase):

    def test_a_run_injects_and_records_what_it_injected(self):
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        *_, info = T.corrida(fam, D, 'batchnorm', semilla=100,
                             mover_fuente=False, cuadros=T.CTRL,
                             hp2={'RHO_EMB': 0.25, 'RADIO_EMB': 4.0})
        self.assertEqual(info['hp2']['RHO_EMB'], 0.25)
        self.assertGreater(info['emb_inyectados'], 0)

    def test_a_zero_rate_run_is_the_clean_run_bit_for_bit(self):
        """The ladder contains its own floor or the damage has no origin."""
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        *_, a = T.corrida(fam, D, 'batchnorm', semilla=100, mover_fuente=False,
                          cuadros=T.CTRL, hp2={'RHO_EMB': 0.0})
        *_, b = T.corrida(fam, D, 'batchnorm', semilla=100, mover_fuente=False,
                          cuadros=T.CTRL)
        self.assertEqual(a['acc_te'], b['acc_te'])
        self.assertEqual(a['emb_inyectados'], 0)

    def test_the_embedding_ceiling_is_a_ceiling_and_not_a_method(self):
        self.assertIn(T.PESO_ORACULO_EMB, T.PESOS)
        self.assertIn(T.PESO_ORACULO_EMB, T.PESOS_TECHO)
        self.assertNotIn(T.PESO_ORACULO_EMB, T.PESOS_BUSCABLES)
        self.assertIn(T.PALABRA_TECHO, T.BRAZO_TECHO_EMB.lower())

    def test_the_ceiling_refuses_when_nothing_was_injected(self):
        """A ceiling that cannot know who the intruders are weights by ones and
        IS the floor running under a ceiling's label."""
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        with self.assertRaises(AssertionError):
            T.corrida(fam, D, 'batchnorm', semilla=100, mover_fuente=False,
                      cuadros=T.CTRL, peso=T.PESO_ORACULO_EMB,
                      hp2={'RHO_EMB': 0.0})


if __name__ == '__main__':
    unittest.main()
