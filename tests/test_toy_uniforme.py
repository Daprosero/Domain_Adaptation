"""La perturbación que `w` SÍ descuenta, y la única que faltaba probar.

El contraejemplo del deliberador la señala sin nombrarla. Con C = 3 y cabeza
afín, dos muestras con el argmax igual de inestable reciben pesos muy
distintos:

    logits (0, 0, -50)   empate exacto entre dos clases   ->  w = 0.369
    logits (0,  0,  0)   empate exacto entre TODAS        ->  w = 0.000

`Σ_c g_c²` no mide el gap entre las dos primeras: mide cuántas clases están en
juego. De modo que el punto que el peso lleva a su MÍNIMO no es el que cae
entre dos clases --- ése es `MODO_FRONTERA`, y recibe peso MEDIO --- sino el
que cae donde se cruzan TODAS las fronteras.

Ninguna de las cuatro construcciones anteriores produce ese punto:
`reubicado` aterriza dentro de otra clase (confiado, peso alto), el empuje por
un rayo satura (peso -> 1), `frontera` empata dos (peso medio) y los intrusos
caen lejos (peso alto). Así que el régimen donde el peso puede descontar al
máximo **nunca se midió**.

Y el conjunto se resuelve en vez de buscarse: la cabeza es un solo `Linear`,
así que `{f : W f + b tiene todas sus componentes iguales}` es un subespacio
AFÍN de codimensión C-1. Se proyecta con una pseudoinversa, lo que da el punto
MÁS CERCANO con posterior uniforme --- desplazamiento mínimo, incertidumbre
máxima, `w = 0` exacto.

Que el desplazamiento sea mínimo importa: separa «el peso descuenta puntos
inciertos» de «el peso descuenta puntos lejanos», que es la confusión que ya
nos costó cuatro construcciones.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA import toy_experiment as T


def _cab(d=16, semilla=0):
    torch.manual_seed(semilla)
    c = nn.Linear(d, T.CLASES)
    for q in c.parameters():
        q.requires_grad_(False)
    return c


class LaProyeccionAlPosteriorUniforme(unittest.TestCase):

    def setUp(self):
        self.cab = _cab()
        torch.manual_seed(1)
        self.f = torch.randn(200, 16) * 2.0

    def test_the_projected_points_have_an_exactly_uniform_posterior(self):
        g, mal = T.uniformar_el_embedding(self.f, self.cab, 0.25, 0)
        z = self.cab(g[mal])
        self.assertLess(float((z - z.mean(1, keepdim=True)).abs().max()), 1e-3)

    def test_their_weight_is_zero_which_is_the_whole_point(self):
        """If `w` is not at its minimum on them, this construction is not the
        regime it exists to reach."""
        g, mal = T.uniformar_el_embedding(self.f, self.cab, 0.25, 0)
        w = T.pesos_declarados(self.cab(g))
        self.assertLess(float(w[mal.numpy()].max()), 0.01)
        self.assertGreater(float(w[~mal.numpy()].mean()), 0.2)

    def test_it_moves_them_LESS_than_a_ray_push_would(self):
        """Minimal displacement is what separates 'uncertain' from 'far'. A
        construction that also moved them far would reopen the confusion that
        cost four earlier designs."""
        g, mal = T.uniformar_el_embedding(self.f, self.cab, 1.0, 0)
        d_uni = float((g - self.f).norm(dim=1).mean())
        r, _ = T.intrusar_el_embedding(self.f, 1.0, 4.0, 0)
        self.assertLess(d_uni, float((r - self.f).norm(dim=1).mean()))

    def test_it_replaces_rather_than_adds_and_leaves_the_rest_untouched(self):
        g, mal = T.uniformar_el_embedding(self.f, self.cab, 0.25, 0)
        self.assertEqual(g.shape, self.f.shape)
        self.assertEqual(int(mal.sum()), 50)
        self.assertTrue(torch.equal(g[~mal], self.f[~mal]))

    def test_a_zero_rate_is_a_no_op(self):
        g, mal = T.uniformar_el_embedding(self.f, self.cab, 0.0, 0)
        self.assertTrue(torch.equal(g, self.f))
        self.assertEqual(int(mal.sum()), 0)

    def test_a_run_injects_and_records_it(self):
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        *_, info = T.corrida(fam, D, 'layernorm', semilla=100,
                             mover_fuente=False, cuadros=T.CTRL,
                             hp2={'RHO_UNI': 0.25})
        self.assertEqual(info['hp2']['RHO_UNI'], 0.25)
        self.assertGreater(info['uni_inyectados'], 0)

    def test_a_zero_rate_run_is_the_clean_run_bit_for_bit(self):
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        *_, a = T.corrida(fam, D, 'layernorm', semilla=100, mover_fuente=False,
                          cuadros=T.CTRL, hp2={'RHO_UNI': 0.0})
        *_, b = T.corrida(fam, D, 'layernorm', semilla=100, mover_fuente=False,
                          cuadros=T.CTRL)
        self.assertEqual(a['acc_te'], b['acc_te'])


if __name__ == '__main__':
    unittest.main()
