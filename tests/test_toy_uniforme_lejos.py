"""Incierto Y lejos: desplazarse DENTRO del subespacio de posterior uniforme.

La proyección sola dejó una tensión que el resultado expuso: desplazamiento
mínimo e incertidumbre máxima implican DAÑO mínimo. El punto se queda donde
estaba y sólo recibe una etiqueta arbitraria, así que envenena poco --- medido,
mejoró el acierto en 4 de 6 celdas.

Para que el peso tenga algo que reparar hace falta un punto que sea incierto Y
dañino. Esos dos requisitos parecían empujar en direcciones opuestas: lo que
daña es estar lejos del bloque, y lo que `w` ve es estar indeciso.

No se oponen. `{f : W f + b uniforme}` tiene codimensión C-1, así que le quedan
D-C+1 direcciones LIBRES. Moverse dentro de ese subespacio aleja el punto
cuanto se quiera y deja el posterior exactamente uniforme: `w = 0` para
siempre, a cualquier distancia.

Eso convierte el desplazamiento en un EJE barrible con el peso clavado en su
mínimo, que es la única forma de separar «el peso no sirve» de «no había daño
que reparar».
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA import toy_experiment as T


class DentroDelSubespacioUniforme(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.cab = nn.Linear(16, T.CLASES)
        for q in self.cab.parameters():
            q.requires_grad_(False)
        torch.manual_seed(1)
        self.f = torch.randn(200, 16) * 2.0

    def test_the_posterior_stays_uniform_at_every_displacement(self):
        """The whole construction. If it drifts, the axis stops being
        displacement-at-constant-weight and becomes two changes."""
        for d in (0.0, 1.0, 4.0, 16.0):
            g, mal = T.uniformar_el_embedding(self.f, self.cab, 0.25, 0,
                                              desplazamiento=d)
            z = self.cab(g[mal])
            self.assertLess(float((z - z.mean(1, keepdim=True)).abs().max()),
                            1e-2, d)

    def test_the_weight_stays_at_zero_at_every_displacement(self):
        for d in (0.0, 1.0, 4.0, 16.0):
            g, mal = T.uniformar_el_embedding(self.f, self.cab, 0.25, 0,
                                              desplazamiento=d)
            w = T.pesos_declarados(self.cab(g))
            self.assertLess(float(w[mal.numpy()].max()), 0.01, d)

    def test_the_distance_grows_with_the_displacement(self):
        mu = self.f.mean(0, keepdim=True)
        prev = -1.0
        for d in (0.0, 1.0, 4.0, 16.0):
            g, mal = T.uniformar_el_embedding(self.f, self.cab, 0.25, 0,
                                              desplazamiento=d)
            cur = float((g[mal] - mu).norm(dim=1).mean())
            self.assertGreater(cur, prev, d)
            prev = cur

    def test_zero_displacement_is_the_plain_projection(self):
        """The ladder contains its own floor."""
        a, _ = T.uniformar_el_embedding(self.f, self.cab, 0.25, 0)
        b, _ = T.uniformar_el_embedding(self.f, self.cab, 0.25, 0,
                                        desplazamiento=0.0)
        self.assertTrue(torch.equal(a, b))

    def test_a_run_records_the_displacement_it_used(self):
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        *_, info = T.corrida(fam, D, 'layernorm', semilla=100,
                             mover_fuente=False, cuadros=T.CTRL,
                             hp2={'RHO_UNI': 0.25, 'DESPL_UNI': 4.0})
        self.assertEqual(info['hp2']['DESPL_UNI'], 4.0)
        self.assertGreater(info['uni_inyectados'], 0)


if __name__ == '__main__':
    unittest.main()
