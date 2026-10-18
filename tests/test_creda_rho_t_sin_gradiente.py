"""Cortar el gradiente del bloque OBJETIVO dentro de la puntuación por clase.

El deliberador propone un candidato para la caída de confianza que mi medición
no explica: la fuerza de `rho_t` dispersa el bloque objetivo, y con el peso
declarado esa fuerza se concentra en los pares de muestras SEGURAS --- entra
como `w_j^2 w_k^2` --- de modo que separa el núcleo confiado de cada clase y
empuja parte de él hacia la frontera. Con el barajado la misma fuerza cae sobre
pares al azar y no ataca al núcleo.

Lo que la medición le exige al mecanismo, y este candidato cumple las tres:
sobrevive SIN bloque cruzado sopesado, sobrevive con `w` sin gradiente, y
depende de la ASIGNACIÓN (no aparece en el barajado).

Su predicción, que puede matarlo: con `rho_t` sin gradiente, la caída de
confianza propia del declarado desaparece.

**El test tiene que ser sobre el GRADIENTE y no sobre el valor.** Un `detach`
no cambia lo que `forward` devuelve, sólo por dónde fluye hacia atrás: un test
sobre la pérdida pasaría con la bandera prendida y con la bandera rota.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA.models import CREDALoss


def _lote(n=96, d=8, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    f_s = torch.randn(n, d, generator=g)
    f_t = (torch.randn(n, d, generator=g)).requires_grad_(True)
    y_s = torch.randint(0, 3, (n,), generator=g)
    g_t = torch.softmax(torch.randn(n, 3, generator=g) * 2.0, 1)
    return f_s, f_t, y_s, g_t


def _grad(cr, lote):
    f_s, f_t, y_s, g_t = lote
    if f_t.grad is not None:
        f_t.grad = None
    cr(f_s, f_t, y_s, g_t).backward()
    return f_t.grad.clone()


class RhoTSinGradiente(unittest.TestCase):

    def _loss(self, **kw):
        return CREDALoss(use_entropy_weighting=True, balance_mix_blocks=True,
                         balance_mass='trace', normalize_by_bounds=True,
                         bounds_form='entropy-cosh', weight_cross_block=True,
                         **kw)

    def test_the_default_is_bit_for_bit_what_the_loss_already_did(self):
        a = _grad(self._loss(), _lote())
        b = _grad(self._loss(detach_target_entropy=False), _lote())
        self.assertTrue(torch.equal(a, b))

    def test_the_value_is_unchanged_which_is_why_the_test_reads_gradients(self):
        lote = _lote()
        v0 = float(self._loss()(*lote))
        v1 = float(self._loss(detach_target_entropy=True)(*lote))
        self.assertAlmostEqual(v0, v1, places=6)

    def test_the_gradient_DOES_change(self):
        a = _grad(self._loss(), _lote())
        b = _grad(self._loss(detach_target_entropy=True), _lote())
        self.assertFalse(torch.allclose(a, b, atol=1e-7))

    def test_the_run_records_the_flag_it_used(self):
        from CREDA import toy_experiment as T
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        *_, a = T.corrida(fam, D, 'layernorm', semilla=100, mover_fuente=False,
                          cuadros=T.CTRL, hp2={'RHO_T_SIN_GRAD': False})
        *_, b = T.corrida(fam, D, 'layernorm', semilla=100, mover_fuente=False,
                          cuadros=T.CTRL)
        *_, c = T.corrida(fam, D, 'layernorm', semilla=100, mover_fuente=False,
                          cuadros=T.CTRL, hp2={'RHO_T_SIN_GRAD': True})
        self.assertEqual(a['acc_te'], b['acc_te'])
        self.assertNotEqual(a['acc_te'], c['acc_te'])


if __name__ == '__main__':
    unittest.main()
