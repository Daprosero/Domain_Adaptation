"""Un término sin clase activa devuelve cero, y ese cero tiene que derivar.

`forward` devolvía `torch.tensor(0.0)` cuando ninguna clase quedaba activa: un
escalar suelto, SIN `grad_fn`. En el régimen declarado (`mover_fuente=False`)
la cross-entropy de la fuente llega detachada, así que el término de Rényi es
el ÚNICO camino de gradiente del paso --- y un cero suelto mata
`backward()` con «element 0 of tensors does not require grad».

Lo encontró un agente de medición al poner un piso de CONTEO por bloque de
clase: el piso de TRAZA que ya existía (`MIN_CLASS_TRACE = 1e-6`) casi nunca
dispara, así que el camino estaba vivo y nunca se recorría.

**Era alcanzable antes del piso**, y esto lo demuestra: basta con que ninguna
clase tenga muestras de los DOS dominios a la vez. La corrección es
`f_t[:0].sum()` --- la suma de una tajada vacía: vale `0.0` exacto, no depende
del contenido de `f_t` (ni un infinito la vuelve NaN), y su gradiente es cero
exacto. «No hubo clase activa» no es «no hubo paso»: Adam sigue moviéndose por
su momento.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA.models import CREDALoss


def _sin_clase_activa():
    """Ninguna clase tiene muestras de los dos dominios a la vez."""
    f_s = torch.randn(8, 4)
    f_t = torch.randn(8, 4).requires_grad_(True)
    y_s = torch.zeros(8, dtype=torch.long)        # la fuente, toda en la clase 0
    g_t = torch.zeros(8, 3); g_t[:, 1] = 1.0      # el blanco, todo en la clase 1
    return f_s, f_t, y_s, g_t


class ElCeroSinClaseActivaDeriva(unittest.TestCase):

    def setUp(self):
        self.cr = CREDALoss()
        self.cr.class_mixing = 1.0

    def test_the_value_is_exactly_one(self):
        """**Era cero hasta 2026-10-18, y r33 lo cambia a proposito.**

        La Ec. 37 de r31 excluia las clases inactivas de la suma y daba 0 con
        `C_act` vacio, de modo que un codificador colapsado hasta dejar una
        sola clase reportaba MENOS por haber abandonado las demas. r32 le cobra
        `l~_c = 1` a cada clase abandonada, y con todas abandonadas los pesos
        siguen sumando uno: el termino vale exactamente 1.

        El valor cambia; la propiedad que este archivo existe para proteger no
        --- `test_backward_does_not_die` abajo, que es la entera.
        """
        f_s, f_t, y_s, g_t = _sin_clase_activa()
        self.assertEqual(float(self.cr(f_s, f_t, y_s, g_t)), 1.0)

    def test_backward_does_not_die(self):
        """La propiedad entera. Sin esto, un lote donde el ruteo deja sin
        clase activa tumba el paso --- y en el régimen declarado tumba el
        ÚNICO camino de gradiente que hay."""
        f_s, f_t, y_s, g_t = _sin_clase_activa()
        self.cr(f_s, f_t, y_s, g_t).backward()
        self.assertIsNotNone(f_t.grad)
        self.assertEqual(float(f_t.grad.abs().sum()), 0.0)

    def test_the_charge_does_not_depend_on_the_contents_of_f_t(self):
        """`1.0 + f_t[:0].sum()` y no `1.0 + 0 * f_t.sum()`: con un
        infinito o un NaN adentro, el producto daría NaN y el paso envenenaría
        los pesos en vez de no moverlos."""
        f_s, f_t, y_s, g_t = _sin_clase_activa()
        with torch.no_grad():
            f_t[0, 0] = float('inf')
        v = self.cr(f_s, f_t, y_s, g_t)
        self.assertEqual(float(v), 1.0)
        v.backward()
        self.assertFalse(bool(torch.isnan(f_t.grad).any()))

    def test_a_batch_WITH_an_active_class_is_untouched(self):
        """La corrección no toca el camino normal."""
        torch.manual_seed(0)
        f_s = torch.randn(32, 4); f_t = torch.randn(32, 4).requires_grad_(True)
        y_s = torch.randint(0, 3, (32,))
        g_t = torch.softmax(torch.randn(32, 3), 1)
        v = self.cr(f_s, f_t, y_s, g_t)
        self.assertNotEqual(float(v), 0.0)
        v.backward()
        self.assertGreater(float(f_t.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
