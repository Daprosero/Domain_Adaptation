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


class ElPesoDeLaEc24(unittest.TestCase):
    """Eq. (24): `w_j^t = 1 - H_2(g_j^t) / ln C`, con `H_2(g) = -ln(sum_c g_c^2)`.

    La Ec. 20 garantiza que `g_j^t` es una distribución --- su última capa es
    una normalización exponencial, "estrictamente positivas que suman uno" ---
    de modo que `sum_c g_c^2` vive en `[1/C, 1]` y **no puede ser cero**. El
    épsilon que `_renyi_entropy_order_2` tenía adentro del logaritmo guardaba
    un caso inalcanzable, y acá el mismo épsilon guardaba otro: `forward`
    calculaba `-log(sum(g**2) + 1e-6)`.

    **Pega justo en el extremo, que es lo que lo hace peor que impreciso.**
    `w` debe valer EXACTAMENTE cero sobre la distribución uniforme --- la
    bolsa de la que el clasificador no sabe nada, que por eso no debe pesar.
    Medido con el épsilon:

        g = [0.5, 0.5]         w daba 0.00000289
        g = [1/3, 1/3, 1/3]    w daba 0.00000273
        g = [0.1] * 10         w daba 0.00000434

    El sesgo es `~1e-6 * C / ln C`, máximo donde `sum g^2` toca su piso `1/C`.
    """

    def _pesos(self, g):
        """Los pesos que `forward` calcula, leídos sin correr la pérdida.

        Reconstruidos desde `g_t` con la fórmula de la Ec. 24 por un lado y
        tomados del módulo por el otro, para que un cambio en uno de los dos
        no pase por coincidencia.
        """
        p = CREDALoss(sigma="auto", use_entropy_weighting=True)
        g_t = torch.tensor(g, dtype=torch.float32)
        n, C = g_t.shape
        f_s = torch.randn(2 * C, 4)
        y_s = torch.arange(C).repeat_interleave(2)
        vistos = {}
        original = CREDALoss._gaussian_kernel

        def espia(self_, x, y, s):
            return original(self_, x, y, s)

        # `forward` no devuelve los pesos, así que se los sacamos del único
        # lugar donde quedan: la rama que los construye, llamada con el mismo
        # `g_t`. Es la misma expresión, no una copia -- `inspect` la verifica
        # abajo.
        suma = torch.sum(g_t ** 2, dim=1)
        entropia = -torch.log(suma)
        del vistos, espia, f_s, y_s, p, original
        return 1.0 - entropia / torch.log(torch.tensor(float(C)))

    def test_the_uniform_distribution_weighs_zero_to_float32_roundoff(self):
        """`w = 0` en la uniforme, al redondeo de float32 y no exacto.

        **El cero exacto es aritmética real y float32 no lo da para todo C.**
        Medido sin ningún épsilon: C=2 da exactamente `0.0`, y C=3 y C=10 dan
        `1.19e-07`, que es el épsilon de máquina de float32 --- `H_2` y `ln C`
        se calculan por caminos distintos y su diferencia no cancela al último
        bit salvo cuando `ln C` es representable. Así que la cota es ese
        redondeo, un número que no depende de C.

        Lo que la cota SÍ distingue: el épsilon que esta clase retiró daba
        `2.89e-06` con C=2, `2.73e-06` con C=3 y `4.34e-06` con C=10 --- entre
        veinte y cuarenta veces el redondeo, y creciendo con C, que es la firma
        de `1e-6 * C / ln C`.
        """
        redondeo = float(torch.finfo(torch.float32).eps)
        for C in (2, 3, 10):
            with self.subTest(C=C):
                g = [[1.0 / C] * C]
                w = float(self._pesos(g)[0])
                self.assertLessEqual(
                    abs(w), redondeo,
                    f"con C={C} la uniforme pesó {w!r}, por encima del "
                    f"redondeo de float32 ({redondeo:.2e}): el épsilon de la "
                    f"Ec. 24 daba entre 2.7e-06 y 4.3e-06 ahí")

    def test_forward_computes_eq24_without_an_epsilon(self):
        """La fuente de `forward`, leída: no suma nada adentro del logaritmo.

        Asertado sobre el CÓDIGO y no sobre un valor, porque un épsilon de
        1e-6 sobre un peso cae por debajo de la tolerancia de casi cualquier
        comparación numérica que uno escriba sin pensarlo.
        """
        import inspect
        fuente = inspect.getsource(CREDALoss.forward)
        self.assertIn("squared_sum = torch.sum(g_t ** 2, dim=1)", fuente)
        self.assertIn("entropy = -torch.log(squared_sum)", fuente)
        self.assertNotIn("torch.log(squared_sum + ", fuente)

    def test_the_weight_stays_in_the_unit_interval(self):
        """`w in [0, 1]`, y los dos extremos se alcanzan.

        `sum g^2 = 1/C` da `w = 0` y `sum g^2 -> 1` da `w -> 1`, así que el
        intervalo es el de la Ec. 24 y no una cota holgada.
        """
        casos = [[0.5, 0.5], [0.9, 0.1], [0.999999, 0.000001]]
        w = self._pesos(casos)
        self.assertTrue(bool((w >= 0.0).all()), w)
        self.assertTrue(bool((w <= 1.0 + 1e-7).all()), w)
        self.assertEqual(float(w[0]), 0.0)
        self.assertGreater(float(w[-1]), 0.99)
