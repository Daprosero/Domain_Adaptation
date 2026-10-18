"""La pureza como estadístico U: sin los pares de cada muestra consigo misma.

`H₂(K) = −log(tr(K²)/tr(K)²)` cuenta los pares `j = k`, que para un kernel de
diagonal unitaria valen 1 y para el bloque del blanco SOPESADO valen `w_j⁴`.
Ese bloque aporta `1/n_eff` con `n_eff` el tamaño efectivo de muestra, de modo
que desparejar los pesos baja `n_eff` e infla la pureza **sin que la
distribución haya cambiado**. Es el mismo sesgo que el MMD insesgado quita
usando estadísticos U, y por la misma razón.

Medido sobre el kernel con la geometría congelada: el castigo por pesos
desiguales es **97% artefacto**, y la correlación con el coeficiente de
variación de `w` cae de r=+0.970 a r=+0.057 al quitar la diagonal.

El bloque CRUZADO no tiene pares consigo mismo --- fuente y blanco son muestras
distintas --- así que el sesgo lo cargan sólo las purezas internas, y en la
mezcla concatenada la diagonal es exactamente la unión de esos dos conjuntos de
pares propios. Quitarla deja los términos cruzados intactos.

Opt-in, y `False` deja la línea base bit a bit.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA.models import CREDALoss


def _bloque(n=60, d=16, semilla=0, w=None):
    g = torch.Generator().manual_seed(semilla)
    f = torch.randn(n, d, generator=g) * 3.0
    d2 = (f*f).sum(1)[:, None] + (f*f).sum(1)[None] - 2*f@f.T
    K = torch.exp(-d2.clamp_min(0) / 128.0)
    if w is not None:
        K = K * torch.outer(w, w)
    return K


class ElInstrumentoRetirado(unittest.TestCase):
    """`unbiased_purity` se retiró el 2026-10-18, y esto lo afirma.

    **Este archivo lleva el nombre de algo que ya no existe, y eso se dice en
    vez de renombrarlo en silencio.** Estimaba la pureza como un estadístico U
    --- saca los pares `j = k` del numerador Y del denominador --- y se fue por
    tres razones acumuladas, ninguna de estilo:

    1. **No está en la formulación.** r34 declara UNA entropía, la Ec. (10):
       `H_2(A) = -ln[tr(A^T A)]`. Un estadístico U de la pureza no aparece en
       el documento, y la declaración del toyset lo rotulaba `# Instrumento`.
    2. **Nunca corrió.** `PUREZA_U` valía `False` en la declaración y `False`
       en cada corrida que el cuaderno ejecutado registra, una por una.
    3. **Y estaba refutado donde se podía pedir.** Las clases que este archivo
       tenía medían 109 de 600 configuraciones fuera de `[0, 1]`, en un rango
       de `-0.7631` a `1.0517`, y una guarda lo rechazaba junto con el único
       `bounds_form` admisible. Rederivar los extremos de las Ecs. (35)-(36)
       para ese estimador es deliberación y no una bandera.

    Lo que SOBREVIVE de este archivo es `LaInvarianciaDeEscala`, que es sobre
    la única forma que queda --- y que se escribió precisamente porque el
    estimador que quedaba tenía su propio defecto.
    """

    def test_the_keyword_is_gone_and_asking_for_it_is_a_TypeError(self):
        """Y es `TypeError` a propósito, no un `ValueError` con nombre.

        `bounds_form` se quedó con un solo valor legal justamente para que una
        corrida que pida el mapa retirado reciba un mensaje que lo nombre. Acá
        la decisión es la contraria: no hay nada que explicarle a quien pida un
        instrumento que nunca estuvo en la matemática, así que la firma lo
        rechaza y el error dice qué argumento no existe.
        """
        import inspect
        firma = inspect.signature(CREDALoss.__init__)
        self.assertNotIn('unbiased_purity', firma.parameters)
        with self.assertRaises(TypeError) as ctx:
            CREDALoss(unbiased_purity=True)
        self.assertIn('unbiased_purity', str(ctx.exception))

    def test_the_branch_and_its_fourth_epsilon_are_gone_from_the_source(self):
        """La rama se fue entera, y con ella el cuarto épsilon.

        Era el ÚNICO de los cuatro que tapaba un caso alcanzable: el
        estadístico U puede salir no positivo de verdad --- medido, 139 de 400
        veces con dos puntos y 54 de 400 con cuarenta --- y ahí el logaritmo no
        está definido. Los otros tres guardaban ceros que no podían ocurrir.
        """
        import inspect
        fuente = inspect.getsource(CREDALoss._renyi_entropy_order_2)
        for muerto in ('self.unbiased_purity', 'propios', 'estadistico U es'):
            self.assertNotIn(muerto, fuente, muerto)
        # El CÓDIGO, sin docstring ni comentarios: la prosa sí nombra el
        # instrumento y sus épsilon, para registrar por qué se fueron, y un
        # chequeo que no los separe reporta la historia como si fuera el
        # mecanismo.
        codigo = [l for l in fuente.splitlines()
                  if l.strip() and not l.strip().startswith('#')]
        codigo = "\n".join(codigo).split('"""')[-1]
        self.assertNotIn('1e-6', codigo, codigo)
        self.assertIn('torch.trace(K_norm.T @ K_norm)', codigo)

    def test_the_toyset_no_longer_declares_it(self):
        """`PUREZA_U` sale de `HP_2`, y el artefacto en disco no la llevaba.

        Comprobado antes de sacarla: el `hp2` que `barrido.json` guarda tiene
        las llaves BUSCADAS, no el merge completo de `HP_2`, así que retirar
        una llave no rompe la lectura de lo ya medido.
        """
        from CREDA import toy_experiment as T
        self.assertNotIn('PUREZA_U', T.HP_2)

    def test_the_other_bounds_form_no_longer_exists(self):
        """El mapa de r25 tampoco existe, y pedirlo se nombra.

        **Esto afirmaba lo contrario hasta 2026-10-18.** Decía que
        `conservative` no quedaba bloqueado por la hipótesis `entropy-cosh`,
        porque sus extremos salen de las cardinalidades y no del normalizador
        compartido --- cierto mientras ese mapa existió. r33 declara UN par de
        extremos y el par de cardinalidades de r25 no está en la formulación,
        así que `conservative_bounds` y `class_global_loss` se retiraron de los
        dos paquetes y la bandera quedó con un solo valor legal.

        La guarda de la COMBINACIÓN que esta clase probaba se fue con el
        instrumento: ya no hay dos estimadores que combinar.
        """
        import CREDA.models as models
        self.assertEqual(models.BOUNDS_FORMS, ('entropy-cosh',))
        with self.assertRaises(ValueError) as ctx:
            CREDALoss(normalize_by_bounds=True, bounds_form='conservative')
        self.assertIn('bounds_form', str(ctx.exception))
        self.assertIn('entropy-cosh', str(ctx.exception))


class LaInvarianciaDeEscala(unittest.TestCase):
    """Eq. (9) normalizes by the trace, so `H_2` is invariant to the kernel's
    SCALE: `H_2(cK) = H_2(K)` for every `c > 0`.

    **The estimator's own epsilon is what broke it.** `trace(K) + 1e-6` is a
    constant added to a quantity that can be far smaller than it: once
    `tr(K)` approaches `1e-6` the epsilon IS the normalizer, and the entropy
    reported is the epsilon's and not the matrix's. Measured on one full-rank
    block in float32, exact entropy 1.14525202:

        tr(K) = 1e+00   ->  1.14525080   (the log's own epsilon)
        tr(K) = 1e-06   ->  2.53153396
        tr(K) = 1e-09   ->  13.53976822
        tr(K) = 1e-20   ->  13.81551075   saturated at -log(1e-6)

    The last one is not merely wrong: a 5x5 density matrix cannot carry more
    than `ln 5 = 1.609` nats, so the estimator leaves the quantity's own
    range. `MIL_CREDA.renyi.trace_normalize` divides by the TRUE trace and
    refuses on a non-positive one, and that form is exact to 7 digits in
    float32 from `tr(K) = 1` down to `1.175e-38`, the smallest normal.

    The sibling `test_the_default_is_bit_for_bit_what_it_already_was` keeps the
    two forks agreeing with each other; this one keeps the default fork
    agreeing with Eq. (9).
    """

    #: Below float32's smallest normal the matrix itself is denormal and the
    #: division loses digits, so this is where the scale claim stops holding
    #: and `MIN_CLASS_TRACE` has to take over.
    ESCALAS = (1.0, 1e-3, 1e-6, 1e-9, 1e-20, 1e-30, 1e-37)

    def _referencia(self, K):
        """Eq. (9) in float64, computed independently of the module."""
        A = K.to(torch.float64)
        A = A / torch.trace(A)
        return float(-torch.log(torch.trace(A @ A)))

    def test_the_biased_entropy_is_invariant_to_the_blocks_scale(self):
        K = _bloque()
        exacta = self._referencia(K)
        # The fixture has to carry real entropy, or 0 == 0 would pass this.
        self.assertGreater(exacta, 0.1, "el bloque no lleva entropía que medir")
        p = CREDALoss()
        for c in self.ESCALAS:
            with self.subTest(escala=c):
                got = float(p._renyi_entropy_order_2(K * c))
                self.assertAlmostEqual(
                    got, exacta, places=4,
                    msg=f"a escala {c:.0e} la entropía leyó {got:.8f} donde "
                        f"Ec. (9) da {exacta:.8f}")

    def test_the_entropy_never_leaves_the_range_its_own_size_allows(self):
        """`0 <= H_2(A) <= ln n` for a unit-trace PSD `A` of size `n`.

        The upper end is the uniform spectrum and the lower is a pure state.
        Asserted across the same scales, because this is the half that shows
        the old estimator was not merely imprecise: it reported 13.82 nats on
        a 60x60 block whose ceiling is `ln 60 = 4.09`.
        """
        K = _bloque()
        techo = float(np.log(K.shape[0]))
        p = CREDALoss()
        for c in self.ESCALAS:
            with self.subTest(escala=c):
                got = float(p._renyi_entropy_order_2(K * c))
                self.assertGreaterEqual(got, -1e-6, f"escala {c:.0e}")
                self.assertLessEqual(
                    got, techo + 1e-6,
                    f"a escala {c:.0e} la entropía leyó {got:.6f} nats, y una "
                    f"matriz de densidad {K.shape[0]}x{K.shape[0]} no puede "
                    f"llevar más de ln {K.shape[0]} = {techo:.6f}")

    def test_a_non_positive_trace_is_refused_and_not_stabilized(self):
        """Eq. (9) is defined only for a positive trace, so the estimator says
        so instead of returning the epsilon's entropy.

        This is `MIL_CREDA.renyi.trace_normalize`'s own discipline -- "a Gram
        matrix with non-positive trace has no normalization" -- and the same
        discipline `unbiased_purity`'s branch already keeps for its own
        denominator.
        """
        p = CREDALoss()
        with self.assertRaises(ValueError) as caja:
            p._renyi_entropy_order_2(torch.zeros(4, 4))
        # El mensaje está en castellano, como el de la rama de arriba
        # (`unbiased_purity necesita al menos dos muestras...`): este módulo
        # habla así y esto lo extiende en vez de introducir un segundo idioma.
        self.assertIn("traza", str(caja.exception).lower())
