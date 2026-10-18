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


class LaPurezaInsesgada(unittest.TestCase):

    def test_the_default_is_bit_for_bit_what_it_already_was(self):
        K = _bloque()
        a = CREDALoss()._renyi_entropy_order_2(K)
        b = CREDALoss(unbiased_purity=False)._renyi_entropy_order_2(K)
        self.assertEqual(float(a), float(b))

    def test_the_unbiased_one_differs(self):
        K = _bloque()
        a = float(CREDALoss()._renyi_entropy_order_2(K))
        b = float(CREDALoss(unbiased_purity=True)._renyi_entropy_order_2(K))
        self.assertNotAlmostEqual(a, b, places=4)

    def test_it_drops_exactly_the_self_pairs(self):
        """No una aproximación: la identidad algebraica."""
        K = _bloque()
        dg = torch.diagonal(K)
        num = float(torch.trace(K @ K) - (dg**2).sum())
        den = float(torch.trace(K)**2 - (dg**2).sum())
        esperado = -np.log(num/den + 1e-6)
        got = float(CREDALoss(unbiased_purity=True)._renyi_entropy_order_2(K))
        self.assertAlmostEqual(got, esperado, places=5)

    def test_it_kills_the_inequality_penalty_ON_THE_COMPOSITE_SCORE(self):
        """La propiedad vive en `creda_c`, NO en un bloque suelto.

        Medido: a nivel de `H2` de un bloque solo, la reducción es errática ---
        14.1x, 1.4x, 1.7x, 3.2x, 10.5x y 0.97x sobre seis semillas. El
        estadístico U no debiasa cada bloque por separado de forma consistente.

        Lo que sí hace es que el sesgo `1/n_eff` entre en `h_t` y en `h_mix` de
        forma COHERENTE y se cancele en la resta
        `creda_c = h_mix - (h_s + h_t)/2`. Esa cancelación es la propiedad, y
        sobre ella la reducción medida es del 97% con la correlación contra el
        coeficiente de variación cayendo de r=+0.970 a r=+0.057.

        Un test sobre un bloque suelto afirma algo que NO es cierto y habría
        pasado por casualidad en dos de seis semillas.
        """
        g = torch.Generator().manual_seed(11)
        for semilla in range(4):
            gg = torch.Generator().manual_seed(semilla)
            ft = torch.randn(60, 16, generator=gg) * 3.0
            fs = torch.randn(40, 16, generator=gg) * 3.0
            d2 = lambda a, b: ((a*a).sum(1)[:, None] + (b*b).sum(1)[None]
                               - 2*a@b.T)
            Ks = torch.exp(-d2(fs, fs).clamp_min(0)/128.0)
            Kt = torch.exp(-d2(ft, ft).clamp_min(0)/128.0)
            Kst = torch.exp(-d2(fs, ft).clamp_min(0)/128.0)
            w = torch.clamp(1.0 + 0.6*torch.randn(60, generator=g), 0.0, 2.0)
            w = w / w.mean()

            def creda(cr, peso):
                Kw = Kt * torch.outer(peso, peso)
                Kc = Kst * peso.unsqueeze(0)
                ts, tt = float(Ks.trace()), float(Kw.trace())
                M = 0.5*torch.cat([
                    torch.cat([Ks/ts, Kc/np.sqrt(ts*tt)], 1),
                    torch.cat([(Kc/np.sqrt(ts*tt)).T, Kw/tt], 1)], 0)
                h = cr._renyi_entropy_order_2
                return float(h(M) - 0.5*(h(Ks) + h(Kw)))

            uno = torch.ones(60)
            mov = {}
            for nom, cr in (('sesgada', CREDALoss()),
                            ('U', CREDALoss(unbiased_purity=True))):
                mov[nom] = abs(creda(cr, w) - creda(cr, uno))
            self.assertGreater(mov['sesgada'], 0.005, (semilla, mov))
            self.assertLess(mov['U'], mov['sesgada'] / 3.0, (semilla, mov))

    def test_a_block_of_one_sample_is_refused_rather_than_divided_by_zero(self):
        """Con n=1 no hay pares distintos: el denominador es cero y el valor
        que saldría sería el del epsilon y no el del dato."""
        K = _bloque(n=1)
        with self.assertRaises(ValueError):
            CREDALoss(unbiased_purity=True)._renyi_entropy_order_2(K)

    def test_the_toy_REFUSES_the_flag_because_of_its_bounds(self):
        """Y ésta es la consecuencia práctica, dicha como lo que es: el
        remedio queda INDISPONIBLE en el toy tal como está configurado.

        El toy corre `FORMA_DE_COTAS = 'entropy-cosh'`, así que pedirle
        `PUREZA_U=True` es pedir la combinación inadmisible. Se niega en vez de
        correr, que es lo correcto: con 109 de 600 configuraciones fuera de
        [0,1], lo que produciría son puntajes que se leen como medidos.

        Para usarlo hay que rederivar los extremos primero, y eso es
        deliberación y no una bandera.
        """
        from CREDA import toy_experiment as T
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        k = dict(semilla=100, mover_fuente=False, cuadros=T.CTRL)
        self.assertEqual(T.FORMA_DE_COTAS, 'entropy-cosh')
        self.assertIs(T.NORMALIZAR_POR_COTAS, True)
        *_, a = T.corrida(fam, D, 'layernorm', hp2={'PUREZA_U': False}, **k)
        *_, b = T.corrida(fam, D, 'layernorm', **k)
        self.assertEqual(a['acc_te'], b['acc_te'])
        with self.assertRaises(ValueError) as ctx:
            T.corrida(fam, D, 'layernorm', hp2={'PUREZA_U': True}, **k)
        self.assertIn('unbiased_purity', str(ctx.exception))


class LaCombinacionInadmisibleSeRefuta(unittest.TestCase):
    """El estadístico U y las cotas de la Ec. 35 no se pueden combinar.

    Las cotas se prueban sobre la mezcla balanceada con los TRES bloques
    compartiendo el normalizador por traza. El estadístico U le da a cada uno
    el suyo --- `S² − Σu_j²`, que depende de los pesos de ESE bloque --- así
    que la hipótesis se rompe.

    Medido sobre 600 configuraciones aleatorias: con la pureza sesgada, 0/600
    caen fuera de [0,1]; con el estadístico U, **109/600**, con rango de
    −0.7631 a 1.0517. El 18%, y ninguna cota lo delataría: un puntaje fuera de
    rango bajo un mapa que declara acotarlo ahí se lee como un valor medido.
    """

    def test_the_combination_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            CREDALoss(unbiased_purity=True, normalize_by_bounds=True,
                      bounds_form='entropy-cosh', balance_mix_blocks=True,
                      balance_mass='trace')
        m = str(ctx.exception)
        self.assertIn('unbiased_purity', m)
        self.assertIn('entropy-cosh', m)

    def test_each_half_alone_is_still_allowed(self):
        """La guarda prohíbe la COMBINACIÓN, no cada mitad."""
        CREDALoss(unbiased_purity=True, normalize_by_bounds=False)
        CREDALoss(unbiased_purity=False, normalize_by_bounds=True,
                  bounds_form='entropy-cosh', balance_mix_blocks=True,
                  balance_mass='trace', weight_cross_block=True)

    def test_the_other_bounds_form_no_longer_exists(self):
        """**Esto afirmaba lo contrario hasta 2026-10-18.**

        Decía que `conservative` no quedaba bloqueado por esta hipótesis,
        porque sus extremos salen de las cardinalidades y no del normalizador
        compartido --- cierto mientras ese mapa existió. r33 declara UN par de
        extremos y el par de cardinalidades de r25 no está en la formulación,
        así que `conservative_bounds` y `class_global_loss` se retiraron de los
        dos paquetes y la bandera quedó con un solo valor legal.

        Lo que esta clase sigue probando es que la guarda es de la
        COMBINACIÓN: `unbiased_purity=True` con el normalizador apagado
        construye, y es `test_each_half_alone_is_still_allowed` el que lo dice.
        """
        with self.assertRaises(ValueError) as ctx:
            CREDALoss(unbiased_purity=True, normalize_by_bounds=True,
                      bounds_form='conservative')
        self.assertIn('bounds_form', str(ctx.exception))
        self.assertIn('entropy-cosh', str(ctx.exception))
