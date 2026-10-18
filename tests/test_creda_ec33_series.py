"""La serie de la Ec. 33 por punto de control: las cuatro cantidades, LEIDAS.

La Ec. 33 es

    J_c = 0.5 * ( H2(K_s,c) + H2(K~_t,c) ) - H2(K~_mix,c)

y `CREDALoss.forward` ya calcula las tres entropias por clase en cada paso,
porque las necesita para `J_c`. Lo que estos tests fijan es que la serie que
`corrida` publica sea ESAS entropias y no una segunda via para el mismo numero:

  * `ec33_por_clase` no reimplementa nada --- observa a `CREDALoss` corriendo
    --- y el lock de CONTENIDO de abajo lo exige replicando la ecuacion a mano
    con los helpers del propio modulo, igual que `test_creda_r28_bounds.py` ya
    hace para la Ec. 36;
  * el ORDEN de la captura es `(h_s, h_t, h_mix)` por clase, y es lo que
    `forward` appendea. Un orden corrido no rompe nada que se vea: la serie
    sigue teniendo la forma correcta, los numeros siguen siendo entropias
    plausibles, y `J` sale de la aritmetica equivocada sin que nada falle;
  * la lectura es POR PUNTO DE CONTROL y no por paso. Un espia ingenuo sobre
    `_renyi_entropy_order_2` dispara 54.861 veces en una corrida de 200 epocas
    --- el termino se evalua en cada paso, dos veces (train y valid) --- asi
    que el lock de abajo cuenta las llamadas y exige exactamente una por punto
    de control;
  * el espia es ACOTADO: al salir, `cr` tiene que quedar como estaba. Una
    instancia que se queda con el metodo parchado seguiria capturando durante
    el entrenamiento, y un `class_mixing` no restaurado cambiaria el TERMINO
    que el paso siguiente optimiza --- o sea, la medicion perturbaria lo que
    mide, que es el defecto que este repositorio ya pago dos veces.

Lo que estos tests NO afirman: que `mean_c h_mix_c`, `mean_c h_t_c` o
`mean_c J_c` sirvan como criterio de parada. Eso es una medicion, no una
propiedad del codigo, y vive en el informe.
"""
import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn.functional as F

from CREDA import toy_experiment as T
from CREDA.bounds import entropy_cosh_bounds
from CREDA.models import CREDALoss


#: La celda mas barata que sirve, la misma que `test_creda_criterio_cef.py`
#: eligio y por la misma razon: la familia y los HP no son del toyset, son del
#: contrato de `corrida`.
_FAM = next(f for f in T.FAMILIAS if f['nombre'] == 'lunas')
_POS = 'ninguna'
_SEM = 0
_DIM = 16
_LOTE_2 = 128


def _hp():
    h1 = dict(T.HP_1)
    h1['DIM'] = _DIM
    h2 = dict(T.HP_2)
    h2['LOTE_2'] = _LOTE_2
    return h1, h2


class _UnaCorrida:
    """Una corrida, cacheada: los tests de abajo leen la misma."""

    _cache = {}

    @classmethod
    def una(cls, pos=_POS):
        if pos not in cls._cache:
            h1, h2 = _hp()
            D = T.datos_semilla(_FAM, _SEM)
            _, cu, _, _, nf = T.corrida(
                _FAM, D, pos, semilla=_SEM, balance=True, mover_fuente=False,
                hp1=h1, hp2=h2, cuadros=T.CTRL)
            cls._cache[pos] = (cu, nf)
        return cls._cache[pos]


def _un_bloque(semilla=0, clases=2, n_s=23, n_t=19, dim=4):
    """Una entrada controlada: dos dominios, `clases` clases presentes.

    No sale de una corrida a proposito: el lock de contenido tiene que poder
    replicar la ecuacion a mano, y para eso necesita las entradas EXACTAS.
    """
    g = torch.Generator().manual_seed(semilla)
    f_s = torch.randn(n_s, dim, generator=g)
    f_t = torch.randn(n_t, dim, generator=g) + 0.7
    y_s = torch.arange(n_s) % clases
    logits = torch.randn(n_t, T.CLASES, generator=g) * 1.5
    # Que las `clases` primeras clases esten presentes en el ruteo y las otras
    # no: una clase sin puntos en un dominio se SALTEA en `forward`, y el lock
    # necesita saber cuantas sobreviven.
    logits[:, clases:] -= 10.0
    g_t = F.softmax(logits, 1)
    return f_s, y_s, f_t, g_t


def _un_loss(sopesado=True):
    return CREDALoss(sigma=1.3,
                     normalize_by_bounds=T.NORMALIZAR_POR_COTAS,
                     bounds_form=T.FORMA_DE_COTAS,
                     weight_cross_block=T.HP_2['CRUZADO_SOPESADO'],
                     detach_target_entropy=T.HP_2['RHO_T_SIN_GRAD'],
                     unbiased_purity=T.HP_2['PUREZA_U'],
                     use_entropy_weighting=sopesado,
                     balance_mix_blocks=True,
                     balance_mass=T.BALANCE_DE_MASA)


class LaEcuacion33PorClase(unittest.TestCase):
    """`ec33_por_clase` sola: que sea la Ec. 33 de `CREDALoss` y no otra."""

    def test_las_cuatro_filas_declaradas_y_su_forma(self):
        """Cuatro filas, en el orden declarado, una columna por clase viva."""
        self.assertEqual(T.CAMPOS_EC33_NOMBRES, ('h_s', 'h_t', 'h_mix', 'J'))
        self.assertEqual(T.CAMPOS_EC33, 4)
        cr = _un_loss()
        for clases in (1, 2, 3):
            with self.subTest(clases=clases):
                A = T.ec33_por_clase(cr, *_un_bloque(clases=clases))
                self.assertEqual(A.shape, (T.CAMPOS_EC33, clases),
                                 f'{clases} clases vivas dieron {A.shape}')
                self.assertTrue(np.isfinite(A).all(), A)

    def test_la_cuarta_fila_es_la_aritmetica_de_la_ecuacion(self):
        """`J = 0.5*(h_s + h_t) - h_mix`, BIT A BIT y no aproximado.

        Es la unica fila que no sale de `CREDALoss`: las tres entropias las
        calcula el modulo del metodo y esta es su combinacion. Un signo
        invertido aca es silencioso --- `J` sigue siendo un numero finito, la
        serie sigue teniendo la forma correcta, y la regla de parada que la
        lea elegiria la epoca de enfrente.
        """
        cr = _un_loss()
        for semilla in range(6):
            with self.subTest(semilla=semilla):
                d = T.ec33(
                    [T.ec33_por_clase(cr, *_un_bloque(semilla=semilla,
                                                      clases=3))], 0)
                np.testing.assert_array_equal(
                    d['J'], 0.5 * (d['h_s'] + d['h_t']) - d['h_mix'])

    def test_las_tres_entropias_son_las_de_credaloss_replicadas_a_mano(self):
        """EL LOCK DE CONTENIDO: la Ec. 33 a mano, con los helpers del modulo.

        Replicada como `test_creda_r28_bounds.py::_normalized` ya la replica
        para la Ec. 36, y por la misma razon: una captura que tomara las tres
        entropias en otro orden, o el bloque del blanco SIN ponderar, o la
        mezcla SIN balancear, daria tres numeros finitos y plausibles. Lo
        unico que lo distingue es rehacer la cuenta.
        """
        for sopesado in (True, False):
            for clases in (2, 3):
                with self.subTest(sopesado=sopesado, clases=clases):
                    cr = _un_loss(sopesado=sopesado)
                    f_s, y_s, f_t, g_t = _un_bloque(clases=clases)
                    A = T.ec33_por_clase(cr, f_s, y_s, f_t, g_t)
                    esperado = self._a_mano(cr, f_s, y_s, f_t, g_t)
                    np.testing.assert_allclose(A[:3], esperado, rtol=0,
                                               atol=0.0)

    @staticmethod
    def _a_mano(cr, f_s, y_s, f_t, g_t):
        """Las tres entropias de la Ec. 33, clase por clase, escritas aqui."""
        y_t = g_t.argmax(1)
        if cr.use_entropy_weighting:
            # La Ec. 19, copiada de `CREDALoss.forward` a proposito: un lock de
            # contenido que llamara a la misma funcion no probaria nada.
            w = 1.0 - (-torch.log((g_t ** 2).sum(1) + 1e-6)
                       / math.log(float(g_t.shape[1])))
        else:
            w = None
        filas = []
        with torch.no_grad():
            for c in torch.unique(y_s, sorted=True):
                f_s_c, f_t_c = f_s[y_s == c], f_t[y_t == c]
                if not len(f_s_c) or not len(f_t_c):
                    continue
                K_s = cr._gaussian_kernel(f_s_c, f_s_c, cr.sigma)
                K_t = cr._gaussian_kernel(f_t_c, f_t_c, cr.sigma)
                K_st = cr._gaussian_kernel(f_s_c, f_t_c, cr.sigma)
                if w is not None:
                    w_c = w[y_t == c]
                    K_t = K_t * torch.outer(w_c, w_c)
                    if cr.weight_cross_block:
                        K_st = K_st * w_c.unsqueeze(0)
                h_s = cr._renyi_entropy_order_2(K_s)
                h_t = cr._renyi_entropy_order_2(K_t)
                h_mix = cr._renyi_entropy_order_2(
                    cr._mix_kernel_concat(K_s, K_t, K_st))
                filas.append([float(h_s), float(h_t), float(h_mix)])
        return np.asarray(filas, dtype=np.float64).T

    def test_el_espia_no_sobrevive_a_la_lectura(self):
        """Al salir, `cr` queda como estaba: ni metodo parchado ni mezcla movida.

        Un espia que sobreviviera seguiria capturando en cada paso del
        entrenamiento --- 54.861 veces en una corrida de 200 epocas, medido ---
        y un `class_mixing` no restaurado cambiaria el TERMINO que el paso
        siguiente optimiza. Lo segundo es lo peligroso: la medicion
        perturbaria lo que mide y la corrida no fallaria.
        """
        cr = _un_loss()
        for mezcla in (0.0, 1.0):
            with self.subTest(class_mixing=mezcla):
                cr.class_mixing = mezcla
                antes = set(cr.__dict__)
                T.ec33_por_clase(cr, *_un_bloque())
                self.assertEqual(cr.class_mixing, mezcla,
                                 'class_mixing no se restauro')
                self.assertEqual(set(cr.__dict__), antes,
                                 'quedo un atributo de instancia nuevo')
                self.assertIs(type(cr)._renyi_entropy_order_2.__get__(
                    cr, type(cr)).__func__,
                    CREDALoss._renyi_entropy_order_2,
                    'el metodo de la clase quedo parchado')

    def test_la_lectura_no_mueve_el_azar_ni_el_gradiente(self):
        """Ni RNG global ni grafo: los dos canales por los que perturbaria.

        El bloque del punto de control corre ANTES del paso del optimizador y
        sobre los mismos sorteos: una lectura que consumiera el RNG global
        correria los lotes de toda la fase 2, y una que dejara grafo cargaria
        `backward()` con un camino que el termino no tiene.
        """
        cr = _un_loss()
        torch.manual_seed(12345)
        np.random.seed(12345)
        t_antes = torch.get_rng_state().clone()
        n_antes = np.random.get_state()[1].copy()
        f_s, y_s, f_t, g_t = _un_bloque()
        f_s.requires_grad_(True)
        T.ec33_por_clase(cr, f_s, y_s, f_t, g_t)
        self.assertTrue(torch.equal(torch.get_rng_state(), t_antes),
                        'la lectura consumio el RNG de torch')
        np.testing.assert_array_equal(np.random.get_state()[1], n_antes)
        self.assertIsNone(f_s.grad, 'la lectura dejo gradiente acumulado')


class LaSerieQueCorridaPublica(unittest.TestCase):
    """`info['ec33_cu']`: una entrada por punto de control, y nada mas."""

    def test_una_entrada_por_punto_de_control(self):
        cu, nf = _UnaCorrida.una()
        self.assertIn('ec33_cu', nf)
        serie = nf['ec33_cu']
        self.assertEqual(len(serie), len(cu),
                         'un punto de control sin su Ec. 33')
        for k in range(len(serie)):
            d = T.ec33(serie, k)
            vivas = len(d['J'])
            self.assertGreaterEqual(vivas, 1, (k, 'un punto sin clase viva'))
            self.assertLessEqual(vivas, T.CLASES, (k, vivas))
            for nombre, fila in d.items():
                self.assertTrue(np.isfinite(fila).all(), (k, nombre, fila))

    def test_la_serie_publicada_cumple_la_ecuacion_y_sus_cotas(self):
        """`J` es la aritmetica de la Ec. 33 y cae dentro de las cotas r28.

        La segunda mitad es lo que hace al test de CONTENIDO: `L_c = -ln 2` y
        `U_c = 2 ln cosh((h_t - h_s)/4)` son un TEOREMA sobre la mezcla
        balanceada de la Ec. 32, asi que una terna capturada en otro orden
        --- `h_mix` donde va `h_t` --- se sale de ellas.
        """
        _, nf = _UnaCorrida.una()
        serie = nf['ec33_cu']
        fuera, total = [], 0
        for k in range(len(serie)):
            d = T.ec33(serie, k)
            np.testing.assert_allclose(
                d['J'], 0.5 * (d['h_s'] + d['h_t']) - d['h_mix'],
                rtol=0, atol=0.0)
            for h_s, h_t, J in zip(d['h_s'], d['h_t'], d['J']):
                L, U = entropy_cosh_bounds(torch.as_tensor(h_s),
                                           torch.as_tensor(h_t))
                total += 1
                if not (L - 1e-6 <= J <= float(U) + 1e-6):
                    fuera.append((k, float(L), float(U), J))
        self.assertGreater(total, 0, 'no se midio ninguna clase')
        self.assertEqual(fuera, [], f'{len(fuera)} de {total} fuera de [L, U]')

    def test_la_lectura_no_usa_la_instancia_que_el_bucle_optimiza(self):
        """La vara de la Ec. 33 es OTRA instancia, y esto lo mide por identidad.

        La lectura necesita `class_mixing = 1.0`. Hacerlo sobre la instancia
        del entrenamiento y restaurarlo deja los numeros intactos y el
        REGISTRO falso: `tests/test_toy_entrega_marginal.py` espia
        `CREDALoss.forward` por instancia y afirma que el termino del bucle es
        marginal PURO antes de `PASO_COND`. Medido, la instancia compartida
        pone dos de sus tests en rojo. Este test lo atrapa desde este lado,
        asi que el defecto no puede volver arreglando aquel archivo: con
        `PASO_COND` inalcanzable, la instancia del bucle tiene que ver SOLO
        `0.0`, y la mezcla `1.0` tiene que existir en OTRA.
        """
        h1, h2 = _hp()
        h2['PASO_COND'] = 10 ** 6
        D = T.datos_semilla(_FAM, _SEM)
        por_id = {}
        propio = CREDALoss.forward

        def espia(self, *a, **k):
            por_id.setdefault(id(self), []).append(float(self.class_mixing))
            return propio(self, *a, **k)

        CREDALoss.forward = espia
        try:
            with mock.patch.object(T, 'PASOS_2', 14):
                T.corrida(_FAM, D, _POS, semilla=_SEM, mover_fuente=False,
                          cuadros=(0,), hp1=h1, hp2=h2)
        finally:
            CREDALoss.forward = propio
        # La del bucle es la que MAS llamadas tiene, por construccion: una por
        # paso de train y otra por paso de valid.
        bucle = max(por_id.values(), key=len)
        self.assertEqual(set(bucle), {0.0},
                         f'la instancia del bucle vio {sorted(set(bucle))} '
                         f'con PASO_COND inalcanzable')
        otras = [v for v in por_id.values() if v is not bucle]
        self.assertTrue(any(1.0 in v for v in otras),
                        'ninguna otra instancia leyo con la mezcla en 1.0')

    def test_se_lee_en_el_punto_de_control_y_en_ningun_otro_momento(self):
        """EL LOCK DEL MOMENTO: exactamente una lectura por punto de control.

        Un espia ingenuo sobre `_renyi_entropy_order_2` dispara 54.861 veces
        en una corrida de 200 epocas, porque el termino se evalua en cada paso
        y otra vez sobre valid. Este test cuenta las lecturas de la Ec. 33 y
        exige una por punto de control: si el registro se corriera al cuerpo
        del paso, el conteo saltaria a `PASOS_2 + 1`.
        """
        h1, h2 = _hp()
        D = T.datos_semilla(_FAM, _SEM)
        veces = []
        propio = T.ec33_por_clase

        def contar(*a, **k):
            veces.append(1)
            return propio(*a, **k)

        T.ec33_por_clase = contar
        try:
            _, cu, _, _, nf = T.corrida(
                _FAM, D, _POS, semilla=_SEM, balance=True, mover_fuente=False,
                hp1=h1, hp2=h2, cuadros=T.CTRL)
        finally:
            T.ec33_por_clase = propio
        self.assertEqual(len(veces), len(cu),
                         f'{len(veces)} lecturas para {len(cu)} puntos de '
                         f'control (pasos corridos: {nf["paso_fin"]})')
        self.assertLess(len(veces), int(nf['paso_fin']),
                        'se leyo tantas veces como pasos: es por paso')


if __name__ == '__main__':
    unittest.main()
