"""El criterio de la fase 2 es el MAXIMO de `cef`, y el vigia dejo de decidir.

`cef` es el numero EFECTIVO de clases en el ruteo del BLANCO:
`exp(H(histograma de argmax))`, en `[1, CLASES]`. No usa `y_t` en ningun paso
--- es el histograma de las predicciones del propio modelo --- y por eso puede
ser criterio sin fugar la etiqueta que la tabla publica.

POR QUE CAMBIO, medido el 2026-10-16 sobre 90 celdas (tres toysets, diez
semillas, tres posiciones). El criterio de ayer --- el MINIMO DEL VIGIA --- era
el peor de todas las reglas medidas, peor que una constante:

    regla                   GLOBAL  |  lunas   toroide   rollo
    ORACULO (techo)         0.7951  | 0.7071   0.8608   0.8175
    MAX cef                 0.7505  | 0.6227   0.8420   0.7868
    constante epoca 20      0.7048  | 0.5861   0.7601   0.7682
    min loss                0.6471  | 0.5379   0.8295   0.5739
    min vigia (AYER)        0.6410  | 0.5219   0.7894   0.6117

Apareado sobre las 90 celdas, `MAX cef` contra el minimo del vigia:
`d = +0.1095`, `IC95 [+0.0796, +0.1394]`, `t = 7.18`, `p = 2.0e-10`,
Wilcoxon `p = 2.3e-10`, gana en 70 de 90.

Y la razon de que sea un CRITERIO y no una regla con suerte: es la unica senal
que apunta al pico. Spearman con la epoca del pico: `MAX cef` `+0.612`
(`p = 1.4e-10`), `min vigia` `-0.345` (`p = 0.062`), `min loss` `-0.011`
(`p = 0.956`). La loss cambia de SIGNO entre familias --- negativa con el
acierto en lunas y toroide, POSITIVA en rollo --- asi que ninguna regla que la
minimice puede servir a las tres.

EL CAMBIO DE DIRECCION ES LO PELIGROSO de este archivo. La paciencia cuenta
epocas sin un NUEVO MAXIMO; antes contaba epocas sin un nuevo minimo. Un error
de signo ahi es SILENCIOSO: la corrida termina, publica una epoca y un numero,
y nada falla. Por eso el lock de abajo es de CONTENIDO y no de forma --- se
REPLICA la regla declarada sobre la columna del registro y se exige que
reproduzca `mejor_paso` y `mejor_cef` BIT A BIT.

Lo que estos tests NO afirman: que `cef` sea mejor en otro toyset, ni que el
mecanismo --- el termino colapsa el ruteo del blanco si se lo deja correr, y el
mejor modelo esta justo antes de que el colapso arranque --- sea una ley. Es
una correlacion medida en este banco.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

from CREDA import toy_experiment as T


#: La celda mas barata que sirve, la misma que `test_creda_criterio_loss.py`
#: eligio y por la misma razon: la familia y los HP no son del toyset, son del
#: contrato de `corrida`. `DIM` bajo para que la corrida quede en segundos.
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
    """Una corrida, cacheada: todos los tests de abajo leen la misma."""

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


def _epocas(nf):
    """La bitacora por epoca, por NOMBRE y nunca por indice."""
    return [dict(zip(T.CAMPOS_P2_NOMBRES, fila)) for fila in nf['pasada_2']]


class ElNumeroEfectivoDeClases(unittest.TestCase):
    """`cef` sola: los tres extremos que la definen y su cota."""

    def test_the_effective_classes_stay_within_one_and_the_class_count(self):
        """`cef` en `[1, CLASES]`, con los dos extremos EXACTOS.

        Un ruteo a una sola clase vale 1 --- el colapso --- y un ruteo
        perfectamente repartido vale `CLASES`. Si la normalizacion se escapa,
        uno de los dos extremos se corre y esto lo atrapa.
        """
        C = T.CLASES
        colapsado = np.zeros(60, dtype=np.int64)
        self.assertAlmostEqual(T.clases_efectivas(colapsado), 1.0, places=6,
                               msg='un ruteo a UNA clase no vale 1')
        repartido = np.arange(60, dtype=np.int64) % C
        self.assertAlmostEqual(T.clases_efectivas(repartido), float(C),
                               places=6,
                               msg='un ruteo uniforme no vale CLASES')
        dos = np.arange(60, dtype=np.int64) % 2
        self.assertAlmostEqual(T.clases_efectivas(dos), 2.0, places=6,
                               msg='un ruteo a dos clases parejas no vale 2')
        rng = np.random.default_rng(0)
        for _ in range(200):
            r = rng.integers(0, C, size=int(rng.integers(2, 200)))
            v = T.clases_efectivas(r)
            self.assertGreaterEqual(v, 1.0 - 1e-9, r)
            self.assertLessEqual(v, float(C) + 1e-9, r)

    def test_la_cef_acepta_torch_y_numpy_y_da_lo_mismo(self):
        """El criterio la llama con un tensor y el artefacto con un array."""
        r = (np.arange(37, dtype=np.int64) * 7) % T.CLASES
        self.assertEqual(T.clases_efectivas(r),
                         T.clases_efectivas(torch.as_tensor(r)))


class ElCriterioEsElMaximoDeLaCef(unittest.TestCase):
    """De CONTENIDO: se replica la regla y tiene que coincidir bit a bit."""

    @staticmethod
    def _replicar_el_maximo(nf):
        """La regla DECLARADA, releida del registro por epoca.

        Tres reglas, las tres en `corrida`: se lee SOLO al final de cada
        epoca, no antes de `MIN_EPOCAS_2` epocas completas, y se queda con el
        MAXIMO ESTRICTO de `cef`. `mejor_paso` se registra como `paso + 1`.
        """
        ppe = int(nf['pasos_por_epoca'])
        mejor_cef, mejor_paso, mejor_epoca = float('-inf'), -1, 0
        for e in _epocas(nf):
            epoca = int(e['epoca'])
            if epoca + 1 < T.MIN_EPOCAS_2:
                continue
            if float(e['cef']) > mejor_cef:
                mejor_cef = float(e['cef'])
                mejor_epoca = epoca + 1
                mejor_paso = (epoca + 1) * ppe
        return mejor_paso, mejor_cef, mejor_epoca

    def test_the_phase_two_criterion_is_the_maximum_of_cef(self):
        _, nf = _UnaCorrida.una()
        paso, cef, epoca = self._replicar_el_maximo(nf)
        self.assertEqual(paso, int(nf['mejor_paso']),
                         'replicar el MAXIMO de cef sobre pasada_2 da otro '
                         'paso: o la columna no es la que se lee, o la regla '
                         'de epoca/guarda/direccion cambio')
        self.assertEqual(cef, float(nf['mejor_cef']),
                         'el cef publicado no es el de la epoca elegida')
        self.assertEqual(epoca, int(nf['mejor_epoca']))

    def test_the_phase_two_patience_counts_epochs_without_a_new_maximum(self):
        """La DIRECCION de la paciencia, que es donde un signo se esconde.

        Si la corrida paro antes del techo, paro por paciencia, y entonces
        tienen que valer las dos mitades: pasaron `PACIENCIA_2_EPOCAS` epocas
        desde la elegida, y NINGUNA de las posteriores supero el maximo. La
        segunda mitad es la que un `<` por un `>` rompe.
        """
        _, nf = _UnaCorrida.una()
        epocas = _epocas(nf)
        mejor_epoca = int(nf['mejor_epoca'])
        self.assertGreater(mejor_epoca, 0, 'el criterio no eligio ninguna')
        # **La fase 2 NO para, desde el 2026-10-19**, asi que la corrida
        # SIEMPRE llega al techo y la rama que leia la paciencia se fue con
        # ella. Lo que queda es la afirmacion mas fuerte: el techo se corre
        # entero, en vez de «si corto temprano, fue por la paciencia».
        self.assertEqual(int(nf['paso_fin']), int(nf['pasos_techo']),
                         (nf['paso_fin'], nf['pasos_techo'],
                          'la fase 2 no corrio su techo entero'))
        self.assertEqual(int(nf['epoca_fin']), T.EPOCAS_2,
                         (nf['epoca_fin'], T.EPOCAS_2))
        posteriores = [float(e['cef']) for e in epocas
                       if int(e['epoca']) + 1 > mejor_epoca]
        if posteriores:
            self.assertLessEqual(
                max(posteriores), float(nf['mejor_cef']),
                (max(posteriores), nf['mejor_cef'],
                 'una epoca POSTERIOR tiene cef mayor que la elegida: la '
                 'paciencia esta contando en la direccion vieja'))

    def test_el_criterio_no_elige_antes_del_minimo_declarado(self):
        _, nf = _UnaCorrida.una()
        self.assertGreaterEqual(int(nf['mejor_epoca']), T.MIN_EPOCAS_2,
                                (nf['mejor_epoca'], T.MIN_EPOCAS_2))

    def test_the_criterion_signal_and_its_direction_are_readable_from_the_flags(self):
        """Que la senal y su direccion se LEAN del artefacto, no del codigo.

        `CRITERIO_2_EN` ya decia la UNIDAD (epocas) y no la senal. Una corrida
        vieja y una nueva publicaban banderas identicas aunque eligieran por
        cantidades opuestas, asi que el artefacto no podia distinguirlas.
        """
        b = T.banderas()
        for k in ('CRITERIO_2_SENAL', 'CRITERIO_2_DIRECCION'):
            self.assertIn(k, b)
        self.assertEqual(b['CRITERIO_2_SENAL'], 'cef')
        self.assertEqual(b['CRITERIO_2_DIRECCION'], 'maximo')
        self.assertEqual(b['CRITERIO_2_EN'], 'epocas')


class ElVigiaSigueRegistradoYNoDecide(unittest.TestCase):
    """Deja de decidir, NO deja de reportarse. Las dos mitades, afirmadas."""

    def test_the_watchman_is_recorded_and_decides_nothing(self):
        _, nf = _UnaCorrida.una()
        # MITAD UNO: sigue estando, por paso y en el resumen.
        self.assertIn('mejor_vg', nf)
        self.assertIn('vigia_fin', nf)
        for r in nf['hist_t']:
            self.assertEqual(len(r), T.CAMPOS_HT, (len(r), T.CAMPOS_HT))
        vg = [float(r[5]) for r in nf['hist_t']]
        self.assertGreater(max(vg) - min(vg), 0.0,
                           'el vigia no varia: la columna que se reporta no '
                           'esta midiendo nada')
        # MITAD DOS: no decide. Se replica la regla VIEJA y elige otra epoca.
        ppe = int(nf['pasos_por_epoca'])
        mejor_vg, paso_vigia = float('inf'), -1
        for r in nf['hist_t']:
            paso = int(r[0])
            epoca, k_ep = divmod(paso, ppe)
            if k_ep != ppe - 1 or epoca + 1 < T.MIN_EPOCAS_2:
                continue
            if float(r[5]) < mejor_vg:
                mejor_vg, paso_vigia = float(r[5]), paso + 1
        self.assertNotEqual(
            paso_vigia, int(nf['mejor_paso']),
            (paso_vigia, nf['mejor_paso'],
             'el minimo del vigia y el maximo de cef eligen el MISMO paso en '
             'esta celda, asi que este test no puede ver la diferencia: hay '
             'que elegir otra celda, no debilitar la asercion'))

    def test_mejor_vg_SIGUE_SIENDO_SU_MINIMO_y_no_la_epoca_elegida(self):
        """`mejor_vg` NO cambio de significado, y es deliberado.

        La tentacion era redefinirlo como 'el vigia en la epoca que cef
        eligio'. Eso habria cambiado EN SILENCIO lo que la BUSQUEDA DE HP
        optimiza: `busqueda_fase2` devuelve `mean(mejor_vg)` como objetivo a
        minimizar (`_obj`), asi que redefinir el campo redefine el objetivo sin
        tocar una sola linea de la busqueda. El criterio de PARADA cambio; el
        objetivo de la BUSQUEDA no se toco, porque nadie lo pidio.

        Asi que `mejor_vg` sigue siendo el MINIMO del vigia sobre las epocas
        elegibles, y `mejor_re` el Renyi de valid apareado con ESE minimo.
        """
        _, nf = _UnaCorrida.una()
        ppe = int(nf['pasos_por_epoca'])
        elegibles = [float(r[5]) for r in nf['hist_t']
                     if divmod(int(r[0]), ppe)[1] == ppe - 1
                     and divmod(int(r[0]), ppe)[0] + 1 >= T.MIN_EPOCAS_2
                     and int(r[0]) < int(nf['paso_fin'])]
        self.assertTrue(elegibles, 'ninguna epoca elegible en el registro')
        self.assertEqual(float(nf['mejor_vg']), min(elegibles),
                         'mejor_vg dejo de ser el minimo del vigia: la '
                         'busqueda de HP minimiza ESE campo y acaba de '
                         'cambiar de objetivo sin decirlo')


class LaCefViajaEnElArtefacto(unittest.TestCase):
    """La evidencia del criterio tiene que viajar con el artefacto."""

    def test_the_checkpoint_cef_is_derived_from_the_recorded_routing(self):
        """Por punto de control, y DERIVADA de `rut` y no recalculada.

        `cu` ya trae `rut` --- el argmax sobre el train del blanco --- asi que
        la cef del punto se saca de ahi. Calcularla por otra via seria una
        segunda copia de la formula, y dos copias son dos oportunidades de que
        una se quede atras sin que nada falle.
        """
        cu, nf = _UnaCorrida.una()
        serie = nf['cef_cu']
        self.assertEqual(len(serie), len(cu), (len(serie), len(cu)))
        for j in range(len(cu)):
            self.assertEqual(
                float(serie[j]),
                T.clases_efectivas(T.punto(cu, j)['rut']),
                (j, 'la cef del punto no sale de su propio rut'))

    def test_la_cef_del_punto_cero_es_la_del_modelo_sin_adaptar(self):
        """El paso 0 es `enc_t == enc`: el ruteo de antes de adaptar nada."""
        cu, nf = _UnaCorrida.una()
        self.assertEqual(int(T.punto(cu, 0)['paso']), 0)
        self.assertEqual(float(nf['cef_cu'][0]),
                         T.clases_efectivas(T.punto(cu, 0)['rut']))


if __name__ == '__main__':
    unittest.main()
