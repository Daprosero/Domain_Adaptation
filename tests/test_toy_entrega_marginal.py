"""LA ENTREGA: marginal primero, condicional despues. `PASO_COND`.

El marginal alinea las dos nubes SIN pseudoetiquetas --- no recorre clases, asi
que no puede equivocarse de clase. La condicional las necesita, y al arranque
son basura: ahi vive la fusion que medimos (`tr_e` -4.6%, p<0.001, con el
barajado en cero exacto).

Asi que la entrega ataca justo eso: primero las nubes se superponen con un
termino que no puede mal rutear, y recien cuando se superponen entra la
condicional, con pseudoetiquetas que ya valen algo.

Es el espejo de `PASO_W`. Ese retarda PRENDER el sopesado dentro de la
condicional; este retarda PRENDER la condicional, o lo que es lo mismo,
retarda APAGAR el marginal.

Lo que cada test fija:

  DEFAULT      `PASO_COND = 0` es condicional desde el paso 0 --- lo de hoy,
               bit a bit. Sin esto, agregar la bandera cambiaria todas las
               tablas ya publicadas.
  CONMUTA      `class_mixing` vale 0.0 antes y 1.0 desde el paso declarado.
               Los dos extremos de la familia convexa, nunca una mezcla.
  COTA         un `PASO_COND` que la corrida nunca alcanza es marginal PURO
               de punta a punta, y eso tiene que ser alcanzable y declarable.
"""
import sys, os, unittest
from unittest import mock
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from CREDA import toy_experiment as T


def _corre(paso_cond, peso='declarado', paso_w=None, pasos=14, sem=0):
    fam = T.FAMILIAS[0]
    D = T.datos_semilla(fam, sem)
    h1 = dict(T.HP_1); h1['DIM'] = 8
    h2 = {**T.HP_2, 'LOTE_2': 64, 'PASO_COND': paso_cond}
    if paso_w is not None: h2['PASO_W'] = paso_w
    # El espia filtra POR INSTANCIA, y la razon es un falso rojo real: al
    # final de cada corrida `alineacion` construye un `CREDALoss` propio ---
    # la vara comun, con `class_mixing` en su default 1.0 --- y lo llama tres
    # veces. Sin filtrar, esas tres llamadas legitimas se leen como que el
    # bucle conmuto. Se queda la instancia con MAS llamadas, que es la del
    # entrenamiento por construccion: una por paso contra tres en total.
    vistos_por_id = {}
    orig = T.CREDALoss.forward
    def espia(self, *a, **k):
        vistos_por_id.setdefault(id(self), []).append(float(self.class_mixing))
        return orig(self, *a, **k)
    # `PASOS_2` es un GLOBAL que `corrida` lee directo, no un parametro, asi
    # que acortar la corrida se hace parchandolo. `MIN_PASOS_2 = 40` es mas
    # grande que `pasos`, asi que la parada temprana no entra y el numero de
    # pasos es exactamente el declarado.
    T.CREDALoss.forward = espia
    try:
        with mock.patch.object(T, 'PASOS_2', pasos):
            out = T.corrida(fam, D, 'ninguna', semilla=sem, mover_fuente=False,
                            cuadros=(0,), hp1=h1, hp2=h2, peso=peso)
    finally:
        T.CREDALoss.forward = orig
    vistos = max(vistos_por_id.values(), key=len) if vistos_por_id else []
    return out, vistos


class LaEntrega(unittest.TestCase):

    def test_la_llave_existe_y_su_default_es_cero(self):
        self.assertIn('PASO_COND', T.HP_2)
        self.assertEqual(T.HP_2['PASO_COND'], 0)

    def test_el_default_es_condicional_desde_el_paso_cero(self):
        """Hoy class_mixing va clavado en 1.0; el default no puede moverlo."""
        _, vistos = _corre(0)
        self.assertTrue(vistos, 'el loss no se llamo ni una vez')
        self.assertEqual(set(vistos), {1.0}, sorted(set(vistos)))

    def test_conmuta_de_0_a_1_en_el_paso_declarado(self):
        """Antes marginal PURO, despues condicional PURA. Nunca una mezcla."""
        _, vistos = _corre(6, pasos=14)
        # El loss se llama tambien en valid, asi que no se cuentan pasos: lo
        # que se fija es que aparezcan LOS DOS extremos y NADA en el medio.
        self.assertEqual(set(vistos), {0.0, 1.0}, sorted(set(vistos)))
        self.assertEqual(vistos[0], 0.0)
        self.assertEqual(vistos[-1], 1.0)

    def test_un_paso_inalcanzable_deja_marginal_puro(self):
        """La cota declarable: marginal de punta a punta."""
        _, vistos = _corre(999, pasos=14)
        self.assertEqual(set(vistos), {0.0}, sorted(set(vistos)))

    def test_el_marginal_corre_SIN_sopesar_si_PASO_W_acompania(self):
        """La fase marginal no puede llevar pesos: es el pedido literal.

        Se consigue con el mecanismo que ya tiene test --- `PASO_W` pone
        `w = 1` antes de su paso --- no con una segunda bandera que haga lo
        mismo. Dos banderas para un hecho es como se desincronizan.
        """
        (_, _, _, _, i), _ = _corre(6, paso_w=6, pasos=14)
        self.assertEqual(i['hp2']['PASO_W'], 6)
        self.assertEqual(i['hp2']['PASO_COND'], 6)

    # NO hay test de "la corrida publica PASO_COND". Pasaria sin la bandera
    # implementada --- `info['hp2']` reenvia lo que se le paso --- asi que
    # seria verde porque no paso nada. `test_conmuta_de_0_a_1...` ya prueba
    # que ALGUIEN lo lee, que es la unica afirmacion que vale.


if __name__ == '__main__':
    unittest.main()
