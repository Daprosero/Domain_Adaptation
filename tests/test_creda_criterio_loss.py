"""La serie por paso YA sale de `corrida`, y el criterio publicado no se mueve.

Esto existe por un probe: el operador pidio comparar la parada de HOY --- el
MINIMO DEL VIGIA, `acc(cab(enc_t(Xs_va)), ys_va)` --- contra la parada en el
MINIMO DE `re_val`, el Renyi de VALID sin ponderar. La forma barata de hacerlo
es recomputar los DOS criterios sobre UNA corrida, de modo que la comparacion
sea apareada por construccion: mismos pesos, misma trayectoria, y lo unico que
cambia es QUE paso se elige.

Para eso hacen falta tres propiedades, y ninguna de las tres estaba afirmada en
ningun test. Las tres se midieron el 2026-10-16 antes de escribir esto:

  1. `hist_t` SALE de `corrida`, dentro de `info`. La premisa con la que se
     encargo el probe era que no sale --- y es falsa en el borde de la funcion
     y verdadera en el ARTEFACTO: `barrido.json` guarda `mejor_re`, `mejor_vg`,
     `paso_fin` y `mejor_paso` por fila y ninguna serie por paso. Medido: las
     llaves de una fila de celda son 17 y `hist_t` no esta entre ellas. Asi que
     NO hubo cambio de fuente: lo que faltaba era el test que fija el contrato,
     no el contrato.

  2. QUE COLUMNA es cual. `hist_t` es una tupla de 6 posiciones
     --- `(paso, t_ce, t_re, term, re_val, vigia)` --- y la aridad bajo de 7 a 6
     el 2026-10-11 al retirarse el warmup, corriendo las dos ultimas columnas.
     Un probe que leyera `r[5]` creyendo que es el vigia cuando es otra cosa
     produciria una tabla con la forma correcta y los numeros de otra cantidad.
     Una prueba de FORMA no lo atrapa: las seis son floats. Por eso el lock de
     abajo es de CONTENIDO --- se REPLICA el criterio declarado sobre las
     columnas y se exige que reproduzca `mejor_paso`, `mejor_vg` y `mejor_re`
     BIT A BIT. Si dos columnas se cruzan, la replica discrepa.

  3. El ACIERTO en un paso elegido sale del punto de control de ese paso. El
     numero publicado viene de restaurar los pesos en `mejor_paso`; recomputar
     QUE paso elegiria otro criterio es gratis, pero el acierto en ese paso no
     lo seria si hubiera que volver a correr. Medido: con `cuadros` denso,
     `punto(cu, i)['acc_te']` en el `paso == mejor_paso` es BIT A BIT igual a
     `info['acc_te']`, el restaurado. Eso es lo que hace al probe barato.

Y la cuarta, que es la que protege lo publicado: pasar un `cuadros` DENSO no
mueve la trayectoria. El bloque de puntos de control corre en `eval()` y bajo
`no_grad()`, y sus permutaciones salen de generadores LOCALES
(`manual_seed(7)`), asi que no consume el RNG global; `proy` esta clavado a la
fuente de la fase 1 y no se recalcula. Eso es un ARGUMENTO, no una medicion, y
por eso el test de abajo lo MIDE: los cinco escalares del criterio y las 170x6
celdas de `hist_t` salen identicas con la rejilla de 11 puntos y con la de 201.

Lo que estos tests NO afirman: que la parada en `re_val` sea mejor. Eso es la
pregunta del probe, no una propiedad del codigo, y el probe vive fuera del
repositorio.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from CREDA import toy_experiment as T


#: La celda mas barata que sirve: `lunas` sin normalizacion, una semilla. La
#: familia y los HP no importan para ninguna de las cuatro propiedades --- son
#: del contrato de `corrida`, no de un toyset --- y `DIM` se baja para que las
#: dos corridas juntas queden en el orden de dos segundos.
_FAM = next(f for f in T.FAMILIAS if f['nombre'] == 'lunas')
_POS = 'ninguna'
_SEM = 0

#: `DIM` y `LOTE_2` ELEGIDOS, y la eleccion es la medicion de abajo y no un
#: gusto. La primera version uso `DIM=8` con el `LOTE_2` por default y la celda
#: salia DEGENERADA para el test del restaurado: paraba en el techo
#: (`paso_fin = 200`) con `mejor_paso = 198`, y el acierto de test en 198 y en
#: 200 era el MISMO numero --- 0.441667 en los dos. O sea que el test pasaba
#: igual si la restauracion no existiera, que es la unica cosa que ese test
#: existe para atrapar. Medido sobre ocho configuraciones el 2026-10-16, esta
#: es la que lo hace morder: `paso_fin = 135` (para TEMPRANO, asi que el camino
#: de la paciencia tambien corre), `mejor_paso = 90`, y el acierto de test en
#: 90 es 0.425000 contra 0.427778 en el ultimo punto. La diferencia es chica y
#: eso no la debilita: el test es de IGUALDAD BIT A BIT, asi que 2.8e-03 lo
#: separa tanto como 0.5.
#:
#: No salen del artefacto a proposito. `CREDA/Results/tesis/toyset/barrido.json`
#: existe en este disco y esta SIN RASTREAR, asi que un clon no lo tiene y un
#: test que lo leyera seria rojo ahi por una razon que no es la suya.
_DIM = 16
_LOTE_2 = 128


def _hp():
    h1 = dict(T.HP_1)
    h1['DIM'] = _DIM
    h2 = dict(T.HP_2)
    h2['LOTE_2'] = _LOTE_2
    return h1, h2


class _UnaCorridaCompartida:
    """El par de corridas --- rejilla publicada y rejilla densa --- por posicion.

    Cuesta ~2s por posicion y todos los tests las leen. Hacerlas por test
    serian veinte corridas para medir lo mismo.

    `batchnorm` esta aca y no solo `ninguna`, y el motivo es una mutacion que
    SOBREVIVIO. La neutralidad de la rejilla densa se probo primero solo sobre
    `ninguna`, que no tiene buffers: la via mas plausible por la que densificar
    MOVERIA la trayectoria es justamente un BatchNorm que reestime
    `running_mean`/`running_var` en el forward del punto de control, y sobre
    una red sin normalizacion esa via no existe. Una prueba de neutralidad que
    no puede ver la via plausible no es una prueba de neutralidad.
    """

    _cache = {}

    @classmethod
    def par(cls, pos=_POS):
        if pos not in cls._cache:
            h1, h2 = _hp()
            D = T.datos_semilla(_FAM, _SEM)
            comun = dict(semilla=_SEM, balance=True, mover_fuente=False,
                         hp1=h1, hp2=h2)
            # La rejilla PUBLICADA, la que `barrer_tablas` pasa.
            _, cu_c, _, br_c, nf_c = T.corrida(_FAM, D, pos,
                                               cuadros=T.CTRL, **comun)
            # Y la densa, un punto por paso, que es la que el probe necesita.
            denso = list(range(0, T.PASOS_2 + 1))
            _, cu_d, _, br_d, nf_d = T.corrida(_FAM, D, pos,
                                               cuadros=denso, **comun)
            cls._cache[pos] = (cu_c, br_c, nf_c, cu_d, br_d, nf_d)
        return cls._cache[pos]


class LaSerieSaleDeCorrida(unittest.TestCase):
    """Propiedad 1: `hist_t` cruza el borde de `corrida`, dentro de `info`."""

    def test_info_trae_la_serie_por_paso(self):
        *_, nf = _UnaCorridaCompartida.par()[3:]
        self.assertIn('hist_t', nf,
                      'la serie por paso no sale de corrida; sin ella los dos '
                      'criterios no se pueden recomputar de UNA corrida y la '
                      'comparacion deja de ser apareada')
        self.assertTrue(nf['hist_t'], 'la serie sale vacia')

    def test_una_fila_por_paso_entrenado_y_sin_huecos(self):
        """`paso_fin` pasos, numerados 0..paso_fin-1, en orden y completos.

        Un hueco no es cosmetico: el argmin de un criterio es un INDICE de esta
        serie, y una serie con huecos devuelve el paso de al lado.
        """
        nf = _UnaCorridaCompartida.par()[5]
        pasos = [int(r[0]) for r in nf['hist_t']]
        self.assertEqual(pasos, list(range(int(nf['paso_fin']))),
                         'la serie por paso no cubre 0..paso_fin-1 exactamente')

    def test_la_aridad_declarada_se_sostiene(self):
        """`CAMPOS_HT` y las filas tienen que decir lo mismo.

        La aridad ya se movio: de 7 a 6 el 2026-10-11, al irse el factor del
        warmup, y las dos columnas de despues se corrieron.
        """
        nf = _UnaCorridaCompartida.par()[5]
        for k, r in enumerate(nf['hist_t']):
            self.assertEqual(len(r), T.CAMPOS_HT, (k, len(r), T.CAMPOS_HT))


class LasColumnasSonLasQueElProbeLee(unittest.TestCase):
    """Propiedad 2, de CONTENIDO: se replica el criterio y tiene que coincidir.

    Este es el unico test del archivo que puede atrapar un cruce de columnas, y
    la razon de que sea de contenido y no de forma es que las seis columnas son
    floats: cualquier permutacion de ellas pasa un test de forma.
    """

    @staticmethod
    def _replicar_el_vigia(nf):
        """El criterio PUBLICADO, releido de la serie.

        Las tres reglas que lo definen, y las tres estan en el codigo de
        `corrida`: se lee SOLO al final de cada epoca, no antes de
        `MIN_EPOCAS_2` epocas completas, y se queda con el MINIMO estricto.
        `mejor_paso` se registra como `paso + 1`.
        """
        ppe = int(nf['pasos_por_epoca'])
        mejor_vg, mejor_re, mejor_paso = float('inf'), float('inf'), -1
        for r in nf['hist_t']:
            paso = int(r[0])
            re_val, vigia = float(r[4]), float(r[5])
            epoca, k_ep = divmod(paso, ppe)
            if k_ep != ppe - 1:
                continue
            if epoca + 1 < T.MIN_EPOCAS_2:
                continue
            if vigia < mejor_vg:
                mejor_vg, mejor_re, mejor_paso = vigia, re_val, paso + 1
        return mejor_paso, mejor_vg, mejor_re

    def test_la_replica_del_vigia_esta_bien_definida_y_no_es_la_publicada(self):
        """**Esto afirmaba `paso == mejor_paso` y la premisa se retiro.**

        Cuando se escribio, el 2026-10-16, el criterio publicado ERA el minimo
        del vigia, asi que replicarlo sobre `hist_t` tenia que dar el mismo
        paso. Ese mismo dia el criterio paso a ser el MAXIMO de `cef`
        (`toy_experiment`: "Era el minimo del vigia hasta el 2026-10-16"), de
        modo que la igualdad quedo afirmando una regla que ya no corre --- y
        aguanto por coincidencia hasta que r33 movio el valor de la perdida y
        con el la trayectoria.

        Quien fija el criterio VIVO, bit a bit y sobre la columna `cef`, es
        `tests/test_creda_criterio_cef.py`. Lo que este archivo sigue
        estableciendo es lo que el probe necesitaba y nada mas: que la columna
        del vigia es la que se cree, que la regla de epoca/guarda se puede
        replicar sin ambiguedad, y que da un paso DISTINTO del publicado ---
        que es justamente la comparacion apareada que se encargo.
        """
        nf = _UnaCorridaCompartida.par()[5]
        paso, vigia, _ = self._replicar_el_vigia(nf)
        ppe = int(nf['pasos_por_epoca'])

        # Bien definida: eligio un paso real, en un borde de epoca, y con un
        # valor del vigia que esta en la propia serie.
        self.assertGreater(paso, 0, 'la replica no eligio ningun paso')
        self.assertEqual(paso % ppe, 0, (paso, ppe))
        self.assertIn(vigia, [float(r[5]) for r in nf['hist_t']])

        # Y NO es el publicado, porque el publicado sale de otro criterio.
        self.assertNotEqual(paso, int(nf['mejor_paso']), (
            'la replica del vigia dio el mismo paso que el criterio publicado '
            '(max cef), asi que esta corrida no distingue los dos criterios y '
            'la comparacion apareada que motivo este archivo no mide nada'))

    def test_la_guarda_de_epocas_minimas_queda_SIN_FIJAR_y_se_dice(self):
        """Lo que estos tests NO pinchan, afirmado en vez de dejado en silencio.

        Se mutó `MIN_EPOCAS_2` a `1` dentro de `corrida` y los doce tests
        siguieron verdes: el MUTANTE SOBREVIVIO. No es que la asercion sea
        floja --- es que la guarda es INERTE en esta celda. Medido, el vigia
        por fin de epoca baja casi monotonamente (0.9528, 0.9528, 0.9583,
        0.9528, 0.9472, ... 0.9306), asi que el minimo de las cuatro primeras
        epocas (0.9528) nunca le gana al de las posteriores (0.9306) y quitar
        la guarda no cambia la eleccion. Y eso no es un accidente de la celda:
        el vigia ARRANCA alto --- `enc_t` empieza siendo `enc`, que es el
        encoder entrenado sobre `Xs` --- y cae a medida que se mueve, asi que
        la guarda solo podria morder en una corrida que bajara y volviera a
        subir dentro de las cuatro primeras epocas.

        Este test fija esa INERTITUD, que es lo que de verdad se midio. Si
        algun dia una celda hiciera viva la guarda, este test se pone rojo y lo
        que hay que hacer entonces es escribir el test que la fija, no aflojar
        esto.
        """
        nf = _UnaCorridaCompartida.par()[5]
        ppe = int(nf['pasos_por_epoca'])
        fin = [(int(r[0]) // ppe + 1, float(r[5])) for r in nf['hist_t']
               if int(r[0]) % ppe == ppe - 1]
        pronto = [v for e, v in fin if e < T.MIN_EPOCAS_2]
        tarde = [v for e, v in fin if e >= T.MIN_EPOCAS_2]
        self.assertTrue(pronto and tarde, (len(pronto), len(tarde)))
        self.assertGreaterEqual(
            min(pronto), min(tarde),
            f'la guarda de MIN_EPOCAS_2 PASO A SER VIVA en esta celda '
            f'(minimo temprano {min(pronto)} < minimo tardio {min(tarde)}). '
            f'Hasta ahora ningun test de este archivo la fijaba, y la mutacion '
            f'que la quita sobrevivia; ahora que puede morder, hace falta el '
            f'test que la fije.')

    def test_la_replica_reproduce_el_vigia_bit_a_bit(self):
        """Bit a bit y no aproximado: es LA MISMA lectura, no un recalculo."""
        nf = _UnaCorridaCompartida.par()[5]
        _, vg, _ = self._replicar_el_vigia(nf)
        self.assertEqual(vg, float(nf['mejor_vg']),
                         'la columna 5 de hist_t no es el vigia que el '
                         'criterio minimiza')

    def test_la_replica_reproduce_el_renyi_de_valid_bit_a_bit(self):
        """Lo que fija que la columna 4 es `re_val`, que es la senal del probe.

        `mejor_re` se registra en el MISMO instante que `mejor_vg` --- es el
        `re_val` del paso que el vigia eligio --- asi que reproducirlo desde la
        columna 4 la identifica sin ambiguedad.
        """
        nf = _UnaCorridaCompartida.par()[5]
        _, _, re = self._replicar_el_vigia(nf)
        self.assertEqual(re, float(nf['mejor_re']),
                         'la columna 4 de hist_t no es el Renyi de VALID que '
                         'la busqueda minimiza')

    def test_el_vigia_discrimina_y_no_satura(self):
        """La premisa de la comparacion, medida y no supuesta.

        En MIL el vigia sobre 15 bolsas de valid tomo DOS valores en toda la
        corrida: un criterio que solo toma dos valores no elige un paso, elige
        un empate. El de CREDA va sobre `Xs_va` entero, asi que deberia
        discriminar --- y eso es justo lo que no se puede asumir, porque es la
        premisa de la que cuelga todo el probe. Medido el 2026-10-16: 57
        valores distintos sobre 170 pasos. El umbral de abajo es flojo a
        proposito: afirma que NO satura, no cuanto discrimina.
        """
        nf = _UnaCorridaCompartida.par()[5]
        vg = [float(r[5]) for r in nf['hist_t']]
        self.assertGreater(len(set(vg)), 10,
                           f'el vigia toma {len(set(vg))} valores distintos en '
                           f'{len(vg)} pasos: esta saturando, y un criterio '
                           f'saturado no elige un paso')


class ElAciertoDelPasoElegidoSaleDelPuntoDeControl(unittest.TestCase):
    """Propiedad 3: el mecanismo que hace al probe barato.

    Si esto no se sostiene, medir el acierto del paso que el otro criterio
    elegiria exige volver a correr y restaurar --- y entonces la comparacion
    deja de ser una recomputacion sobre una trayectoria.
    """

    def test_el_paso_elegido_tiene_su_punto_en_la_rejilla_densa(self):
        """Con la rejilla PUBLICADA puede no tenerlo, y de hecho no lo tiene.

        `CTRL` va de 20 en 20 y `mejor_paso` es un multiplo de
        `pasos_por_epoca`, que en esta celda es 17: medido, el paso elegido fue
        85 y 85 no esta en CTRL. Esa es la razon por la que el probe necesita
        la rejilla densa en vez de leer el artefacto.
        """
        _, _, nf_c, cu_d, _, nf_d = _UnaCorridaCompartida.par()
        m = int(nf_d['mejor_paso'])
        self.assertIn(m, [int(r[0]) for r in cu_d],
                      'la rejilla densa no trae el paso elegido')
        ppe = int(nf_d['pasos_por_epoca'])
        self.assertNotEqual(
            ppe % 20, 0,
            'la rejilla de control y la de epocas se volvieron conmensurables; '
            'si eso pasa a proposito, este test sobra, pero mientras no lo sea '
            'es la razon de que el probe no pueda leer el artefacto')

    def test_el_acierto_del_punto_iguala_al_restaurado_bit_a_bit(self):
        """Lo que autoriza a leer el acierto de CUALQUIER paso de la rejilla.

        El punto de control de `paso = p` se mide ARRIBA de la iteracion `p`,
        antes del paso del optimizador, o sea sobre el modelo despues de `p`
        pasos. Y `mejor_paso = paso + 1` se registra DESPUES del paso, o sea
        tambien despues de `p` pasos. Los dos nombran el mismo modelo, y por
        eso los dos numeros son el mismo bit a bit.

        Vale porque esta fila corre con `mover_fuente=False`: `enc` y `cab` no
        se mueven, asi que restaurar solo `enc_t` restaura el modelo entero. Si
        la fuente se soltara, el restaurado y el punto de control podrian
        diferir y este test tiene que verse fallar antes de creerle.
        """
        _, _, _, cu_d, _, nf_d = _UnaCorridaCompartida.par()
        self.assertFalse(nf_d['mover_fuente'],
                         'esta igualdad se afirma para la fila de la fuente '
                         'quieta; con la fuente suelta hay que volver a medirla')
        m = int(nf_d['mejor_paso'])
        i = next(k for k, r in enumerate(cu_d) if int(r[0]) == m)
        # LA GUARDA DE LA FIXTURE, antes de la igualdad. Si el acierto del paso
        # elegido coincidiera con el del ULTIMO paso, la igualdad de abajo
        # pasaria tambien sin restauracion --- la fixture seria igual al mutante
        # y el test estaria verde sin medir nada. Eso paso de verdad con la
        # primera fixture (`DIM=8`): 0.441667 en el paso 198 y en el 200.
        ultimo = T.punto(cu_d, len(cu_d) - 1)['acc_te']
        self.assertNotEqual(
            T.punto(cu_d, i)['acc_te'], ultimo,
            f'la fixture es degenerada: el acierto en el paso elegido ({m}) y '
            f'en el ultimo ({int(cu_d[-1][0])}) son el mismo numero, asi que la '
            f'igualdad de abajo pasa con o sin restauracion. Hay que elegir '
            f'otra celda, no aflojar la asercion.')
        self.assertEqual(T.punto(cu_d, i)['acc_te'], float(nf_d['acc_te']),
                         'el acierto del punto de control en el paso elegido no '
                         'es el acierto restaurado: el probe no puede leer el '
                         'acierto de un paso de la serie')


class LaRejillaDensaNoMueveLoPublicado(unittest.TestCase):
    """Propiedad 4: la prueba de que el default no cambio.

    El probe corre con `cuadros` denso. Si la rejilla moviera la trayectoria,
    todo lo que el probe mida seria de otro experimento que el publicado --- y
    la comparacion con el numero de la tesis no querria decir nada.

    Corre sobre DOS posiciones y no una. `ninguna` es la celda barata;
    `batchnorm` es la que tiene BUFFERS, que es la unica via plausible por la
    que densificar movería algo --- un punto de control que no fuera a `eval()`
    reestimaria `running_mean`/`running_var` en su forward, y 201 puntos lo
    harian veinte veces mas que 11. Sobre `ninguna` esa via no existe, asi que
    el test ahi pasa sin poder verla.
    """

    #: Lo que el criterio publicado produce, y que no puede moverse.
    CAMPOS = ('paso_fin', 'mejor_paso', 'mejor_epoca', 'epoca_fin',
              'mejor_vg', 'mejor_re', 'acc_te', 'acc_s_te', 'vigia_fin',
              'pasos_por_epoca')

    #: Las dos posiciones sobre las que se mide. `layernorm` no agrega una via
    #: nueva --- no tiene estadisticas corridas --- y cuesta otra corrida.
    POSICIONES = ('ninguna', 'batchnorm')

    def test_los_escalares_del_criterio_son_identicos(self):
        for pos in self.POSICIONES:
            with self.subTest(pos=pos):
                _, _, nf_c, _, _, nf_d = _UnaCorridaCompartida.par(pos)
                for k in self.CAMPOS:
                    self.assertEqual(
                        nf_c[k], nf_d[k],
                        f'{pos}: {k} se movio al densificar la rejilla de '
                        f'puntos de control: {nf_c[k]!r} -> {nf_d[k]!r}')

    def test_la_serie_por_paso_es_identica_en_batchnorm(self):
        """La posicion CON buffers, que es la que puede derivar de verdad."""
        _, _, nf_c, _, _, nf_d = _UnaCorridaCompartida.par('batchnorm')
        a = np.asarray(nf_c['hist_t'], dtype=float)
        b = np.asarray(nf_d['hist_t'], dtype=float)
        self.assertEqual(a.shape, b.shape, (a.shape, b.shape))
        self.assertEqual(float(np.abs(a - b).max()), 0.0,
                         'con BatchNorm la serie por paso derivo al densificar '
                         'la rejilla: el bloque de puntos de control esta '
                         'reestimando los buffers en su forward')

    def test_la_posicion_con_buffers_de_verdad_los_tiene(self):
        """Si no los tuviera, el test de arriba no estaria viendo nada.

        La via que ese test existe para cubrir es un buffer que se reestima.
        Una posicion sin buffers lo deja verde sin poder fallar, que es como
        una prueba se vuelve decorativa.
        """
        h1, _ = _hp()
        con = T.MLP(_FAM['amb'], 'batchnorm', dim=h1['DIM'])
        sin = T.MLP(_FAM['amb'], 'ninguna', dim=h1['DIM'])
        self.assertTrue(
            [n for n, _ in con.named_buffers()],
            'la posicion batchnorm no declara ningun buffer, asi que el test '
            'de identidad en batchnorm no esta cubriendo la via que existe '
            'para cubrir')
        self.assertFalse(
            [n for n, _ in sin.named_buffers()],
            'la posicion ninguna declara buffers, asi que ya no es el control '
            'sin estado que esta clase supone')

    def test_la_serie_por_paso_es_identica_celda_a_celda(self):
        """No `allclose`: identica. Cualquier deriva es consumo de RNG."""
        _, _, nf_c, _, _, nf_d = _UnaCorridaCompartida.par()
        a = np.asarray(nf_c['hist_t'], dtype=float)
        b = np.asarray(nf_d['hist_t'], dtype=float)
        self.assertEqual(a.shape, b.shape, (a.shape, b.shape))
        self.assertEqual(float(np.abs(a - b).max()), 0.0,
                         'la serie por paso derivo al densificar la rejilla: '
                         'el bloque de puntos de control esta consumiendo '
                         'aleatoriedad o moviendo buffers')

    def test_la_rejilla_densa_trae_un_punto_por_paso(self):
        """Y la publicada no. Si las dos trajeran lo mismo, el test de arriba
        compararia una corrida contra si misma y pasaria sin medir nada.

        El largo es `paso_fin` o `paso_fin + 1`, y las dos cuentas son
        correctas: el bloque de control corre ARRIBA de la iteracion, asi que
        el paso `PASOS_2` recibe su punto y despues el bucle corta --- una
        corrida que llega al techo trae el punto del techo. Una que para
        temprano corta DESPUES del paso, antes del control siguiente, y trae
        `paso_fin`. Medido: esta celda con `DIM=8` no para temprano
        (`paso_fin = 200 = PASOS_2`) y trae 201 puntos. La primera version de
        este test afirmaba `paso_fin` a secas y se vio fallar con `201 != 200`:
        la cuenta mala era la del test.
        """
        _, _, nf_c, cu_d, _, nf_d = _UnaCorridaCompartida.par()
        cu_c = _UnaCorridaCompartida.par()[0]
        fin = int(nf_d['paso_fin'])
        self.assertIn(len(cu_d), (fin, fin + 1),
                      f'la rejilla densa dio {len(cu_d)} puntos para '
                      f'paso_fin={fin}')
        # Lo que el probe necesita de verdad: el paso `p` esta en el indice `p`.
        self.assertEqual([int(r[0]) for r in cu_d], list(range(len(cu_d))),
                         'la rejilla densa no esta indexada por paso')
        self.assertLess(len(cu_c), len(cu_d),
                        'las dos rejillas tienen el mismo largo, asi que el '
                        'test de identidad no esta comparando dos rejillas')


if __name__ == '__main__':
    unittest.main()
