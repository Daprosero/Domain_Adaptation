"""LA FASE 2 CORRE EPOCAS, y su criterio se lee al final de cada una.

El defecto que esto cierra. La fase 1 (`train_source_ce`) es un bucle de
entrenamiento ordinario: `orden = torch.randperm(len(x))` y despues
`for ini in range(0, len(x), batch_size)`, o sea una PASADA sobre los datos.
La fase 2 no lo era: sorteaba `randperm(len(Xs_tr))[:LOTE_2]` DE CERO en cada
paso, que es muestreo CON REEMPLAZO entre pasos --- un punto podia salir dos
veces seguidas y otro quedarse veinte pasos sin salir. Sin pasada no hay
epoca, y entonces la parada temprana (`MIN_PASOS_2`, `PACIENCIA_2`) decidia
POR PASO, sobre un modelo que habia visto un lote arbitrario en vez de uno
que completo una pasada.

Lo que cada test fija:

  PASADA       en UNA epoca cada punto de train sale EXACTAMENTE UNA vez, en
               los dos dominios. Es la propiedad que el codigo viejo no tenia
               y es todo el punto del cambio. Se lee del registro de la
               corrida, no de la funcion sola: la funcion podia devolver una
               particion perfecta y el bucle consumirla mal.
  FINAL        `mejor_paso` y `paso_fin` son multiplos exactos de
               `pasos_por_epoca`, porque el criterio se lee al terminar la
               epoca y en ningun otro momento.
  DECLARADA    la paciencia esta DECLARADA en epocas y no derivada del lote.
               Es la decision del operador y el motivo es que el lote SE
               BUSCA: una paciencia derivada de pasos heredaria la busqueda, y
               la resolucion del criterio terminaria decidida por un
               hiperparametro que no sabe que la esta decidiendo.
  BLANCO       el BLANCO define la epoca y la fuente tiene que coincidir. Si
               alguna vez no coinciden la corrida se niega, en vez de envolver
               la permutacion corta y dejar la propiedad de PASADA falsa sin
               que nada lo diga.

Y un quinto, que no es un invariante del metodo sino del registro: que los
campos que el criterio publica tengan todos un lector. Un campo agregado sin
lector pasa verde para siempre.
"""
import math
import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from CREDA import toy_experiment as T


#: El lote de los tests y el techo en EPOCAS. `LOTE_2 = 256` sobre un train de
#: 1080 da 5 bloques --- cuatro de 256 y uno de 56 --- asi que el ULTIMO
#: bloque es parcial y la propiedad de pasada se mide con el caso dificil y no
#: con uno que divide justo. Ocho epocas pasan `MIN_EPOCAS_2 = 5` y dejan tres
#: puntos elegibles, que es lo minimo para que elegir signifique algo.
#:
#: Era `PASOS_TEST = 40` --- 8 epocas a 5 pasos --- y ahora es el numero de
#: epocas directo: la cuenta que el test hacia a mano la hace el techo.
LOTE_TEST, EPOCAS_TEST = 256, 8


def _corre(*, lote=LOTE_TEST, epocas=EPOCAS_TEST, sem=0, fam=None):
    """Una corrida corta. `EPOCAS_2` es un GLOBAL que `corrida` lee directo."""
    fam = T.FAMILIAS[0] if fam is None else fam
    D = T.datos_semilla(fam, sem)
    h1 = dict(T.HP_1); h1['DIM'] = 8
    h2 = {**T.HP_2, 'LOTE_2': lote}
    with mock.patch.object(T, 'EPOCAS_2', epocas):
        out = T.corrida(fam, D, 'ninguna', semilla=sem, mover_fuente=False,
                        cuadros=(0,), hp1=h1, hp2=h2)
    return D, out[4]


class LaPasada(unittest.TestCase):

    def test_the_phase_two_epoch_draws_every_training_point_exactly_once(self):
        """La propiedad que el codigo viejo no tenia.

        Se afirman LAS DOS mitades y no una: `distintos == n` sola admite que
        un punto salga dos veces y otro se caiga, y `suma == n` sola admite la
        misma cosa al reves. Juntas obligan a exactamente una vez.

        Y se lee del registro de la CORRIDA, no de `pasada_en_lotes` sola: la
        funcion puede devolver una particion impecable y el bucle consumir dos
        veces el mismo bloque. Ese es el cruce que el test tiene que atravesar.
        """
        D, nf = _corre()
        n_s, n_t = len(D['Xs_tr']), len(D['Xt_tr'])
        pasada = nf['pasada_2']

        # El control: si no corrio ninguna epoca completa, todo lo de abajo se
        # afirma sobre el conjunto vacio y pasa sin medir nada.
        self.assertGreaterEqual(len(pasada), 2, 'menos de dos epocas completas')

        for fila in pasada:
            e = dict(zip(T.CAMPOS_P2_NOMBRES, fila))
            self.assertEqual(e['pasos'], nf['pasos_por_epoca'], e)
            self.assertEqual(e['distintos_t'], n_t, ('blanco distintos', e))
            self.assertEqual(e['suma_t'], n_t, ('blanco suma', e))
            self.assertEqual(e['distintos_s'], n_s, ('fuente distintos', e))
            self.assertEqual(e['suma_s'], n_s, ('fuente suma', e))

    def test_el_sorteo_viejo_no_pasaria_este_test(self):
        """El polo: que el instrumento pueda registrar la falla.

        Se construye aca el sorteo que el codigo tenia --- `randperm(n)[:lote]`
        de cero por paso --- y se mide sobre el MISMO numero de pasos que una
        epoca. Si esto diera `n` distintos, el test de arriba no distinguiria
        nada y seria verde contra las dos implementaciones.
        """
        n, lote = 1080, LOTE_TEST
        pasos = T.pasos_de_una_epoca(n, n, lote)
        g = torch.Generator().manual_seed(0)
        viejo = set()
        suma = 0
        for _ in range(pasos):
            idx = torch.randperm(n, generator=g)[:lote]
            viejo |= set(int(k) for k in idx)
            suma += len(idx)
        self.assertLess(len(viejo), n,
                        'el sorteo con reemplazo cubrio todo el train')
        self.assertGreater(suma, 0)

        # Y la pasada nueva, sobre los mismos n y lote, si lo cubre.
        bloques = T.pasada_en_lotes(n, lote, torch.Generator().manual_seed(0))
        self.assertEqual(len(bloques), pasos)
        todos = torch.cat(bloques)
        self.assertEqual(len(todos), n)
        self.assertEqual(sorted(int(k) for k in todos), list(range(n)))


class ElFinalDeLaEpoca(unittest.TestCase):

    def test_the_phase_two_criterion_is_read_at_the_end_of_an_epoch(self):
        """`mejor_paso` y `paso_fin` caen en multiplos de la epoca.

        Los dos son `paso + 1` en el ULTIMO paso de una epoca, asi que son
        `(epoca + 1) * pasos_por_epoca` exacto. Antes eran cualquier paso: el
        artefacto publicado trae `mejor_paso` 40, 53, 62, 70 --- ninguno
        multiplo de los 9 pasos que una epoca tenia a ese lote.
        """
        _, nf = _corre()
        k = nf['pasos_por_epoca']

        # DOS controles, y los dos hacen falta. Con `k == 1` cada paso es un
        # final de epoca y la congruencia no puede fallar; con `mejor_paso`
        # en 0 la congruencia tampoco, porque 0 es multiplo de todo.
        self.assertGreater(k, 1, 'con un paso por epoca el test es vacio')
        self.assertGreater(nf['mejor_paso'], 0,
                           'el criterio no eligio ningun paso')

        self.assertEqual(nf['mejor_paso'] % k, 0, (nf['mejor_paso'], k))
        self.assertEqual(nf['mejor_epoca'], nf['mejor_paso'] // k)
        # `epoca_fin` son las epocas COMPLETAS que corrieron, y con el techo
        # en epocas son TODAS: no hay paciencia que corte y no hay cola de
        # pasos que se entrene sin lectura. La rama «si corto temprano» se fue
        # con la paciencia, y lo que la reemplaza es mas fuerte.
        self.assertEqual(nf['epoca_fin'], nf['paso_fin'] // k)
        self.assertEqual(nf['epoca_fin'], EPOCAS_TEST, nf['epoca_fin'])
        self.assertEqual(nf['paso_fin'], EPOCAS_TEST * k)
        self.assertEqual(nf['paso_fin'], nf['pasos_techo'])

    def test_el_criterio_no_elige_antes_del_minimo_declarado(self):
        """`MIN_EPOCAS_2` es una guarda en epocas, y se mide que muerde."""
        _, nf = _corre()
        self.assertGreaterEqual(nf['mejor_epoca'], T.MIN_EPOCAS_2,
                                (nf['mejor_epoca'], T.MIN_EPOCAS_2))
        self.assertEqual(nf['min_epocas_2'], T.MIN_EPOCAS_2)


class LaPacienciaRetirada(unittest.TestCase):
    """Era `LaPacienciaDeclarada` y la clase entera cambio de signo.

    Sus tres tests afirmaban que `PACIENCIA_2_EPOCAS` estaba declarada y no
    derivada, que `MIN_EPOCAS_2` SI se derivaba de `MIN_PASOS_2`, y que los dos
    campos viejos en pasos conservaban su valor porque el banco de
    `MIL_CREDA_Benchmark` los leia. Las tres premisas se fueron el 2026-10-19:
    la fase 2 no tiene paciencia, el minimo esta declarado porque su insumo se
    retiro, y el banco paso a epocas en el mismo cambio.

    Se invierte en vez de borrarse. Un archivo sin estos tests no distingue
    «la paciencia se retiro» de «nadie la probo nunca», y lo que se rompe de
    verdad es que alguien la reintroduzca de un lado solo.
    """

    def test_la_paciencia_de_la_fase_dos_esta_retirada_y_no_solo_sin_usar(self):
        """No existe la constante, no viaja en las banderas, no para nada.

        Las tres cosas, porque una constante que sobrevive a su mecanismo es
        exactamente como volvio la ultima vez: quedo declarada, el registro
        publicaba `paciencia_epocas=5`, y nada paraba por eso.
        """
        self.assertFalse(hasattr(T, 'PACIENCIA_2_EPOCAS'))
        self.assertFalse(hasattr(T, 'PACIENCIA_2'))
        b = T.banderas()
        self.assertNotIn('PACIENCIA_2_EPOCAS', b)
        self.assertNotIn('PACIENCIA_2', b)
        _, nf = _corre()
        self.assertNotIn('paciencia_epocas', nf)
        # Y la afirmacion positiva: la corrida llego al techo. Sin esto, un
        # modulo que parara por otra razon pasaria estas ausencias.
        self.assertEqual(nf['epoca_fin'], EPOCAS_TEST)

    def test_la_fase_uno_SI_para_y_eso_es_la_asimetria_declarada(self):
        """La otra mitad, porque la ausencia de arriba se lee como «no hay
        parada en ninguna fase» y eso seria falso.

        `PACIENCIA_1` existe, vale 30 y gobierna el final de la fase 1: su
        senal es la exactitud de validacion de la FUENTE, que es supervisada.
        La bandera lo dice en vez de dejar que un lector lo infiera de que
        `PACIENCIA_1` exista.
        """
        self.assertEqual(T.PACIENCIA_1, 30)
        b = T.banderas()
        self.assertEqual(b['PACIENCIA_1'], T.PACIENCIA_1)
        self.assertIn('fase 1 SI', b['PARADA_TEMPRANA'])
        self.assertIn('Fase 2 NO', b['PARADA_TEMPRANA'])

    def test_el_minimo_esta_declarado_y_su_insumo_ya_no_existe(self):
        """`MIN_EPOCAS_2` se derivaba de `MIN_PASOS_2` y ahora esta declarado.

        El VALOR no se movio --- 5 es lo que `ceil(40/9)` daba --- y eso es
        deliberado: declararlo y cambiarlo en el mismo paso mezclaria dos
        cosas. Lo que se afirma es que el insumo se fue, porque una derivacion
        que lee un numero que nadie mantiene se lee como derivada y no lo es.
        """
        self.assertEqual(T.MIN_EPOCAS_2, 5)
        self.assertFalse(hasattr(T, 'MIN_PASOS_2'))
        b = T.banderas()
        self.assertEqual(b['MIN_EPOCAS_2'], T.MIN_EPOCAS_2)
        self.assertIn('declarada', b['MIN_EPOCAS_2_ES'])
        self.assertNotIn('MIN_PASOS_2', b)
        # Y la guarda MUERDE, que es lo unico que vuelve a este numero
        # interesante: la bandera publica la medicion en vez de afirmar que
        # es una guarda y nada mas.
        self.assertIn('42%', b['MIN_EPOCAS_2_MUERDE'])


class ElBlancoDefineLaEpoca(unittest.TestCase):

    def test_the_target_defines_the_epoch_and_the_source_must_agree(self):
        """Quien define la epoca, dicho en el registro, y la negativa medida.

        Los dos dominios NO tienen el mismo numero de puntos: medido,
        `lunas` sortea 1080/1080 y `toroide` y `rollo` 1078/1079, porque el
        piso por clase del split estratificado pierde un punto. Lo que
        coincide a todo lote de la rejilla es el numero de BLOQUES, y es eso
        lo que una pasada necesita.
        """
        D, nf = _corre()
        self.assertEqual(nf['epoca_definida_por'], 'blanco')
        self.assertEqual(nf['bloques_t'], nf['pasos_por_epoca'])
        self.assertEqual(nf['bloques_s'], nf['pasos_por_epoca'])

        # Que los dos dominios puedan DIFERIR en puntos y seguir coincidiendo
        # en bloques, medido sobre las tres familias y no supuesto.
        vistos = set()
        for f in T.FAMILIAS:
            Df = T.datos_semilla(f, 0)
            n_s, n_t = len(Df['Xs_tr']), len(Df['Xt_tr'])
            vistos.add((n_s, n_t))
            self.assertEqual(T.pasos_de_una_epoca(n_s, n_t, T.LOTE),
                             math.ceil(n_t / T.LOTE))
        self.assertIn((1078, 1079), vistos,
                      ('los dos dominios coinciden en todas las familias, asi '
                       'que este test no mide la asimetria', sorted(vistos)))

    def test_se_niega_cuando_los_bloques_no_coinciden(self):
        """La guarda, y que sea ALCANZABLE.

        Envolver la permutacion corta dejaria la propiedad de PASADA falsa sin
        que nada fallara, asi que el desacuerdo se niega. `1024` y `1025` a
        lote 128 dan 8 bloques contra 9: el caso existe.
        """
        self.assertEqual(math.ceil(1024 / 128), 8)
        self.assertEqual(math.ceil(1025 / 128), 9)
        with self.assertRaises(AssertionError):
            T.pasos_de_una_epoca(1024, 1025, 128)
        # Y que no se niegue cuando SI coinciden, o la guarda negaria todo.
        self.assertEqual(T.pasos_de_una_epoca(1078, 1079, 128), 9)


class ElRegistro(unittest.TestCase):

    def test_el_registro_nombra_cada_campo_que_el_criterio_publica(self):
        """El roster, porque un campo sin lector pasa verde para siempre.

        No es un invariante del metodo: es del registro. Si alguien agrega un
        campo al criterio y no lo nombra aca, nada se pone rojo; si saca uno
        que los tests de arriba leen, se ponen rojos ellos. Lo que este cierra
        es la mitad que no: un campo que se publica y nadie lee.
        """
        _, nf = _corre()
        esperados = ('pasos_por_epoca', 'epoca_definida_por', 'bloques_s',
                     'bloques_t', 'min_epocas_2', 'epocas_2', 'pasos_techo',
                     'mejor_epoca', 'epoca_fin', 'pasada_2')
        faltan = [k for k in esperados if k not in nf]
        self.assertEqual(faltan, [], faltan)
        # Y la otra mitad del roster: los dos campos que SE FUERON con sus
        # mecanismos. Sin esto, reintroducir uno pasaria verde --- el roster
        # de arriba solo atrapa los que faltan, nunca los que sobran.
        sobran = [k for k in ('paciencia_epocas', 'pasos_sin_leer') if k in nf]
        self.assertEqual(sobran, [], sobran)

        self.assertEqual(T.CAMPOS_P2, len(T.CAMPOS_P2_NOMBRES))
        for fila in nf['pasada_2']:
            self.assertEqual(len(fila), T.CAMPOS_P2, (fila, T.CAMPOS_P2))

    def test_las_banderas_publican_la_conversion(self):
        """La conversion tiene que poder LEERSE del artefacto, no del codigo."""
        b = T.banderas()
        for k in ('MIN_EPOCAS_2', 'CRITERIO_2_EN', 'CRITERIO_2_SENAL',
                  'CRITERIO_2_DIRECCION', 'TECHO_EN', 'EPOCAS_1', 'EPOCAS_2',
                  'BLOQUES_AL_LOTE_DECLARADO', 'N_POR_DOMINIO'):
            self.assertIn(k, b)
        self.assertEqual(b['CRITERIO_2_EN'], 'epocas')
        self.assertEqual(b['CRITERIO_2_SENAL'], 'cef')
        self.assertEqual(b['CRITERIO_2_DIRECCION'], 'maximo')
        self.assertEqual(b['MIN_EPOCAS_2'], T.MIN_EPOCAS_2)
        # EL TECHO, en la misma unidad en las dos fases, que es lo que el
        # artefacto no podia decir: `EPOCAS_1` estaba en epocas y `PASOS_2` en
        # pasos, asi que un lector no sabia cuantas pasadas habia corrido la
        # fase 2 sin derivarlo a mano --- y lo habria derivado con el lote
        # DECLARADO en vez del buscado, o sea mal.
        self.assertIn('epocas', b['TECHO_EN'])
        self.assertEqual(b['EPOCAS_1'], T.EPOCAS_1)
        self.assertEqual(b['EPOCAS_2'], T.EPOCAS_2)
        self.assertNotIn('PASOS_2', b)


if __name__ == '__main__':
    unittest.main()
