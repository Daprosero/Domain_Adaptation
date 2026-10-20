"""El barrido de contaminacion corre UN escalon, y `RHOS` sigue declarando cuatro.

**Tercera vuelta de la misma decision, y las tres quedan escritas porque la
proxima la va a tomar alguien que no vio las anteriores.** El 2026-10-10 el
operador decidio un solo nivel, el mas alto, porque lo que la seccion comparaba
eran los tres pesos entre si a contaminacion fija. El 2026-10-11 pidio la curva
contra `rho` y el default volvio a la escalera entera. El 2026-10-19 volvio al
nivel unico, y esta vez sin el cero.

Lo que esto fija ahora, y por que cada cosa:

  DEFAULT      sin `rhos`, el barrido corre `ESCALERA_CONTAMINACION`, que es un
               escalon: `(0.30,)`. El default NO es `RHOS`, y eso es la
               decision --- `RHOS` quedo fijada en
               `odd/tasks/toyset-tests-estadisticos-y-ruido.md` antes de
               implementar y no se mueve; lo que se eligio es CUALES de sus
               escalones corre esta seccion.
  EL ROTULO    y por eso las FILAS de las tablas salen de la misma constante.
               El defecto que la version anterior de este archivo nombraba
               --- "el cuaderno IMPRIMIA `RHOS` y corria uno solo: el rotulo
               decia cuatro niveles sobre datos de uno" --- es REAL y sigue
               siendo el riesgo de esta forma. Lo que lo cierra es que el
               barrido y el renderizador leen UNA constante, asi que el rotulo
               no puede decir cuatro sobre datos de uno.
  CERO         el `0.0` NO esta en la escalera que corre, por decision del
               operador: "sin la base limpia, eso ya es la seccion 1". Sigue
               estando en `RHOS`, que es lo que el resto del arbol lee.

               La razon que lo defendia estaba MUERTA y se deja escrita: decia
               que "la fila limpia de la seccion 1 NO sirve de reemplazo: la
               contaminacion busca fase 2 por escenario, asi que empalmar la
               corrida limpia mezclaria dos regimenes de HP en una tabla". La
               busqueda por escenario se apago el 2026-10-18 --- `busquedas ==
               0`, `hp2_inyectado == True` --- asi que los dos lados corren los
               mismos HP sobre la misma posicion y no hay dos regimenes.
  ENSAYO       `ESCALERA_UNICA` sigue existiendo y sigue siendo de un escalon,
               porque el ensayo a escala chica la usa. Coincide en VALOR con
               `ESCALERA_CONTAMINACION` y son dos constantes a proposito: una
               dice "lo que corre el ensayo" y la otra "lo que corre esta
               seccion", y el dia que una se mueva la otra no tiene por que
               seguirla.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from CREDA import toy_experiment as T


class LaEscaleraQueCorre(unittest.TestCase):

    # NO hay test que lea el TEXTO de `barrer_contaminacion` buscando el
    # nombre del default. Lo escribi y lo saque: un test sobre la fuente pasa
    # con el nombre correcto y la conducta equivocada, y al fallar vuelca la
    # funcion entera. El de abajo CORRE el barrido y lee `C['rhos']`, que es
    # la afirmacion que importa.

    def test_el_barrido_devuelve_el_escalon_declarado_y_no_la_escalera(self):
        """Medido corriendo, no leyendo: `trials` al minimo para que sea barato.

        Asertaba `== RHOS` --- los cuatro niveles --- hasta el 2026-10-19.
        Ahora aserta las dos mitades de la decision: lo que corre es
        `ESCALERA_CONTAMINACION`, y NO es `RHOS`. La segunda hace falta: con
        una escalera de un escalon las dos serian iguales el dia que alguien
        recortara `RHOS`, y este test pasaria sin notarlo.
        """
        C = T.barrer_contaminacion(
            familias=[T.FAMILIAS[0]], semillas=[0], trials1=1, trials2=1,
            al_paso=lambda *a: None)
        self.assertEqual([float(r) for r in C['rhos']],
                         [float(r) for r in T.ESCALERA_CONTAMINACION])
        self.assertNotEqual([float(r) for r in T.ESCALERA_CONTAMINACION],
                            [float(r) for r in T.RHOS],
                            'la escalera declarada y la que corre quedaron '
                            'iguales: este test ya no distingue el default de '
                            'la escalera')

    def test_el_cero_sigue_declarado_y_ya_no_corre(self):
        """Las dos mitades: `RHOS` lo declara y la seccion no lo corre.

        La primera la lee el resto del arbol --- `barrer_intrusos` y el conteo
        de indices de las pruebas --- y la segunda es la decision del operador
        del 2026-10-19.
        """
        self.assertIn(0.0, [float(r) for r in T.RHOS])
        self.assertNotIn(0.0, [float(r) for r in T.ESCALERA_CONTAMINACION])

    def test_la_escalera_del_ensayo_sigue_siendo_de_un_escalon(self):
        self.assertEqual(len(T.ESCALERA_UNICA), 1)
        self.assertEqual(float(T.ESCALERA_UNICA[0]), float(max(T.RHOS)))


if __name__ == '__main__':
    unittest.main()
