"""El barrido de contaminacion recorre la escalera ENTERA, no un escalon.

El 2026-10-10 el operador habia decidido lo contrario: un solo nivel, el mas
alto, porque lo que la seccion comparaba eran los tres pesos entre si a
contaminacion fija. El 2026-10-11 el diseno del cuaderno cambio y pide la curva
contra `rho` --- filas de nivel de ruido en la figura y en el gif, y una tabla
"ampliada a los diferentes niveles" --- asi que la escalera vuelve.

Lo que esto fija, y por que cada cosa:

  DEFAULT      sin `rhos`, el barrido recorre los CUATRO niveles de `RHOS`.
               Mientras el default era de un escalon, el cuaderno IMPRIMIA
               `RHOS` y corria uno solo: el rotulo decia cuatro niveles sobre
               datos de uno. Es el mismo defecto que ya aparecio con "5
               semillas" sobre numeros de 10.
  CERO         el `0.0` esta en la escalera que corre. Sin el no hay contra que
               afirmar que el ruido hizo dano, y la fila limpia de la seccion 1
               NO sirve de reemplazo: la contaminacion busca fase 2 por
               escenario, asi que empalmar la corrida limpia mezclaria dos
               regimenes de HP en una tabla.
  ENSAYO       `ESCALERA_UNICA` sigue existiendo y sigue siendo de un escalon,
               porque el ensayo a escala chica la usa. Lo que ya no es, es el
               default de produccion.
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

    def test_el_barrido_devuelve_los_cuatro_niveles(self):
        """Medido corriendo, no leyendo: `trials` al minimo para que sea barato."""
        C = T.barrer_contaminacion(
            familias=[T.FAMILIAS[0]], semillas=[0], trials1=1, trials2=1,
            al_paso=lambda *a: None)
        self.assertEqual([float(r) for r in C['rhos']],
                         [float(r) for r in T.RHOS])

    def test_el_cero_esta_en_la_escalera(self):
        self.assertIn(0.0, [float(r) for r in T.RHOS])

    def test_la_escalera_del_ensayo_sigue_siendo_de_un_escalon(self):
        self.assertEqual(len(T.ESCALERA_UNICA), 1)
        self.assertEqual(float(T.ESCALERA_UNICA[0]), float(max(T.RHOS)))


if __name__ == '__main__':
    unittest.main()
