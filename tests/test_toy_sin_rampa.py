"""El warmup y el coeficiente NO estan en la formulacion, asi que no corren.

Verificado contra la revision gestionada: `research-concept-r29.md` tiene CERO
menciones de rampa, ramp, schedule o warmup. El factor que multiplicaba a CREDA
--- `(1-e^-delta*p)/(1+e^-delta*p)`, de 0.000 en el paso 0 a 0.780 en el 200 ---
era un agregado de implementacion, no del paper.

Y el coeficiente tampoco: con `mover_fuente=False` la CE de la fuente llega
DETACHADA, asi que el unico termino con gradiente es CREDA. Un lambda existe
para balancear DOS terminos; sobre uno solo es una tasa de aprendizaje con otro
nombre, redundante con `LR_2`, con el que ademas se multiplica. Medido: entre
`1e-3` y `10` --- cuatro ordenes --- el acierto no se mueve (0.69-0.85), porque
Adam actualiza con `lr*m/(sqrt(v)+eps)` y escalar la perdida por `c` escala `m`
y `v` por igual.

Lo que esto cuesta, medido y declarado: apagar el warmup costaba -0.0102 de
media, concentrado en las tres celdas de BatchNorm (-0.081, -0.076, -0.033),
porque le compraba pasos a BatchNorm para reestimar sus corridas. Eso es una
necesidad de AdaBN, no de CREDA. En `rollo` sin normalizacion, que no tiene
buffers, apagarlo GANA +0.148.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from CREDA import toy_experiment as T


class SinRampaNiCoeficiente(unittest.TestCase):

    def test_las_llaves_retiradas_no_existen(self):
        for k in ('TECHO', 'DELTA'):
            self.assertNotIn(k, T.HP_2, f'{k} sigue en HP_2')

    def test_pasarlas_se_refuta_por_nombre(self):
        """Un `barrido.json` viejo las trae y quien las manda cree que decide."""
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 0)
        h1 = dict(T.HP_1); h1['DIM'] = 8
        for k in ('TECHO', 'DELTA'):
            with self.assertRaises(Exception, msg=f'{k} pasa en silencio'):
                T.corrida(fam, D, 'ninguna', semilla=0, mover_fuente=False,
                          cuadros=(0,), hp1=h1,
                          hp2={**T.HP_2, 'LOTE_2': 64, k: 1.0}, peso='apagado')

    def test_el_modulo_no_importa_creda_ramp(self):
        import inspect
        src = inspect.getsource(T)
        self.assertNotIn('creda_ramp', src, 'el modulo sigue importando la rampa')

    def test_la_historia_de_fase_2_ya_no_lleva_columna_de_rampa(self):
        """Seis campos, no siete: la rampa se fue y el total era CE + CREDA."""
        self.assertEqual(T.CAMPOS_HT, 6)

    def test_el_renyi_registrado_es_CREDA_CRUDO(self):
        """Sin factor, la curva de train y la de valid viven en la MISMA escala.

        Era el defecto que hacia ilegible el panel: train iba multiplicado por
        el factor y valid crudo, dibujados juntos.
        """
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 0)
        h1 = dict(T.HP_1); h1['DIM'] = 8
        _, _, _, _, i = T.corrida(fam, D, 'ninguna', semilla=0,
                                  mover_fuente=False, cuadros=(0,), hp1=h1,
                                  hp2={**T.HP_2, 'LOTE_2': 64}, peso='apagado')
        h = np.asarray(i['hist_t'])
        self.assertEqual(h.shape[1], T.CAMPOS_HT)
        # total == CE + CREDA, sin ningun factor en el medio.
        self.assertTrue(np.allclose(h[:, 1] + h[:, 2], h[:, 3], atol=1e-5))
        # Y el CREDA del paso 0 NO es cero: con el warmup lo era por
        # construccion, y eso era el factor y no el termino.
        self.assertGreater(float(h[0, 2]), 0.0,
                           'el CREDA del paso 0 sigue anulado por un factor')


if __name__ == '__main__':
    unittest.main()
