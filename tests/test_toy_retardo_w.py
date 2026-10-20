"""El sopesado ARRANCA TARDE: `w ≡ 1` hasta el paso T, y el declarado después.

La hipótesis es del operador y tiene respaldo medido antes de probarla: en el
paso 0 `enc_t` es un CLON de `enc`, así que el ruteo del blanco es el
clasificador de la FUENTE aplicado al blanco --- exactamente el brazo
`.predict`, que mide 0.3703 / 0.4157 / 0.3528 sobre tres clases, o sea cerca
del azar en una de las tres.

Y el ruteo del arranque está medido como peor: los bloques de clase tienen
mediana 2 a 6 con el cabezal de fase 1 y 4 a 10 durante la corrida.

Si `w` sale de esa posterior, al principio apaga muestras por una
incertidumbre que es del MODELO y no de la muestra. El retardo le da tiempo al
blanco a separarse antes de que el peso empiece a decidir.

**Es un RETARDO y no una rampa, a propósito.** Una rampa mezcla dos preguntas
--- cuándo empezar y con qué pendiente --- y acá la que se prueba es la
primera. Si el retardo compra algo, la rampa es su versión suave y se mide
después; si no compra nada, no hay pendiente que lo salve.

**Y NO es la rampa que ya existe.** `creda_ramp` ramplea `lambda`, el
coeficiente del término. Esto ramplea el PESO, que es otra cosa: hoy el
término entra de a poco y el peso entra entero desde el paso 0.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA import toy_experiment as T


class ElSopesadoArrancaTarde(unittest.TestCase):

    def test_the_default_is_bit_for_bit_what_it_already_was(self):
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        k = dict(semilla=100, mover_fuente=False, cuadros=T.CTRL, peso='declarado')
        *_, a = T.corrida(fam, D, 'layernorm', **k)
        *_, b = T.corrida(fam, D, 'layernorm', hp2={'EPOCA_W': 0}, **k)
        self.assertEqual(a['acc_te'], b['acc_te'])
        self.assertEqual(a['mejor_paso'], b['mejor_paso'])

    def test_before_the_epoch_the_weight_the_term_used_is_all_ones(self):
        """La propiedad entera: antes de T el brazo declarado ES el apagado.

        `EPOCA_W` es el techo entero, asi que NINGUNA epoca de la corrida
        sopesa y las once lecturas de la rejilla tienen que traer unos. Era
        `PASO_W = 100` sobre un techo de 200 pasos, o sea la mitad del
        experimento; ahora es el techo completo, que es la unica forma de
        afirmar «antes de T» sobre toda la corrida.
        """
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        *_, i = T.corrida(fam, D, 'layernorm', semilla=100, mover_fuente=False,
                          cuadros=T.CTRL, peso='declarado',
                          hp2={'EPOCA_W': T.EPOCAS_2})
        for k, paso in enumerate(T.CTRL):
            if k >= len(i['w_usado_cu']):
                break
            u = np.asarray(i['w_usado_cu'][k])
            if paso < 100:
                self.assertTrue(np.allclose(u, 1.0), (paso, u.min(), u.max()))
            else:
                self.assertLess(float(u.min()), 0.999, paso)

    def test_a_delay_past_the_end_IS_the_off_arm_bit_for_bit(self):
        """El piso de la escalera. Si no coincide, el retardo hace algo más
        que retardar y los peldaños miden dos cosas."""
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        k = dict(semilla=100, mover_fuente=False, cuadros=T.CTRL)
        *_, a = T.corrida(fam, D, 'layernorm', peso='apagado', **k)
        *_, b = T.corrida(fam, D, 'layernorm', peso='declarado',
                          hp2={'EPOCA_W': T.EPOCAS_2 + 1}, **k)
        self.assertEqual(a['acc_te'], b['acc_te'])
        self.assertEqual(a['mejor_paso'], b['mejor_paso'])

    def test_a_middling_delay_differs_from_BOTH_ends(self):
        """Si no difiere de los dos extremos, la escalera tiene un solo punto.

        **Se leia sobre `mejor_paso` y ese campo es PLANO en este eje.** El
        criterio vivo es el MAXIMO de `cef`, y el retardo del sopesado no mueve
        en que epoca cae ese maximo --- mueve su VALOR. La igualdad que este
        test afirmaba aguantaba por coincidencia y r33 la deshizo al mover el
        termino.

        **Re-medido el 2026-10-19 al techo en EPOCAS**, porque el anterior era
        sobre 200 pasos y la conversion mueve cada numero. Sobre esta misma
        celda, con `EPOCA_W` en 0, 2, 20, 80 y `EPOCAS_2+1`:

            EPOCA_W     mejor_epoca   mejor_re     mejor_cef
            0                     6   0.4365964    1.9999451306
            2                     6   0.4371354    1.9999862826
            20                    6   0.4429708    1.9999691359
            80                    6   0.4332136    1.9999691359
            101                   6   0.4332136    1.9999691359

        Tres cosas que esa tabla fija y que este test lee:

        1. `mejor_epoca` vale 6 en las CINCO: el eje es plano, como era.
        2. `mejor_re` es un MINIMO sobre toda la corrida y no la lectura de la
           epoca elegida, asi que el eje SI lo alcanza aunque la seleccion
           caiga en la epoca 6: 0.4365964 / 0.4429708 / 0.4332136 para
           0 / 20 / nunca.
        3. `EPOCA_W = 80` y `EPOCA_W = 101` dan `mejor_re` y `mejor_cef`
           IDENTICOS bit a bit, o sea que sopesar las ultimas veinte epocas no
           mueve nada. Es consistente con que `cef` sature cerca de 2 para la
           epoca 6 --- con el ruteo ya confiado, `w` vale practicamente uno y
           sopesar no cambia el termino --- y es lo que hace de 80 un control
           valido del extremo "nunca".
        """
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        k = dict(semilla=100, mover_fuente=False, cuadros=T.CTRL, peso='declarado')
        *_, a = T.corrida(fam, D, 'layernorm', hp2={'EPOCA_W': 0}, **k)
        *_, m = T.corrida(fam, D, 'layernorm', hp2={'EPOCA_W': 20}, **k)
        *_, z = T.corrida(fam, D, 'layernorm', hp2={'EPOCA_W': T.EPOCAS_2+1}, **k)

        self.assertNotEqual(m['mejor_re'], a['mejor_re'])
        self.assertNotEqual(m['mejor_re'], z['mejor_re'])
        # La planitud, afirmada: el eje no mueve QUE epoca se elige.
        self.assertEqual(a['mejor_epoca'], z['mejor_epoca'], (
            'mejor_epoca dejo de ser plano en este eje, asi que la medicion '
            'del 2026-10-19 ya no describe la corrida y hay que re-medirla'))
        # Y un retardo posterior a la epoca elegida es el extremo "nunca".
        *_, tarde = T.corrida(fam, D, 'layernorm', hp2={'EPOCA_W': 80}, **k)
        self.assertEqual(tarde['mejor_cef'], z['mejor_cef'], (
            'un retardo posterior a la epoca elegida dejo de coincidir con '
            'no sopesar, asi que 80 ya no es el control que esta prueba cree'))

    def test_the_record_publishes_the_delay_it_used(self):
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 100)
        *_, i = T.corrida(fam, D, 'layernorm', semilla=100, mover_fuente=False,
                          cuadros=T.CTRL, hp2={'EPOCA_W': 60})
        self.assertEqual(i['hp2']['EPOCA_W'], 60)


if __name__ == '__main__':
    unittest.main()
