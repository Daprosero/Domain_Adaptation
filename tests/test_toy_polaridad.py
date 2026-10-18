"""Cada cantidad declara en QUE direccion se gana, y la tabla la respeta.

`_par` contaba `int((dif > 0).sum())` para todo, polaridad unica. Para `tr_e`
eso esta bien --- mas dispersion ENTRE clases es mas separacion --- pero para
`tr_d` esta AL REVES: mas dispersion DENTRO de clase son nubes mas
desparramadas, o sea peor.

Medido en la tabla publicada antes de este arreglo: `rollo`, frontera,
rho=0.10, `tr_d` con delta **+0.8287** salia rotulado **"gana 6/10"**, cuando
+0.83 de dispersion intraclase es la direccion mala. Un conteo de victorias con
el signo invertido es peor que no tenerlo, porque se lee con confianza.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from CREDA import toy_plots as P


class Polaridad(unittest.TestCase):

    def test_cada_campo_declara_su_direccion(self):
        for campo in P.CAMPOS_AGRUPAMIENTO + (P.CAMPO_ACIERTO,):
            self.assertIn(campo, P.MEJOR_SI, f'{campo} no declara polaridad')
            self.assertIn(P.MEJOR_SI[campo], ('sube', 'baja'), campo)

    def test_las_direcciones_declaradas_son_las_correctas(self):
        self.assertEqual(P.MEJOR_SI['tr_e'], 'sube')   # mas separacion
        self.assertEqual(P.MEJOR_SI['tr_d'], 'baja')   # menos desparramo
        self.assertEqual(P.MEJOR_SI[P.CAMPO_ACIERTO], 'sube')

    def test_el_conteo_sigue_la_direccion_y_no_el_signo(self):
        """El caso exacto que salio mal: delta POSITIVO en `tr_d` es DERROTA."""
        a = np.array([11.0, 12.0, 13.0])      # con sopesado
        b = np.array([10.0, 11.0, 12.0])      # sin: el con es MAYOR en los 3
        sube = P._par('con', 'sin', a, b, 3, campo='tr_e')
        baja = P._par('con', 'sin', a, b, 3, campo='tr_d')
        self.assertEqual(sube['gana'], 3, 'tr_e: subir es ganar')
        self.assertEqual(baja['gana'], 0, 'tr_d: subir es PERDER')
        # Y el delta es el mismo numero en los dos: lo que cambia es la lectura.
        self.assertAlmostEqual(sube['delta'], baja['delta'])

    def test_no_hay_default_y_un_campo_sin_declarar_se_refuta(self):
        """`campo` es obligatorio: un default que coincide con la mitad de las
        cantidades es exactamente como `tr_d` quedo contando al reves."""
        a = np.array([1.0, 2.0]); b = np.array([0.0, 0.0])
        with self.assertRaises(TypeError):
            P._par('c', 's', a, b, 2)
        with self.assertRaises(ValueError):
            P._par('c', 's', a, b, 2, campo='una_cantidad_sin_declarar')

    def test_la_tabla_nombra_la_direccion_de_cada_fila(self):
        """Un conteo sin su direccion al lado se lee con el signo de uno."""
        for campo in P.CAMPOS_AGRUPAMIENTO:
            self.assertIn(P.MEJOR_SI[campo], P.GLOSA_CAMPO[campo],
                          f'{campo}: la glosa no nombra su direccion')


if __name__ == '__main__':
    unittest.main()
