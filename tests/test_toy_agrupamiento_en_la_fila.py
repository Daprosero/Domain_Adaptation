"""La fila de cada brazo tiene que llevar el AGRUPAMIENTO, no solo el acierto.

La seccion 2 del cuaderno compara con y sin sopesado sobre dos cosas: acierto y
agrupamiento. `_fila_brazo` llevaba solo acierto, asi que la tabla de
agrupamiento no tenia de donde leer.

Y las lleva SUELTAS, `tr_e` y `tr_d`, nunca el cociente: el cociente confunde
dos fenomenos OPUESTOS --- `tr_e` cayendo son clases que se FUSIONAN, `tr_d`
creciendo son nubes que se DESPARRAMAN. Medido en el barrido del sopesado,
`tr_e` cae 4.6% (p<0.001) mientras `tr_d` no se mueve (-0.7%, p=0.052): el
cociente lo habria reportado como un solo numero y la lectura habria sido otra.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from CREDA import toy_experiment as T


class LaFilaDelBrazo(unittest.TestCase):

    def _nf(self):
        """Lo minimo que `_fila_brazo` lee, con valores DISTINTOS entre si.

        Distintos a proposito: con `tr_e == tr_d` un campo cruzado pasaria el
        test sin que nada falle, que es exactamente el defecto que este
        modulo ya coleccionó cinco veces.
        """
        return dict(acc_te=0.71, acc_s_te=0.93, paso_fin=122, mejor_paso=118,
                    vigia_fin=0.42, hp2={'LR_2': 1e-3},
                    tr_e=17.0566, tr_d=22.1766)

    def test_lleva_tr_e_y_tr_d(self):
        f = T._fila_brazo(self._nf(), [(0,)*T.CAMPOS_CU], T.PESO_DECLARADO)
        self.assertIn('tr_e', f); self.assertIn('tr_d', f)
        self.assertIsInstance(f['tr_e'], float)
        self.assertIsInstance(f['tr_d'], float)

    def test_no_los_cruza(self):
        """Con valores distintos, un campo cruzado se ve."""
        f = T._fila_brazo(self._nf(), [(0,)*T.CAMPOS_CU], T.PESO_DECLARADO)
        self.assertAlmostEqual(f['tr_e'], 17.0566, places=4)
        self.assertAlmostEqual(f['tr_d'], 22.1766, places=4)

    def test_no_guarda_el_cociente(self):
        """El cociente NO puede estar: su presencia invita a leerlo."""
        f = T._fila_brazo(self._nf(), [(0,)*T.CAMPOS_CU], T.PESO_DECLARADO)
        for k in ('conc', 'tr_ratio', 'agrupamiento'):
            self.assertNotIn(k, f, f'el cociente no va en la fila: {k}')

    def test_una_corrida_real_los_entrega(self):
        """Y `corrida` los pone en `info`, que es de donde la fila los saca."""
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 0)
        h1 = dict(T.HP_1); h1['DIM'] = 8
        _, cu, _, _, i = T.corrida(fam, D, 'ninguna', semilla=0,
                                   mover_fuente=False, cuadros=(0,), hp1=h1,
                                   hp2={**T.HP_2, 'LOTE_2': 64},
                                   peso='declarado')
        f = T._fila_brazo(i, cu, T.PESO_DECLARADO)
        self.assertAlmostEqual(f['tr_e'], float(i['tr_e']), places=6)
        self.assertAlmostEqual(f['tr_d'], float(i['tr_d']), places=6)


if __name__ == '__main__':
    unittest.main()
