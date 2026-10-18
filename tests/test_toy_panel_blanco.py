"""Las columnas 1 y 2 del gif del sopesado: SOLO el blanco, estilo seccion 1.

Dos defectos medidos en la version anterior, los dos del mismo origen ---
reusar `panel_embedding`, que es de otra figura:

  ESCALA   dibujaba la FUENTE y el blanco en el mismo panel. La fuente esta
           CONGELADA (`mover_fuente=False`), asi que es identica en los dos
           brazos y ocupa 58.6 de ancho; lo que de verdad se compara, el
           blanco, difiere 2.10 --- el 3.6% del panel. Las dos columnas
           mostraban dato distinto y se veian iguales.
  ESTILO   otra paleta y otra gramatica que la figura de entradas de la
           seccion 1: ejes numerados, verde/azul/rombos.

`_panel_blanco` dibuja el blanco y nada mas, con la misma gramatica que
`_panel_entradas`: color por CLASE REAL, circulo hueco para el mal ruteado,
sin numeros en los ejes.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from CREDA import toy_plots as P
from CREDA import toy_experiment as T


class PanelDelBlanco(unittest.TestCase):

    def setUp(self):
        import matplotlib; matplotlib.use('Agg')
        self.r = np.random.default_rng(0)

    def test_existe(self):
        self.assertTrue(hasattr(P, '_panel_blanco'))

    def test_dibuja_UNA_coleccion_por_clase_y_nada_mas(self):
        """Exactamente `CLASES`. Ni la fuente ni los circulos negros.

        Los circulos los retiro el operador el 2026-10-12: esta figura compara
        los DOS brazos, y una marca que los dos llevan igual no distingue nada
        y tapa los puntos. Una coleccion de mas es la fuente o la marca
        colandose otra vez.
        """
        import matplotlib.pyplot as plt
        n = 30
        Pt = self.r.normal(size=(n, 2)); yt = self.r.integers(0, T.CLASES, n)
        for rut in (yt.copy(), (yt + 1) % T.CLASES):   # sin y CON mal ruteo
            fig, a = plt.subplots()
            P._panel_blanco(a, Pt, yt, rut=rut, rotulo='x')
            self.assertEqual(len(a.collections), T.CLASES, len(a.collections))
            plt.close(fig)

    def test_usa_la_paleta_de_la_seccion_1(self):
        """`COL_DURA`: un color por CLASE, el mismo que `familias_duras`."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
        n = 30
        Pt = self.r.normal(size=(n, 2)); yt = np.arange(n) % T.CLASES
        fig, a = plt.subplots()
        P._panel_blanco(a, Pt, yt, rut=yt.copy(), rotulo='x')
        for c, col in enumerate(a.collections):
            self.assertEqual(to_hex(col.get_facecolor()[0]).lower(),
                             T.COL_DURA[c].lower(), c)
        plt.close(fig)

    def test_sin_numeros_en_los_ejes_como_la_seccion_1(self):
        import matplotlib.pyplot as plt
        n = 20
        Pt = self.r.normal(size=(n, 2)); yt = self.r.integers(0, T.CLASES, n)
        fig, a = plt.subplots()
        P._panel_blanco(a, Pt, yt, rut=yt.copy(), rotulo='x')
        self.assertEqual([t.get_text() for t in a.get_xticklabels()],
                         [''] * len(a.get_xticklabels()))
        plt.close(fig)

    def test_el_titulo_cuenta_los_mal_ruteados(self):
        import matplotlib.pyplot as plt
        n, k = 40, 10
        Pt = self.r.normal(size=(n, 2)); yt = self.r.integers(0, T.CLASES, n)
        mal = yt.copy(); mal[:k] = (mal[:k] + 1) % T.CLASES
        fig, a = plt.subplots()
        P._panel_blanco(a, Pt, yt, rut=mal, rotulo='rótulo')
        t = a.get_title()
        plt.close(fig)
        self.assertIn('rótulo', t)
        self.assertIn(f'{k} de {n}', t)

    def test_el_gif_del_sopesado_lo_usa(self):
        """Lo que estaba roto: las columnas 1 y 2 llamaban `panel_embedding`."""
        import inspect
        src = inspect.getsource(P.cuadro_sopesado_panel)
        self.assertIn('_panel_blanco', src)
        self.assertNotIn('panel_embedding', src,
                         'sigue dibujando la fuente en el mismo panel')


if __name__ == '__main__':
    unittest.main()
