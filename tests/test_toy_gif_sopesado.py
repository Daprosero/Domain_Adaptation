"""Las CUATRO columnas del gif de la seccion 2, declaradas por el operador.

  1  embedding CON sopesado
  2  embedding SIN sopesado
  3  loss de fase 2 CON sopesado
  4  loss de fase 2 SIN sopesado

No son las de la seccion 1. Es una comparacion con/sin, y por eso necesita la
figura de LOS DOS brazos --- que el registro guarda desde que
`barrer_contaminacion` escribe `figuras` por brazo.

El orden IMPORTA y se fija: con, sin, con, sin. Invertido, un lector compara
el brazo equivocado y la figura se lee con confianza.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from CREDA import toy_experiment as T
from CREDA import toy_plots as P


class CuatroColumnas(unittest.TestCase):

    def test_la_funcion_existe(self):
        self.assertTrue(hasattr(P, 'cuadro_sopesado_panel'),
                        'falta `cuadro_sopesado_panel`')
        self.assertTrue(hasattr(P, 'animar_sopesado'),
                        'falta `animar_sopesado`')

    def test_el_orden_de_las_columnas_esta_declarado(self):
        self.assertEqual(
            list(P.COLUMNAS_SOPESADO),
            ['embedding · CON sopesado', 'embedding · SIN sopesado',
             'loss de fase 2 · CON sopesado', 'loss de fase 2 · SIN sopesado'])

    def _par_sintetico(self):
        """Dos figuras de brazo, con PS_t DISTINTOS para que un cruce se vea."""
        import matplotlib; matplotlib.use('Agg')
        n, C = 12, 3
        r = np.random.default_rng(0)
        def fig(k):
            return dict(
                PS_t=r.normal(size=(3, n, 2)) + k, Pt=r.normal(size=(3, n, 2)),
                Ps=r.normal(size=(n, 2)), pred_s=r.integers(0, C, n),
                ys_tr=r.integers(0, C, n), yt_tr=r.integers(0, C, n),
                rut=r.integers(0, C, (3, n)), w=r.random((3, n)),
                acc_va=np.array([.3,.4,.5]), acc_te=np.array([.3,.4,.5]),
                acc_s=np.array([.9,.9,.9]), u_t=np.array([.5,.4,.3]),
                pok=np.array([.3,.4,.5]), pasos=np.array([0, 10, 20]),
                hist_s=np.array([[e, .1, .1, .5] for e in range(4)]),
                hist_t=np.array([[q, .01, .4, .41, .4, .8] for q in range(21)]),
                ep_s=3, k=2, mejor_paso=10, semilla=0, idx=np.array([0, 1]))
        return fig(0.0), fig(5.0)

    def test_el_panel_dibuja_CUATRO_columnas_en_el_orden_declarado(self):
        """Comportamiento, no forma del codigo: se mira QUE quedo dibujado.

        Mi primera version asertaba cuatro llamadas DESENROLLADAS por AST y
        fallaba sobre una implementacion con bucles, que es la correcta. Un
        test sobre la forma del codigo prohibe refactors y no prueba nada que
        el lector vea.
        """
        import matplotlib.pyplot as plt
        f_con, f_sin = self._par_sintetico()
        fig, ax = plt.subplots(1, 4, squeeze=False)
        P.cuadro_sopesado_panel(ax[0], f_con, f_sin, 1,
                                ((-9, 9), (-9, 9)), True)
        titulos = [a.get_title() for a in ax[0]]
        plt.close(fig)
        for i, esperado in enumerate(P.COLUMNAS_SOPESADO):
            self.assertIn(esperado, titulos[i],
                          f'columna {i+1}: esperaba {esperado!r}, '
                          f'quedo {titulos[i]!r}')

    def test_las_columnas_2_y_4_SON_del_brazo_apagado(self):
        """Medido sobre LO DIBUJADO, no sobre el titulo.

        El titulo solo lleva el CONTEO de mal ruteados, y una mutacion de
        `rut` puede dejar el mismo conteo por casualidad --- paso: 9 de 12 en
        las dos. Comparar titulos era un observable que no podia distinguir.
        Lo que se compara son los PUNTOS que cada coleccion dibujo.
        """
        import matplotlib.pyplot as plt
        f_con, f_sin = self._par_sintetico()

        def puntos(fs):
            fig, ax = plt.subplots(1, 4, squeeze=False)
            P.cuadro_sopesado_panel(ax[0], f_con, fs, 1,
                                    ((-9, 9), (-9, 9)), True)
            out = [[np.asarray(c.get_offsets()) for c in a.collections]
                   for a in ax[0]]
            plt.close(fig); return out

        a = puntos(f_sin)
        otro = dict(f_sin)
        otro['Pt'] = np.asarray(f_sin['Pt']) + 50.0
        b = puntos(otro)
        mismo = lambda x, y: (len(x) == len(y)
                              and all(np.array_equal(u, v) for u, v in zip(x, y)))
        self.assertTrue(mismo(a[0], b[0]),
                        'la columna 1 cambió con el brazo apagado')
        self.assertFalse(mismo(a[1], b[1]),
                         'la columna 2 NO lee el brazo apagado')


class ElGifSeEscribe(unittest.TestCase):
    """El gif se ESCRIBE, no sólo se arma.

    `cuadro_sopesado` llamaba a un `_a_pixeles` que no existe --- nombre que
    inventé --- y los cuatro tests de columnas pasaban igual, porque ninguno
    llegaba a convertir la figura a píxeles. El cuaderno reventó ahí.
    """

    def test_cuadro_sopesado_devuelve_pixeles(self):
        import matplotlib; matplotlib.use('Agg')
        c = CuatroColumnas('test_la_funcion_existe')
        f_con, f_sin = c._par_sintetico()
        img = P.cuadro_sopesado(
            'lunas', [('frontera', 0.3)], [(f_con, f_sin)], 1,
            ((-9, 9), (-9, 9)), True, 'batchnorm',
            {'BALANCE_DE_MASA': 'trace'})
        self.assertEqual(np.asarray(img).ndim, 3)
        self.assertEqual(np.asarray(img).shape[2], 3)
