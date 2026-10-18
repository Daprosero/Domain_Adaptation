"""La demostracion DIRECTA del peso: tres puntos que caminan a un centro.

Todo lo que tenemos del sopesado es estadistica de RESULTADOS. Ninguna pieza
mostraba el MECANISMO. Esta figura lo muestra por construccion, sin semillas,
sin tests y sin ruido: es una propiedad de la Ec. 19 y del cabezal ya
entrenado, no una medicion con varianza.

Lo medido ANTES de escribirla, sobre el cabezal real de `lunas`:

    frontera de DOS clases (empate 0.5/0.5)    w = 0.369072
    frontera TRIPLE (las tres en 1/3)          w = 0.000003
    centro de clase (one-hot)                  w = 1.000001
    minimo sobre todo el train de la fuente    w = 0.36899

Por eso el tercer punto arranca en la frontera TRIPLE y no en una tercera
frontera de dos: tres puntos en fronteras dobles arrancan los tres en 0.369 y
las tres curvas son la misma. El triple es el unico lugar donde `w` se anula
de verdad, y es donde la premisa del operador se ve literal --- el punto puede
irse a cualquiera de las tres clases y `w` sube igual.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import torch
from CREDA import toy_experiment as T


class FronterasYPesos(unittest.TestCase):

    def setUp(self):
        import matplotlib; matplotlib.use('Agg')

    def _cabezal(self):
        """Un cabezal de verdad: dim latente conocida y pesos no triviales."""
        torch.manual_seed(0)
        cab = torch.nn.Linear(2, T.CLASES)
        with torch.no_grad():
            cab.weight.copy_(torch.tensor([[1.0, 0.0], [-0.5, 0.9], [-0.5, -0.9]]))
            cab.bias.zero_()
        return cab

    def test_existe(self):
        self.assertTrue(hasattr(T, 'puntos_de_frontera'))
        self.assertTrue(hasattr(T, 'caminata_al_centro'))

    def test_el_punto_de_frontera_DOBLE_empata_dos_logits(self):
        cab = self._cabezal()
        p = T.puntos_de_frontera(cab, (0, 1))
        with torch.no_grad():
            lg = cab(torch.as_tensor(p, dtype=torch.float32).reshape(1, -1))[0]
        self.assertAlmostEqual(float(lg[0]), float(lg[1]), places=4)
        # Y los dos empatados son los MAYORES: un empate en el fondo no es
        # una frontera, es un punto cualquiera donde otra clase gana.
        self.assertGreaterEqual(float(lg[0]), float(lg.max()) - 1e-4)

    def test_el_punto_TRIPLE_empata_los_tres(self):
        cab = self._cabezal()
        p = T.puntos_de_frontera(cab, tuple(range(T.CLASES)))
        with torch.no_grad():
            lg = cab(torch.as_tensor(p, dtype=torch.float32).reshape(1, -1))[0]
        self.assertLess(float(lg.max() - lg.min()), 1e-4, lg)

    def test_el_peso_en_cada_frontera_es_el_MEDIDO(self):
        """0.369072 en la doble, 0.000003 en la triple. Son los numeros que
        la figura promete y salen de la Ec. 19, no de una corrida."""
        cab = self._cabezal()
        # La tolerancia es `1e-3` y no `1e-4` a proposito: 0.369072 es el
        # limite con la tercera clase EXACTAMENTE en cero, que es inalcanzable
        # --- el punto construido deja 7e-05 de masa ahi y da 0.368945. Exigir
        # el limite exacto seria exigir algo que la construccion no promete.
        for clases, esperado in (((0, 1), 0.369072), ((0, 1, 2), 0.000003)):
            p = T.puntos_de_frontera(cab, clases)
            with torch.no_grad():
                lg = cab(torch.as_tensor(p, dtype=torch.float32).reshape(1, -1))
            w = float(T.pesos_declarados(lg)[0])
            self.assertAlmostEqual(w, esperado, delta=1e-3, msg=(clases, w))
            # Y que la masa de las NO nombradas sea despreciable: sin esto,
            # un `w` parecido por casualidad pasaria sobre un punto que no es
            # esa frontera.
            g = torch.softmax(lg, 1)[0].numpy()
            otras = [c for c in range(T.CLASES) if c not in clases]
            self.assertLess(float(g[otras].sum()) if otras else 0.0, 1e-3,
                            (clases, g))

    def test_la_caminata_SUBE_monotona_y_llega_a_uno(self):
        cab = self._cabezal()
        ini = T.puntos_de_frontera(cab, (0, 1))
        centro = np.array([6.0, 0.0])          # bien adentro de la clase 0
        pas, w = T.caminata_al_centro(cab, ini, centro, pasos=40)
        self.assertEqual(len(pas), len(w))
        # `delta` y no `places`, por lo mismo que arriba: 0.369072 es el
        # limite inalcanzable y la construccion da 0.368945.
        self.assertAlmostEqual(float(w[0]), 0.369072, delta=1e-3)
        self.assertGreater(float(w[-1]), 0.999)
        # Monotona: la premisa de la figura es que `w` SUBE al acercarse.
        d = np.diff(w)
        self.assertGreaterEqual(float(d.min()), -1e-6, 'la curva baja en algún tramo')

    def test_la_caminata_desde_el_TRIPLE_arranca_en_cero(self):
        cab = self._cabezal()
        ini = T.puntos_de_frontera(cab, tuple(range(T.CLASES)))
        pas, w = T.caminata_al_centro(cab, ini, np.array([6.0, 0.0]), pasos=40)
        self.assertLess(float(w[0]), 1e-4, float(w[0]))
        self.assertGreater(float(w[-1]), 0.999)


if __name__ == '__main__':
    unittest.main()


class ElGifDeLasFronteras(unittest.TestCase):
    """La caminata ANIMADA: el punto avanza y la curva se dibuja detrás.

    Es lo que el operador pidió desde el principio --- «que ese se mueva con
    las épocas a un centro de una clase» --- y lo que una figura estática no
    muestra: que `w` sube MIENTRAS el punto camina, no que al final valga 1.
    """

    def setUp(self):
        import matplotlib; matplotlib.use('Agg')

    def _datos(self):
        import torch
        r = np.random.default_rng(0)
        H = r.normal(size=(120, 2)) * 2.0
        ys = np.arange(120) % T.CLASES
        for c in range(T.CLASES):
            H[ys == c] += np.array([np.cos(2*np.pi*c/3), np.sin(2*np.pi*c/3)])*5
        cab = torch.nn.Linear(2, T.CLASES)
        with torch.no_grad():
            cab.weight.copy_(torch.tensor([[1.0, 0.0], [-0.5, 0.9], [-0.5, -0.9]]))
            cab.bias.zero_()
        return H, ys, cab

    def test_existe_y_escribe_un_gif_de_mas_de_un_cuadro(self):
        import tempfile
        from pathlib import Path
        from CREDA import toy_plots as P
        self.assertTrue(hasattr(P, 'animar_fronteras_w'))
        H, ys, cab = self._datos()
        with tempfile.TemporaryDirectory() as d:
            r = P.animar_fronteras_w('lunas', H, ys, cab, Path(d), pasos=8,
                                     al_paso=lambda *a: None)
            self.assertTrue(Path(r).exists(), r)
            self.assertGreater(Path(r).stat().st_size, 5_000, 'gif sospechoso')

    def test_el_cuadro_j_dibuja_la_curva_HASTA_j_y_no_entera(self):
        """Si cada cuadro trajera la curva completa, el gif sería un PNG
        repetido: lo que la animación muestra es el avance."""
        from CREDA import toy_plots as P
        import matplotlib.pyplot as plt
        H, ys, cab = self._datos()
        largos = []
        for j in (1, 6):
            fig = P.figura_cuadro_fronteras('lunas', H, ys, cab, j, pasos=8)
            lineas = [l for l in fig.axes[1].get_lines()
                      if len(l.get_xdata()) > 1]
            largos.append(max(len(l.get_xdata()) for l in lineas))
            plt.close(fig)
        self.assertLess(largos[0], largos[1],
                        f'la curva no crece con el cuadro: {largos}')
