"""Las fronteras EXACTAS del cabezal y la caminata de confianza hacia un centro.

`CREDA.tutorial_visuals` dibuja, en el cuaderno tutorial, `w` a lo largo del
segmento que va de una frontera del cabezal al centro de una clase. Esa curva
es una propiedad de la Ec. 24 y del cabezal, no una medicion con varianza, y
estos tests la fijan sobre `puntos_de_frontera` y `caminata_al_centro`.

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

    def _cabezal(self):
        """Un cabezal de verdad: dim latente conocida y pesos no triviales."""
        torch.manual_seed(0)
        cab = torch.nn.Linear(2, T.CLASES)
        with torch.no_grad():
            cab.weight.copy_(torch.tensor([[1.0, 0.0], [-0.5, 0.9], [-0.5, -0.9]]))
            cab.bias.zero_()
        return cab

    def test_una_sola_clase_no_tiene_frontera_y_se_refuta(self):
        with self.assertRaises(AssertionError):
            T.puntos_de_frontera(self._cabezal(), (0,))

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


if __name__ == '__main__':
    unittest.main()
