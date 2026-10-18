"""El COLAPSO: lo que un modelo sin etiquetas hace de verdad.

El acierto no es la lente. Con la cabeza siendo un solo `Linear`, el extractor
no puede ser general y discriminante a la vez, asi que el acierto mide esa
tension y no la calidad de la alineacion. Lo que SI hace un modelo alimentado
con puntos sin guia es COLAPSAR --- a una clase equivocada o no --- y eso es un
problema real y medible.

Dos de los tres numeros son SIN ETIQUETAS a proposito, asi que no son
diagnostico: son observables que el metodo podria leer.

  `cls_ef`   clases efectivas, `exp(H(g_promedio))`. Un modelo que mete todo
             en una clase da 1; uno que reparte parejo da C. Sin etiquetas.
  `rank_ef`  dimension efectiva del embedding del blanco, `(sum l)^2/sum l^2`.
             Una nube que se aplasta a una recta da 1. Sin etiquetas.
  `tr_e`/`tr_d`  ya existian, PERO solo como cociente en los barridos. El
             cociente confunde dos fenomenos OPUESTOS: `tr_e` cayendo son
             clases que se fusionan, `tr_d` creciendo son nubes que se
             desparraman. Separados distinguen; juntos no.
"""
import sys, os, unittest
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from CREDA import toy_experiment as T


class Colapso(unittest.TestCase):

    def test_clases_efectivas_es_C_con_posterior_parejo(self):
        for C in (2, 3, 5):
            g = torch.full((40, C), 1.0 / C)
            self.assertAlmostEqual(T.colapso(np.zeros((40, 4)), g)['cls_ef'],
                                   float(C), places=5, msg=f'C={C}')

    def test_clases_efectivas_es_1_cuando_todo_cae_en_una(self):
        g = torch.zeros(40, 4); g[:, 2] = 1.0
        self.assertAlmostEqual(T.colapso(np.zeros((40, 4)), g)['cls_ef'],
                               1.0, places=5)

    def test_clases_efectivas_lee_el_PROMEDIO_no_la_confianza(self):
        """Mitad segura de la clase 0 y mitad segura de la 1 da DOS, no una.

        Es la distincion entera: un modelo confiado en cada punto pero que usa
        dos clases NO colapso. Confundirlas convertiria `cls_ef` en una
        medida de confianza, que ya tenemos y es otra cosa.
        """
        g = torch.zeros(40, 4); g[:20, 0] = 1.0; g[20:, 1] = 1.0
        self.assertAlmostEqual(T.colapso(np.zeros((40, 4)), g)['cls_ef'],
                               2.0, places=5)

    def test_rango_efectivo_es_1_sobre_una_recta(self):
        r = np.random.default_rng(0)
        t = r.normal(size=(60, 1)); v = r.normal(size=(1, 6))
        H = (t @ v).astype(np.float32)
        g = torch.full((60, 3), 1 / 3)
        self.assertAlmostEqual(T.colapso(H, g)['rank_ef'], 1.0, places=4)

    def test_rango_efectivo_es_d_sobre_una_nube_isotropa(self):
        r = np.random.default_rng(1)
        H = r.normal(size=(4000, 5)).astype(np.float32)
        g = torch.full((4000, 3), 1 / 3)
        self.assertAlmostEqual(T.colapso(H, g)['rank_ef'], 5.0, delta=0.25)

    def test_rango_efectivo_no_se_mueve_con_la_escala(self):
        """Una nube que se encoge NO perdio rango. Sin esto, `rank_ef` leeria
        un cambio de unidades como un colapso."""
        r = np.random.default_rng(2)
        H = r.normal(size=(200, 6)).astype(np.float32)
        g = torch.full((200, 3), 1 / 3)
        a = T.colapso(H, g)['rank_ef']; b = T.colapso(H * 0.01, g)['rank_ef']
        self.assertAlmostEqual(a, b, places=4)

    def test_una_corrida_real_publica_los_dos_y_las_trazas_sueltas(self):
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 0)
        h1 = dict(T.HP_1); h1['DIM'] = 8
        _, _, _, _, i = T.corrida(fam, D, 'ninguna', semilla=0,
                                  mover_fuente=False, cuadros=(0,), hp1=h1,
                                  hp2={**T.HP_2, 'LOTE_2': 64}, peso='declarado')
        self.assertEqual(set(i['colapso']), {'cls_ef', 'rank_ef'})
        for k in ('tr_e', 'tr_d'):          # ya existian; que sigan sueltos
            self.assertIsInstance(i[k], float)


if __name__ == '__main__':
    unittest.main()
