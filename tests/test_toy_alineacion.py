"""La ALINEACION, que es lo que el sopesado promete y nunca se midio.

Las cuatro metricas del barrido --- acierto, confianza, concentracion, ruteo ---
son del blanco solo o del clasificador. Ninguna mide si la nube del blanco se
acerco a la de la fuente, que es literalmente lo que la Ec. 19 promete hacer
usando solo las muestras confiables.

Lo que cada test fija:

  VARA COMUN   el numero se calcula con `w = 1`, SIEMPRE, cualquiera sea el
               brazo. Puntuar cada brazo con su propio objetivo seria comparar
               dos funciones distintas y llamarlo una mejora.
  CARDINALIDAD alto y bajo se miden sobre subconjuntos del MISMO tamanio. El
               artefacto diagonal escala como 1/n --- medido, reducible 10.7x
               --- asi que dos subconjuntos de tamanio distinto no son
               comparables ni cuando el modelo no cambia.
  SELECCION    `alto` depende de los embeddings de los confiables y de nadie
               mas. Si mover los NO seleccionados lo mueve, no selecciono.
  ESCALA       `cls` es adimensional: escalar las dos nubes no lo mueve. Sin
               eso, una nube que se encoge se lee como una que se alineo.
"""
import sys, os, unittest
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from CREDA import toy_experiment as T
from CREDA.models import CREDALoss


def _datos(ns=40, nt=36, d=5, C=3, sem=0):
    r = np.random.default_rng(sem)
    Hs = r.normal(size=(ns, d)).astype(np.float32)
    Ht = r.normal(size=(nt, d)).astype(np.float32)
    ys = torch.as_tensor(r.integers(0, C, ns))
    yt = r.integers(0, C, nt)
    lg = torch.as_tensor(r.normal(size=(nt, C)).astype(np.float32))
    return Hs, Ht, ys, yt, lg


class Alineacion(unittest.TestCase):

    def test_devuelve_las_cuatro_claves(self):
        Hs, Ht, ys, yt, lg = _datos()
        a = T.alineacion(Hs, Ht, ys, lg, yt, sigma=1.0, cfg={})
        self.assertEqual(set(a), {'creda', 'creda_alto', 'creda_bajo', 'cls'})
        for k, v in a.items():
            self.assertTrue(np.isfinite(v), (k, v))

    def test_vara_comun_es_el_termino_SIN_sopesar(self):
        """El numero tiene que ser el del loss con `w = 1`, exacto."""
        Hs, Ht, ys, yt, lg = _datos()
        a = T.alineacion(Hs, Ht, ys, lg, yt, sigma=0.9, cfg={})
        esperado = float(CREDALoss(sigma=0.9, use_entropy_weighting=False)(
            torch.as_tensor(Hs), torch.as_tensor(Ht), ys,
            torch.softmax(lg, 1)))
        self.assertAlmostEqual(a['creda'], esperado, places=6)

    def test_la_vara_ignora_el_sopesado_que_le_pasen(self):
        """`use_entropy_weighting=True` en cfg no puede cambiar el numero."""
        Hs, Ht, ys, yt, lg = _datos()
        base = T.alineacion(Hs, Ht, ys, lg, yt, sigma=1.0, cfg={})
        con = T.alineacion(Hs, Ht, ys, lg, yt, sigma=1.0,
                           cfg={'use_entropy_weighting': True})
        self.assertAlmostEqual(base['creda'], con['creda'], places=7)

    def test_los_cuartiles_son_exactamente_los_k_declarados(self):
        """Fija CUALES entran y CUANTOS, contra el calculo independiente.

        Mi primera version de este test afirmaba que con pesos planos `alto` y
        `bajo` coinciden. Es falso: con el orden estable caen en los k
        primeros y los k ultimos, que son conjuntos DISTINTOS. Lo correcto es
        reconstruir los dos subconjuntos afuera y exigir el numero exacto ---
        eso fija la cardinalidad y la seleccion a la vez.
        """
        for nt in (36, 37, 38, 39):
            Hs, Ht, ys, yt, lg = _datos(nt=nt)
            k = T._k_cuartil(nt)
            self.assertEqual(k, max(2, nt // 4), nt)
            a = T.alineacion(Hs, Ht, ys, lg, yt, sigma=1.1, cfg={})
            w = T.pesos_declarados(lg)
            o = np.argsort(w, kind='stable')
            vara = CREDALoss(sigma=1.1, use_entropy_weighting=False)
            g = torch.softmax(lg, 1)
            for rot, idx in (('creda_alto', o[nt - k:]), ('creda_bajo', o[:k])):
                self.assertEqual(len(idx), k, (nt, rot))
                esp = float(vara(torch.as_tensor(Hs),
                                 torch.as_tensor(Ht[idx]), ys, g[idx]))
                self.assertAlmostEqual(a[rot], esp, places=6, msg=f'{rot} nt={nt}')

    def test_alto_solo_mira_a_los_confiables(self):
        """Mover a los NO seleccionados no puede mover `creda_alto`."""
        Hs, Ht, ys, yt, lg = _datos(nt=40)
        # `w` estrictamente creciente en el indice: logits cada vez mas picudos.
        lg = torch.zeros(40, 3)
        lg[:, 0] = torch.linspace(0.0, 8.0, 40)
        a0 = T.alineacion(Hs, Ht, ys, lg, yt, sigma=1.0, cfg={})
        Ht2 = Ht.copy()
        k = T._k_cuartil(40)
        Ht2[:40 - k] += 50.0            # destrozo a todos menos el cuartil alto
        a1 = T.alineacion(Hs, Ht2, ys, lg, yt, sigma=1.0, cfg={})
        self.assertAlmostEqual(a0['creda_alto'], a1['creda_alto'], places=6)
        self.assertNotAlmostEqual(a0['creda_bajo'], a1['creda_bajo'], places=3)

    def test_cls_es_adimensional(self):
        """Escalar las DOS nubes por igual no mueve la alineacion de clase."""
        Hs, Ht, ys, yt, lg = _datos()
        a0 = T.alineacion(Hs, Ht, ys, lg, yt, sigma=1.0, cfg={})
        a1 = T.alineacion(Hs * 7.0, Ht * 7.0, ys, lg, yt, sigma=7.0, cfg={})
        self.assertAlmostEqual(a0['cls'], a1['cls'], places=5)

    def test_cls_es_cero_cuando_las_clases_coinciden(self):
        Hs, Ht, ys, yt, lg = _datos(ns=30, nt=30)
        yt = ys.numpy().copy()
        a = T.alineacion(Hs, Hs.copy(), ys, lg[:30], yt, sigma=1.0, cfg={})
        self.assertLess(a['cls'], 1e-6)

    def test_una_corrida_real_publica_align(self):
        fam = T.FAMILIAS[0]
        D = T.datos_semilla(fam, 0)
        h1 = dict(T.HP_1); h2 = dict(T.HP_2)
        h1['DIM'] = 8
        _, _, _, _, i = T.corrida(fam, D, 'ninguna', semilla=0,
                                  mover_fuente=False, cuadros=(0,),
                                  hp1=h1, hp2={**h2, 'LOTE_2': 64},
                                  peso='declarado')
        self.assertEqual(set(i['align']),
                         {'creda', 'creda_alto', 'creda_bajo', 'cls'})


if __name__ == '__main__':
    unittest.main()
