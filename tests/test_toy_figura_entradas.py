"""The first panel of the weighting section: the INPUT clouds, atypicals marked.

Asked three times and never delivered, and the reason it could not be was
measured rather than stylistic: with an injected perturbation the artifact
recorded WHICH points moved but never WHERE they landed, and `frontera`'s
landings depend on a phase-1 head trained in-process and persisted nowhere.

On clean data that obstacle is gone, because the atypical points are no longer
injected -- they are the MISROUTED ones, and the sweep's own npz names them:
`rut[k]` is the predicted class at the stopping checkpoint and `yt_tr` the true
one, so the set is `rut[k] != yt_tr`. The clouds themselves come from
`guardar_entradas`, because `toy_plots` imports no `torch` and cannot generate
them.

Two columns, source and target, because these toysets ARE the shift: a target
cloud on its own shows a shape and not a transfer.
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA import toy_experiment as T
from CREDA import toy_plots as P


def _contam(nombre='lunas', n=60, semilla=0, mal=(1, 5, 9)):
    rut = np.zeros((3, n), dtype=np.int16)
    yt = np.zeros(n, dtype=np.int16)
    for j in mal:
        rut[:, j] = 1                      # predicha != real -> mal ruteado
    # Clave de TRES partes y en TUPLA, que es lo que el barrido escribe en
    # memoria. El pipe-joined es la forma en DISCO, porque JSON no tiene
    # claves de tupla: leer una con la convencion de la otra es el error que
    # este fixture cometio primero.
    return {'rhos': [0.0], 'semillas': [semilla],
            'celdas': {(nombre, 'reubicado', 0.0): {'figura': {
                'k': 2, 'pasos': np.array([0, 20, 40]), 'rut': rut,
                'yt_tr': yt, 'semilla': semilla}}}}


class LaFiguraDeEntradas(unittest.TestCase):

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dest = Path(self.tmp.name)
        self.ruta_e = T.guardar_entradas(self.dest, semillas=[0])
        self.ENTR = dict(np.load(self.ruta_e))

    def tearDown(self):
        self.tmp.cleanup()

    def test_it_draws_one_file_per_toyset_and_says_where(self):
        D = T.datos_semilla(T.FAMILIAS[0], 0)
        n = len(D['Xt_tr'])
        C = _contam('lunas', n=n, mal=(1, 5, 9))
        fg = C['celdas'][('lunas', 'reubicado', 0.0)]['figura']
        fg['yt_tr'] = D['yt_tr'].numpy().astype(np.int16)
        rut = D['yt_tr'].numpy().astype(np.int16)[None, :].repeat(3, 0).copy()
        for j in (1, 5, 9):
            rut[:, j] = (rut[:, j] + 1) % T.CLASES
        fg['rut'] = rut
        ruta = P.figura_entradas('lunas', C, self.ENTR, T.banderas(),
                                 destino=self.dest)
        self.assertTrue(ruta.exists())
        self.assertIn('lunas', ruta.name)
        self.assertGreater(ruta.stat().st_size, 10_000)

    def test_it_refuses_a_cloud_whose_length_is_not_the_routing_length(self):
        """A figure that silently zipped a 1080-point cloud against a
        1079-point routing would mark the wrong points and look fine."""
        C = _contam('toroide', n=11)
        with self.assertRaises(AssertionError):
            P.figura_entradas('toroide', C, self.ENTR, T.banderas(),
                              destino=self.dest)

    def test_it_refuses_when_the_seed_has_no_persisted_cloud(self):
        D = T.datos_semilla(T.FAMILIAS[0], 0)
        C = _contam('lunas', n=len(D['Xt_tr']), semilla=7)
        C['celdas'][('lunas', 'reubicado', 0.0)]['figura']['yt_tr'] = \
            D['yt_tr'].numpy().astype(np.int16)
        with self.assertRaises(AssertionError):
            P.figura_entradas('lunas', C, self.ENTR, T.banderas(),
                              destino=self.dest)

    def test_the_drawing_half_still_imports_no_torch(self):
        """The invariant this whole detour exists to respect."""
        import inspect
        self.assertNotIn('import torch', inspect.getsource(P))


if __name__ == '__main__':
    unittest.main()
