"""The INPUT clouds, persisted, so the drawing half can show them.

The weighting section's first panel is the input data with the atypical points
marked, and it could not be drawn before for a reason that is structural rather
than an oversight: `toy_plots` imports no `torch` -- a declared invariant with
a test -- so it cannot call `datos_semilla`, and the 3-D inputs were in no
artifact. Only the PROJECTED latent was, in two dimensions.

Which points are atypical IS already derivable: the sweep's npz carries
`yt_tr` and, per checkpoint, `rut` -- the predicted class -- so the misrouted
set is `rut[k] != yt_tr`. The clouds themselves were the only missing half.

They live in their own artifact rather than inside each sweep's npz because
they are a property of `(familia, semilla)` and of nothing else: the same seed
gives the same clouds for every section, every rho and every arm, and copying
three 3-D clouds into every sweep would be the same bytes under several names
-- which is how two copies of one thing drift apart.
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from CREDA import toy_experiment as T


class LasEntradasSePersisten(unittest.TestCase):

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dest = Path(self.tmp.name)
        self.ruta = T.guardar_entradas(self.dest, semillas=[0, 1])

    def tearDown(self):
        self.tmp.cleanup()

    def test_it_writes_one_file_and_says_where(self):
        self.assertTrue(self.ruta.exists())
        self.assertEqual(self.ruta.suffix, '.npz')

    def test_every_family_and_seed_carries_its_four_arrays(self):
        A = np.load(self.ruta)
        for fam in T.FAMILIAS:
            for s in (0, 1):
                for campo in ('Xs_tr', 'Xt_tr', 'ys_tr', 'yt_tr'):
                    self.assertIn(f"{fam['nombre']}|{s}|{campo}", A.files)

    def test_the_clouds_are_the_ambient_dimension_not_a_projection(self):
        """The panel asked for is the INPUT data. A 2-D array here would mean
        the projected latent sneaked in under the clouds' name."""
        A = np.load(self.ruta)
        for fam in T.FAMILIAS:
            x = A[f"{fam['nombre']}|0|Xt_tr"]
            self.assertEqual(x.ndim, 2)
            self.assertEqual(x.shape[1], fam['amb'])

    def test_what_is_written_is_what_datos_semilla_gives(self):
        """Not a second generation path that resembles the first. If these
        drift, the figure shows points the run never saw."""
        A = np.load(self.ruta)
        for fam in T.FAMILIAS:
            D = T.datos_semilla(fam, 1)
            for campo in ('Xs_tr', 'Xt_tr', 'ys_tr', 'yt_tr'):
                self.assertTrue(
                    np.array_equal(A[f"{fam['nombre']}|1|{campo}"],
                                   D[campo].numpy()),
                    (fam['nombre'], campo))

    def test_the_labels_align_with_their_cloud(self):
        A = np.load(self.ruta)
        for fam in T.FAMILIAS:
            n = fam['nombre']
            self.assertEqual(len(A[f'{n}|0|yt_tr']), len(A[f'{n}|0|Xt_tr']))
            self.assertEqual(len(A[f'{n}|0|ys_tr']), len(A[f'{n}|0|Xs_tr']))


if __name__ == '__main__':
    unittest.main()
