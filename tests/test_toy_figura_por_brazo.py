"""El escenario guarda la figura de LOS DOS brazos de sopesado, no de uno.

El gif de la seccion 2 lleva cuatro columnas que el operador declaro: con
sopesado, sin sopesado, loss de fase 2 con, loss de fase 2 sin. Las columnas 2
y 4 son del brazo APAGADO, y el registro guardaba un solo `figura` --- el del
brazo con sopesado --- asi que la mitad del gif no tenia de donde salir.

`figura` (singular) se conserva como ALIAS del brazo con sopesado: el lado que
dibuja la seccion vieja lo lee, y romperlo seria un cambio que nadie pidio.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from CREDA import toy_experiment as T


class FiguraPorBrazo(unittest.TestCase):

    def test_el_barrido_guarda_una_figura_por_brazo_de_peso(self):
        C = T.barrer_contaminacion(
            familias=[T.FAMILIAS[0]], semillas=[0], rhos=(T.RHO_UNICO,),
            trials1=1, trials2=1, al_paso=lambda *a: None)
        cel = next(iter(C['celdas'].values()))
        self.assertIn('figuras', cel, 'la celda no guarda figuras por brazo')
        for b, _ in T.BRAZOS_POR_PESO:
            self.assertIn(b, cel['figuras'], f'falta la figura de {b!r}')
            self.assertIsNotNone(cel['figuras'][b], b)

    def test_las_dos_figuras_son_DISTINTAS(self):
        """Dos brazos distintos no pueden dejar la misma trayectoria.

        Sin esto, guardar el mismo objeto dos veces bajo dos claves pasaria el
        test de arriba y el gif dibujaria la misma columna dos veces.
        """
        import numpy as np
        C = T.barrer_contaminacion(
            familias=[T.FAMILIAS[0]], semillas=[0], rhos=(T.RHO_UNICO,),
            trials1=1, trials2=1, al_paso=lambda *a: None)
        cel = next(iter(C['celdas'].values()))
        a = np.asarray(cel['figuras'][T.BRAZO_CON_SOP]['Pt'])
        b = np.asarray(cel['figuras'][T.BRAZO_SIN_SOP]['Pt'])
        # NO se exige la misma forma: los dos brazos paran en pasos distintos,
        # asi que sus trayectorias tienen distinto numero de cuadros --- medido,
        # 9 contra 5 en lunas/rho=0.30. Esa diferencia YA prueba que no son la
        # misma figura; mi primera version lo asertaba al reves y fallaba sobre
        # codigo correcto.
        if a.shape == b.shape:
            self.assertFalse(np.array_equal(a, b),
                             'los dos brazos dejaron la MISMA trayectoria')
        # Y el primer cuadro SI es comparable: es el paso 0, donde los dos
        # parten de la misma fase 1. Tiene que ser identico, o no comparten
        # fase 1 y la comparacion apareada deja de serlo.
        import numpy as _np
        self.assertTrue(_np.allclose(a[0], b[0]),
                        'el paso 0 difiere: los brazos no comparten la fase 1')

    def test_la_clave_vieja_sigue_siendo_la_del_brazo_sopesado(self):
        C = T.barrer_contaminacion(
            familias=[T.FAMILIAS[0]], semillas=[0], rhos=(T.RHO_UNICO,),
            trials1=1, trials2=1, al_paso=lambda *a: None)
        cel = next(iter(C['celdas'].values()))
        self.assertIs(cel['figura'], cel['figuras'][T.BRAZO_CON_SOP])

    def test_las_dos_SOBREVIVEN_al_viaje_por_disco(self):
        """El hueco que me mordio: probe el CONTAM en MEMORIA y di por hecho
        que el `.npz` lo persistia. No lo hacia --- guardaba una sola figura
        con la clave `perturbacion|rho|campo` --- asi que el generador corrio
        bien, el artefacto salio sin `figuras`, y el gif reventó al dibujar.

        Una mitad verificada no es la prueba del todo.
        """
        import tempfile, numpy as np
        from pathlib import Path
        from CREDA import toy_plots as P
        C = T.barrer_contaminacion(
            familias=[T.FAMILIAS[0]], semillas=[0], rhos=(T.RHO_UNICO,),
            trials1=1, trials2=1, al_paso=lambda *a: None)
        with tempfile.TemporaryDirectory() as d:
            T.guardar_contaminacion(Path(d), C)
            leido = P.cargar_contaminacion(Path(d))
        cel = next(iter(leido['celdas'].values()))
        self.assertIn('figuras', cel, 'el .npz no persistió las figuras por brazo')
        for b, _ in T.BRAZOS_POR_PESO:
            self.assertIn(b, cel['figuras'], f'falta {b!r} tras el viaje')
        a = np.asarray(cel['figuras'][T.BRAZO_CON_SOP]['PS_t'])
        bb = np.asarray(cel['figuras'][T.BRAZO_SIN_SOP]['PS_t'])
        if a.shape == bb.shape:
            self.assertFalse(np.array_equal(a, bb),
                             'tras el viaje los dos brazos son la misma figura')


if __name__ == '__main__':
    unittest.main()
