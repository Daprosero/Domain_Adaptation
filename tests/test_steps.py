"""Los pasos locales que este repositorio le ofrece a la forja.

Nivel 1 - invariantes de estructura. No ejecuta ningún cuaderno: ejecutar uno
tarda minutos y lo que acá se afirma es qué se ofrece y qué no, no qué imprime.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import MIL_CREDA_Benchmark as paquete
from MIL_CREDA_Benchmark import steps


class PasosDeclaradosTests(unittest.TestCase):

    def test_cada_paso_declarado_resuelve_a_un_llamable(self):
        for nombre, entrada in paquete.__steps__.items():
            with self.subTest(paso=nombre):
                self.assertEqual(entrada["module"], "MIL_CREDA_Benchmark.steps")
                funcion = getattr(steps, entrada["function"], None)
                self.assertTrue(callable(funcion),
                                f"{entrada['function']} no es llamable")

    def test_cada_paso_toma_cero_argumentos(self):
        """La forja llama sin argumentos: la declaración no lleva `kwargs`."""
        for nombre, entrada in paquete.__steps__.items():
            with self.subTest(paso=nombre):
                firma = inspect.signature(getattr(steps, entrada["function"]))
                self.assertEqual(list(firma.parameters), [])

    def test_los_cuadernos_que_los_pasos_nombran_existen(self):
        for nombre in ("verification.ipynb", "Benchmark_Search_v1.ipynb",
                       "Benchmark_Report_v1.ipynb", "Benchmark_Latent_v1.ipynb"):
            with self.subTest(cuaderno=nombre):
                self.assertTrue((steps.CUADERNOS / nombre).is_file(),
                                f"{steps.CUADERNOS / nombre} no existe")

    def test_la_campana_no_se_ofrece_como_paso_local(self):
        """La ausencia es la decisión, no un olvido.

        `Benchmark_Campaign_v1.ipynb` lanza al servicio remoto. Un paso local
        que lo corriera saltearía la aprobación humana por envío, que es la
        única precondición que un llamador no puede satisfacer por su cuenta.
        Sin esta afirmacion la omisión se lee como un descuido y alguien la
        "completa".
        """
        self.assertTrue((steps.CUADERNOS / "Benchmark_Campaign_v1.ipynb").is_file(),
                        "el cuaderno de campaña no está: esta prueba no probaría nada")
        nombrados = {Path(fuente).name
                     for fuente in _cuadernos_nombrados_por_los_pasos()}
        self.assertNotIn("Benchmark_Campaign_v1.ipynb", nombrados)

    def test_un_cuaderno_ausente_se_rechaza_en_vez_de_correr(self):
        with self.assertRaises(FileNotFoundError):
            steps._ejecutar("no_existe.ipynb")


def _cuadernos_nombrados_por_los_pasos() -> list[str]:
    """Los literales de cuaderno que el módulo de pasos menciona, leídos del
    código y no de una lista escrita a mano que puede quedar vieja."""
    import ast

    fuente = Path(steps.__file__).read_text(encoding="utf-8")
    return [nodo.value for nodo in ast.walk(ast.parse(fuente))
            if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str)
            and nodo.value.endswith(".ipynb")]


if __name__ == "__main__":
    unittest.main()
