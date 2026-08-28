"""Los pasos locales que la forja puede ejecutar por su cuenta.

Cada función de acá ejecuta un cuaderno de este repositorio y nada más. La
forja las nombra estáticamente en `__steps__` (módulo + función), las resuelve
sin importarlas, y recién las importa y llama dentro del intérprete de este
repositorio.

**Por qué no se escribe acá el prefijo de `PATH`.** El kernelspec que graban
los cuadernos arranca un `python` pelado, sin ruta absoluta, así que resuelve
contra el `PATH` del proceso que lo lanza. Componer ese `PATH` es trabajo de la
forja, que ya antepone `<repo>/.venv/bin` antes de llamar a estas funciones. Si
cada autor lo escribiera de nuevo acá, la corrección dependería de que ninguno
se olvide -- y ya se olvidó una vez, con quince fallas fantasma contra una
suite que pasa 297/297.

`sys.executable` y nunca `python3`: adentro de este proceso ya es el intérprete
de este repositorio, y nombrarlo directamente no depende de que el `PATH` que
lo rodea sea el correcto.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CUADERNOS = Path(__file__).resolve().parents[2] / "MIL-CREDA" / "Notebooks"


def _ejecutar(nombre: str) -> str:
    """Ejecuta un cuaderno en el lugar y devuelve su ruta.

    `--inplace` a propósito: la salida ejecutada ES el informe, y un cuaderno
    que se ejecuta a un archivo nuevo deja el original en el árbol diciendo
    algo distinto. Sin `--allow-errors`, así que una celda que falla corta acá
    en lugar de dejar un informe que parece completo.
    """
    cuaderno = CUADERNOS / nombre
    if not cuaderno.is_file():
        raise FileNotFoundError(f"no existe el cuaderno {cuaderno}")
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert",
         "--to", "notebook", "--execute", "--inplace", str(cuaderno)],
        check=True)
    return str(cuaderno)


def verificacion() -> str:
    """Nivel 1: la suite corre adentro del cuaderno y el sello queda actual."""
    return _ejecutar("verification.ipynb")


def busqueda() -> str:
    """La búsqueda de techos, con su propio registro."""
    return _ejecutar("Benchmark_Search_v1.ipynb")


def informe() -> str:
    """Las tablas y conclusiones sobre el registro fusionado."""
    return _ejecutar("Benchmark_Report_v1.ipynb")


def latente() -> str:
    """El análisis latente sobre los checkpoints promovidos."""
    return _ejecutar("Benchmark_Latent_v1.ipynb")
