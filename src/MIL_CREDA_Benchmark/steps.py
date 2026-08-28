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


def ensayo_de_busqueda() -> dict:
    """Corre la búsqueda de techos a escala de ensayo y devuelve su registro.

    `pilot=True` y sin manera de pedir otra cosa. La búsqueda a escala completa
    es una decisión con su propia autorización, y un paso local que aceptara un
    argumento para elegirla dejaría esa decisión en manos de quien escribe la
    invocación. Escribe `ceilings.pilot.json`, que el registro completo siempre
    le gana cuando existe.

    Existe porque no existía: la única forma de correr esta búsqueda era un
    `python -c` a mano contra `harness`, por fuera de la skill y sin dejar
    rastro en el ledger. Un paso que falta no es disciplina que falta.
    """
    from MIL_CREDA_Benchmark import harness

    return harness.run_search(pilot=True)


def campana() -> dict:
    """Corre la campaña en esta máquina, a la escala que declara `config`.

    Arma la reducción igual que el cuaderno de campaña -- dispositivo,
    ambiente, y los techos vigentes -- y llama a `harness.campaign()`. Lo que
    la vuelve un ensayo no es un parámetro de acá sino `config`: las épocas y
    las semillas que el repositorio declara. Este paso no las elige, y por eso
    no toma argumentos.

    No hay que ordenar nada acá: `campaign()` se niega sola sin un registro de
    techos buscado, así que la búsqueda va antes por construcción y no por
    convención.

    Esto NO envía nada al servicio remoto. `Benchmark_Campaign_v1.ipynb` sigue
    fuera de `__steps__` por eso mismo: un envío necesita una aprobación humana
    por lanzamiento, y ningún paso local puede dársela.
    """
    from MIL_CREDA_Benchmark import harness

    dispositivo = harness.resolve_device()
    reduccion = harness.Reduction(device=str(dispositivo),
                                  environment=harness.environment())
    reduccion = harness.with_ceilings_in_force(reduccion, dispositivo)
    return harness.campaign(reduccion, dispositivo)


def informe_de_busqueda() -> str:
    """El INFORME de la búsqueda de techos, no la búsqueda.

    El cuaderno lee un registro que ya existe (`harness.search_record()`) y lo
    presenta; la llamada que corre la búsqueda está comentada adentro, a
    propósito -- un cuaderno de informe que además ejecutara horas de cómputo
    haría que abrirlo cueste lo que cuesta correrlo.

    El nombre dice cuál de las dos cosas es. La primera versión de esta función
    se llamaba `busqueda` y prometía en su docstring "la búsqueda de techos",
    que es exactamente lo que NO hace: un lector la habría llamado esperando un
    registro nuevo y habría recibido el viejo, presentado.
    """
    return _ejecutar("Benchmark_Search_v1.ipynb")


def informe() -> str:
    """Las tablas y conclusiones sobre el registro fusionado."""
    return _ejecutar("Benchmark_Report_v1.ipynb")


def latente() -> str:
    """El análisis latente sobre los checkpoints promovidos."""
    return _ejecutar("Benchmark_Latent_v1.ipynb")
