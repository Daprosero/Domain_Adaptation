"""MIL-CREDA: reference implementation of the managed proposal.

One module per mathematical object. Every module declares `__provenance__`
binding it to the revision it was written against.

The numeric backend is PyTorch. That is a change of backend, not of method:
every equation is the same equation, but each term of Eq. (39) now carries a
gradient and can be placed on a device, which is the difference between an
objective that is evaluated and one that is trained.

This file holds the only thing here that is not a mathematical object — how a
caller's input becomes a tensor — precisely so that no module of mathematics has
to spend lines on plumbing.

Double precision is deliberate. The invariants of Section 2 are checked at
tolerances near 1e-12, and single precision cannot resolve them: a bound that
holds only to 1e-6 is not the bound the proposal states. A caller who trains in
single precision passes float32 tensors and gets them back respected.
"""

from __future__ import annotations

import torch

#: The dtype used for anything this package has to materialize itself.
DTYPE = torch.float64


def as_tensor(value: object, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Accept whatever the caller has; return a tensor without disturbing one.

    A tensor is returned untouched. Re-casting it would silently detach it from
    the graph that produced it, or pull a device-resident term back to the host,
    and either one turns a trainable implementation back into a calculator.
    Anything else — a list, an array, a Python float — becomes a tensor.
    """
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=DTYPE if dtype is None else dtype)


def as_matrix(value: object) -> torch.Tensor:
    """`as_tensor`, then at least two dimensions: one row per sample."""
    tensor = as_tensor(value)
    return tensor.reshape(1, -1) if tensor.dim() < 2 else tensor


def as_index(value: object, *, device: torch.device | None = None) -> torch.Tensor:
    """An index set or a label vector as a flat `long` tensor.

    `device` moves it next to the matrix it will index: advanced indexing
    requires both to live in the same place, and the caller's index set is
    usually built on the host even when the kernel matrices are not.
    """
    if isinstance(value, torch.Tensor):
        index = value.reshape(-1).long()
    else:
        index = torch.as_tensor(value, dtype=torch.long).reshape(-1)
    return index if device is None else index.to(device)

__implementation__ = {
    "revision": "research-concept-r44.md",
    "premises": {
        "prediction": "a single class per bag, chosen from CLASSES alternatives",
        "unit": "the bag, for every arm",
        "metric": "accuracy over the evaluation bags of the target domain",
        "direction": "higher is better",
    },
}

# La escalera de escalones que un paso de la sección de posición puede
# alcanzar, en las palabras de este repositorio. Tres, y el primero no es un
# adorno: `_record_scale_level` usa el escalón más bajo para decir «todavía no
# corrió nada», y coloca «corrió, pero corto de la escala declarada» un escalón
# debajo del tope solo si la escalera tiene tres o más. Con dos, «no hay
# búsqueda» y «la búsqueda corrió en ensayo» caerían en el mismo escalón.
#
# `none`   -- no corrió nada. Es un escalón, no la ausencia de uno: «no miramos»
#             es otra cosa y la forja la reporta aparte, sin medir.
# `pilot`  -- corrió acá, a escala de ensayo. Prueba que el cable lleva
#             corriente y no habilita a citar ningún número.
# `remote` -- corrió a la escala que el protocolo declara. Es el único escalón
#             cuyos resultados se reportan.
#
# El orden es el de la lista y lo compara la forja por posición, sin conocer
# ninguno de los tres nombres. Una primera pasada apunta a `pilot` y ahí
# termina; pasar de `pilot` a `remote` es una decisión aparte, con su
# autorización y su porqué.
__levels__: list = ["none", "pilot", "remote"]

# Lo que hay que instalar en el worker antes de que corra nada, en la forma
# EXACTA del bloque `environment` de `run-config.json`. La forja lo pasa entrada
# por entrada (`generate-job --environment-requirement`) y el arranque lo corre
# como `python -m pip install <entradas>`, sin shell: cada entrada es un
# especificador posicional y uno que empiece con `-` se rechaza, así que no hay
# forma de meter una bandera acá.
#
# **Por qué recién ahora.** Mientras un envío llamaba a una FUNCIÓN este paso
# del arranque era un no-op y no se notaba: el worker importaba
# `MIL_CREDA_Benchmark.harness` y lo que esa importación necesita ya venía en la
# imagen. Ejecutar un CUADERNO agrega requisitos que ninguna de esas corridas
# ejerció, y la celda 0 se niega cuando no los encuentra --- nombrando
# justamente esta declaración como el remedio. Sin ella el primer envío se
# rechaza antes de ejecutar una celda.
#
# **Las cuatro, y qué hace cada una.** Las tres primeras son las que la celda 0
# IMPORTA por su nombre para decidir si el runtime puede ejecutar un cuaderno:
# `nbformat` lo lee y lo vuelve a escribir, `nbclient` lo ejecuta, y
# `jupyter-client` es lo que resuelve un NOMBRE de kernel contra un kernelspec
# instalado. La tercera no sobra por venir de arrastre con la segunda: un
# runtime puede tener los dos lectores y ningún kernelspec, y ésa es la falla
# que ya costó un envío entero. `ipykernel` es la cuarta y no la nombra ningún
# rechazo: es lo que instala el kernelspec `python3`, que es el que los diez
# cuadernos de este repositorio declaran en su `metadata.kernelspec.name`. Sin
# él las tres primeras importan y el kernel no arranca igual.
#
# **Y nada más, que es la otra mitad de la decisión.** No va `torch`, ni
# `optuna`, ni nada del stack científico: eso lo satisface la imagen ---medido y
# no supuesto, por los envíos que corrieron `run_search` importando ese mismo
# árbol--- y un `pip install torch` encima podría reemplazar la compilación CUDA
# de la imagen por una rueda de CPU. Declarar de más no es prudencia acá: es
# cambiarle el entorno al worker sin haber medido que hacía falta.
#
# **Sin versiones y sin `indexUrl`, a propósito.** Un instalador ya satisfecho
# es un no-op y no cuesta nada; uno con pin fuerza una subida o una bajada sobre
# el stack de Jupyter que la imagen ya trae. Y un espejo sería una decisión
# sobre el servicio, que no es conocimiento de este repositorio.
__environment__: dict = {
    "install": {
        "requirements": ["nbformat", "nbclient", "jupyter-client", "ipykernel"],
    },
}
