"""El esquema del registro de techos, y la regla que elige adentro de una meseta.

Un solo lugar donde se define qué forma tiene una entrada del registro y cómo se
lee, porque el registro lo escribe la búsqueda y lo leen seis consumidores — los
cuatro renderizadores del informe y los dos resolutores de `config`. Cuando la
forma vivía repartida entre el escritor y cada lector, cambiar el motor de
búsqueda significaba que el informe siguiera imprimiendo «elegido por desempate
entre 3 techos empatados» sobre una búsqueda que no empata.

**Dos formas, y las dos se leen.** La rejilla (`grid`) es la que ya está en disco
y la que gobierna la campaña vigente; borrarle el lector sería tirar el registro
de una campaña de 1800 corridas para estrenar un motor. La búsqueda por trials
(`optuna`) es la nueva. `kind_of` las distingue por lo que la entrada trae, no
por una versión declarada: un número de versión es un dato más que puede mentir,
y la presencia de `grid` o de `search` es la cosa misma.

**Lo que se conserva idéntico** es todo lo que un consumidor ya lee: `ceiling`,
`byTransfer` como `etiqueta -> float`, `epochs`, `role`, `criterion`,
`atRequiredScale` y `requiredScale`. Cambiar `byTransfer` a un diccionario
anidado habría sido más prolijo y habría roto `config.ceilings_by_transfer_on_record`
y `harness.ceiling_for` a la vez. El detalle por transferencia va en `perTransfer`,
que es aditivo: un lector viejo no lo mira y sigue funcionando.
"""

from __future__ import annotations

KIND_GRID = "grid"
KIND_OPTUNA = "optuna"

#: Qué dice el registro sobre cómo se eligió adentro de una zona plana. Se
#: guarda como texto en la entrada, igual que `tieRule` en la forma vieja,
#: porque una regla que solo vive en el código es una regla que el lector del
#: informe no puede discutir.
FLAT_RULE = (
    "dentro de la meseta gana el techo más chico: el mismo resultado con menos "
    "adaptación es la afirmación más débil, y una búsqueda no debería darle a un "
    "término más peso del que la medición pidió. La meseta la define la resolución "
    "del criterio sobre el rol de búsqueda — una bolsa de las que tiene — y no una "
    "cantidad que el GP ajustó: atribuírsela al modelo haría que el ancho de la "
    "meseta dependiera de qué tan bien ajustó, que es la propiedad equivocada. "
    "Tampoco es una igualdad exacta: sobre un rango continuo dos evaluaciones nunca "
    "empatan y una regla de empate exacto no se activaría jamás"
)


def kind_of(entry: dict) -> str:
    """Qué forma tiene esta entrada, leída de lo que trae.

    `search` la escribe solo el motor por trials; `grid` solo el de rejilla. Una
    entrada que traiga las dos es ambigua y se declara así en vez de resolverse
    por precedencia: dos motores no escriben la misma entrada, así que verlo
    significa que algo la fusionó.
    """
    tiene_trials = isinstance(entry.get("search"), dict)
    tiene_grid = "grid" in entry
    if tiene_trials and tiene_grid:
        raise ValueError(
            "entrada de techo ambigua: trae `search` y `grid` a la vez. Dos "
            "motores no escriben la misma entrada; esto es una fusión."
        )
    if tiene_trials:
        return KIND_OPTUNA
    if tiene_grid:
        return KIND_GRID
    raise ValueError(
        "entrada de techo sin forma reconocible: no trae ni `search` ni `grid`."
    )


def plateau(trials: list[dict], noise: float) -> list[dict]:
    """Los trials que la resolución del criterio no distingue del mejor.

    `trials` son `{"ceiling": float, "value": float}` y el criterio se maximiza.
    Con `noise` cero la meseta es el máximo exacto, que es el caso degenerado y
    se deja pasar: un objetivo verdaderamente determinista no tiene por qué
    inventarse una banda.
    """
    if not trials:
        return []
    mejor = max(t["value"] for t in trials)
    return [t for t in trials if t["value"] >= mejor - max(0.0, noise)]


def choose(trials: list[dict], noise: float) -> dict:
    """El techo elegido y por qué, aplicando `FLAT_RULE`.

    Devuelve el detalle y no solo el número, porque «cuál ganó» y «la medición
    pudo distinguir algo» son dos hechos distintos y el segundo es el que dice cuánto
    sostiene el primero. Un techo elegido dentro de una meseta ancha sostiene
    mucho menos de lo que un número elegido aparenta, y ese fue exactamente el
    caso de `creda` con la rejilla.
    """
    if not trials:
        raise ValueError("no hay trials: no hay techo que elegir")
    banda = plateau(trials, noise)
    ganador = min(banda, key=lambda t: t["ceiling"])
    return {
        "ceiling": ganador["ceiling"],
        "value": ganador["value"],
        "best": max(t["value"] for t in trials),
        "trials": len(trials),
        "noise": noise,
        "plateau": sorted(t["ceiling"] for t in banda),
        # Que la meseta tenga más de uno es lo que dice que la búsqueda no
        # distinguió: ahí el ganador lo puso la regla y no el criterio.
        "decidedByFlatRule": len(banda) > 1,
    }


def scale_of(entry: dict) -> dict:
    """La escala a la que corrió esta entrada, en los ejes que su forma tiene.

    Una rejilla se mide en semillas y una búsqueda por trials en trials. Devolver
    un eje que la forma no tiene lo haría indistinguible de uno que vale cero.
    """
    forma = kind_of(entry)
    if forma == KIND_OPTUNA:
        return {"epochs": entry.get("epochs"),
                "trials": (entry.get("search") or {}).get("trials")}
    return {"epochs": entry.get("epochs"), "seeds": len(entry.get("seeds") or [])}


def choice_of(entry: dict) -> dict:
    """Cómo se eligió este techo, en hechos y no en prosa.

    El informe redacta; esta función dice qué pasó. Separadas porque son dos
    cosas que cambian por motivos distintos: la forma del registro cambia cuando
    cambia el motor, y la redacción cuando cambia el idioma o el lector.

    `byRule` es el hecho que más pesa y significa lo mismo en las dos formas: el
    ganador lo puso la regla y no el criterio, así que sostiene menos de lo que
    un número elegido aparenta. `amongst` dice cuántos no se distinguían.

    `axis` no se normaliza a un nombre común. Una rejilla repite semillas y una
    búsqueda por trials no repite nada: llamarlas igual haría que un lector
    comparara tres repeticiones con treinta evaluaciones de puntos distintos.
    """
    forma = kind_of(entry)
    if forma == KIND_OPTUNA:
        busqueda = entry.get("search") or {}
        meseta = entry.get("plateau") or []
        return {"kind": forma,
                "byRule": bool(entry.get("decidedByFlatRule")),
                "amongst": len(meseta),
                "axis": "trials", "count": busqueda.get("trials"),
                "agreement": None, "noise": entry.get("noise"),
                "rule": FLAT_RULE}
    return {"kind": forma,
            "byRule": bool(entry.get("decidedByTieBreak")),
            "amongst": len(entry.get("tied") or []),
            "axis": "seeds", "count": len(entry.get("seeds") or []),
            "agreement": entry.get("seedsAgree"), "noise": None,
            "rule": entry.get("tieRule")}
