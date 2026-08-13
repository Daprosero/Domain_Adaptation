"""Niveles, método por método y transferencia por transferencia.

`verdict.py` responde *quién gana cada peldaño*. Esto responde la otra pregunta,
la que un lector se hace primero — *dónde queda cada método* — y es un módulo
aparte a propósito: una tabla de niveles no puede atribuir nada, y una tabla de
peldaños no puede decir qué tan bueno es nadie. Se imprimen las dos, una después
de la otra.

Solo biblioteca estándar, como `verdict.py`: producir los números necesita torch,
ordenarlos no.

El texto visible está en español porque el informe se lee en español. Las claves
de los registros, los identificadores y los nombres de los métodos siguen en
inglés: son contrato de datos, no prosa.

Tres cosas que este módulo se niega a hacer, cada una por una razón que vale la
pena decir una vez:

* Imprime un decimal. Con `EVAL_BAGS` bolsas de evaluación la exactitud se mueve
  de a `100 / EVAL_BAGS` puntos, así que un segundo decimal describiría una
  precisión que la medición no tiene.
* Su `±` es la dispersión **entre semillas**, que es la que consume la regla de
  veredicto. El `±` del artículo de referencia es la dispersión entre lotes de
  evaluación dentro de una corrida: un número que se ve razonable por pocas
  repeticiones que se hayan corrido, que es exactamente la propiedad equivocada
  para una tabla que se lee al lado de un piloto.
* Nunca esconde el sello de piloto. Por debajo del piso de repeticiones
  declarado lo dice en el encabezado e imprime la tabla igual, porque el piloto
  tiene que ser el mismo programa que la campaña.
"""

from __future__ import annotations

import math
from typing import Iterable

from MIL_CREDA_Benchmark import config

#: Métricas que se informan como porcentaje en lugar de como fracción.
PERCENT = ("targetAccuracy", "sourceAccuracy")

#: Cómo se llama cada dimensión en el informe, y en qué unidad.
SPANISH = {
    "targetAccuracy": ("exactitud en destino", "%"),
    "sourceAccuracy": ("exactitud en fuente", "%"),
    "seconds": ("tiempo de entrenamiento", "s"),
    "peakMiB": ("memoria pico", "MiB"),
    "parameters": ("parámetros", ""),
    "contribution": ("peso del término de adaptación", ""),
}

BETTER = {config.HIGHER: "más alto es mejor", config.LOWER: "más bajo es mejor",
          None: "descriptivo: se informa, no se disputa"}


def spread(values: list[float]) -> dict:
    """Media, dispersión y pico de una celda, sobre sus repeticiones."""
    n = len(values)
    if not n:
        return {"mean": float("nan"), "stdev": 0.0, "max": float("nan"), "n": 0}
    mean = sum(values) / n
    stdev = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"mean": mean, "stdev": stdev, "max": max(values), "n": n}


def cells(runs: Iterable[dict], metric: str) -> dict:
    """{(brazo, transferencia): spread} sobre cada repetición de esa celda."""
    gathered: dict[tuple[str, str], list[float]] = {}
    for run in runs:
        gathered.setdefault((run["arm"], run["transfer"]), []).append(float(run[metric]))
    return {key: spread(values) for key, values in gathered.items()}


def table(runs: Iterable[dict], metric: str) -> list[dict]:
    """Una fila por método: cada transferencia, el promedio y el pico.

    `avg` promedia las medias por transferencia, así una transferencia cuenta una
    vez por muchas repeticiones que haya corrido. `max` promedia los máximos por
    transferencia — el método en su semilla más afortunada de cada una — y existe
    para que el pico quede documentado como número acá, en lugar de colarse en
    una figura eligiendo el mejor modelo.
    """
    runs = list(runs)
    grid = cells(runs, metric)
    shares = cells(runs, "contribution")
    labels = [f"{s}->{t}" for s, t in config.TRANSFERS]

    rows = []
    for arm in config.ARM_ORDER:
        present = [grid[(arm, label)] for label in labels if (arm, label) in grid]
        if not present:
            continue
        rows.append({
            "arm": arm,
            "name": config.NAME_OF[arm],
            "byTransfer": {label: grid.get((arm, label)) for label in labels},
            "avg": sum(c["mean"] for c in present) / len(present),
            "max": sum(c["max"] for c in present) / len(present),
            "share": (sum(shares[(arm, label)]["mean"] for label in labels
                          if (arm, label) in shares)
                      / max(1, sum(1 for label in labels if (arm, label) in shares))),
        })
    return rows


def _scaled(value: float, metric: str) -> float:
    return value * 100.0 if metric in PERCENT else value


def _stamp(reduction: dict) -> list[str]:
    """El sello que separa un piloto de un resultado, en todas las tablas."""
    seeds = reduction.get("seeds", config.SEEDS)
    n_seeds = len(seeds) if isinstance(seeds, (list, tuple)) else int(seeds or 0)
    notes = []
    if n_seeds < 3:
        notes.append(f"!! {n_seeds} repetición(es): el ± de abajo es cero por "
                     f"construcción, no por acuerdo. Son estimaciones puntuales, "
                     f"no resultados.")
    if n_seeds < len(config.FULL_SEEDS):
        notes.append(f"!! piloto: el protocolo declara {len(config.FULL_SEEDS)} "
                     f"repeticiones y {config.FULL_EPOCHS} épocas. Nada de esto es "
                     f"un resultado.")
    return notes


def render(runs: Iterable[dict], metric: str, reduction: dict,
           markdown: bool = False, extras: tuple[str, ...] = ("max", "share")) -> str:
    """La tabla, con los límites que la produjeron en el encabezado."""
    rows = table(runs, metric)
    labels = [f"{s}->{t}" for s, t in config.TRANSFERS]
    title, unit = SPANISH.get(metric, (metric, ""))
    seeds = reduction.get("seeds", config.SEEDS)
    n_seeds = len(seeds) if isinstance(seeds, (list, tuple)) else int(seeds or 0)

    notes = [f"{title} ({unit})  ·  {reduction.get('backbone', config.BACKBONE)}  ·  "
             f"{reduction.get('epochs', config.EPOCHS)} épocas  ·  "
             f"{n_seeds} semilla(s)  ·  {reduction.get('revision', config.REVISION)}",
             *_stamp(reduction)]
    if metric in PERCENT:
        notes.append(f"la exactitud se mueve de a {100 / config.EVAL_BAGS:.2f} puntos "
                     f"sobre {config.EVAL_BAGS} bolsas de evaluación")

    decimals = 1 if metric in PERCENT else 2

    def cell(entry) -> str:
        if entry is None:
            return "—"
        return (f"{_scaled(entry['mean'], metric):.{decimals}f} ± "
                f"{_scaled(entry['stdev'], metric):.{decimals}f}")

    columns = ["Método"] + labels + ["Prom."] + [
        {"max": "máx", "share": "peso"}[e] for e in extras]

    def summary_cells(row) -> list[str]:
        out = [f"{_scaled(row['avg'], metric):.{decimals}f}"]
        for extra in extras:
            out.append(f"{_scaled(row['max'], metric):.{decimals}f}" if extra == "max"
                       else f"{row['share']:.3f}")
        return out

    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            values = [cell(row["byTransfer"][label]) for label in labels]
            head, *rest = summary_cells(row)
            lines.append("| " + " | ".join(
                [f"`{row['name']}`", *values, f"**{head}**", *rest]) + " |")
        return "\n".join(notes) + "\n\n" + "\n".join(lines)

    width = max(14, max((len(r["name"]) for r in rows), default=14) + 2)
    lines = [f"{'Método':<{width}}" + "".join(f"{label:>16}" for label in labels)
             + f"{'Prom.':>9}" + "".join(f"{c:>8}" for c in columns[-len(extras):]
                                         if extras)]
    for row in rows:
        values = "".join(f"{cell(row['byTransfer'][label]):>16}" for label in labels)
        head, *rest = summary_cells(row)
        lines.append(f"{row['name']:<{width}}{values}{head:>9}"
                     + "".join(f"{value:>8}" for value in rest))
    return "\n".join(notes) + "\n\n" + "\n".join(lines)


def conclusion(runs: Iterable[dict], metric: str, reduction: dict) -> str:
    """La lectura más relevante de la tabla, calculada y no escrita a mano.

    Una conclusión escrita a mano es una segunda fuente de verdad: se desactualiza
    en silencio y se le cree igual. Esta se recalcula con la tabla, así que no
    puede alejarse de lo que describe.
    """
    rows = table(runs, metric)
    if not rows:
        return "Sin corridas: no hay nada que concluir."
    better = config.DIMENSIONS.get(metric, config.HIGHER)
    reverse = better == config.HIGHER
    ordered = sorted(rows, key=lambda r: r["avg"], reverse=reverse)
    best, worst = ordered[0], ordered[-1]
    title, unit = SPANISH.get(metric, (metric, ""))

    def show(value: float) -> str:
        return (f"{_scaled(value, metric):.1f}{unit}" if metric in PERCENT
                else f"{value:.2f}{unit}")

    by_name = {row["name"]: row for row in rows}
    lines = [f"Mejor promedio: **{best['name']}** con {show(best['avg'])}; "
             f"peor: {worst['name']} con {show(worst['avg'])}."]

    # Cada método completo contra su propio piso, que es la única lectura que
    # separa lo que aporta la adaptación de lo que aporta la representación.
    for arm, floor in (("D", "A"), ("G", "B")):
        pair = (config.NAME_OF[arm], config.NAME_OF[floor])
        if pair[0] in by_name and pair[1] in by_name:
            delta = by_name[pair[0]]["avg"] - by_name[pair[1]]["avg"]
            direction = "por encima de" if (delta > 0) == reverse else "por debajo de"
            lines.append(f"{pair[0]} queda {show(abs(delta))} {direction} su piso "
                         f"{pair[1]}.")

    if _stamp(reduction):
        lines.append("Con esta cantidad de repeticiones nada de lo anterior es un "
                     "resultado: son estimaciones puntuales y la dispersión es cero "
                     "por construcción.")
    return " ".join(lines)


# ----------------------------------------------------------------- los peldaños

def rung_name(left: str, right: str) -> str:
    """`Baseline → MIL-Baseline`, no `A->B`. Un identificador no es un nombre."""
    return f"{config.NAME_OF.get(left, left)} → {config.NAME_OF.get(right, right)}"


def render_rungs(summary: dict, metric: str, markdown: bool = False) -> str:
    """Un peldaño por fila, una transferencia por columna: la diferencia medida.

    Es la misma forma que la tabla de niveles, con el mismo orden de lectura. La
    diferencia se informa con el signo hacia el brazo de la derecha, así que un
    valor positivo significa que el de la derecha quedó por encima.
    """
    grid = summary["grid"]
    labels = [t for t in [f"{s}->{d}" for s, d in config.TRANSFERS] if t in grid]
    title, unit = SPANISH.get(metric, (metric, ""))
    notes = [f"peldaños · {title} ({unit}) · diferencia con signo hacia la derecha",
             *_stamp(summary["reduction"])]

    rows = []
    for left, right, reading in config.LADDER:
        values = {}
        for label in labels:
            cell = grid[label]
            if left in cell and right in cell:
                values[label] = (cell[right][metric]["mean"] - cell[left][metric]["mean"])
        if not values:
            continue
        rows.append({"name": rung_name(left, right), "reading": reading,
                     "values": values,
                     "avg": sum(values.values()) / len(values),
                     "leans": sum(1 for v in values.values() if v > 0)})
    if not rows:
        return "\n".join(notes) + "\n\n(sin peldaños medibles)"

    def show(value) -> str:
        return "—" if value is None else f"{_scaled(value, metric):+.1f}"

    columns = ["Peldaño"] + labels + ["Prom.", "a favor"]
    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(
                [row["name"], *(show(row["values"].get(l)) for l in labels),
                 f"**{show(row['avg'])}**", f"{row['leans']}/{len(row['values'])}"]) + " |")
        return "\n".join(notes) + "\n\n" + "\n".join(lines)

    width = max(len(r["name"]) for r in rows) + 2
    lines = [f"{'Peldaño':<{width}}" + "".join(f"{l:>10}" for l in labels)
             + f"{'Prom.':>10}{'a favor':>10}"]
    for row in rows:
        lines.append(f"{row['name']:<{width}}"
                     + "".join(f"{show(row['values'].get(l)):>10}" for l in labels)
                     + f"{show(row['avg']):>10}"
                     + f"{row['leans']}/{len(row['values']):<8}")
        lines.append(f"{'':<{width}}    {row['reading']}")
    return "\n".join(notes) + "\n\n" + "\n".join(lines)


def conclusion_rungs(summary: dict, metric: str) -> str:
    """Qué peldaño se movió más y cuál coincidió en todas las transferencias."""
    grid = summary["grid"]
    labels = [t for t in [f"{s}->{d}" for s, d in config.TRANSFERS] if t in grid]
    readings = []
    for left, right, reading in config.LADDER:
        values = [grid[l][right][metric]["mean"] - grid[l][left][metric]["mean"]
                  for l in labels if left in grid[l] and right in grid[l]]
        if values:
            readings.append({"name": rung_name(left, right), "reading": reading,
                             "avg": sum(values) / len(values),
                             "leans": sum(1 for v in values if v > 0), "n": len(values)})
    if not readings:
        return "Sin peldaños medibles."

    strongest = max(readings, key=lambda r: abs(r["avg"]))
    unanimous = [r for r in readings if r["leans"] in (0, r["n"])]
    lines = [f"El peldaño que más se movió es **{strongest['name']}** "
             f"({_scaled(strongest['avg'], metric):+.1f}), que lee "
             f"{strongest['reading']}."]
    if unanimous:
        lines.append("Coinciden en las " + str(readings[0]["n"]) + " transferencias: "
                     + ", ".join(f"{r['name']} ({_scaled(r['avg'], metric):+.1f})"
                                 for r in unanimous) + ".")
    else:
        lines.append("Ningún peldaño se inclina igual en todas las transferencias, "
                     "que es lo que haría falta para leer algo de un solo sentido.")
    if _stamp(summary["reduction"]):
        lines.append("Con una sola repetición la coincidencia entre transferencias es "
                     "lo único que carga peso, y no reemplaza a las repeticiones.")
    return " ".join(lines)


# --------------------------------------------------------------- fase dos

def _reach(record: dict, path: str):
    """`geometry.ratio` dentro de una lectura anidada, o nada si el brazo no la tiene."""
    value = record
    for step in path.split("."):
        if not isinstance(value, dict) or step not in value:
            return None
        value = value[step]
    return float(value) if isinstance(value, (int, float)) else None


def render_readings(readings: Iterable[dict], path: str, title: str,
                    markdown: bool = False) -> str:
    """Una medición de fase dos en la misma forma que las tablas de fase uno.

    El `±` de acá **no** es el de las tablas de fase uno: es la dispersión entre
    los checkpoints guardados de esa celda. La fase uno mide la variabilidad de la
    exactitud; esto mide la variabilidad de la geometría, y nada establece que una
    siga a la otra.

    Un brazo sin lectura para esta cantidad — uno sin término local no tiene
    correspondencia — recibe una celda vacía, no un cero.
    """
    readings = list(readings)
    labels = [f"{s}->{t}" for s, t in config.TRANSFERS]
    gathered: dict[tuple[str, str], list[float]] = {}
    for reading in readings:
        value = _reach(reading, path)
        if value is not None:
            gathered.setdefault((reading["arm"], reading["transfer"]), []).append(value)

    rows = []
    for arm in config.ARM_ORDER:
        by_label = {label: (spread(gathered[(arm, label)]) if (arm, label) in gathered
                            else None) for label in labels}
        present = [c for c in by_label.values() if c]
        if not present:
            continue
        rows.append({"name": config.NAME_OF[arm], "cells": by_label,
                     "avg": sum(c["mean"] for c in present) / len(present),
                     "n": max(c["n"] for c in present)})

    notes = [f"{title}  ·  {max((r['n'] for r in rows), default=0)} checkpoint(s) por "
             f"celda  ·  {config.REVISION}",
             "el ± es la dispersión entre checkpoints guardados, no entre semillas: "
             "esto mide geometría y la fase uno mide exactitud",
             *_stamp({"seeds": config.SEEDS})]

    def cell(entry) -> str:
        return "—" if entry is None else f"{entry['mean']:.3f} ± {entry['stdev']:.3f}"

    columns = ["Método"] + labels + ["Prom."]
    if markdown:
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(
                [f"`{row['name']}`", *(cell(row["cells"][l]) for l in labels),
                 f"**{row['avg']:.3f}**"]) + " |")
        return "\n".join(notes) + "\n\n" + "\n".join(lines)

    width = max(14, max((len(r["name"]) for r in rows), default=14) + 2)
    lines = [f"{'Método':<{width}}" + "".join(f"{l:>18}" for l in labels) + f"{'Prom.':>10}"]
    for row in rows:
        lines.append(f"{row['name']:<{width}}"
                     + "".join(f"{cell(row['cells'][l]):>18}" for l in labels)
                     + f"{row['avg']:>10.3f}")
    return "\n".join(notes) + "\n\n" + "\n".join(lines)


def best_transfers(runs: Iterable[dict], count: int | None = None,
                   metric: str = "targetAccuracy") -> list[str]:
    """Las transferencias donde los métodos llegan más alto, calculadas de la campaña.

    Es una elección hecha por el resultado, y por eso va declarada en cada pie de
    figura. Lo que la hace defendible en esta figura y no en otras: el espacio
    latente de una transferencia donde todos los métodos quedan cerca del azar es
    la foto de un modelo que no aprendió, y de ahí no se lee nada sobre alineación.

    Lo que esta elección nunca toca es **qué sorteo** se muestra. Eso sigue siendo
    la semilla de exhibición, elegida por una regla que no favorece a nadie:
    elegir el sorteo por el resultado es como una figura deja de poder salir mal.
    """
    count = count or config.FIGURE_TRANSFER_COUNT
    grid = cells(runs, metric)
    by_transfer: dict[str, list[float]] = {}
    for (_, transfer), entry in grid.items():
        by_transfer.setdefault(transfer, []).append(entry["mean"])
    ordered = sorted(by_transfer, key=lambda t: sum(by_transfer[t]) / len(by_transfer[t]),
                     reverse=True)
    return ordered[:count]


def ranking(runs: Iterable[dict], metric: str = "targetAccuracy",
            exclude: Iterable[str] = ()) -> list[str]:
    """Los brazos por media de `metric`, el mejor primero."""
    excluded = set(exclude)
    return [row["arm"] for row in
            sorted(table(runs, metric), key=lambda r: r["avg"], reverse=True)
            if row["arm"] not in excluded]


# ------------------------------------------- lo que dice cada figura, calculado

def _paired(by_arm_transfer: dict, left: str, right: str) -> list[tuple[str, float]]:
    """La diferencia entre dos brazos **dentro de cada transferencia**.

    Promediar valores crudos entre transferencias mete la dificultad de cada una
    en la dispersión y ahoga el efecto: una transferencia donde todo el mundo saca
    0,27 y otra donde todo el mundo saca 0,80 no son comparables sumadas. La
    diferencia medida dentro de una misma transferencia cancela esa dificultad, y
    lo que carga peso es que las transferencias **coincidan**, no el promedio.
    """
    shared = sorted(set(by_arm_transfer.get(left, {})) & set(by_arm_transfer.get(right, {})))
    return [(transfer, by_arm_transfer[right][transfer] - by_arm_transfer[left][transfer])
            for transfer in shared]


def _agreement(differences: list[tuple[str, float]], tolerance: float) -> str:
    """Cómo se leen varias diferencias apareadas, sin promediarlas a la nada."""
    if not differences:
        return "sin transferencias en común"
    up = [t for t, d in differences if d > tolerance]
    down = [t for t, d in differences if d < -tolerance]
    flat = [t for t, d in differences if abs(d) <= tolerance]
    detail = ", ".join(f"{t} {d:+.3f}" for t, d in differences)
    if len(up) == len(differences):
        return f"a favor en las {len(differences)} transferencias ({detail})"
    if len(down) == len(differences):
        return f"en contra en las {len(differences)} transferencias ({detail})"
    if len(flat) == len(differences):
        return f"plano en todas, dentro de ±{tolerance:.3f} ({detail})"
    return (f"las transferencias no coinciden — {len(up)} a favor, {len(down)} en "
            f"contra, {len(flat)} planas ({detail}). Un promedio acá diría 'no hace "
            f"nada' y estaría tapando que una transferencia sí se movió")


def conclusion_geometry(readings: Iterable[dict]) -> str:
    """Qué dice la grilla, sacado de las mediciones y no de mirarla.

    Una línea por método contra su propio piso, y las diferencias son apareadas
    dentro de cada transferencia: una transferencia donde todos sacan 0,27 y otra
    donde todos sacan 0,80 no son comparables sumadas.

    Alineación y colapso se separan acá y no en la razón sola. La razón es
    `cruzada / entre clases`, así que baja tanto si los dominios se juntaron como
    si el espacio entero se encogió. Se llama colapso cuando la distancia entre
    clases cayó fuerte **y** la razón no mejoró: ahí lo que se juntó fue todo.
    """
    readings = list(readings)
    ratio: dict[str, dict[str, float]] = {}
    apart: dict[str, dict[str, float]] = {}
    for reading in readings:
        geometry = reading.get("geometry", {})
        if "ratio" in geometry:
            ratio.setdefault(reading["arm"], {})[reading["transfer"]] = float(geometry["ratio"])
        if "betweenClasses" in geometry:
            apart.setdefault(reading["arm"], {})[reading["transfer"]] = float(
                geometry["betweenClasses"])
    if not ratio:
        return "Sin lecturas de geometría: no hay nada que concluir."

    lines = ["Cada método contra su propio piso, transferencia por transferencia. "
             "«Alinea» = la razón bajó; «colapsa» = la razón no bajó y la distancia "
             "entre clases cayó más del 10%.",
             "",
             f"{'Método':<14}{'piso':<14}{'alinea':>8}{'plano':>7}{'empeora':>9}"
             f"{'colapsa':>9}"]

    for arm, floor in config.FLOOR_OF.items():
        if arm not in ratio or floor not in ratio:
            continue
        aligns = flat = worse = collapses = 0
        for transfer, change in _paired(ratio, floor, arm):
            shrink = apart.get(arm, {}).get(transfer, 0.0)
            base = apart.get(floor, {}).get(transfer, 0.0)
            shrank = bool(base) and shrink < base * 0.9
            if change < -0.02:
                aligns += 1
            elif change > 0.02:
                worse += 1
                collapses += shrank
            else:
                flat += 1
                collapses += shrank
        lines.append(f"{config.NAME_OF[arm]:<14}{config.NAME_OF[floor]:<14}"
                     f"{aligns:>8}{flat:>7}{worse:>9}{collapses:>9}")

    lines.append("")
    lines.append("Lo que carga peso no es el promedio sino que las transferencias "
                 "coincidan: un método que alinea en una y empeora en otra no está "
                 "diciendo nada todavía.")
    if len(config.SEEDS) < len(config.FULL_SEEDS):
        lines.append("Piloto: son estimaciones puntuales, no resultados.")
    return "\n".join(lines)


def conclusion_correspondence(scored: Iterable[dict]) -> str:
    """Qué dice la figura de bolsas, sacado de sus propios aciertos y masas."""
    scored = list(scored)
    if not scored:
        return "Sin paneles medidos: no hay nada que concluir."
    mass: dict[str, dict[str, float]] = {}
    hits: dict[str, dict[str, str]] = {}
    for row in scored:
        mass.setdefault(row["arm"], {})[row["transfer"]] = float(row["mass"])
        hits.setdefault(row["arm"], {})[row["transfer"]] = f"{row['hits']}/{row['classes']}"

    chance = 1.0 / config.CLASSES
    lines = [f"Aciertos por transferencia (azar {chance:.3f}): " + " · ".join(
        f"{config.NAME_OF[arm]} " + ", ".join(f"{t} {hits[arm][t]}" for t in hits[arm])
        for arm in config.BAG_PANELS if arm in hits) + "."]

    # El peldaño: el término local solo se ve si el completo separa del mismo
    # método sin él, y separa *dentro* de cada transferencia.
    with_local = next((a for a in config.BAG_PANELS
                       if a in mass and config.ARMS_BY_ID[a]["local"]), None)
    without = next((a for a in config.BAG_PANELS
                    if a in mass and config.ARMS_BY_ID[a]["adaptation"]
                    and not config.ARMS_BY_ID[a]["local"]), None)
    if with_local and without:
        lines.append(f"El término local, aislado como {config.NAME_OF[with_local]} "
                     f"contra {config.NAME_OF[without]}: "
                     f"{_agreement(_paired(mass, without, with_local), 0.02)}.")
    if len(config.SEEDS) < len(config.FULL_SEEDS):
        lines.append("Piloto: son estimaciones puntuales, no resultados.")
    return " ".join(lines)


def conclusions(record: dict) -> dict:
    """Cada conclusión del informe, a partir de un registro y de nada más.

    Un solo punto de entrada y no una lista de funciones, porque una verificación
    que tiene que adivinar firmas termina informando «no se pudo ejercitar» y eso
    se lee como un aprobado. Acá el destino cablea sus propias conclusiones detrás
    de una llamada, y quien verifica solo tiene que invocarla dos veces: una con el
    registro y otra con sus números permutados. Si el texto sale igual, la
    conclusión no está atada a nada.

    Devuelve solo las que el registro puede alimentar: pedir una conclusión sobre
    algo que no se midió es distinto de una que no cambia, y confundirlas haría que
    una fase todavía no corrida se informe como un defecto.
    """
    produced: dict[str, str] = {}
    runs = record.get("runs")
    reduction = record.get("reduction") or {}
    if isinstance(runs, list) and runs:
        for metric in ("seconds", "sourceAccuracy", "targetAccuracy"):
            produced[f"niveles:{metric}"] = conclusion(runs, metric, reduction)
    readings = record.get("readings")
    if isinstance(readings, list) and readings:
        produced["geometría"] = conclusion_geometry(readings)
    scored = record.get("correspondence")
    if isinstance(scored, list) and scored:
        produced["correspondencia"] = conclusion_correspondence(scored)
    return produced
