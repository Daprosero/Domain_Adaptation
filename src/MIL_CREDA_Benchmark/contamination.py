"""The noise axis: which levels have a record, and what changed between two of them.

Reading a campaign is `harness`'s job and rendering one is `tables`'s. What lives
here is the one question neither of them asks: *the same measurement at two
contamination rates*. It is a join, and joins are where a repository quietly grows
two answers to one question -- so it has a module rather than a paragraph in a
notebook, and the notebooks call it instead of each carrying its own copy.

Nothing here decides a level. `config` declares which levels exist, which one the
report shows and which the diagnostic uses, and every one of those was fixed
before anything ran. This only asks the filesystem what actually arrived.
"""

from __future__ import annotations

import json
from pathlib import Path

from MIL_CREDA_Benchmark import config
from MIL_CREDA_Benchmark import pooling


#: Which run the axis reads. The degradation sweep is ONE transfer across every
#: level; a campaign is every transfer at one level. Same rate, different shape,
#: so different trees -- see `config.results_for`.
KIND = "curve"


def level_dir(rate: float, kind: str = KIND, pilot: bool | None = None) -> Path:
    """Where the run of this kind at one rate writes.

    `pilot=None` significa *la que rige*: la corrida completa si existe, y el
    ensayo si no. Es el valor por omisión y no un caso especial, porque leer el
    árbol completo a secas es lo que hacía que un ensayo entero apareciera como
    «no hay datos» --- y «no corrió» y «corrió y no lo encontré» son la misma
    tabla vacía para quien la lee.
    """
    if pilot is not None:
        return config.results_for(rate, kind, pilot)
    for candidato in (False, True):
        raiz = config.results_for(rate, kind, candidato)
        if (raiz / "runs.jsonl").exists() and (raiz / "summary.json").exists():
            return raiz
    return config.results_for(rate, kind, False)


def load(rate: float, kind: str = KIND, pilot: bool | None = None) -> dict | None:
    """One level's runs and summary, or `None` when that level has not run.

    `None` and *an empty campaign* are different facts and must not collapse into
    one: a notebook that renders an empty table for a level nobody ran has
    reported a result. Every caller here is expected to say "this level has not
    run" in words rather than draw a blank row.
    """
    root = level_dir(rate, kind, pilot)
    runs_path, summary_path = root / "runs.jsonl", root / "summary.json"
    if not runs_path.exists() or not summary_path.exists():
        return None
    runs = [json.loads(line) for line in
            runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"rate": rate, "kind": kind, "runs": runs, "summary": summary,
            "root": root,
            # De cuál de los dos salió, para que quien lo muestre pueda decirlo.
            "pilot": root == config.results_for(rate, kind, True)}


def available() -> list[float]:
    """The declared levels that actually have a record on disk, in declared order."""
    return [rate for rate in config.NOISE_LEVELS if load(rate) is not None]


def missing() -> list[float]:
    """The declared levels that have not run. Reported, never silently skipped."""
    return [rate for rate in config.NOISE_LEVELS if load(rate) is None]


def stated_rate(level: dict) -> float | None:
    """What the level's own record says it ran at, read rather than assumed.

    A directory name is not evidence. The rate that governs a table is the one the
    campaign stamped into its own bounds, and if the two disagree the directory is
    the thing that is wrong -- so the disagreement is surfaced rather than
    resolved in favour of whichever is more convenient.
    """
    reduction = (level.get("summary") or {}).get("reduction") or {}
    return reduction.get("labelNoise")


def mismatched(level: dict) -> bool:
    """Whether a level's record contradicts the tree it was found in."""
    stated = stated_rate(level)
    return stated is not None and float(stated) != float(level["rate"])


def by_arm(runs, metric: str) -> dict[str, float]:
    """One number per arm: its mean over every transfer and repetition present.

    The axis this collapses is *transfer and repetition together*, and it is named
    here because that is what the degradation curve is a function of -- rate, and
    nothing else. A curve that also varied in which transfers it averaged would be
    reading two axes and reporting one.
    """
    pooling.refuse(metric)
    totals: dict[str, list[float]] = {}
    for run in runs:
        value = run.get(metric)
        if value is None:
            continue
        totals.setdefault(run["arm"], []).append(float(value))
    return {arm: sum(values) / len(values) for arm, values in totals.items() if values}


def curve(metric: str, levels: list[float] | None = None) -> dict:
    """Every arm's `metric` against the contamination rate.

    Returns the rates that actually ran, the arms present in all of them, and one
    series per arm. Arms are intersected rather than unioned on purpose: a series
    with a hole in it drawn beside a complete one differs in how many points it
    carries, and the eye reads density as coverage.
    """
    pooling.refuse(metric)
    wanted = config.NOISE_LEVELS if levels is None else levels
    loaded = [(rate, load(rate)) for rate in wanted]
    present = [(rate, level) for rate, level in loaded if level is not None]
    if not present:
        return {"metric": metric, "rates": [], "arms": [], "series": {},
                "missing": list(wanted), "dropped": []}

    per_rate = {rate: by_arm(level["runs"], metric) for rate, level in present}
    everywhere = set.intersection(*(set(values) for values in per_rate.values()))
    seen = set().union(*(set(values) for values in per_rate.values()))
    arms = [arm for arm in config.ARM_ORDER if arm in everywhere]
    rates = [rate for rate, _ in present]
    return {
        "metric": metric,
        "rates": rates,
        "arms": arms,
        "series": {arm: [per_rate[rate][arm] for rate in rates] for arm in arms},
        "missing": [rate for rate, level in loaded if level is None],
        # Named rather than dropped in silence: an arm that ran at one rate and
        # not at another is a gap in the campaign, and a curve that quietly
        # omitted it would look complete.
        "dropped": sorted(seen - everywhere),
    }


def degradation(metric: str, levels: list[float] | None = None) -> list[dict]:
    """How far each arm fell from the clean level to the worst one that ran.

    The clean level is the reference because it is the only rate at which nothing
    was corrupted; reading the fall against anything else would be measuring the
    distance between two contaminations. An arm the clean level never ran gets no
    row rather than a fall computed from nothing.
    """
    pooling.refuse(metric)
    drawn = curve(metric, levels)
    if not drawn["rates"] or drawn["rates"][0] != config.NOISE_LEVELS[0]:
        return []
    rows = []
    for arm in drawn["arms"]:
        series = drawn["series"][arm]
        rows.append({
            "arm": arm,
            "name": config.NAME_OF.get(arm, arm),
            "clean": series[0],
            "worst": series[-1],
            "fall": series[0] - series[-1],
            "series": list(series),
        })
    return rows


def in_force(rate: float = 0.0, kind: str = "campaign") -> dict | None:
    """El registro que rige para esta forma: la completa, y si no, el ensayo.

    Alias de `load` con la forma de campaña por omisión. Existía aparte cuando
    `load` leía sólo el árbol completo, y esa separación era el defecto: quien
    llamaba a `available()` o a `curve()` --- el cuaderno de ruido, entre otros
    --- no pasaba por acá y veía vacío un ensayo completo.
    """
    return load(rate, kind)


def source_note(level: dict | None, rate: float = 0.0) -> str:
    """De dónde salieron los números, en una línea, para encabezar lo que sigue.

    Se escribe siempre y no sólo cuando es un ensayo. Un aviso que aparece
    únicamente en el caso malo enseña a nadie qué es lo que vigila, y la primera
    vez que falta se lee como que no había nada que avisar.
    """
    if level is None:
        return (f"**Sin registro para ρ={rate:g}.** Ni corrida completa ni "
                f"ensayo: no hay nada que mostrar, que no es lo mismo que una "
                f"tabla vacía.")
    if level["pilot"]:
        reduction = level["summary"].get("reduction") or {}
        return (f"**Estos números son de un ENSAYO** (ρ={rate:g}, "
                f"{reduction.get('epochs')} épocas, "
                f"{len(reduction.get('seeds') or [])} repetición/es), porque no "
                f"hay corrida completa para esta forma. No se citan como "
                f"resultados: ni en el informe, ni en el resumen, ni en "
                f"conversación.")
    return f"Corrida completa, ρ={rate:g}."
