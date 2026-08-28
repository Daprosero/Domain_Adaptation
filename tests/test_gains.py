"""La tabla de ganancias: cada método menos su propio piso.

Tres cosas que ninguna de las otras tablas puede decir, y una de ellas es la que
nos costó tres intercambios encontrar: que un promedio de `+0.56` puede ser el
promedio de `+15.1` y `-5.5`, y que leerlo solo produce una creencia falsa.
"""

from __future__ import annotations

import pytest

from MIL_CREDA_Benchmark import config, tables


def _run(arm, transfer, seed, target, source=0.5):
    return {"arm": arm, "transfer": transfer, "seed": seed,
            "targetAccuracy": target, "sourceAccuracy": source}


def _pair(arm, floor, transfer, deltas, base=0.50):
    """Un piso constante y el brazo desplazado, semilla por semilla."""
    out = []
    for seed, d in enumerate(deltas):
        out.append(_run(floor, transfer, seed, base))
        out.append(_run(arm, transfer, seed, base + d))
    return out


LABELS = [f"{a}->{b}" for a, b in config.VERDICT_TRANSFERS]


def test_the_difference_is_paired_within_a_transfer():
    """Una transferencia difícil y una fácil no se suman crudas.

    El piso vale 0.20 en una y 0.80 en la otra, y el brazo suma exactamente 0.05
    en las dos. Sin aparear, la media de los valores crudos mezcla la dificultad;
    apareado tiene que dar 5.00 puntos limpios en ambas.
    """
    runs = (_pair("D", "A", LABELS[0], [0.05] * 4, base=0.20)
            + _pair("D", "A", LABELS[1], [0.05] * 4, base=0.80))
    row = tables.paired_gains(runs, "targetAccuracy")[0]
    assert row["cells"][LABELS[0]]["mean"] == pytest.approx(5.0)
    assert row["cells"][LABELS[1]]["mean"] == pytest.approx(5.0)


def test_the_error_is_between_transfers_not_over_the_pooled_pairs():
    """La transferencia es un escenario, no una repetición.

    Dos transferencias sin ninguna dispersión interna —cada semilla da lo mismo—
    pero muy separadas entre sí: +15 y -5. Agrupar los pares daría error cero y
    declararía significativo un promedio que no resuelve nada. Entre
    transferencias el error es la mitad de la distancia que las separa.
    """
    runs = (_pair("D", "A", LABELS[0], [0.15] * 5)
            + _pair("D", "A", LABELS[1], [-0.05] * 5))
    row = tables.paired_gains(runs, "targetAccuracy")[0]
    assert row["mean"] == pytest.approx(5.0)
    assert row["error"] == pytest.approx(10.0)
    assert row["error"] > 0, "agrupar los pares daria cero acá"


def test_the_two_averages_can_disagree_in_sign():
    """El % divide por el piso de cada transferencia, y los pisos difieren.

    Piso 20 con -4 puntos y piso 80 con +5: la media de puntos es `+0.50` y la de
    porcentajes `-6.87`. Los dos son correctos y contestan preguntas distintas,
    que es exactamente por qué se reportan los dos.
    """
    runs = (_pair("D", "A", LABELS[0], [-0.04] * 4, base=0.20)
            + _pair("D", "A", LABELS[1], [0.05] * 4, base=0.80))
    row = tables.paired_gains(runs, "targetAccuracy")[0]
    assert row["mean"] > 0
    assert row["pct"] < 0
    assert row["mean"] == pytest.approx(0.5)
    assert row["pct"] == pytest.approx((-20.0 + 6.25) / 2)


def test_the_agreement_counts_each_transfer_beyond_its_own_noise():
    runs = (_pair("D", "A", LABELS[0], [0.15] * 5)
            + _pair("D", "A", LABELS[1], [-0.05] * 5)
            + _pair("D", "A", LABELS[2], [0.0] * 5))
    row = tables.paired_gains(runs, "targetAccuracy")[0]
    assert row["agreement"] == {"gana": 1, "pierde": 1, "empata": 1}


def test_the_span_reports_the_two_extremes():
    """Lo que frena a un lector que solo mira el promedio."""
    runs = (_pair("D", "A", LABELS[0], [0.15] * 5)
            + _pair("D", "A", LABELS[1], [-0.05] * 5))
    row = tables.paired_gains(runs, "targetAccuracy")[0]
    assert row["span"] == pytest.approx((15.0, -5.0))


def test_the_rendered_row_carries_all_four_summaries():
    """Ninguno de los cuatro puede faltar: cada uno tapa una lectura falsa."""
    runs = (_pair("D", "A", LABELS[0], [0.15] * 5)
            + _pair("D", "A", LABELS[1], [-0.05] * 5))
    text = tables.render_gains(runs, "targetAccuracy", "prueba", markdown=True)
    assert "Media (pts)" in text and "% medio" in text
    assert "Rango" in text and "de +15.0 a -5.0" in text
    assert "gana en 1 de 2, pierde en 1, empata en 0" in text


def test_a_floorless_arm_has_no_row():
    """`Baseline` y `MIL-Baseline` no son la ganancia de nadie."""
    runs = _pair("D", "A", LABELS[0], [0.05] * 4)
    arms = {r["arm"] for r in tables.paired_gains(runs, "targetAccuracy")}
    assert "A" not in arms and "B" not in arms


def test_it_reads_the_dimension_it_is_given():
    """Source y target son la misma tabla con otro instrumento."""
    runs = []
    for seed in range(4):
        runs.append({"arm": "A", "transfer": LABELS[0], "seed": seed,
                     "targetAccuracy": 0.5, "sourceAccuracy": 0.9})
        runs.append({"arm": "D", "transfer": LABELS[0], "seed": seed,
                     "targetAccuracy": 0.6, "sourceAccuracy": 0.8})
    tgt = tables.paired_gains(runs, "targetAccuracy")[0]
    src = tables.paired_gains(runs, "sourceAccuracy")[0]
    assert tgt["cells"][LABELS[0]]["mean"] == pytest.approx(10.0)
    assert src["cells"][LABELS[0]]["mean"] == pytest.approx(-10.0)


def test_the_mean_is_bold_by_the_same_rule_the_legend_states():
    """La leyenda dice que la negrita es superar dos errores estándar.

    Ponerla siempre en la columna `Media` hace que esa leyenda mienta sobre su
    propia tabla: el lector la aplica y lee como significativo un promedio que
    está adentro del ruido.
    """
    ruido = (_pair("D", "A", LABELS[0], [0.10] * 5)
             + _pair("D", "A", LABELS[1], [-0.10] * 5))
    texto = tables.render_gains(ruido, "targetAccuracy", "t", markdown=True)
    fila = [l for l in texto.splitlines() if "CREDA" in l][0]
    assert "**+0.00" not in fila and "**-0.00" not in fila

    claro = (_pair("D", "A", LABELS[0], [0.10] * 5)
             + _pair("D", "A", LABELS[1], [0.10] * 5))
    fila = [l for l in tables.render_gains(claro, "targetAccuracy", "t", markdown=True
                                           ).splitlines() if "CREDA" in l][0]
    assert "**+10.00 ± 0.00**" in fila
