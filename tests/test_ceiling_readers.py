"""Los lectores del informe, contra las dos formas del registro.

La forma vieja gobierna una campaña de 1800 corridas y se sigue leyendo; la
nueva es la búsqueda por trials. Lo que estos tests impiden es lo que motivó
separar el esquema: que un registro de trials se renderice con vocabulario de
rejilla — «elegido por desempate entre 3 techos empatados» sobre una búsqueda
que, sobre un rango continuo, no empata nunca.
"""

from __future__ import annotations

from MIL_CREDA_Benchmark import ceiling_record as cr
from MIL_CREDA_Benchmark import tables

REJILLA = {"creda": {
    "arm": "D", "ceiling": 0.0001, "criterion": "targetAccuracy", "role": "valid",
    "epochs": 20, "seeds": [0, 1, 2], "atRequiredScale": True, "neutral": 1.0,
    "grid": [{"ceiling": 0.0001, "targetAccuracy": 0.85},
             {"ceiling": 0.01, "targetAccuracy": 0.85}],
    "tied": [0.0001, 0.01], "decidedByTieBreak": True, "seedsAgree": False,
    "byTransfer": {"M->U": 0.0001}}}

TRIALS = {"milcreda": {
    "arm": "G", "ceiling": 0.01, "criterion": "targetAccuracy", "role": "valid",
    "epochs": 20, "atRequiredScale": True, "neutral": 1.0,
    "search": {"kind": "optuna", "sampler": "GPSampler", "trials": 30},
    "decidedByFlatRule": True, "plateau": [0.004, 0.01], "noise": 0.021,
    "byTransfer": {"M->U": 0.01, "S->M": 0.002},
    "perTransfer": {
        "M->U": {"ceiling": 0.01, "value": 0.62, "best": 0.63,
                 "plateau": [0.004, 0.01], "decidedByFlatRule": True, "trials": 30},
        "S->M": {"ceiling": 0.002, "value": 0.44, "best": 0.44,
                 "plateau": [0.002], "decidedByFlatRule": False, "trials": 30}}}}


def test_un_registro_de_trials_no_habla_de_semillas_ni_de_empates():
    """El defecto que motivó todo esto."""
    texto = tables.conclusion_ceilings(TRIALS)
    assert "semillas" not in texto
    assert "empatados" not in texto
    assert "trial(s)" in texto and "meseta" in texto


def test_un_registro_de_trials_informa_el_ruido_en_lugar_del_acuerdo():
    """El ruido estimado es lo que ocupa el lugar de «las semillas coinciden».

    Prestarle a una búsqueda continua una evidencia de repetición que no produjo
    sería exactamente la explicación inventada que el docstring original prohíbe.
    """
    texto = tables.conclusion_ceilings(TRIALS)
    assert "0.021" in texto
    # Por lo que es. La meseta la define la resolución del instrumento, no una
    # cantidad que el modelo ajustó: atribuírsela al GP haría que el lector
    # creyera que el ancho depende de qué tan bien ajustó, que es justo la
    # propiedad que se evitó.
    assert "resolución del criterio" in texto
    assert "GP estimó" not in texto


def test_la_regla_que_el_registro_se_lleva_adentro_no_le_atribuye_la_meseta_al_gp():
    """La mitad protegida no probaba la otra.

    `conclusion_ceilings` ya tenia su guarda y pasaba; `FLAT_RULE` -- la frase que
    cada entrada del registro se lleva adentro, y la que el lector del informe
    encuentra -- seguia diciendo que la meseta la definia el ruido que el GP
    estimo. Es la misma afirmacion falsa, en el lugar donde nadie la leia: el
    codigo pasa `SEARCH_RESOLUTION`, que es una bolsa de `VALID_BAGS`, y usar una
    cantidad ajustada haria que el ancho dependiera de que tan bien ajusto.
    """
    assert "GP estimó" not in cr.FLAT_RULE
    assert "resolución del criterio" in cr.FLAT_RULE


def test_la_rejilla_sigue_diciendo_lo_que_siempre_dijo():
    """No es compatibilidad de cortesía: es el registro que gobierna la campaña."""
    texto = tables.conclusion_ceilings(REJILLA)
    assert "desempate entre 2 techos empatados" in texto
    assert "3 repetición(es)" in texto
    assert "las semillas **no** coinciden" in texto


def test_la_tabla_de_trials_muestra_el_ancho_de_la_meseta():
    """Un techo elegido entre uno es una medición; el mismo entre nueve es la
    regla hablando, y se imprimen igual si la tabla muestra solo el ganador."""
    texto = tables.render_ceilings(TRIALS, markdown=True)
    assert "Meseta" in texto and "Trials" in texto
    assert "| 2 |" in texto and "| 1 |" in texto
    assert "**0.01**" in texto, "el que puso la regla va marcado"


def test_la_tabla_de_rejilla_conserva_sus_columnas_de_techos():
    texto = tables.render_ceilings(REJILLA, markdown=True)
    assert "0.0001" in texto and "0.01" in texto
    assert "Meseta" not in texto


def test_el_accesor_no_normaliza_los_dos_ejes_a_un_nombre_comun():
    """Tres repeticiones y treinta evaluaciones de puntos distintos no son la
    misma cantidad de evidencia, y llamarlas igual invita a compararlas."""
    assert cr.choice_of(REJILLA["creda"])["axis"] == "seeds"
    assert cr.choice_of(TRIALS["milcreda"])["axis"] == "trials"
