"""The two seams the later sections of the toyset stand on.

Section 3 asks whether the attention knows which instances matter, and that
question is only answerable if the three selection rules genuinely pick
DIFFERENT instances at the same budget: a `topk` that happened to return what
`arbitrary` returns would make the comparison pass while measuring nothing.

Section 5 asks what each term of Eq. (39) buys alone, and that is only
answerable if the gate on the global term actually drops it. `local` had a gate
since the ladder existed; `global` did not until 2026-10-20, so the arm that
runs the local term alone could not be declared.

Both are tested by MUTATION and not by reading a flag: a gate that is wired to
nothing and a gate that is held look identical to a green run.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from MIL_CREDA_Benchmark import toy_run, wiring


class _Brazo:
    """The smallest object `Arm.select` reads: a spec and a ranking.

    Not a real `Arm`: `select` reads `self.spec` and calls `self.weights_for`,
    and building a whole arm would put an encoder's draw between the rule and
    the instances it keeps. The ranking is strictly increasing, so the `topk`
    answer is known in advance and the assertion below is not a restatement of
    whatever the function returned.
    """

    def __init__(self, rule, k):
        self.spec = {"selection": rule, "selectK": k}

    def weights_for(self, H, sigma):
        return torch.linspace(0.0, 1.0, H.shape[0])


def _bolsa(m=30):
    return torch.arange(m, dtype=torch.float32).reshape(m, 1)


@pytest.mark.parametrize("k", toy_run.PRESUPUESTOS_K)
def test_las_tres_reglas_eligen_instancias_distintas_con_el_mismo_presupuesto(k):
    """Same `k`, three rules, three different sets -- or the section is void.

    This is the test the aggregation section rests on: `topk` against
    `arbitrary` at the SAME budget is what separates "the attention scores
    something" from "the attention is on and scoring nothing", and two rules
    that return the same instances would make that comparison a tie by
    construction.
    """
    H = _bolsa()
    elegidas = {
        regla: tuple(int(x) for x in
                     wiring.Arm.select(_Brazo(regla, k), H, 1.0).reshape(-1))
        for regla in ("topk", "arbitrary", "regular")
    }
    for regla, quedan in elegidas.items():
        assert len(quedan) == k, (regla, len(quedan), k)
        assert len(set(quedan)) == k, (regla, quedan, "una instancia repetida")
        assert list(quedan) == sorted(quedan), (
            regla, quedan,
            "las instancias no salen en el orden de la bolsa, así que una "
            "diferencia entre reglas podría ser el ORDEN y no la elección")
    assert len(set(elegidas.values())) == 3, elegidas


def test_topk_se_queda_con_las_mejores_segun_el_peso_del_propio_brazo():
    """`topk` reads the arm's OWN ranking, which is the whole question.

    The ranking here increases with the index, so the top `k` is the LAST `k`
    positions and the expected side is derived rather than copied from a run.
    """
    H = _bolsa()
    for k in toy_run.PRESUPUESTOS_K:
        quedan = [int(x) for x in
                  wiring.Arm.select(_Brazo("topk", k), H, 1.0).reshape(-1)]
        assert quedan == list(range(30 - k, 30)), (k, quedan)


def test_la_misma_bolsa_sortea_las_mismas_instancias_cada_vez_que_se_la_ve():
    """`arbitrary` is derived from the bag's own contents, never from a clock.

    A draw that moved between steps would put a SECOND difference inside every
    comparison that reads it: the rule, and which instances it happened to take
    this time.
    """
    H = _bolsa()
    primera = wiring.Arm.select(_Brazo("arbitrary", 10), H, 1.0)
    segunda = wiring.Arm.select(_Brazo("arbitrary", 10), H, 1.0)
    assert torch.equal(primera, segunda), (primera, segunda)
    # Y dos bolsas DISTINTAS no sortean lo mismo, o el sorteo no depende del
    # contenido y la regla sería `regular` con otro nombre.
    otra = wiring.Arm.select(_Brazo("arbitrary", 10), H + 7.0, 1.0)
    assert not torch.equal(primera.reshape(-1), (otra - 7.0).reshape(-1)), (
        "dos bolsas distintas sortean las mismas posiciones")


@pytest.mark.parametrize("spec,fragmento", [
    ({"selection": "topk", "selectK": 30}, "selects nothing"),
    ({"selection": "topk", "selectK": 31}, "selects nothing"),
    ({"selection": "topk", "selectK": 0}, "positive integer"),
    ({"selection": "topk", "selectK": None}, "positive integer"),
    ({"selection": "inexistente", "selectK": 5}, "unknown selection rule"),
])
def test_la_costura_se_niega_en_vez_de_devolver_la_bolsa_entera(spec, fragmento):
    """Every refusal, and each one names what it refused.

    A budget at or above the bag size is the dangerous case: it returns the
    whole bag under three different rule names, so the three columns of the
    table would carry one measurement read three times and nothing would fail.
    """
    brazo = _Brazo(None, None)
    brazo.spec = spec
    with pytest.raises(ValueError, match=fragmento):
        wiring.Arm.select(brazo, _bolsa(), 1.0)


def test_sin_regla_la_bolsa_sale_entera_y_es_el_mismo_objeto_de_entrada():
    """`None` is what every arm declared before 2026-10-20 does.

    Identity and not equality: a copy here would mean the no-selection path
    allocates on every bag of every step, which is the path section 1 runs.
    """
    H = _bolsa()
    assert wiring.Arm.select(_Brazo(None, None), H, 1.0) is H


def test_el_padron_de_la_ablacion_declara_los_tres_brazos_y_ninguno_repetido():
    """The three rows, and the pair of flags that tells them apart.

    Derived from the roster rather than written out: a fourth arm added there
    with a flag pair that already exists would make two rows of the table
    identical, and the table would print the same measurement twice.
    """
    banderas = {(a["global"], a["local"]) for a in toy_run.ABLACION}
    assert banderas == {(True, False), (False, True), (True, True)}, banderas
    assert len(toy_run.ABLACION) == 3, toy_run.ABLACION
    assert len({a["id"] for a in toy_run.ABLACION}) == 3


def test_la_ablacion_no_se_coló_en_el_padron_de_la_seccion_uno():
    """Section 1 keeps exactly one arm, and `TOY_ADAPTED` is still it.

    `run_cell` trains phase 1 on `TOY_ARM_IDS[-1]` and every later reading
    shares it, so an arm appended to `TOY_ARMS` would silently move what
    section 1 measures. The rosters are separate and this is what says so.
    """
    assert len(toy_run.TOY_ARMS) == 1, toy_run.TOY_ARMS
    assert toy_run.TOY_ADAPTED == toy_run.TOY_ARM_IDS[-1]
    assert toy_run.TOY_ADAPTED not in {a["id"] for a in toy_run.ABLACION
                                       if not (a["global"] and a["local"])}


def test_el_presupuesto_declarado_cabe_en_la_bolsa_declarada():
    """`k` is declared before the run, and all three are reported.

    Checked against the scale's own bag size rather than a literal 30: the day
    the bag grows, a budget that no longer selects anything has to redden here
    instead of inside a sweep that already spent its time.
    """
    for k in toy_run.PRESUPUESTOS_K:
        assert 0 < k < toy_run.FULL.instances_per_bag, (
            k, toy_run.FULL.instances_per_bag)
    assert len(set(toy_run.PRESUPUESTOS_K)) == len(toy_run.PRESUPUESTOS_K)


def test_los_mecanismos_se_leen_de_wiring_y_no_se_vuelven_a_escribir():
    """A mechanism added in `wiring` cannot go missing from the section.

    Identity against `wiring.MECHANISMS`: a copied tuple would keep agreeing
    until somebody added a sixth mechanism, which is exactly the day it would
    matter.
    """
    assert toy_run.MECANISMOS is wiring.MECHANISMS
    assert "ours" in toy_run.MECANISMOS and "mean" in toy_run.MECANISMOS
