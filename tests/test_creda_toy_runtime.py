"""What `CREDA.toy_experiment` still is, now that the tutorial owns training.

The toyset notebook `CREDA/Notebooks/Toyset.ipynb` writes its own training
loops, its own loss construction and its own figures. What it imports from
`CREDA.toy_experiment` is the toyset itself -- the three families, their
splits, the encoder, the normalized source term, the frozen bandwidth, the
effective-class monitor -- and the exact boundary/confidence-walk helpers
`CREDA.tutorial_visuals` draws with. This file holds both halves of that:

1. **The retired benchmark is gone, not hidden.** The old experiment runner
   (`corrida`, `train_source_ce`), its Optuna searches, its sweeps, its
   artifact IO, its contamination/oracle/intruder/ablation machinery and the
   drawing module `CREDA.toy_plots` were reachable from no live notebook.
   Each name below is asserted absent, so none of them returns as a wrapper.
2. **The retained runtime still computes what the tutorial reads.** The
   supervised term is Eq. (21)'s normalized form, the effective number of
   classes stays in `[1, CLASES]`, the encoder builds the variant it is asked
   for and refuses one it does not know, and the seeded splits are disjoint.

Every fixture asserts its own property before the thing under test is called.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from CREDA import toy_experiment as T
from CREDA.bounds import source_bound, source_loss
from CREDA.models import CREDALoss

RAIZ = Path(__file__).resolve().parents[1]
PAQUETE = RAIZ / "src" / "CREDA"
MODULO = PAQUETE / "toy_experiment.py"

#: The retired experiment surface, by name. Runner and searches, sweeps,
#: artifact IO, perturbation/oracle/ablation arms, and the old registries.
RETIRADOS = (
    # the runner and the phase-1 trainer
    "corrida", "train_source_ce", "fase1_congelada", "adabn",
    # the searches
    "buscar_hp1", "buscar_hp2", "_optuna", "SEMILLAS_BUSQUEDA",
    "LOTES_2_BUSCADOS", "HP_1",
    # the sweeps and the artifact
    "barrer_animaciones", "barrer_tablas", "barrer_gradiente",
    "barrer_contaminacion", "barrer_intrusos", "guardar", "cargar",
    "guardar_entradas", "guardar_contaminacion", "cargar_contaminacion",
    "fusionar_contaminacion", "banderas",
    # contamination, oracle, intruder and weight-injection machinery
    "contaminar", "intrusar", "intrusar_el_embedding",
    "uniformar_el_embedding", "pesos_oraculo", "pesos_oraculo_intrusos",
    "pesos_oraculo_ruidoso", "pesos_oraculo_emb", "pesos_compuerta",
    "pesador_del_cabezal", "MODO_REUBICADO", "MODO_FRONTERA",
    "MODOS_CONTAMINACION", "RHOS", "PESOS", "PESOS_TECHO",
    "BRAZOS_CONTAMINACION", "alineacion", "colapso", "ec33_por_clase",
    # the old series and layout registries
    "FILAS", "TABLA", "CONFIGS", "verificar_coherencia", "punto", "ec33",
    "CUADROS", "CTRL", "CAMPOS_CU", "CAMPOS_HT", "CAMPOS_P2_NOMBRES",
    "NOMBRE_FILA", "B_FUENTE", "preparar", "construir_dura",
    "pasos_de_una_epoca", "pasada_en_lotes",
    # the old epoch-runner scale and the descriptive extras, which lost their
    # last reader when the MIL-CREDA benchmark runner was retired
    "EPOCAS_2", "PACIENCIA_1", "MIN_EPOCAS_2", "BLOQUES_AL_LOTE_DECLARADO",
    "LOTE", "POS_FILAS", "NOTA_DURA", "dim_ef_dura", "entre_dura",
)

#: What the tutorial notebooks, `CREDA.tutorial_visuals`, the MIL-CREDA toyset
#: and the tests that pin the two packages together import from this module.
RETENIDOS = (
    "DURAS", "VISTA_DURA", "COL_DURA", "perturbar", "sigma_de_la_fuente",
    "ASPECTO_MAX", "FAMILIAS", "N_POR_DOMINIO", "datos_semilla", "partir",
    "NORMAS", "MLP", "CLASES", "caminata_al_centro", "puntos_de_frontera",
    "pesos_declarados", "incertidumbre_declarada", "perdida_fuente",
    "clases_efectivas", "HP_2", "NORMALIZAR_POR_COTAS", "FORMA_DE_COTAS",
    "BALANCE_DE_MASA",
)


def _tree(path: Path) -> ast.Module:
    """Parse, failing on any SyntaxWarning: no vendored code remains to tolerate."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not caught, (
        f"{path.name} raised {[str(w.message) for w in caught]} while being "
        "parsed, so it is ours to fix")
    return tree


def _called_names(node: ast.AST) -> list[str]:
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                out.append(func.id)
            elif isinstance(func, ast.Attribute):
                out.append(func.attr)
    return out


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} has no function {name!r}")


def _imported_modules(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module)
    return imported


# ============================================== 1. the retired surface is gone


def test_the_drawing_module_of_the_old_benchmark_is_gone():
    assert importlib.util.find_spec("CREDA.toy_plots") is None
    assert not (PAQUETE / "toy_plots.py").exists()


@pytest.mark.parametrize("nombre", RETIRADOS)
def test_the_retired_experiment_name_is_gone(nombre):
    assert not hasattr(T, nombre), (
        f"toy_experiment.{nombre} is part of the retired benchmark; no live "
        "notebook reaches it")


def test_the_phase_two_declaration_keeps_only_the_keys_something_reads():
    """`CRUZADO_SOPESADO` and `RHO_T_SIN_GRAD` are read by the tutorial's
    loss test and by MIL-CREDA's pinned laws; every other key fed the retired
    runner and its search."""
    assert set(T.HP_2) == {"CRUZADO_SOPESADO", "RHO_T_SIN_GRAD"}, sorted(T.HP_2)
    assert T.HP_2["CRUZADO_SOPESADO"] is True
    assert T.HP_2["RHO_T_SIN_GRAD"] is False


def test_the_module_no_longer_needs_the_search_or_the_artifact_io():
    importados = _imported_modules(MODULO)
    for retirado in ("json", "time", "copy", "pathlib", "optuna",
                     "CREDA.two_phase"):
        assert retirado not in importados, (retirado, sorted(importados))


@pytest.mark.parametrize("nombre", RETENIDOS)
def test_the_retained_name_is_still_there(nombre):
    assert hasattr(T, nombre), nombre


# ====================================== 2. the toyset and its encoder still work


def test_each_family_draws_disjoint_seeded_splits_that_cover_both_domains():
    for fam in T.FAMILIAS:
        D = T.datos_semilla(fam, 0)
        n_s = sum(len(D[f"ys_{r}"]) for r in ("tr", "va", "te"))
        n_t = sum(len(D[f"yt_{r}"]) for r in ("tr", "va", "te"))
        assert n_s == n_t == T.N_POR_DOMINIO, (fam["nombre"], n_s, n_t)
        for clave in ("Xs_tr", "Xt_tr", "Xs_te", "Xt_te"):
            assert D[clave].dtype == torch.float32, (clave, D[clave].dtype)
            assert D[clave].shape[1] == fam["amb"], (clave, D[clave].shape)
        assert set(D["ys_tr"].tolist()) == set(range(T.CLASES))
        otra = T.datos_semilla(fam, 0)
        assert all(torch.equal(D[k], otra[k]) for k in D), fam["nombre"]
        distinta = T.datos_semilla(fam, 1)
        assert not torch.equal(D["Xs_tr"], distinta["Xs_tr"]), fam["nombre"]


def test_the_split_refuses_fractions_that_leave_no_test():
    y = np.arange(30) % 3
    with pytest.raises(AssertionError):
        T.partir(y, fracs=(0.6, 0.4))


def test_the_encoder_builds_each_declared_variant_and_refuses_another():
    torch.manual_seed(0)
    x = torch.randn(12, 3)
    esperado = {"ninguna": None, "batchnorm": nn.BatchNorm1d,
                "layernorm": nn.LayerNorm}
    assert tuple(esperado) == T.NORMAS
    for pos, capa in esperado.items():
        enc = T.MLP(3, pos)
        assert enc(x).shape == (12, T.DIM), pos
        ultima = list(enc.net)[-1]
        if capa is None:
            assert isinstance(ultima, nn.ReLU), (pos, ultima)
        else:
            assert isinstance(ultima, capa), (pos, ultima)
    assert T.MLP(3, "ninguna", dim=5)(x).shape == (12, 5)
    with pytest.raises(AssertionError):
        T.MLP(3, "groupnorm")


# ====================================== 3. Eq. (21) is what the toyset trains on


def test_the_supervised_term_is_equation_twenty_ones_normalized_form():
    """No raw `cross_entropy` anywhere in the module, and `perdida_fuente`
    delegates to the bounds copy instead of writing Eq. (21) a second time."""
    assert "cross_entropy" not in _called_names(_tree(MODULO))
    cuerpo = _function(MODULO, "perdida_fuente")
    assert "source_loss" in _called_names(cuerpo), (
        "perdida_fuente does not call source_loss: an inline copy of Eq. (21) "
        "is a silent fork of the formulation")


def test_the_supervised_term_matches_a_hand_written_equation_twenty_one():
    """`(g + eps) / (1 + eps)` inside the logarithm and `B_src` outside it."""
    eps = 1e-8
    bound = math.log(1.0 + 1.0 / eps)
    assert source_bound() == pytest.approx(bound, rel=0, abs=0)
    assert bound == pytest.approx(18.4207, abs=1e-4), bound

    torch.manual_seed(0)
    logits = torch.randn(7, T.CLASES) * 2.5
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0])

    G = F.softmax(logits, 1)
    Y = F.one_hot(y, T.CLASES).to(G.dtype)
    a_mano = float(-(Y * torch.log((G + eps) / (1.0 + eps))).sum() / bound / len(y))

    medido = float(T.perdida_fuente(logits, y))
    assert medido == pytest.approx(a_mano, rel=1e-6), (medido, a_mano)
    assert medido == pytest.approx(float(source_loss(G, Y)), rel=1e-12)


def test_the_supervised_term_lives_in_the_unit_interval():
    """`[0, 1]`, with the upper end attained exactly in floating point."""
    y = torch.tensor([0, 1, 2])
    certero = torch.tensor([[60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 60.0]])
    equivocado = torch.tensor([[0.0, 0.0, 60.0], [60.0, 0.0, 0.0], [0.0, 60.0, 0.0]])

    assert float(T.perdida_fuente(certero, y)) == pytest.approx(0.0, abs=1e-9)
    assert float(T.perdida_fuente(equivocado, y)) == 1.0
    dentro = float(T.perdida_fuente(
        torch.tensor([[0.0, 0.0, 20.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0]]), y))
    assert 0.5 < dentro < 1.0, dentro
    medio = float(T.perdida_fuente(torch.zeros(3, T.CLASES), y))
    assert medio == pytest.approx(math.log(T.CLASES) / source_bound(), rel=1e-6)


def test_the_supervised_term_refuses_logits_of_the_wrong_shape():
    with pytest.raises(AssertionError):
        T.perdida_fuente(torch.zeros(4, T.CLASES + 1), torch.zeros(4, dtype=torch.long))
    with pytest.raises(AssertionError):
        T.perdida_fuente(torch.zeros(T.CLASES), torch.zeros(1, dtype=torch.long))


#: The ONE module of `MIL_CREDA` that `src/CREDA` may import: Eq. (24)'s
#: confidence weight is the method's treatment, ruled admissible on 2026-10-18.
IMPORT_PERMITIDO = {"MIL_CREDA.confidence"}


def test_the_toy_never_imports_the_method_it_is_the_baseline_for():
    """Over every module of the package: `src/CREDA` is the baseline
    `src/MIL_CREDA` is measured against, so an import pointing that way makes
    the control depend on what it controls."""
    revisados = 0
    for path in sorted(PAQUETE.rglob("*.py")):
        culpables = sorted(
            m for m in _imported_modules(path)
            if (m == "MIL_CREDA" or m.startswith("MIL_CREDA."))
            and m not in IMPORT_PERMITIDO)
        assert culpables == [], (path.name, culpables)
        revisados += 1
    assert revisados > 5, revisados
    assert "MIL_CREDA" not in vars(T)


# ============================== 4. the label-free monitor the tutorial prints


def test_the_effective_classes_stay_within_one_and_the_class_count():
    """`cef` in `[1, CLASES]`, with both ends EXACT."""
    C = T.CLASES
    assert T.clases_efectivas(np.zeros(60, dtype=np.int64)) == pytest.approx(1.0, abs=1e-6)
    repartido = np.arange(60, dtype=np.int64) % C
    assert T.clases_efectivas(repartido) == pytest.approx(float(C), abs=1e-6)
    dos = np.arange(60, dtype=np.int64) % 2
    assert T.clases_efectivas(dos) == pytest.approx(2.0, abs=1e-6)
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = rng.integers(0, C, size=int(rng.integers(2, 200)))
        v = T.clases_efectivas(r)
        assert 1.0 - 1e-9 <= v <= float(C) + 1e-9, r


def test_the_effective_classes_accept_torch_and_numpy_alike():
    r = (np.arange(37, dtype=np.int64) * 7) % T.CLASES
    assert T.clases_efectivas(r) == T.clases_efectivas(torch.as_tensor(r))


def test_the_effective_classes_refuse_an_empty_or_out_of_range_routing():
    with pytest.raises(AssertionError):
        T.clases_efectivas(np.array([], dtype=np.int64))
    with pytest.raises(AssertionError):
        T.clases_efectivas(np.array([0, T.CLASES]))


# ================== 5. the uniform-confidence invariance (Eq. 28), with the toy's
#                       own declared bounds map and balanced mass


def _g_para_w(objetivo, C, n, dtype=torch.float32):
    """`g` with every row at the SAME entropy, so every `w` equals one `c`."""
    lo, hi = 1.0 / C, 1.0 - 1e-15
    for _ in range(300):
        p = (lo + hi) / 2
        fila = np.array([p] + [(1 - p) / (C - 1)] * (C - 1))
        w = 1 - (-math.log((fila ** 2).sum() + 1e-6)) / math.log(C)
        if w < objetivo:
            lo = p
        else:
            hi = p
    p = (lo + hi) / 2
    base = np.array([p] + [(1 - p) / (C - 1)] * (C - 1))
    rows = np.stack([np.roll(base, j % C) for j in range(n)])
    return torch.tensor(rows, dtype=dtype), float(w)


def _perdida_por_clase(C=3, n_s=300, n_t=900):
    torch.manual_seed(0)
    f_s = torch.randn(n_s, 8)
    f_t = torch.randn(n_t, 8) + 0.5
    y_s = torch.arange(n_s) % C
    cr = CREDALoss(sigma="auto", sigma_from="batch-source",
                   normalize_by_bounds=T.NORMALIZAR_POR_COTAS,
                   bounds_form=T.FORMA_DE_COTAS, weight_cross_block=True,
                   balance_mix_blocks=True, balance_mass=T.BALANCE_DE_MASA)
    cr.class_mixing = 1.0

    def medir(objetivo):
        g, w = _g_para_w(objetivo, C, n_t)
        cuad = (g ** 2).sum(1)
        assert float(cuad.max() - cuad.min()) < 1e-6, "the rows differ in entropy"
        assert len(torch.unique(g.argmax(1))) == C, "a class came out empty"
        with torch.no_grad():
            return cr(f_s, f_t, y_s, g, reduction="none").numpy(), w

    return medir, C, n_t


def test_the_term_is_invariant_to_the_common_confidence_level():
    """With every `w` equal to one `c`, the `c^2` cancels under the trace
    balance and the `c` cancels in the cross block: the per-class loss is the
    same number at every `c`, asserted over two decades."""
    medir, C, n_t = _perdida_por_clase()
    cs = [1.0 - 1e-12, 0.5623, 0.3162, 0.1778, 0.1, 0.0562, 0.0316, 0.0178, 0.01]
    assert math.log10(cs[0] / cs[-1]) >= 2.0 - 1e-6, (cs[0], cs[-1])

    ref, _ = medir(cs[0])
    assert len(ref) == C
    bajo_uno = 0
    for c in cs:
        l, w = medir(c)
        assert w == pytest.approx(c, rel=1e-6), (w, c)
        traza = w ** 2 * n_t / C
        assert np.abs(l - ref).max() <= 1e-6, (c, traza, np.abs(l - ref).max())
        if traza < 1.0:
            bajo_uno += 1
    assert bajo_uno >= 3, bajo_uno


def test_the_residual_of_that_invariance_no_longer_follows_the_epsilons_law():
    """Below `tr(K~_t) = 1` the deviation is flat float32 round-off, not `1/c^2`."""
    medir, C, n_t = _perdida_por_clase()
    ref, _ = medir(1.0 - 1e-12)
    devs = []
    for c in (0.0562, 0.0316, 0.0178, 0.01):
        l, _ = medir(c)
        dev = float(np.abs(l - ref).max())
        assert dev <= 1e-6, (c, dev)
        devs.append(dev)
    ulp = float(np.finfo(np.float32).eps)
    assert max(devs) <= 4.0 * ulp, devs
    assert devs[-1] <= devs[0] + ulp, devs
