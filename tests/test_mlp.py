"""`mlp.py`'s three backbones differ in EXACTLY their normalization layer.

`"mlp-nobn"`, `"mlp"` and `"mlp-layernorm"` are the one instrument that
measures whether a bare normalization statistic is doing the adapting. If any
two of them differ in anything else -- width, activation, initialization, draw
order -- the comparison measures that difference instead of normalization.

**And the third one is not symmetry with CREDA's figure: it is the control.**
BatchNorm does two things at once, it normalizes AND it carries running
statistics the target domain can re-estimate without an optimizer, which is
what AdaBN is. LayerNorm normalizes and holds nothing re-estimable. So a
BatchNorm row reading above the un-normalized one can be attributed only with
the LayerNorm row beside it: without it, *normalizing helped* and
*re-estimating helped* are the same reading.

`tests/test_mlp_backbone.py` holds the pair that existed before this file and is
not touched by it; its claims about `"mlp"` and `"mlp-nobn"` are unchanged and
still hold. What is here is the THIRD kind and the claims that only exist once
there are three: that the structural axis is one, that the normalization layer
is the whole of it, and that AdaBN's reachability is read off the module tree by
type rather than off a backbone's name.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from MIL_CREDA_Benchmark import config, mlp, toy_run


def _linear_layers(net: nn.Sequential) -> list[nn.Linear]:
    return [layer for layer in net if isinstance(layer, nn.Linear)]


def _normalization_layers(net: nn.Sequential) -> list[nn.Module]:
    return [layer for layer in net
            if isinstance(layer, (nn.modules.batchnorm._BatchNorm,
                                  nn.LayerNorm, nn.GroupNorm))]


def _activations(net: nn.Sequential) -> list[type]:
    return [type(layer) for layer in net if isinstance(layer, nn.ReLU)]


def _md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def _mlp_mutado(tmp_path: Path, antes: str, despues: str, nombre: str):
    """A second `mlp`, loaded from a SAME-LENGTH mutation of its source.

    Three measured reasons for this shape, all of them bitten in this
    repository before:

    * a replacement of the same length leaves the byte count unchanged, so what
      proves the mutation reached the file is the **md5** and never the size;
    * a cached `.pyc` is validated against (size, mtime), and a same-size
      rewrite inside the same second has already matched both here -- so
      `__pycache__` is removed before loading;
    * the module is loaded under its own name and deregistered afterwards, so
      the installed `mlp` every other test holds a reference to is untouched.
    """
    original = Path(mlp.__file__)
    fuente = original.read_text(encoding="utf-8")
    assert len(antes) == len(despues), (len(antes), len(despues))
    assert fuente.count(antes) == 1, (antes, fuente.count(antes))
    destino = tmp_path / f"{nombre}.py"
    destino.write_text(fuente.replace(antes, despues), encoding="utf-8")
    assert destino.stat().st_size == original.stat().st_size
    assert _md5(destino) != _md5(original), (
        "la mutación no movió el md5: el reemplazo no llegó al archivo")
    shutil.rmtree(tmp_path / "__pycache__", ignore_errors=True)
    spec = importlib.util.spec_from_file_location(nombre, str(destino))
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    try:
        spec.loader.exec_module(modulo)
    finally:
        sys.modules.pop(nombre, None)
    return modulo


# ------------------------------------------------ tres, y el conteo es derivado

def test_there_are_three_backbones_and_the_roster_is_derived():
    """Three kinds, three backbones, and neither list is written twice.

    `MLP_BACKBONES` is derived from `BACKBONE_NORM`, which is the only place a
    backbone name is joined to a normalization kind. Two lists of the same
    thing drift, and the one that drifts is the one nobody re-reads -- this
    module shipped exactly that defect once, with the toyset's row names in one
    file and the backbone names in another.

    The count is asserted against CREDA's own row list and not against a three
    written here: three is what CREDA draws, so the number has a source.
    """
    from CREDA import toy_experiment as creda

    assert mlp.MLP_BACKBONES == tuple(mlp.BACKBONE_NORM)
    assert set(mlp.BACKBONE_NORM.values()) == set(mlp.NORM_LAYERS)
    assert set(mlp.NORM_LAYERS) == set(creda.POS_FILAS), (
        f"las normalizaciones de `mlp` son {sorted(mlp.NORM_LAYERS)} y CREDA "
        f"dibuja {sorted(creda.POS_FILAS)}: la grilla de este paquete no es la "
        "de al lado")
    assert len(mlp.MLP_BACKBONES) == len(creda.POS_FILAS)


def test_every_declared_backbone_builds_and_names_its_own_kind():
    """A backbone that built the wrong kind would pass every shape assertion
    below -- they are about widths -- so the kind each one carries is read off
    the instance and compared with the declaration."""
    for backbone, kind in mlp.BACKBONE_NORM.items():
        encoder = mlp.build_mlp(backbone, input_dim=3)
        assert encoder.normalization == kind, (backbone, kind,
                                               encoder.normalization)
        esperada = mlp.NORM_LAYERS[kind]
        capas = _normalization_layers(encoder.net)
        if esperada is None:
            assert capas == [], (backbone, capas)
        else:
            assert capas and all(isinstance(c, esperada) for c in capas), (
                backbone, [type(c).__name__ for c in capas])


# ------------------------- el único eje: la capa de normalización y nada más

def test_the_three_backbones_share_every_linear_shape_and_every_activation():
    """Same widths and same activations across all three, pairwise.

    Pairwise and not against the first: a defect that moved the middle one
    would be invisible to a test that only compared the ends, which is the
    shape of mistake that let a three-column comparison skip its own middle
    column one file over.
    """
    construidos = {b: mlp.build_mlp(b, input_dim=3) for b in mlp.MLP_BACKBONES}
    formas = {b: [(l.in_features, l.out_features)
                  for l in _linear_layers(e.net)]
              for b, e in construidos.items()}
    activaciones = {b: _activations(e.net) for b, e in construidos.items()}

    assert all(len(f) == 3 for f in formas.values()), formas
    assert all(len(a) == 2 for a in activaciones.values()), activaciones
    referencia = formas[mlp.MLP_BACKBONES[0]]
    assert referencia == [(3, config.MLP_HIDDEN_WIDTH),
                          (config.MLP_HIDDEN_WIDTH, config.MLP_HIDDEN_WIDTH),
                          (config.MLP_HIDDEN_WIDTH, config.MLP_OUTPUT_DIM)]
    for backbone in mlp.MLP_BACKBONES[1:]:
        assert formas[backbone] == referencia, (backbone, formas)
        assert activaciones[backbone] == activaciones[mlp.MLP_BACKBONES[0]]
    assert all(e.output_dim == config.MLP_OUTPUT_DIM
               for e in construidos.values())


def test_the_module_sequence_differs_in_the_normalization_layer_and_nowhere_else():
    """Strip the normalization layers and the three sequences become one.

    This is the structural form of «nothing else different»: it reads the whole
    `nn.Sequential` as a list of TYPES, removes whatever normalizes, and
    requires the remainders to be equal. A widened layer, a swapped activation
    or a fourth `Linear` in one variant all fail here, and none of them is
    visible to a parameter count alone.

    The control is the second half: the sequences must NOT be equal before the
    normalization layers are removed, or the assertion above would hold of
    three identical backbones and measure nothing.
    """
    secuencias = {b: [type(l) for l in mlp.build_mlp(b, 3).net]
                  for b in mlp.MLP_BACKBONES}
    normaliza = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm)
    desnudas = {b: [t for t in s if not issubclass(t, normaliza)]
                for b, s in secuencias.items()}
    assert len({tuple(s) for s in desnudas.values()}) == 1, desnudas
    assert len({tuple(s) for s in secuencias.values()}) == len(secuencias), (
        "dos backbones tienen la MISMA secuencia de módulos, así que la "
        "aserción de arriba no distingue nada: el eje de la normalización no "
        "está haciendo ninguna diferencia")


def test_the_parameter_count_is_equal_where_the_layer_type_allows_it():
    """`nn.BatchNorm1d(n)` and `nn.LayerNorm(n)` both carry `2n` learnables, so
    those two backbones hold the SAME total; the un-normalized one holds exactly
    the two layers' worth less.

    Asserted as an identity computed from the widths and not as the number this
    run happens to produce: a count written here would stop being true the day
    `config.MLP_HIDDEN_WIDTH` moves, and it would stop being true silently.
    """
    cuentas = {b: sum(p.numel() for p in mlp.build_mlp(b, 3).parameters())
               for b in mlp.MLP_BACKBONES}
    con_estado = cuentas["mlp"]
    con_layernorm = cuentas["mlp-layernorm"]
    sin_nada = cuentas["mlp-nobn"]
    assert con_estado == con_layernorm, (
        f"`mlp` tiene {con_estado} parámetros y `mlp-layernorm` "
        f"{con_layernorm}: las dos capas llevan 2n aprendibles cada una, así "
        "que una diferencia acá es una diferencia que nadie declaró")
    # Dos capas, dos tensores aprendibles cada una, del ancho oculto.
    assert con_estado - sin_nada == 2 * 2 * config.MLP_HIDDEN_WIDTH, (
        con_estado, sin_nada, config.MLP_HIDDEN_WIDTH)
    assert sin_nada < con_estado


def test_no_normalization_layer_draws_from_the_random_generator():
    """Inserting, removing or SWAPPING the normalization must not shift which
    draws the `Linear` layers receive.

    `nn.BatchNorm1d`'s and `nn.LayerNorm`'s default initializations are weight
    ones and bias zeros, which draw nothing -- so under one seed the three
    weight matrices come out bit-identical across all three variants. That is
    what makes this a controlled comparison of normalization ALONE rather than
    of normalization-plus-whatever-else-the-generator-moved, and it is measured
    here rather than assumed: a layer type added later whose init did draw
    would break the control in silence.

    Pairwise over all three, and the whole parameter -- weight and bias -- of
    every `Linear`.
    """
    pesos = {}
    for backbone in mlp.MLP_BACKBONES:
        torch.manual_seed(0)
        pesos[backbone] = [(l.weight.clone(), l.bias.clone())
                           for l in _linear_layers(mlp.build_mlp(backbone, 3).net)]
    referencia = pesos[mlp.MLP_BACKBONES[0]]
    for backbone in mlp.MLP_BACKBONES[1:]:
        assert len(pesos[backbone]) == len(referencia) == 3
        for (wa, ba), (wb, bb) in zip(pesos[backbone], referencia):
            assert torch.equal(wa, wb), backbone
            assert torch.equal(ba, bb), backbone
    # El control: con OTRA semilla los pesos tienen que moverse, o la igualdad
    # de arriba sería la de un generador que no se usa.
    torch.manual_seed(1)
    otra = [l.weight.clone()
            for l in _linear_layers(mlp.build_mlp("mlp", 3).net)]
    assert not all(torch.equal(a, b[0]) for a, b in zip(otra, referencia))


# ---------------------------- el control: LayerNorm no tiene qué reestimar

def test_layernorm_normalizes_and_holds_no_reestimable_buffer():
    """The whole reason the third row exists, measured on the module tree.

    Three readings and the three together are the claim: the LayerNorm backbone
    HAS normalization layers (so it normalizes), has ZERO buffers of any kind
    (so there is nothing to carry between batches), and `_reestimable_layers`
    finds none (so AdaBN cannot act). The BatchNorm backbone is the contrast on
    all three, and the un-normalized one is the contrast on the first -- which
    is what keeps «no re-estimable buffer» from reading the same for the row
    that does not normalize at all.
    """
    lecturas = {}
    for backbone in mlp.MLP_BACKBONES:
        encoder = mlp.build_mlp(backbone, 3)
        lecturas[backbone] = {
            "normaliza": len(toy_run._normalization_layers(encoder)),
            "reestimable": len(toy_run._reestimable_layers(encoder)),
            "buffers": sum(b.numel() for b in encoder.buffers()),
        }

    assert lecturas["mlp-nobn"] == {"normaliza": 0, "reestimable": 0,
                                    "buffers": 0}
    assert lecturas["mlp-layernorm"]["normaliza"] == 2
    assert lecturas["mlp-layernorm"]["reestimable"] == 0, (
        "`mlp-layernorm` trae capas que AdaBN podría reestimar: entonces no es "
        "el control que separa «normalizar ayuda» de «reestimar ayuda»")
    assert lecturas["mlp-layernorm"]["buffers"] == 0
    assert lecturas["mlp"]["normaliza"] == 2
    assert lecturas["mlp"]["reestimable"] == 2
    assert lecturas["mlp"]["buffers"] > 0
    # Y las dos filas que normalizan normalizan la MISMA cantidad de veces: si
    # una llevara una capa y la otra dos, el par no sería un control.
    assert (lecturas["mlp"]["normaliza"]
            == lecturas["mlp-layernorm"]["normaliza"])


def test_adabn_says_why_it_has_no_rung_and_the_two_reasons_are_apart():
    """`adabn_reading` reports a REASON on the LayerNorm row, never a number.

    A number there would be `.predict` under another name, published as if a
    second thing had been measured. And the two reasons are different facts
    about the backbone -- «does not normalize» against «normalizes and keeps
    nothing» -- so they do not share a sentence: collapsing them would make the
    control row and the bare row read alike, which is the confusion the control
    exists to end.

    Driven through the early-return path, which reads `model.target_encoder` and
    nothing else, so a namespace carrying that one attribute exercises exactly
    what runs. The BatchNorm pole is measured by the suite's own cell fixtures
    and by the artefact the toyset tests read; what this adds is that the two
    `None` branches are distinguishable.
    """
    lecturas = {}
    for backbone in ("mlp-nobn", "mlp-layernorm"):
        modelo = types.SimpleNamespace(
            target_encoder=mlp.build_mlp(backbone, 3))
        lecturas[backbone] = toy_run.adabn_reading(
            modelo, None, None, torch.device("cpu"))

    for backbone, lectura in lecturas.items():
        assert lectura["accuracy"] is None, backbone
        assert lectura["layers"] == 0, backbone
        assert lectura["batches"] == 0, backbone
        assert lectura["note"], backbone

    assert lecturas["mlp-nobn"]["normLayers"] == 0
    assert lecturas["mlp-layernorm"]["normLayers"] == 2
    assert (lecturas["mlp-nobn"]["note"]
            != lecturas["mlp-layernorm"]["note"]), (
        "las dos razones comparten la frase: «no normaliza» y «normaliza y no "
        "guarda nada» son hechos distintos del backbone y la fila del control "
        "se leería igual que la fila pelada")
    assert "NINGUNA capa de normalización" in lecturas["mlp-nobn"]["note"]
    control = lecturas["mlp-layernorm"]["note"]
    assert "control" in control, control
    assert "LayerNorm" in control, control
    assert "reestimar" in control, control


# ---------------------------------------------- las dos puertas, y su negativa

def test_the_legacy_boolean_door_and_the_knob_cannot_both_be_given():
    """Exactly one of `batchnorm` and `norm`, and the refusal is reachable both
    ways.

    `batchnorm=` is the two-state door this class had while there were two
    kinds and is kept because callers outside this module construct through it.
    It is an alias and not a second axis, so giving both is two answers to one
    question and giving neither would make the normalization a DEFAULT -- which
    is the one thing a controlled comparison may not have.
    """
    assert mlp.MLPEncoder(input_dim=3, batchnorm=True).normalization == "batchnorm"
    assert mlp.MLPEncoder(input_dim=3, batchnorm=False).normalization == "ninguna"
    assert mlp.MLPEncoder(input_dim=3, norm="layernorm").normalization == "layernorm"
    with pytest.raises(ValueError, match="exactly one"):
        mlp.MLPEncoder(input_dim=3)
    with pytest.raises(ValueError, match="exactly one"):
        mlp.MLPEncoder(input_dim=3, batchnorm=True, norm="layernorm")
    with pytest.raises(ValueError, match="unknown normalization"):
        mlp.MLPEncoder(input_dim=3, norm="instancenorm")
    with pytest.raises(ValueError, match="unknown MLP backbone"):
        mlp.build_mlp("mlp-groupnorm", input_dim=3)


# -------------------------------------------- el rojo alcanzable, por mutación

def test_a_layernorm_row_that_carried_running_statistics_is_caught(tmp_path):
    """Point the third kind at `nn.BatchNorm1d` and the control collapses.

    The red for `test_layernorm_normalizes_and_holds_no_reestimable_buffer` and
    for `test_adabn_says_why_it_has_no_rung_and_the_two_reasons_are_apart` at
    once, which is what makes those two measurements rather than descriptions.
    Under the mutation the `layernorm` row would hold buffers AdaBN re-estimates
    and `adabn_reading` would try to report a NUMBER on it -- so the row that is
    supposed to answer «did BatchNorm win by normalizing» would be a second
    BatchNorm row, and every other assertion in this file would stay green: the
    widths, the activations, the draw order and the parameter count are all
    unchanged by it.
    """
    mutado = _mlp_mutado(
        tmp_path,
        '    "batchnorm": nn.BatchNorm1d,\n    "layernorm": nn.LayerNorm,',
        '    "batchnorm": nn.BatchNorm1d,\n   "layernorm":nn.BatchNorm1d,',
        "mlp_layernorm_con_corridas")

    encoder = mutado.build_mlp("mlp-layernorm", 3)
    assert toy_run._reestimable_layers(encoder), (
        "la mutación no le dio capas reestimables a la fila de layernorm, así "
        "que el verde de arriba no prueba nada")
    assert sum(b.numel() for b in encoder.buffers()) > 0
    modelo = types.SimpleNamespace(target_encoder=encoder)
    # `adabn_reading` ya no toma el camino del `None`: entra al camino que
    # reestima, que sobre este namespace falla en la primera llamada que un
    # `nn.Module` real contesta --- lo que importa es que DEJÓ de contestar «no
    # hay nada que reestimar», que es lo que la fila del control informa.
    with pytest.raises(AttributeError):
        toy_run.adabn_reading(modelo, None, None, torch.device("cpu"))
    # Y el control: el módulo sin mutar sigue contestando la razón.
    sano = types.SimpleNamespace(
        target_encoder=mlp.build_mlp("mlp-layernorm", 3))
    assert toy_run.adabn_reading(
        sano, None, None, torch.device("cpu"))["accuracy"] is None


def test_a_third_kind_that_stopped_normalizing_is_caught(tmp_path):
    """Point the third kind at `nn.Identity` and it stops being a control.

    The other direction, and the one a well-meaning edit would take: a layer
    that is cheap, has no state and «does nothing» satisfies «no re-estimable
    buffer» perfectly, and a row built on it answers nothing -- the question is
    whether NORMALIZING helps, so the control has to normalize. Under this
    mutation the row keeps its name, keeps its three `Linear` layers and keeps
    AdaBN's `None`, and what changes is that it no longer normalizes and no
    longer carries the two layers' parameters.
    """
    mutado = _mlp_mutado(
        tmp_path,
        '    "layernorm": nn.LayerNorm,',
        '    "layernorm": nn.Identity, ',
        "mlp_layernorm_que_no_normaliza")

    encoder = mutado.build_mlp("mlp-layernorm", 3)
    assert toy_run._normalization_layers(encoder) == [], (
        "la mutación no sacó la normalización, así que el verde de arriba no "
        "prueba nada")
    cuentas = sum(p.numel() for p in encoder.parameters())
    assert cuentas == sum(p.numel()
                          for p in mutado.build_mlp("mlp-nobn", 3).parameters()), (
        "la fila mutada sigue llevando los parámetros de una capa que "
        "normaliza")
    modelo = types.SimpleNamespace(target_encoder=encoder)
    lectura = toy_run.adabn_reading(modelo, None, None, torch.device("cpu"))
    assert lectura["normLayers"] == 0
    assert "NINGUNA capa de normalización" in lectura["note"], (
        "la lectura sigue diciendo que esta fila es el control, y la mutación "
        "le quitó justo lo que la hacía serlo")
