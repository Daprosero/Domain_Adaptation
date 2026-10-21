"""What the sections after the first stand on: a record they read and a
perturbation that perturbs.

Three claims are tested here that nothing else can catch:

1. **They read and never search.** Every hyperparameter comes out of section
   1's own artefact, so a section cannot win by being tuned harder than the one
   before it. Tested against the artefact ON DISK, not a fixture: the full run
   declares which architecture won each family in `MIL-CREDA/AGREED-TOYSET.md`,
   and the reader has to derive those same three.

2. **The ground truth of this toyset is an identity, not a ranking.** The two
   domains split the three roles IDENTICALLY, which is measured here rather
   than assumed -- it is what makes the exact partner always present on the
   train/train pair and always absent on train/eval, and it refutes the premise
   `toy.true_partners` states to justify ranking.

3. **A perturbation that moves nothing looks exactly like a null result.**
   Both destination rules are exercised, and the one that runs is asserted to
   land on a NAMED bag rather than on an average -- because an average over a
   class lands near the global centre, which is where a frontier bag already
   is.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from MIL_CREDA_Benchmark import bags, toy_run

#: The artefact the readers are tested against. The FULL one, because it is the
#: one whose winners the agreement writes down.
ARTEFACTO_COMPLETO = (Path(__file__).resolve().parents[1]
                      / "MIL-CREDA" / "Results" / "Toyset" / "full")

#: What `MIL-CREDA/AGREED-TOYSET.md` declares the full run measured, quoted:
#: "`lunas` none (0.4056), `toroide` BatchNorm (0.7944), `rollo` BatchNorm
#: (0.3944)". Written here as data so the reader is held to the agreement and
#: not to whatever it returns.
GANADORES_DECLARADOS = {"lunas": "ninguna", "toroide": "batchnorm",
                        "rollo": "batchnorm"}


def _registro_sintetico(medias: dict) -> dict:
    """A record carrying only what these readers read."""
    filas = {}
    for familia, por_norma in medias.items():
        for norma, media in por_norma.items():
            filas[toy_run.key(familia, norma, toy_run.TOY_ADAPTED)] = {
                "summary": {"targetAccuracy": {"mean": media, "stdev": 0.0,
                                               "n": 1}},
                "scale": {"name": "sintetica"},
            }
    familias = list(medias)
    return {
        "hp1": {f: {"lr1": 0.1, "dim": 8} for f in familias},
        "hp2": {f: {"lr2": 0.2, "tauLocal": 1.5} for f in familias},
        "filas": filas,
        "escala": {"name": "sintetica"},
    }


# ------------------------------------------------- lo que se lee de la seccion 1

def test_el_lector_se_niega_cuando_no_hay_artefacto(tmp_path):
    with pytest.raises(FileNotFoundError) as caido:
        toy_run.artefacto_seccion_uno(tmp_path)
    assert toy_run.NOMBRE_ARTEFACTO in str(caido.value)


def test_el_lector_se_niega_ante_un_artefacto_de_otra_forma(tmp_path):
    (tmp_path / toy_run.NOMBRE_ARTEFACTO).write_text(
        json.dumps({"hp1": {}, "filas": {}}), encoding="utf-8")
    with pytest.raises(KeyError) as caido:
        toy_run.artefacto_seccion_uno(tmp_path)
    assert "hp2" in str(caido.value)


def test_los_hp_son_la_union_con_la_fase_dos_ultima():
    registro = _registro_sintetico({"lunas": {"ninguna": 0.5}})
    registro["hp1"]["lunas"]["lr2"] = 999.0     # la fase 1 no decide lr2
    leido = toy_run.hp_de_la_seccion_uno(registro, "lunas")
    assert leido["lr1"] == 0.1
    assert leido["lr2"] == 0.2, "la fase 2 va ultima y gana"
    assert leido["tauLocal"] == 1.5


def test_los_hp_se_niegan_para_una_familia_que_el_registro_no_trae():
    registro = _registro_sintetico({"lunas": {"ninguna": 0.5}})
    with pytest.raises(KeyError) as caido:
        toy_run.hp_de_la_seccion_uno(registro, "toroide")
    assert "toroide" in str(caido.value)


def test_la_mejor_arquitectura_es_la_de_media_mas_alta():
    registro = _registro_sintetico(
        {"lunas": {"ninguna": 0.40, "batchnorm": 0.71, "layernorm": 0.55}})
    assert toy_run.mejor_arquitectura(registro, "lunas") == "batchnorm"


def test_un_empate_lo_rompe_el_orden_declarado_y_no_el_del_dict():
    """Dos lecturas del mismo registro tienen que elegir lo mismo."""
    registro = _registro_sintetico(
        {"lunas": {"ninguna": 0.7, "batchnorm": 0.7, "layernorm": 0.7}})
    primera = toy_run.mejor_arquitectura(registro, "lunas")
    assert primera == next(iter(toy_run.NORMALIZATIONS))
    revuelto = copy.deepcopy(registro)
    revuelto["filas"] = dict(reversed(list(revuelto["filas"].items())))
    assert toy_run.mejor_arquitectura(revuelto, "lunas") == primera


def test_la_mejor_arquitectura_se_niega_sin_ninguna_fila():
    registro = _registro_sintetico({"lunas": {"ninguna": 0.5}})
    with pytest.raises(KeyError) as caido:
        toy_run.mejor_arquitectura(registro, "rollo")
    assert "rollo" in str(caido.value)


def test_la_arquitectura_fija_se_niega_porque_nadie_la_nombro(monkeypatch):
    """`ARQUITECTURA_POR_FAMILIA` en False no puede elegir sola."""
    registro = _registro_sintetico({"lunas": {"ninguna": 0.5}})
    monkeypatch.setattr(toy_run, "ARQUITECTURA_POR_FAMILIA", False)
    with pytest.raises(NotImplementedError):
        toy_run.arquitectura_de_la_seccion(registro, "lunas")


@pytest.mark.skipif(not (ARTEFACTO_COMPLETO / toy_run.NOMBRE_ARTEFACTO).exists(),
                    reason="el artefacto de la corrida completa no esta en disco")
def test_el_lector_deriva_los_ganadores_que_el_acuerdo_declara():
    """La prueba que cruza el codigo con el documento, sobre datos de verdad.

    El acuerdo escribe los tres ganadores de la corrida completa. Si el lector
    derivara otros, o el lector esta mal o el acuerdo quedo viejo --- y las dos
    son cosas que hay que saber antes de correr cuatro secciones encima.
    """
    registro = toy_run.artefacto_seccion_uno(ARTEFACTO_COMPLETO)
    derivados = {f: toy_run.arquitectura_de_la_seccion(registro, f)
                 for f in GANADORES_DECLARADOS}
    assert derivados == GANADORES_DECLARADOS


# --------------------------------------------------- la verdad de fondo del toyset

def test_los_dos_dominios_cortan_los_roles_identicos():
    """La medicion que obliga al par train/train de la seccion 4.

    Es la premisa que `toy.true_partners` niega en su propio docstring, y en
    ESTE toyset es falsa: los cortes coinciden. De ahi sale que `fuente-train`
    y `blanco-eval` son disjuntos por construccion.
    """
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp")
    fuente, blanco = material["source"], material["target"]
    for rol in ("train_idx", "valid_idx", "eval_idx"):
        assert (sorted(getattr(fuente, rol).tolist())
                == sorted(getattr(blanco, rol).tolist())), rol
    assert not (set(fuente.train_idx.tolist()) & set(blanco.eval_idx.tolist()))
    assert (set(fuente.train_idx.tolist()) == set(blanco.train_idx.tolist()))


def test_el_socio_exacto_es_la_identidad_en_clase_y_sujeto():
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp")
    socios = toy_run.socio_exacto(material["source"], material["target"],
                                  toy_run.PILOT.bags_per_class)
    fuente = material["source"].train_idx.numpy()
    blanco = material["target"].train_idx.numpy()
    assert (socios >= 0).all(), "sobre train/train la cobertura es total"
    # Y el socio de cada bolsa tiene la MISMA posicion, no una parecida.
    assert (fuente[socios] == blanco).all()


def test_el_socio_exacto_devuelve_menos_uno_cuando_la_fuente_no_lo_sorteo():
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp")
    socios = toy_run.socio_exacto(material["source"], material["target"],
                                  toy_run.PILOT.bags_per_class,
                                  rol_blanco="eval_idx")
    assert (socios == -1).all(), (
        "los cortes coinciden, asi que train de la fuente y eval del blanco no "
        "comparten ninguna bolsa")


def test_el_socio_exacto_se_niega_sin_bolsas_por_clase():
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp")
    with pytest.raises(ValueError):
        toy_run.socio_exacto(material["source"], material["target"], 0)


def test_las_posiciones_de_bolsa_salen_de_la_escala_y_no_del_indice_de_sujeto():
    """El defecto medido: `max(familia.bt)` es 26 y la posicion mas alta es 80."""
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp")
    total = toy_run.posiciones_de_bolsa(toy_run.PILOT)
    assert total >= int(material["target"].members.shape[0])
    assert int(np.max(familia.bt)) < total, (
        "si fueran la misma numeracion esta prueba no distinguiria nada")


def test_el_puente_al_bagset_asigna_cada_fila_sorteada_y_nada_mas():
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp")
    blanco = material["target"]
    asignacion = toy_run.bolsas_desde_el_bagset(blanco, familia.Xt.shape[0])
    assert asignacion.shape == (familia.Xt.shape[0],)
    miembros = blanco.members.numpy()
    for posicion in range(miembros.shape[0]):
        assert (asignacion[miembros[posicion]] == posicion).all()
    sorteadas = int(miembros.size)
    assert int((asignacion >= 0).sum()) == sorteadas
    assert int((asignacion == -1).sum()) == familia.Xt.shape[0] - sorteadas


# ------------------------------------------------------ el destino del desplazamiento

def _material_plano():
    """Tres bolsas de dos clases en una recta, con el destino conocido de antemano.

    Clase 0: bolsa 0 en x=0. Clase 1: bolsa 1 en x=1, bolsa 2 en x=9. El sujeto
    mas cercano a la bolsa 0 es la bolsa 1 (a 1.0); el promedio de la clase 1
    esta en x=5 (a 5.0). Las dos reglas no pueden dar el mismo numero.
    """
    X = np.array([[0.0], [0.0], [1.0], [1.0], [9.0], [9.0]])
    bolsas = np.array([0, 0, 1, 1, 2, 2])
    y = np.array([0, 0, 1, 1, 1, 1])
    scores = np.array([[0.5, 0.5], [0.1, 0.9], [0.1, 0.9]])
    return X, bolsas, y, scores


def test_el_sujeto_mas_cercano_nombra_una_bolsa_y_el_promedio_no():
    X, bolsas, y, scores = _material_plano()
    _, cercano = toy_run.desplazar_bolsas(
        X, bolsas, y, np.array([0]), scores, 1.0,
        destino_regla="sujeto-mas-cercano")
    _, promedio = toy_run.desplazar_bolsas(
        X, bolsas, y, np.array([0]), scores, 1.0,
        destino_regla="centroide-de-clase")
    assert cercano[0]["bolsaDestino"] == 1
    assert cercano[0]["distanciaTotal"] == pytest.approx(1.0)
    assert promedio[0]["bolsaDestino"] == -1
    assert promedio[0]["distanciaTotal"] == pytest.approx(5.0)
    assert cercano[0]["reglaDestino"] == "sujeto-mas-cercano"
    assert promedio[0]["reglaDestino"] == "centroide-de-clase"


def test_la_regla_que_corre_es_la_declarada_en_el_modulo():
    X, bolsas, y, scores = _material_plano()
    _, por_defecto = toy_run.desplazar_bolsas(X, bolsas, y, np.array([0]),
                                              scores, 1.0)
    assert por_defecto[0]["reglaDestino"] == toy_run.DESTINO_DEL_DESPLAZAMIENTO
    assert toy_run.DESTINO_DEL_DESPLAZAMIENTO in toy_run.DESTINOS


def test_una_regla_de_destino_desconocida_se_niega():
    X, bolsas, y, scores = _material_plano()
    with pytest.raises(ValueError) as caido:
        toy_run.desplazar_bolsas(X, bolsas, y, np.array([0]), scores, 1.0,
                                 destino_regla="la-que-me-convenga")
    assert "destino" in str(caido.value)


def test_una_posicion_negativa_no_entra_como_destino():
    """Las filas que ningun rol sorteo no son una bolsa mas.

    Se las pone en x=100 y de la misma clase que el destino: si entraran al
    promedio o a la busqueda del mas cercano, el destino se iria para alla.
    """
    X = np.array([[0.0], [0.0], [1.0], [1.0], [100.0], [100.0]])
    bolsas = np.array([0, 0, 1, 1, -1, -1])
    y = np.array([0, 0, 1, 1, 1, 1])
    scores = np.array([[0.5, 0.5], [0.1, 0.9]])
    for regla in toy_run.DESTINOS:
        _, registros = toy_run.desplazar_bolsas(
            X, bolsas, y, np.array([0]), scores, 1.0, destino_regla=regla)
        assert registros[0]["distanciaTotal"] == pytest.approx(1.0), regla


def test_la_dispersion_de_la_nube_viaja_al_lado_de_la_distancia():
    """Sin ella, «recorrio 1.14» no se puede leer."""
    X, bolsas, y, scores = _material_plano()
    _, registros = toy_run.desplazar_bolsas(X, bolsas, y, np.array([0]),
                                            scores, 1.0)
    assert registros[0]["dispersionDeLaNube"] == pytest.approx(
        float(np.linalg.norm(X.std(axis=0))))


# -------------------------------------------------------- los padrones de la seccion 3

def test_hay_un_brazo_por_mecanismo_y_todos_sobre_todas_las_instancias():
    brazos = toy_run.brazos_de_mecanismo()
    assert len(brazos) == len(toy_run.MECANISMOS)
    assert [b["mechanism"] for b in brazos] == list(toy_run.MECANISMOS)
    assert all(b["selection"] is None for b in brazos)
    assert all(b["global"] and b["local"] for b in brazos), (
        "el eje es la agregacion; apagar un termino mezclaria dos secciones")


def test_las_selecciones_son_las_cuatro_reglas_por_cada_presupuesto_declarado():
    brazos = toy_run.brazos_de_seleccion("ours")
    esperados = 1 + (len(toy_run.SELECCIONES) - 1) * len(toy_run.PRESUPUESTOS_K)
    assert len(brazos) == esperados
    sin_k = [b for b in brazos if b["selection"] is None]
    assert len(sin_k) == 1 and "selectK" not in sin_k[0]
    for regla in toy_run.SELECCIONES:
        if regla is None:
            continue
        ks = sorted(b["selectK"] for b in brazos if b["selection"] == regla)
        assert ks == sorted(toy_run.PRESUPUESTOS_K), regla


def test_las_selecciones_se_niegan_sobre_un_mecanismo_que_no_existe():
    with pytest.raises(ValueError):
        toy_run.brazos_de_seleccion("el-que-me-guste")


def test_registrar_no_puede_redefinir_un_id_que_ya_existe():
    brazo = {**toy_run.TOY_ARMS[0], "id": "PRUEBA-REGISTRO"}
    try:
        toy_run._registrar((brazo,))
        toy_run._registrar((brazo,))         # identico: no pasa nada
        with pytest.raises(ValueError) as caido:
            toy_run._registrar(({**brazo, "local": False},))
        assert "PRUEBA-REGISTRO" in str(caido.value)
    finally:
        toy_run.TOY_ARMS_BY_ID.pop("PRUEBA-REGISTRO", None)


def test_registrar_los_padrones_no_mueve_lo_que_mide_la_seccion_uno():
    """`TOY_ADAPTED` es el ultimo de `TOY_ARMS` y la fase 1 entrena ese."""
    antes = len(toy_run.TOY_ARMS), toy_run.TOY_ADAPTED
    toy_run._registrar(toy_run.brazos_de_mecanismo())
    toy_run._registrar(toy_run.brazos_de_seleccion("ours"))
    assert (len(toy_run.TOY_ARMS), toy_run.TOY_ADAPTED) == antes
    assert len(toy_run.TOY_ARMS) == 1
    assert toy_run.TOY_ARMS_BY_ID[toy_run.TOY_ADAPTED] is toy_run.TOY_ARMS[0]


def test_un_brazo_de_mecanismo_construye_con_la_especificacion_del_toyset():
    """Y no con la de `G`, que es la declaracion del banco de digitos."""
    ids = toy_run._registrar(toy_run.brazos_de_mecanismo())
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp-nobn")
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    with toy_run.material_bound(familia, toy_run.PILOT, "mlp-nobn", hyper):
        brazo = toy_run.build_arm(ids[-1], material, toy_run.PILOT, hyper,
                                  torch.device("cpu"), 0)
    assert brazo.mechanism == toy_run.TOY_ARMS_BY_ID[ids[-1]]["mechanism"]
    assert brazo.spec["id"] == ids[-1], "la especificacion es la del padron"
    assert brazo.spec["global"] is True and brazo.spec["local"] is True


# ------------------------------------------------------------- el gancho de lectura

def test_sin_gancho_run_cell_no_lee_nada():
    """Es lo que hace el gancho aditivo: la seccion 1 nunca pasa uno."""
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    celda = toy_run.run_cell(familia, "ninguna", 0, toy_run.PILOT, hyper,
                             torch.device("cpu"), arms=(toy_run.TOY_ADAPTED,))
    assert celda["lecturas"] == {}


def test_el_gancho_corre_una_vez_por_brazo_y_ve_el_modelo_entrenado():
    vistos = []

    def _gancho(model, material, device, arm_id):
        vistos.append(arm_id)
        assert isinstance(model, torch.nn.Module)
        assert set(material) == {"source", "target"}
        return {"visto": arm_id}

    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    celda = toy_run.run_cell(familia, "ninguna", 0, toy_run.PILOT, hyper,
                             torch.device("cpu"), arms=toy_run.ABLACION_IDS,
                             lecturas=_gancho)
    assert vistos == list(toy_run.ABLACION_IDS)
    assert set(celda["lecturas"]) == set(toy_run.ABLACION_IDS)


def test_la_lectura_de_bolsas_se_niega_ante_un_dominio_que_no_existe():
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp-nobn")
    with toy_run.material_bound(familia, toy_run.PILOT, "mlp-nobn", hyper):
        brazo = toy_run.build_arm(toy_run.TOY_ADAPTED, material, toy_run.PILOT,
                                  hyper, torch.device("cpu"), 0)
        with pytest.raises(ValueError):
            toy_run.lectura_de_bolsas(brazo, material["target"],
                                      torch.device("cpu"), "el-de-al-lado")


def test_la_lectura_de_bolsas_devuelve_w_en_cero_uno_y_una_fila_por_bolsa():
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp-nobn")
    with toy_run.material_bound(familia, toy_run.PILOT, "mlp-nobn", hyper):
        brazo = toy_run.build_arm(toy_run.TOY_ADAPTED, material, toy_run.PILOT,
                                  hyper, torch.device("cpu"), 0)
        leido = toy_run.lectura_de_bolsas(brazo, material["target"],
                                          torch.device("cpu"), "target")
    total = int(material["target"].members.shape[0])
    assert leido["scores"].shape == (total, toy_run.PILOT.classes)
    assert leido["w"].shape == (total,)
    assert np.all(leido["w"] >= 0.0) and np.all(leido["w"] <= 1.0)
    assert leido["scores"].sum(axis=1) == pytest.approx(
        np.ones(total), abs=1e-5)


def test_la_masa_se_niega_ante_un_nombre_de_rol_abreviado():
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp-nobn")
    with toy_run.material_bound(familia, toy_run.PILOT, "mlp-nobn", hyper):
        brazo = toy_run.build_arm(toy_run.TOY_ADAPTED, material, toy_run.PILOT,
                                  hyper, torch.device("cpu"), 0)
        with pytest.raises(ValueError):
            toy_run.masa_de_correspondencia(brazo, material["source"],
                                            material["target"],
                                            torch.device("cpu"),
                                            rol_blanco="train")


def test_la_masa_suma_uno_por_bolsa_del_blanco():
    """La Ec. (29) reparte la masa de `g_j` entre los sujetos de cada clase, y
    `g_j` ya suma uno, asi que no hace falta una segunda normalizacion."""
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp-nobn")
    with toy_run.material_bound(familia, toy_run.PILOT, "mlp-nobn", hyper):
        brazo = toy_run.build_arm(toy_run.TOY_ADAPTED, material, toy_run.PILOT,
                                  hyper, torch.device("cpu"), 0)
        leido = toy_run.masa_de_correspondencia(brazo, material["source"],
                                                material["target"],
                                                torch.device("cpu"))
    pi = leido["pi"]
    assert pi.shape == (len(leido["bolsasBlanco"]), len(leido["bolsasFuente"]))
    assert pi.sum(axis=1) == pytest.approx(
        np.ones(pi.shape[0]), abs=1e-4)


# -------------------------------------------------------------- escribir la seccion

def test_guardar_una_seccion_relee_lo_que_escribio(tmp_path):
    barrido = {"seccion": 4, "celdas": {}, "escala": {"name": "x"},
               "valor": np.float32(0.5), "arreglo": np.arange(3)}
    ruta = toy_run.guardar_seccion(tmp_path, barrido)
    assert ruta.name == "seccion4.json"
    vuelta = json.loads(ruta.read_text(encoding="utf-8"))
    assert vuelta["valor"] == pytest.approx(0.5)
    assert vuelta["arreglo"] == [0, 1, 2]


def test_el_escritor_se_niega_ante_un_tipo_que_no_sabe_convertir(tmp_path):
    """Un default que devolviera `str(valor)` dejaria el artefacto verde con
    una cadena donde iba una lista."""
    with pytest.raises(TypeError):
        toy_run.guardar_seccion(tmp_path, {"seccion": 2,
                                           "raro": {1, 2, 3}})

def test_los_cinco_mecanismos_son_UN_brazo_bajo_cinco_pooling():
    """La cerradura que `spec_de_mecanismo_ligado` promete cumplir.

    `tests/test_attention_mechanisms.py` protege, sobre el banco de digitos,
    que una comparacion de mecanismos no se vuelva cinco brazos distintos ---
    y lo hace prohibiendo que `MechanismArm.__init__` reciba otro spec que
    `G`. El toyset liga el spec en vez de agregarle un parametro, asi que esa
    cerradura queda intacta y lo que hay que asegurar aca es lo que ella
    protegia: que los cinco salgan de UNA sola base.
    """
    libres = {"mechanism", "id", "name", "label", "why"}
    base = toy_run.TOY_ARMS[0]
    for brazo in toy_run.brazos_de_mecanismo():
        difieren = {k for k in set(base) | set(brazo)
                    if base.get(k) != brazo.get(k)}
        assert difieren <= libres, (brazo["id"], sorted(difieren - libres))


@pytest.mark.parametrize("mecanismo", list(toy_run.wiring.MECHANISMS))
def test_la_clase_propia_pesa_igual_que_la_de_wiring(mecanismo):
    """La duplicacion que dos cerraduras obligan, clavada por COMPORTAMIENTO.

    `toy_run.BrazoDeMecanismo` copia la asignacion de parametros de
    `wiring.MechanismArm` porque ninguna de las dos vias directas esta abierta
    --- ver el docstring de la clase. Una prueba de forma (que los dos `_params`
    tengan las mismas claves) dejaria pasar una copia que asigna distinto; esta
    exige que los dos devuelvan el MISMO tensor sobre la misma entrada, con la
    misma semilla.
    """
    wiring = toy_run.wiring
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp-nobn")
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    dev = torch.device("cpu")
    with toy_run.material_bound(familia, toy_run.PILOT, "mlp-nobn", hyper):
        fuente = material["source"]
        blanco = material["target"]
        pools = (toy_run.harness.pool_of(fuente, fuente.train_idx, dev),
                 toy_run.harness.pool_of(blanco, blanco.train_idx, dev))
        torch.manual_seed(7)
        de_wiring = wiring.build_mechanism(mecanismo, toy_run.PILOT.classes,
                                           *pools, hyper=hyper)
        torch.manual_seed(7)
        propio = toy_run.BrazoDeMecanismo(mecanismo, toy_run.TOY_ARMS[0],
                                          toy_run.PILOT.classes, *pools,
                                          hyper=hyper)
        H = torch.randn(toy_run.PILOT.instances_per_bag,
                        de_wiring.encoder.output_dim,
                        generator=torch.Generator().manual_seed(11))
        assert sorted(de_wiring._params()) == sorted(propio._params()), mecanismo
        esperado = de_wiring.weights_for(H, de_wiring.sigma)
        obtenido = propio.weights_for(H, propio.sigma)
    assert torch.allclose(esperado, obtenido, atol=0.0), mecanismo


def test_la_clase_propia_no_acepta_un_mecanismo_que_no_existe():
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp-nobn")
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    dev = torch.device("cpu")
    with toy_run.material_bound(familia, toy_run.PILOT, "mlp-nobn", hyper):
        fuente, blanco = material["source"], material["target"]
        with pytest.raises(ValueError):
            toy_run.BrazoDeMecanismo(
                "el-que-me-guste", toy_run.TOY_ARMS[0], toy_run.PILOT.classes,
                toy_run.harness.pool_of(fuente, fuente.train_idx, dev),
                toy_run.harness.pool_of(blanco, blanco.train_idx, dev),
                hyper=hyper)


def test_el_resolvedor_de_codificador_se_niega_sin_un_dominio_declarado():
    class _Falso:
        instance_embeddings = "fuente"
        target_embeddings = "blanco"

    falso = _Falso()
    assert toy_run.codificador_de(falso, "source") == "fuente"
    assert toy_run.codificador_de(falso, "target") == "blanco"
    with pytest.raises(ValueError):
        toy_run.codificador_de(falso, "el-de-al-lado")


# --------------------------------------------- los padrones declarados de la seccion 3

def test_hay_un_solo_presupuesto_y_es_el_que_el_operador_eligio():
    """`k = 10` de 30, elegido ANTES de la corrida completa.

    El numero esta escrito aca y no derivado del modulo: derivarlo haria la
    prueba una tautologia --- diria que `PRESUPUESTOS_K` es igual a si mismo ---
    y lo que hay que sostener es que es EL numero que se declaro.
    """
    assert toy_run.PRESUPUESTOS_K == (10,)
    assert 0 < toy_run.PRESUPUESTOS_K[0] < toy_run.FULL.instances_per_bag


def test_los_pares_de_mecanismo_siguen_la_declaracion_y_no_el_efecto():
    """El orden fijo que CREDA nombra: si siguiera al ganador, un signo siempre
    positivo no diria nada.

    Se asegura contra el orden de `MECANISMOS` y no contra una lista escrita a
    mano: un mecanismo agregado alla entra aca en su lugar, y uno que se
    reordene mueve los pares con el.
    """
    esperado = tuple(("ours", otro) for otro in toy_run.MECANISMOS
                     if otro != "ours")
    assert toy_run.PARES_MECANISMO == esperado
    assert all(a == "ours" for a, _ in toy_run.PARES_MECANISMO), (
        "el brazo propio va primero en todos")
    # Y NO son los diez pares que existen entre los cinco: esa familia nadie
    # la declaro.
    import itertools
    assert len(toy_run.PARES_MECANISMO) < len(
        list(itertools.combinations(toy_run.MECANISMOS, 2)))


def test_los_pares_de_seleccion_dejan_a_las_tres_reglas_con_test():
    assert toy_run.PARES_SELECCION == (("topk", "arbitrary"),
                                       ("topk", "regular"),
                                       ("regular", "arbitrary"))
    nombradas = {r for par in toy_run.PARES_SELECCION for r in par}
    assert nombradas == set(toy_run.SELECCIONES) - {None}


def test_regular_es_determinista_y_arbitrary_no_lo_es_sobre_la_misma_bolsa():
    """La diferencia que el tercer par mide: varianza de sorteo, no cobertura.

    `regular` guarda indices equiespaciados en el orden de la bolsa, asi que
    dos bolsas distintas guardan las MISMAS posiciones; `arbitrary` siembra del
    contenido de la bolsa, asi que dos bolsas distintas guardan otras. Es lo
    que hace a `regular` el control sin ruido de sorteo.
    """
    k = toy_run.PRESUPUESTOS_K[0]
    m = toy_run.PILOT.instances_per_bag
    g = torch.Generator().manual_seed(3)
    H1 = torch.randn(m, 4, generator=g)
    H2 = torch.randn(m, 4, generator=g)

    class _Brazo:
        def __init__(self, regla):
            self.spec = {"selection": regla, "selectK": k}

        def weights_for(self, H, sigma):
            del sigma
            return torch.arange(H.shape[0], dtype=torch.float32)

    reg = _Brazo("regular")
    esperados = torch.tensor([int(i * m / k) for i in range(k)])
    uno = toy_run.wiring.Arm.select(reg, H1, 1.0)
    dos = toy_run.wiring.Arm.select(reg, H2, 1.0)
    assert torch.equal(uno, H1[esperados])
    assert torch.equal(dos, H2[esperados]), "regular guarda las MISMAS posiciones"

    arb = _Brazo("arbitrary")
    a1 = toy_run.wiring.Arm.select(arb, H1, 1.0)
    a2 = toy_run.wiring.Arm.select(arb, H2, 1.0)
    # Las posiciones que eligio cada una, recuperadas por pertenencia.
    pos1 = [i for i in range(m) if any(torch.equal(H1[i], f) for f in a1)]
    pos2 = [i for i in range(m) if any(torch.equal(H2[i], f) for f in a2)]
    assert pos1 != pos2, "arbitrary sortea por bolsa; dos bolsas no coinciden"



# ===========================================================================
# El rediseno del 2026-10-20: la seccion 4 de dos brazos, la 2 contaminada, y
# las dos caminatas construidas.
# ===========================================================================

def test_la_seccion_cuatro_lee_los_brazos_que_la_ablacion_declara():
    """Un padron propio con los mismos dos interruptores seria la misma
    declaracion escrita dos veces, y las dos divergen."""
    assert toy_run.BRAZOS_CORRESPONDENCIA == (toy_run.BRAZO_SOLO_GLOBAL,
                                              toy_run.BRAZO_AMBOS)
    for brazo in toy_run.BRAZOS_CORRESPONDENCIA:
        assert brazo in toy_run.ABLACION_IDS
    # Y el orden importa: la referencia va antes que lo que se compara contra
    # ella, porque el par declarado es `TGL - TG`.
    sin, con = toy_run.BRAZOS_CORRESPONDENCIA
    assert toy_run.TOY_ARMS_BY_ID[sin]["local"] is False
    assert toy_run.TOY_ARMS_BY_ID[con]["local"] is True
    assert toy_run.TOY_ARMS_BY_ID[sin]["global"] is True


def test_los_pasos_del_barrido_nacen_en_el_device_del_tensor():
    """El defecto medido: un `linspace` sin `device=` nace en CPU, y un tensor
    0-dim de CPU multiplicando uno de MPS DEVUELVE EL ORIGINAL sin fallar.

    Las dos caminatas salian perfectamente planas y eso se lee como "el termino
    no responde", que es un resultado, en vez de como un defecto.
    """
    for dtype in (torch.float32, torch.float64):
        como = torch.zeros(3, dtype=dtype)
        pasos = toy_run._pasos_del_barrido(7, como)
        assert pasos.dtype == como.dtype
        assert pasos.device == como.device
        assert len(pasos) == 7
        assert float(pasos[0]) == 0.0 and float(pasos[-1]) == 1.0
    # Y el movimiento que el defecto anulaba: con el ultimo paso, el punto
    # llega al final y no se queda en el arranque.
    inicio, final = torch.zeros(4), torch.ones(4) * 5.0
    pasos = toy_run._pasos_del_barrido(4, inicio)
    movido = inicio + pasos[-1] * (final - inicio)
    assert torch.allclose(movido, final)


def test_las_dos_caminatas_usan_el_mismo_constructor_de_pasos():
    """El arreglo no se puede aplicar a una y olvidar en la otra."""
    import inspect
    for funcion in (toy_run.caminata_de_w, toy_run.caminata_de_correspondencia):
        fuente = inspect.getsource(funcion)
        assert "_pasos_del_barrido(" in fuente, funcion.__name__
        assert "torch.linspace" not in fuente, (
            funcion.__name__,
            "una caminata que arma su propio linspace puede volver a nacer en "
            "CPU; el constructor compartido es lo que lo impide")


def test_el_modo_de_contaminacion_es_uno_de_los_que_creda_declara():
    """`frontera` es inalcanzable aca --- necesita un `w` por PUNTO y `w_j`
    esta definido sobre la bolsa --- asi que la seccion corre `reubicado`."""
    from CREDA.toy_experiment import MODO_REUBICADO, MODOS_CONTAMINACION
    assert toy_run.MODO_SOPESADO in MODOS_CONTAMINACION
    assert toy_run.MODO_SOPESADO == MODO_REUBICADO
    assert 0.0 < toy_run.RHO_SOPESADO < 1.0


def test_los_dos_brazos_del_sopesado_difieren_en_un_solo_interruptor():
    brazos = toy_run.brazos_de_sopesado()
    assert len(brazos) == 2
    sin, con = brazos
    assert sin["weighting"] is False and con["weighting"] is True
    # Todo lo demas igual: la diferencia entre las dos filas es `weighting` y
    # nada mas, o la tabla compara dos cosas a la vez.
    distintas = {k for k in set(sin) | set(con)
                 if sin.get(k) != con.get(k)}
    assert distintas == {"id", "name", "label", "weighting", "why"}, distintas


def test_el_costo_del_sopesado_es_nan_y_no_cero_sin_sopesado():
    """`weighting_cost` devuelve `None` para un brazo sin sopesado: no hay
    costo que medir. Un cero diria "no costo nada", que es una medicion."""
    import inspect
    fuente = inspect.getsource(toy_run.run_cell)
    assert 'float("nan") if cost is None' in fuente
    assert "weightingCostNote" in fuente
    # Y la razon viaja: sin ella un `nan` se lee igual que una medicion fallida.
    assert "weighting: False" in fuente


def test_la_contaminacion_toca_solo_filas_del_train_del_blanco():
    """MEDIDO sobre el material de verdad y no leyendo la fuente.

    La metrica se queda sobre datos que nadie movio, que es la unica razon por
    la que el acierto de la tabla es legible: lo que mide es cuanto dano hacen
    los puntos malos y no cuanto se corrio el conjunto de evaluacion. Una
    perturbacion que tocara tambien el eval daria una curva igual midiendo su
    propia perturbacion.
    """
    familia = toy_run.build_family("lunas", 0, toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp")
    Xn, registro = toy_run.contaminar_el_train_del_blanco(
        familia, material, toy_run.RHO_SOPESADO, 0)

    asignacion = toy_run.bolsas_desde_el_bagset(material["target"],
                                                familia.Xt.shape[0])
    del_train = np.isin(asignacion,
                        material["target"].train_idx.cpu().numpy())
    movidas = ~np.isclose(Xn, np.asarray(familia.Xt)).all(axis=1)

    assert movidas.sum() > 0, "no se movio nada"
    assert not movidas[~del_train].any(), (
        "se movio una fila que NO cae en una bolsa de TRAIN del blanco")
    # Y la fraccion es la declarada, contada sobre las filas del train.
    assert registro["filasMovidas"] == round(
        toy_run.RHO_SOPESADO * registro["filasDelTrain"])
    assert int(movidas.sum()) == registro["filasMovidas"]
    assert registro["modo"] == toy_run.MODO_SOPESADO


def test_contaminar_no_mueve_el_sorteo_de_bolsas():
    """`bags.build` siembra de `(seed, code)` y sortea sobre las ETIQUETAS, asi
    que cambiar coordenadas no mueve ni la permutacion ni los miembros.

    Si se moviera, las filas que se contaminaron ya no serian las del train de
    la corrida que de verdad se mide, y "solo el train del blanco" dejaria de
    ser verdad sin que nada fallara.
    """
    familia = toy_run.build_family("lunas", 0, toy_run.PILOT)
    limpio = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp")
    Xn, _ = toy_run.contaminar_el_train_del_blanco(
        familia, limpio, toy_run.RHO_SOPESADO, 0)
    sucia = toy_run._reemplazar(familia, Xt=Xn)
    sucio = toy_run.material_for(sucia, 0, toy_run.PILOT, "mlp")

    for rol in ("train_idx", "valid_idx", "eval_idx"):
        assert torch.equal(getattr(limpio["target"], rol),
                           getattr(sucio["target"], rol)), rol
    assert torch.equal(limpio["target"].members, sucio["target"].members)
    # Y la fuente no se toco en absoluto.
    assert torch.equal(limpio["source"].images, sucio["source"].images)


def test_el_barrido_del_sopesado_comprueba_que_el_sorteo_no_se_movio():
    """Sin esa comprobacion, "solo el train del blanco" es una intencion."""
    import inspect
    fuente = inspect.getsource(toy_run.barrido_sopesado)
    assert "torch.equal" in fuente
    for rol in ("train_idx", "valid_idx", "eval_idx"):
        assert rol in fuente, rol
    assert "members" in fuente


def test_la_seccion_tres_escribe_su_eje_en_la_fila():
    """Deducirlo de `selection is None` estaba mal y se midio: el brazo
    `S-<ganador>-todas` de la etapa DOS tambien lo lleva en `None`."""
    import inspect
    fuente = inspect.getsource(toy_run._fila_de_agregacion)
    assert '"eje": eje' in fuente
    # Y el brazo que causaba el choque sigue existiendo, o el test no prueba
    # nada: la referencia de la etapa dos ES `selection: None`.
    referencia = [b for b in toy_run.brazos_de_seleccion("ours")
                  if b["selection"] is None]
    assert len(referencia) == 1, referencia


def test_cada_barrido_publica_la_etiqueta_de_sus_brazos():
    """`toy_tables` no puede importar `toy_run` --- lo importa el --- asi que
    el rotulo tiene que viajar con el artefacto o la tabla imprime el id."""
    import inspect
    for barrido in (toy_run.barrido_sopesado, toy_run.barrido_agregacion,
                    toy_run.barrido_correspondencia, toy_run.barrido_ablacion):
        assert '"rotulos"' in inspect.getsource(barrido), barrido.__name__


def test_la_caminata_de_correspondencia_no_materializa_psi_ni_mu():
    """La Ec. (31) se expande ENTERA en el kernel justamente porque `Psi` y
    `mu` no se materializan. Una version que promediaba `(H, w)` para fabricar
    un `Psi` habria dibujado una curva plausible de algo que la formulacion no
    nombra."""
    import inspect
    fuente = inspect.getsource(toy_run.caminata_de_correspondencia)
    assert "K_ss" in fuente and "bag_gram" in fuente
    assert "local_distance(" in fuente
    # La brecha sale de la misma expansion y no de una resta de coordenadas.
    assert "pi @ K_ss @ pi" in fuente
    assert "propio_del_centro" in fuente


def test_las_cuatro_secciones_se_niegan_ante_el_barrido_en_memoria():
    """El mismo objeto tiene TRES nombres y solo uno sirve: `sweep` devuelve
    `rows`, `guardar` escribe `filas`, `cargar` vuelve a `rows`.

    Pasarle el de memoria moria con `KeyError: 'filas'` adentro de
    `mejor_arquitectura`, DESPUES de que el cuaderno ya habia pagado la seccion
    1 entera. La negativa tiene que decir cual de los tres hay que pasar.
    """
    with pytest.raises(KeyError) as excedido:
        toy_run.exigir_artefacto_de_la_seccion_uno({"rows": {}, "escala": {}})
    dicho = str(excedido.value)
    assert "artefacto_seccion_uno" in dicho
    for nombre in ("rows", "filas", "sweep", "guardar", "cargar"):
        assert nombre in dicho, nombre

    # Y la guardia esta en la PUERTA de las cuatro, antes de pagar nada.
    import inspect
    for barrido in (toy_run.barrido_sopesado, toy_run.barrido_agregacion,
                    toy_run.barrido_correspondencia, toy_run.barrido_ablacion):
        cuerpo = inspect.getsource(barrido)
        assert "exigir_artefacto_de_la_seccion_uno(registro)" in cuerpo, \
            barrido.__name__
        # Antes de resolver el dispositivo, que es la primera linea que cuesta.
        assert (cuerpo.index("exigir_artefacto_de_la_seccion_uno")
                < cuerpo.index("harness.resolve_device")), barrido.__name__
