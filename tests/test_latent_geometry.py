"""La geometría de fase dos se mide donde el método alinea, no donde se dibuja.

El acuerdo lo dice y el código no lo hacía: `latent.geometry` restaba centroides
con `torch.norm`, que es la distancia euclidiana sobre el embedding de la
Ec. (19). El método no alinea ahí. Alinea en el RKHS que induce el kernel de la
Ec. (14), donde una clase se representa por la Ec. (17) y dos representaciones
se comparan por la Ec. (18).

Las dos lecturas se ven iguales impresas ---un número por celda, más chico es
mejor--- y responden preguntas distintas: una mide qué tan cerca quedaron dos
nubes en un espacio que el método nunca usó.

Nada acá recompone la matemática: el valor esperado sale de llamar a
`MIL_CREDA.bag_kernel.bag_kernel`, que es el módulo que implementa la Ec. (18) y
declara su provenance. Una prueba que escribiera la doble suma por su cuenta
compararía dos implementaciones mías y no diría nada sobre la del método.
"""

from __future__ import annotations

import math

import pytest
import torch

from MIL_CREDA.bag_kernel import bag_kernel
from MIL_CREDA_Benchmark import config, latent


def _dominios(desplazamiento: float = 0.3, clases: int = 3, por_clase: int = 8):
    """Dos dominios con las mismas clases, uno corrido respecto del otro."""
    torch.manual_seed(0)
    filas_f = torch.cat([torch.randn(por_clase, 4) + c for c in range(clases)])
    filas_d = torch.cat([torch.randn(por_clase, 4) + c + desplazamiento
                         for c in range(clases)])
    etiquetas = torch.tensor([c for c in range(clases) for _ in range(por_clase)])
    return filas_f, etiquetas, filas_d, etiquetas.clone()


def _distancia_rkhs(A: torch.Tensor, B: torch.Tensor, sigma) -> float:
    """d = 1 - K_AB / sqrt(K_AA * K_BB), con el kernel del propio método."""
    peso_a = torch.full((len(A),), 1.0 / len(A), dtype=A.dtype)
    peso_b = torch.full((len(B),), 1.0 / len(B), dtype=B.dtype)
    k_ab = float(bag_kernel(A, peso_a, B, peso_b, sigma))
    k_aa = float(bag_kernel(A, peso_a, A, peso_a, sigma))
    k_bb = float(bag_kernel(B, peso_b, B, peso_b, sigma))
    return 1.0 - k_ab / math.sqrt(k_aa * k_bb)


def _esperado(filas_f, etiquetas_f, filas_d, etiquetas_d):
    sigma = config.KERNEL_SIGMA
    por_clase_f = {int(c): filas_f[etiquetas_f == c] for c in etiquetas_f.unique()}
    por_clase_d = {int(c): filas_d[etiquetas_d == c] for c in etiquetas_d.unique()}
    compartidas = sorted(set(por_clase_f) & set(por_clase_d))

    cruce = [_distancia_rkhs(por_clase_f[c], por_clase_d[c], sigma)
             for c in compartidas]

    def entre(por_clase):
        claves = sorted(por_clase)
        return [_distancia_rkhs(por_clase[a], por_clase[b], sigma)
                for i, a in enumerate(claves) for b in claves[i + 1:]]

    aparte = entre(por_clase_f) + entre(por_clase_d)
    return (sum(cruce) / len(cruce), sum(aparte) / len(aparte))


def test_the_reading_is_the_kernel_distance_the_method_aligns_in() -> None:
    """El polo exacto: los tres números salen de la Ec. (18), no de `torch.norm`.

    Rojo alcanzable: volver a restar centroides. Los dos valores existen y
    difieren, así que la prueba no puede pasar por coincidencia --- lo afirma la
    prueba de abajo, que es su complemento.
    """
    argumentos = _dominios()
    leido = latent.geometry(*argumentos)
    cruce, aparte = _esperado(*argumentos)

    assert leido["crossDomainSameClass"] == pytest.approx(cruce, rel=1e-9)
    assert leido["betweenClasses"] == pytest.approx(aparte, rel=1e-9)
    assert leido["ratio"] == pytest.approx(cruce / aparte, rel=1e-9)


def test_the_kernel_reading_and_the_euclidean_one_are_not_the_same_number() -> None:
    """Sin esto, la prueba de arriba pasaría igual si nada hubiera cambiado.

    Es el complemento: que la lectura sea del kernel sólo significa algo si la
    euclidiana ---que sigue en el registro--- da distinto. Si dieran igual, no
    habría forma de saber cuál de las dos se está leyendo.
    """
    leido = latent.geometry(*_dominios())

    assert "euclidean" in leido, "las distancias crudas se van del registro"
    for clave in ("crossDomainSameClass", "betweenClasses", "ratio"):
        assert leido["euclidean"][clave] != pytest.approx(leido[clave], rel=1e-6), \
            f"`{clave}`: la lectura del kernel y la euclidiana coinciden"


def test_the_kernel_reading_is_bounded_to_the_unit_interval() -> None:
    """Cauchy-Schwarz sobre un kernel PSD, comprobado y no citado.

    Puede fallar de verdad: una raíz sobre el producto equivocado o un sigma
    tomado sobre otro material se salen del intervalo, y las dos mutaciones lo
    muestran.

    Lo que NO lo saca es que los pesos no sumen uno, y vale escribirlo porque es
    contraintuitivo: multiplicar un vector de pesos por una constante multiplica
    `K_AB` por ella y `K_AA` por su cuadrado, así que el cociente ---y con él la
    distancia--- no se mueve. Medido, no razonado: con pesos `1/n` y con pesos
    `1` da el mismo número a dieciséis dígitos. Lo que sí carga el peso de la
    Ec. (17) es que sean **uniformes entre instancias**; una masa concentrada en
    una sola instancia cambia la lectura y la prueba de arriba se pone roja.
    """
    for desplazamiento in (0.0, 0.3, 5.0):
        leido = latent.geometry(*_dominios(desplazamiento))
        for clave in ("crossDomainSameClass", "betweenClasses"):
            assert 0.0 <= leido[clave] <= 1.0, (desplazamiento, clave, leido[clave])


def test_the_raw_distances_are_kept_and_never_declared_as_a_dimension() -> None:
    """Quedan en el registro y no se dibujan: la declaración no las nombra.

    Es la mitad que un `assert` sobre la lectura no cubre. Guardarlas está bien;
    lo que el acuerdo prohíbe es que se rendericen, y lo que decide eso es que
    no sean una dimensión declarada.
    """
    from MIL_CREDA_Benchmark import __benchmark__

    dimensiones = __benchmark__["report"]["dimensions"]
    for clave in dimensiones:
        assert not clave.startswith("geometry.euclidean"), clave
    assert "geometry.ratio" in dimensiones, \
        "la lectura sí se dibuja, y ahora es la del kernel"


# ------------------------------- las bolsas fuente más cercanas, y su clase

def _referencia(kernel: list[list[float]], clases_fuente: list[int],
                clases_destino: list[int]) -> dict:
    """La forma que `bag_pairs` deja y `top_k_source_bags` lee, escrita a mano.

    Sólo los tres campos que este lector toca. El kernel se escribe en vez de
    computarse porque lo que se mide acá es qué sale por vecino, no la Ec. (18):
    esa ya la miden las pruebas de arriba, contra el módulo del método.
    """
    return {"kernel": torch.tensor(kernel),
            "sourceLabels": torch.tensor(clases_fuente),
            "targetLabels": torch.tensor(clases_destino)}


def test_every_neighbour_carries_the_source_bags_own_class() -> None:
    """Lo que la tabla 5d muestra por vecino es la clase de ESA bolsa fuente.

    No la del destino, que ya está en su propia columna, y no la comparación
    entre las dos, que es `trueClass` y la usa la conclusión. La distinción se
    ve sólo con clases que no coinciden: con todas iguales, `sourceLabel` y
    `targetLabel` imprimen lo mismo y la prueba no separaría un lector del otro.

    Rojo alcanzable: escribir `targetLabel` en `sourceLabel`, o leer
    `source_labels[rank]` en vez de `source_labels[source_index]`.
    """
    referencia = _referencia(
        kernel=[[0.1, 0.9],      # bolsa fuente 0, clase 2
                [0.8, 0.2],      # bolsa fuente 1, clase 0
                [0.5, 0.4]],     # bolsa fuente 2, clase 1
        clases_fuente=[2, 0, 1],
        clases_destino=[0, 1])

    filas = latent.top_k_source_bags(referencia, k=3)
    clases = {0: 2, 1: 0, 2: 1}

    for fila in filas:
        assert [n["sourceBag"] for n in fila["neighbours"]] == \
            sorted((n["sourceBag"] for n in fila["neighbours"]),
                   key=lambda b: -referencia["kernel"][b][fila["targetBag"]]), \
            "el orden dejó de ser el del kernel"
        for vecino in fila["neighbours"]:
            assert vecino["sourceLabel"] == clases[vecino["sourceBag"]], (
                f"el vecino {vecino['sourceBag']} se declaró de clase "
                f"{vecino['sourceLabel']} y es de la {clases[vecino['sourceBag']]}")
            assert vecino["trueClass"] == (
                vecino["sourceLabel"] == fila["targetLabel"]), \
                "la comparación ya hecha dejó de ser la de esas dos clases"


def test_the_neighbour_table_prints_the_class_and_not_the_kernel_value() -> None:
    """La mitad del documento: lo que queda entre paréntesis es la clase.

    El valor del kernel sigue calculándose ---fija el orden--- y deja de
    imprimirse: cinco números por fila son veinticinco por pantalla que nadie
    compara. Un lector compara clases.

    Rojo alcanzable: volver a `f"{n['sourceBag']} ({n['kernel']:.3f}, ✓)"`, o
    dejar el encabezado viejo mientras la celda cambia.
    """
    from MIL_CREDA_Benchmark import tables

    referencia = _referencia(kernel=[[0.796], [0.795]],
                             clases_fuente=[3, 1], clases_destino=[3])
    filas = latent.top_k_source_bags(referencia, k=2)
    rendered = tables.render_bag_neighbors(filas, markdown=True)

    assert "Vecinos fuente (clase)" in rendered, \
        "el encabezado sigue prometiendo el kernel"
    assert "0 (3), 1 (1)" in rendered, rendered
    assert "0.796" not in rendered and "0.795" not in rendered, \
        "el valor del kernel volvió a la fila"
    assert "✓" not in rendered and "✗" not in rendered, \
        "la marca de coincidencia volvió a la fila"
