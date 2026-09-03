"""La geometría de fase dos se mide donde el método alinea, no donde se dibuja.

El acuerdo lo dice y el código no lo hacía: `latent.geometry` restaba centroides
con `torch.norm`, que es la distancia euclidiana sobre el embedding de la
Ec. (16). El método no alinea ahí. Alinea en el RKHS que induce el kernel de la
Ec. (19), donde una clase se representa por la Ec. (20) y dos representaciones
se comparan por la Ec. (21).

Las dos lecturas se ven iguales impresas ---un número por celda, más chico es
mejor--- y responden preguntas distintas: una mide qué tan cerca quedaron dos
nubes en un espacio que el método nunca usó.

Nada acá recompone la matemática: el valor esperado sale de llamar a
`MIL_CREDA.bag_kernel.bag_kernel`, que es el módulo que implementa la Ec. (21) y
declara su provenance. Una prueba que escribiera la doble suma por su cuenta
compararía dos implementaciones mías y no diría nada sobre la del método.
"""

from __future__ import annotations

import math

import pytest
import torch

from MIL_CREDA.bag_kernel import bag_kernel
from MIL_CREDA_Benchmark import latent, wiring


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
    sigma = wiring._median_sigma(torch.cat([filas_f, filas_d]))
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
    """El polo exacto: los tres números salen de la Ec. (21), no de `torch.norm`.

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
    Ec. (20) es que sean **uniformes entre instancias**; una masa concentrada en
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
