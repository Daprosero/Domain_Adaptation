"""El guarda que vigila el árbol de producto del dueño.

Nivel 1 --- invariantes del propio guarda. No corre ninguna campaña: lo que acá
se afirma es qué diferencias sabe VER `conftest._product_fingerprint`, porque un
guarda que mira poco y uno apagado dejan exactamente la misma suite verde.
"""

from __future__ import annotations

import os

import conftest
from MIL_CREDA_Benchmark import config


def test_la_huella_ve_una_reescritura_que_no_cambia_ningun_tamano(
        tmp_path, monkeypatch) -> None:
    """Un archivo reescrito con sus mismos bytes es la forma ORDINARIA de un
    pisón: `runs.jsonl` se abre en `"w"` y otra corrida de la misma grilla deja
    el mismo tamaño. El conjunto de rutas no se movió y el tamaño tampoco, así
    que lo único que queda es la fecha.

    Rojo alcanzable: sacar `st_mtime_ns` de la huella y dejar sólo el tamaño ---
    la suite entera sigue verde y este pisón deja de verse.
    """
    monkeypatch.setattr(conftest, "_PRODUCT_TREE", tmp_path)
    marca = tmp_path / "runs.jsonl"
    marca.write_bytes(b"")

    antes = conftest._product_fingerprint()
    # `os.utime` y no otra escritura: lo que se afirma es que la huella LEE la
    # fecha, y dos escrituras seguidas podrían caer en el mismo nanosegundo y
    # volver este test verde por el reloj y no por el guarda.
    fecha = marca.stat().st_mtime_ns
    os.utime(marca, ns=(fecha + 1_000_000, fecha + 1_000_000))

    assert marca.stat().st_size == 0, "el tamaño no se movió: sólo cambió la fecha"
    assert conftest._product_fingerprint() != antes


def test_la_huella_cuenta_los_directorios_y_no_solo_los_archivos(
        tmp_path, monkeypatch) -> None:
    """`campaign()` hace sus dos `mkdir` ANTES de sus tres rechazos, así que lo
    único que una campaña rechazada llega a escribir es un directorio vacío. Una
    huella de archivos solos pasaría de largo justo por ahí.

    Rojo alcanzable: no registrar las entradas de directorio.
    """
    monkeypatch.setattr(conftest, "_PRODUCT_TREE", tmp_path)
    antes = conftest._product_fingerprint()

    (tmp_path / "Benchmark").mkdir()

    despues = conftest._product_fingerprint()
    assert despues != antes
    assert "Benchmark" in despues and "Benchmark" not in antes


def test_el_arbol_vigilado_es_el_real_y_no_el_que_el_test_redirigio(
        tmp_path, monkeypatch) -> None:
    """El guarda mira el árbol del dueño incluso mientras el test bajo
    observación redirige `config.PRODUCT`, que es lo que casi todos hacen. Si lo
    preguntara adentro del fixture leería el scratch, y un guarda que vigila el
    scratch no vigila nada.

    Rojo alcanzable: reemplazar `_PRODUCT_TREE` por una lectura de
    `config.PRODUCT` adentro de `_product_fingerprint`.
    """
    real = conftest._PRODUCT_TREE
    monkeypatch.setattr(config, "PRODUCT", tmp_path)

    assert conftest._PRODUCT_TREE == real != tmp_path
    # Y la huella sigue saliendo del árbol real: el scratch está vacío, el real
    # no, así que una huella vacía sería la del scratch.
    assert conftest._product_fingerprint()
