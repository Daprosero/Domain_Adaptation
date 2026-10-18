#!/usr/bin/env python3
"""Assemble the "Ramas hacia CREDA" page from its sections.

Writes two outputs next to this folder:
  ../ramas-hacia-creda.html     standalone file (doctype + charset, opens in a browser)
  publish/ramas-hacia-creda.html  body-only version for publishing as a claude.ai artifact
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTIONS = [
    "01-encabezado-y-estilos.html",  # <title>, styles, MathJax, header, tabs, tree panel
    "02-embebimiento.html",          # tab: Del embebimiento a CREDA
    "03-matematica.html",            # tab: Matemática
    "04-comparacion.html",           # tab: Comparación final
    "05-arbol-script.html",          # tree data, rendering and tab switching
]
PREFIX = ('<!doctype html>\n<meta charset="utf-8">\n'
          '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')

body = "".join((HERE / name).read_text(encoding="utf-8") for name in SECTIONS)
(HERE / "publish").mkdir(exist_ok=True)
(HERE / "publish" / "ramas-hacia-creda.html").write_text(body, encoding="utf-8")
(HERE.parent / "ramas-hacia-creda.html").write_text(PREFIX + body, encoding="utf-8")
print("built", len(body), "chars from", len(SECTIONS), "sections")
