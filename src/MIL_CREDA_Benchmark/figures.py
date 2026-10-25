"""Figures as images a notebook cell actually shows.

`inline` is the one helper the toyset notebooks call. The digit report's
curves and bands, drawn from `runs.jsonl`, were retired with the image
benchmark.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def inline(figure: plt.Figure):
    """La figura como imagen, para que la celda la muestre de verdad.

    `display(figure)` no dibuja nada por sí solo: depende de que el runtime haya
    registrado el formateador de matplotlib, y bajo `Agg` —que es el backend que
    este módulo fija para poder dibujar sin pantalla— no está registrado. La celda
    imprime entonces `<Figure size ...>`, que es otra vez un cuaderno informando
    una descripción en lugar de un resultado, con el agravante de que parece que
    algo se mostró.

    Rasterizar acá no depende de ningún backend ni de ninguna magia del cuaderno,
    así que se ve igual local, en Colab y bajo `nbconvert`. El PNG es la miniatura
    del cuaderno y no un archivo: el único formato que llega al disco sigue siendo
    el PDF que escribe `emit`.
    """
    from io import BytesIO

    from IPython.display import Image

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140)
    return Image(data=buffer.getvalue())
