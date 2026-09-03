"""What the trajectories look like, drawn from the record phase one already wrote.

The tables in the notebook report the minimum, the maximum and the width of each
adaptation term. That answers the question. It is not what a reader looks at, and
a claim about scale — that a quantity stays inside its bounds, that it behaves the
same whichever pair of domains it measures — is seen before it is checked.

Nothing here re-runs anything. Every point comes from `runs.jsonl`, which the
campaign wrote step by step, so the figures describe exactly the runs the tables do.

Every curve is the **median across seeds** with an interquartile band, never one
run's trajectory and never the seeds concatenated. A single trajectory cannot show
whether the shape is the method's or the draw's, and gluing the repetitions end to
end would draw thirty runs as one long one.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MIL_CREDA_Benchmark import config
from MIL_CREDA_Benchmark import pooling


def emit(figure: plt.Figure, path: Path) -> plt.Figure:
    """Escribe la figura y la devuelve, para que la celda la muestre además de guardarla.

    Las dos cosas a la vez, y a propósito. Guardar sin mostrar deja un cuaderno que
    informa un nombre de archivo donde debería haber un resultado: hay que abrir
    otra cosa para ver lo que la celda dice haber producido, y un informe que exige
    abrir otra cosa no está informando. Mostrar sin guardar deja la figura adentro
    del cuaderno y fuera del registro.

    PDF y ningún otro formato. Es vectorial, es lo que ya escribe el trabajo previo
    de este repositorio, y una sola extensión evita el estado en el que conviven dos
    versiones de la misma figura y nadie sabe cuál quedó vieja. La miniatura que se
    ve en el cuaderno no es un archivo: la escribe el propio cuaderno al ejecutarse.
    """
    path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return figure


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


def load_curves(path: Path | None = None) -> dict:
    """{transfer: {arm: [curve_of_seed_0, curve_of_seed_1, ...]}}.

    Grouped by repetition and not flattened. Concatenating them would make a
    figure of thirty seeds indistinguishable from a figure of one run that took
    thirty times as long.
    """
    path = path or (config.RESULTS / "runs.jsonl")
    curves: dict[str, dict[str, list[list[dict]]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            run = json.loads(line)
            curves.setdefault(run["transfer"], {}).setdefault(run["arm"], []).append(
                run["curve"])
    return curves


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    """Median and the two quartiles, by plain interpolation on the sorted values."""
    ordered = sorted(values)
    n = len(ordered)

    def at(fraction: float) -> float:
        if n == 1:
            return ordered[0]
        position = fraction * (n - 1)
        low = int(position)
        high = min(low + 1, n - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return at(0.25), at(0.5), at(0.75)


def band(repetitions: list[list[dict]], key: str) -> tuple[list[float], list[float], list[float]]:
    """The median trajectory of `key` across repetitions, and its interquartile band.

    Repetitions of unequal length are truncated to the shortest, which only happens
    when a run stopped early; extending the short ones would invent steps.
    """
    if not repetitions:
        return [], [], []
    length = min(len(curve) for curve in repetitions)
    low, mid, high = [], [], []
    for step in range(length):
        q1, q2, q3 = _quantiles([curve[step][key] for curve in repetitions])
        low.append(q1)
        mid.append(q2)
        high.append(q3)
    return low, mid, high


def bounded_transfers(available: list[str],
                      requested: list[str] | None = None) -> list[str]:
    """Cuántas transferencias entran en una figura, y quién elige cuáles.

    Dos reglas que se confunden fácil y tienen dueños distintos. **Cuántas** es
    de la figura: seis paneles a un tamaño legible no entran en una página, y
    las tablas ya llevan las seis, así que el tope vive en
    `FIGURE_TRANSFER_COUNT` y se aplica acá. **Cuáles** no es de la figura: es
    `tables.best_transfers`, por el resultado y declarado en el epígrafe. Por eso
    se niega en lugar de recortar --- recortar elegiría por orden de aparición,
    que es una elección disfrazada de detalle de implementación.

    El tope existía en `config` y no lo miraba ninguna figura. Que la rejilla
    latente saliera con tres era una propiedad del cuaderno, que llama a
    `best_transfers`; las tres figuras de curvas del informe salían con seis.

    Una transferencia pedida que el registro no corrió también se niega. Pasa de
    verdad ---la campaña contaminada puede haber corrido menos que la limpia, y
    el cuaderno les pasa la misma lista a las dos--- y dibujar la intersección
    dejaría dos figuras con distinto número de paneles, que el ojo compara como
    si fueran comparables.
    """
    chosen = list(available if requested is None else requested)
    if requested is not None:
        missing = [t for t in requested if t not in available]
        if missing:
            raise ValueError(
                f"el registro no dejó curvas de {', '.join(missing)}, así que la "
                f"figura tendría menos paneles que la que se le compara al lado.")
    if len(chosen) > config.FIGURE_TRANSFER_COUNT:
        raise ValueError(
            f"una figura lleva {config.FIGURE_TRANSFER_COUNT} transferencias "
            f"(`FIGURE_TRANSFER_COUNT`) y se le pasaron {len(chosen)}. Cuáles no "
            f"lo decide la figura: `tables.best_transfers(runs)` las elige por el "
            f"resultado, y el epígrafe lo declara.")
    return chosen


def _panelled(path: Path, arms: tuple[str, ...], key: str, ylabel: str,
              shade_unit: bool = False,
              ylim: tuple[float, float] | None = None,
              runs: Path | None = None,
              transfers: list[str] | None = None) -> plt.Figure:
    """One panel per transfer, one median-with-band per arm. The shape all three share.

    Carries no title and no footer. The heading above the figure already says what
    is being measured and the notebook already states the bounds the run was made
    under; repeating either here is the same measurement in two places, which is
    the failure the duplication rule exists for — in two media instead of one.
    """
    # `runs` explícito, porque `config.RESULTS` es el árbol de la corrida
    # completa y un informe de ensayo dibuja las curvas de OTRA corrida: el
    # cuaderno resuelve de cuál lee y lo pasa. Sin esto la figura salía del
    # árbol equivocado o no salía --- y en un repositorio con las dos, habría
    # salido de la equivocada en silencio.
    curves = load_curves(runs)
    transfers = bounded_transfers(list(curves), transfers)
    columns = min(3, len(transfers)) or 1
    rows = -(-len(transfers) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.2 * rows),
                                squeeze=False, sharey=True)

    # Whether the panels share an x axis is a claim about the data, never a layout
    # preference. The curves are indexed by optimizer step, and a transfer that ran
    # a different number of steps shares nothing with the others — collapsing the
    # axis anyway would draw them as if one ruler governed all of them. So it gets
    # measured, and collapses only if it is true. The y axis needs no measurement:
    # `sharey` makes every panel carry literally the same limits.
    lengths = {len(band(reps, key)[1])
               for transfer in transfers
               for arm in arms
               if (reps := curves[transfer].get(arm))}
    shared_x = len(lengths) <= 1

    # The bottom-most drawn panel of each column, which is where the x tick labels
    # belong. Counting rows would put them on a blank cell whenever the grid does
    # not divide evenly.
    lowest = {}
    for index in range(len(transfers)):
        lowest[index % columns] = index

    for index, transfer in enumerate(transfers):
        column = index % columns
        axis = axes[index // columns][column]
        if shade_unit:
            axis.axhspan(0.0, 1.0, color="0.88", zorder=0,
                         label="[0, 1]" if index == 0 else None)
            axis.axhline(0.0, color="0.6", linewidth=0.8, zorder=1)
        for arm in arms:
            repetitions = curves[transfer].get(arm)
            if not repetitions:
                continue
            low, mid, high = band(repetitions, key)
            steps = range(len(mid))
            line, = axis.plot(steps, mid, linewidth=1.3, zorder=3,
                              label=config.NAME_OF[arm] if index == 0 else None)
            if len(repetitions) > 1:
                axis.fill_between(steps, low, high, alpha=0.18, zorder=2,
                                  color=line.get_color(), linewidth=0)
        # The panel title names the transfer, which is the panel's own identity and
        # not a restatement of anything above the figure.
        axis.set_title(transfer, fontsize=10)
        axis.tick_params(labelsize=8)
        if not shared_x:
            axis.set_xlabel("optimizer step", fontsize=8)
        elif index != lowest[column]:
            axis.tick_params(labelbottom=False)
        if column > 0:
            axis.tick_params(labelleft=False)
        if ylim:
            axis.set_ylim(*ylim)

    for spare in range(len(transfers), rows * columns):
        axes[spare // columns][spare % columns].axis("off")

    # One label for the whole figure instead of one per panel. Decoration that every
    # panel repeats stops being read after the second panel.
    figure.supylabel(ylabel, fontsize=9)

    # La leyenda abajo del todo y la etiqueta del eje encima de ella, cada una en
    # su franja. Por omisión las dos se colocan abajo al centro y se superponen:
    # los nombres de los métodos quedaban escritos encima de «optimizer step».
    figure.legend(loc="lower center", ncol=6, fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, 0.0))
    if shared_x:
        figure.supxlabel("optimizer step", fontsize=9, y=0.055)
    figure.tight_layout(rect=(0, 0.11 if shared_x else 0.06, 1, 1))
    return emit(figure, path)


def adaptation_curves(path: Path,
                      arms: tuple[str, ...] = ("C", "D", "E", "F", "G"),
                      runs: Path | None = None,
                      transfers: list[str] | None = None) -> plt.Figure:
    """Each adaptation term across training, one panel per transfer.

    The shaded band is [0, 1]. Section 5 normalizes MIL-CREDA's terms onto exactly
    that interval, and the prior work's score has no such bound — so whether a curve
    stays inside the band, and whether it occupies the same part of it from one
    transfer to the next, is the claim itself rather than an illustration of it.
    """
    return _panelled(path, arms, "adaptation", "adaptation term",
                     shade_unit=True, runs=runs, transfers=transfers)


def supervised_curves(path: Path,
                      arms: tuple[str, ...] = ("A", "D", "B", "G"),
                      runs: Path | None = None,
                      transfers: list[str] | None = None) -> plt.Figure:
    """The supervised term beside the adaptation one.

    This is where an adaptation term that destabilizes the fit shows up. Reading the
    adaptation curve alone would call a term well-behaved while the classification it
    shares an objective with comes apart underneath it.
    """
    return _panelled(path, arms, "supervised", "supervised term", runs=runs,
                     transfers=transfers)


def contribution_curves(path: Path,
                        arms: tuple[str, ...] = ("C", "D", "E", "F", "G"),
                        runs: Path | None = None,
                      transfers: list[str] | None = None) -> plt.Figure:
    """What share of the objective each declared term actually commands.

    Without this panel, "the term had no effect" and "the term had no weight" are
    the same picture. The coefficient is fixed at `RAMP_CEILING` for every arm, and
    fixing the coefficient does not fix the share: a term whose magnitude differs
    by an order of magnitude between arms is a difference nobody declared, and a
    rung that ignores it credits the mechanism with what the scale did.
    """
    return _panelled(path, arms, "contribution", "lambda x adaptation",
                     runs=runs, transfers=transfers)


def noise_curves(metric: str = "targetAccuracy", path: Path | None = None):
    """Cada método contra la tasa de contaminación, en un solo panel.

    Un panel y no una rejilla: el eje es la tasa, y separar por transferencia
    pondría dos ejes en una figura que afirma ser función de uno. Las
    transferencias ya están colapsadas dentro de cada punto, y `contamination.by_arm`
    nombra ese colapso donde ocurre.

    El azar se dibuja porque es el piso real de la lectura: un método que llega
    ahí dejó de decidir, y sin la línea eso se lee como una caída más entre
    otras. No lleva título — el encabezado de arriba ya dice qué se mide — ni
    repite las cotas, que el sello ya declaró una vez.
    """
    pooling.refuse(metric)
    from MIL_CREDA_Benchmark import config, contamination as noise_axis

    drawn = noise_axis.curve(metric)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    if not drawn["rates"]:
        # Decirlo dentro de la figura y no dibujar ejes vacíos: una figura en
        # blanco se lee como un resultado plano.
        axis.text(0.5, 0.5, "ningún nivel declarado dejó registro todavía",
                  ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return emit(figure, path) if path else figure

    for arm in drawn["arms"]:
        axis.plot(drawn["rates"], drawn["series"][arm], marker="o",
                  label=config.NAME_OF.get(arm, arm))
    axis.axhline(1.0 / config.CLASSES, linestyle=":", linewidth=1, color="0.5")
    axis.annotate(f"azar ({1.0 / config.CLASSES:.2f})",
                  xy=(drawn["rates"][0], 1.0 / config.CLASSES),
                  xytext=(2, 3), textcoords="offset points", fontsize=8, color="0.4")
    axis.set_xlabel("tasa de contaminación ρ")
    axis.set_ylabel(metric)
    axis.set_xticks(drawn["rates"])
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return emit(figure, path) if path else figure
