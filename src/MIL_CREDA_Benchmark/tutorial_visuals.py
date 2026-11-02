"""The two teaching animations of the MIL-CREDA toyset tutorial, in memory.

`MIL-CREDA/Notebooks/Toyset.ipynb` trains one model per family and must not
write anything to disk, so the animations are drawn here from the tutorial's
OWN run and handed back as GIF bytes:

- **The adaptation.** `Trayectoria` is an observer the tutorial's phase-2 loop
  calls at epoch 0 (theta_t still the clone of theta_s) and at the end of each
  sampled epoch. It reads bag positions, decisions and Eq. (24) confidences off
  a deep copy of the model, so the training model -- its weights, BatchNorm
  buffers, train/eval flags and the generator -- is never touched.
- **The confidence walk.** `toy_run.caminata_de_w` walks three target bags
  from a decision boundary to a source class centre and reads `w` through the
  frozen trained head. Its unit is a SWEEP of a constructed path, never an
  optimizer epoch.

Every frame shares one PCA basis fitted once (`toy_run._base_pca`), fixed
limits (`toy_run._limites`) and partial curves. Target labels only colour the "true class" panel and the accuracy
reading; nothing here feeds back into training.
"""

from __future__ import annotations

import copy
import io

import numpy as np
import torch

from CREDA.toy_experiment import COL_DURA
from MIL_CREDA_Benchmark import toy_run

#: Rasterization resolution of every frame: small enough to keep six GIFs
#: embedded in the notebook readable without bloating it.
DPI = 80


def epocas_a_capturar(epocas: int, maximo: int = toy_run.TRAJECTORY_FRAMES
                      ) -> list[int]:
    """Epoch 0 and at most `maximo` evenly spaced epochs up to `epocas`."""
    if epocas <= maximo:
        return list(range(epocas + 1))
    return sorted({int(round(e)) for e in np.linspace(0, epocas, maximo + 1)})


class Trayectoria:
    """Bag-level snapshots of the target domain along phase 2.

    Called as `observar(modelo, epoca)`. Every reading runs on a deep copy of
    the model, so capturing a frame cannot move a BatchNorm buffer, flip a
    train/eval flag or draw from the training generator.
    """

    def __init__(self, material: dict, dispositivo: torch.device,
                 epocas: int, maximo: int = toy_run.TRAJECTORY_FRAMES) -> None:
        self.material = material
        self.dispositivo = dispositivo
        self.objetivo = set(epocas_a_capturar(epocas, maximo))
        self.epocas: list[int] = []
        self.blanco: list[np.ndarray] = []
        self.prediccion_blanco: list[np.ndarray] = []
        self.confianza_blanco: list[np.ndarray] = []
        self.fuente: dict | None = None
        self.clase_blanco: np.ndarray | None = None

    def __call__(self, modelo, epoca: int) -> None:
        if epoca not in self.objetivo:
            return
        sombra = copy.deepcopy(modelo)
        with torch.no_grad():
            blanco = toy_run.bag_positions(sombra, self.material["target"],
                                           self.dispositivo, "target")
            lectura = toy_run.lectura_de_bolsas(sombra, self.material["target"],
                                                self.dispositivo, "target")
            if self.fuente is None:
                self.fuente = toy_run.bag_positions(
                    sombra, self.material["source"], self.dispositivo, "source")
        self.epocas.append(int(epoca))
        self.blanco.append(blanco["positions"])
        self.prediccion_blanco.append(blanco["predictions"])
        self.confianza_blanco.append(lectura["w"])
        self.clase_blanco = blanco["classes"]

    def como_dict(self) -> dict:
        return {"epocas": list(self.epocas),
                "fuente": self.fuente["positions"],
                "claseFuente": self.fuente["classes"],
                "prediccionFuente": self.fuente["predictions"],
                "blanco": list(self.blanco),
                "claseBlanco": self.clase_blanco,
                "prediccionBlanco": list(self.prediccion_blanco),
                "confianzaBlanco": list(self.confianza_blanco)}


def representaciones_del_blanco(modelo, material: dict,
                                dispositivo: torch.device) -> np.ndarray:
    """Eq. (19) for every target bag through theta_t, by bagset index."""
    return toy_run.bag_positions(copy.deepcopy(modelo), material["target"],
                                 dispositivo, "target")["positions"]


# ------------------------------------------------------------------- frames


def _marco(fuente: np.ndarray, otros: list[np.ndarray]) -> dict:
    """One PCA basis fitted on the source and one box for every frame."""
    mu, base, explicada = toy_run._base_pca(np.asarray(fuente, dtype=float))
    proyectar = lambda Z: (np.asarray(Z, dtype=float) - mu) @ base
    puntos = np.vstack([proyectar(fuente)] + [proyectar(o) for o in otros])
    limites, _, cuadrada = toy_run._limites(puntos)
    return {"proyectar": proyectar, "limites": limites, "cuadrada": cuadrada,
            "explicada": explicada}


def _ajustar(ax, marco: dict) -> None:
    ax.set_xlim(*marco["limites"][0])
    ax.set_ylim(*marco["limites"][1])
    if marco["cuadrada"]:
        ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)


def _por_epoca(historia: list[dict], pasos_por_epoca: int, clave: str) -> dict:
    valores = {}
    for fila in historia:
        epoca = (fila["paso"] - 1) // pasos_por_epoca + 1
        valores.setdefault(epoca, []).append(fila[clave])
    return {e: float(np.mean(v)) for e, v in valores.items()}


def figura_de_adaptacion(trayectoria: dict, fase_dos: dict, nombre: str,
                         k: int):
    """Frame `k` of the adaptation: real class, decision + confidence, curves."""
    import matplotlib.pyplot as plt

    marco = _marco(trayectoria["fuente"], trayectoria["blanco"])
    fuente = marco["proyectar"](trayectoria["fuente"])
    blanco = marco["proyectar"](trayectoria["blanco"][k])
    clase_f = np.asarray(trayectoria["claseFuente"])
    clase_b = np.asarray(trayectoria["claseBlanco"])
    pred_f = np.asarray(trayectoria["prediccionFuente"])
    pred_b = np.asarray(trayectoria["prediccionBlanco"][k])
    w_b = np.asarray(trayectoria["confianzaBlanco"][k], dtype=float)
    epocas = trayectoria["epocas"]
    clases = sorted(set(clase_f.tolist()) | set(clase_b.tolist()))

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for ax, por_decision in ((a, False), (b, True)):
        col_f = pred_f if por_decision else clase_f
        col_b = pred_b if por_decision else clase_b
        for clase in clases:
            color = COL_DURA[int(clase)]
            ms, mt = col_f == clase, col_b == clase
            if ms.any():
                ax.scatter(fuente[ms, 0], fuente[ms, 1], marker="x", c=[color],
                           s=36, linewidths=1.2, alpha=0.9, zorder=3)
            if mt.any():
                if por_decision:
                    # Opacity and size follow the bag's confidence w (Eq. 24).
                    rgba = np.tile(np.array(plt.matplotlib.colors.to_rgba(color)),
                                   (int(mt.sum()), 1))
                    rgba[:, 3] = 0.15 + 0.85 * w_b[mt]
                    ax.scatter(blanco[mt, 0], blanco[mt, 1], marker="o",
                               c=rgba, s=20 + 60 * w_b[mt], edgecolors="0.2",
                               linewidths=0.4, zorder=4)
                else:
                    ax.scatter(blanco[mt, 0], blanco[mt, 1], marker="o",
                               c=[color], s=44, edgecolors="0.15",
                               linewidths=0.5, alpha=0.95, zorder=4)
            # Centroids are always of the TRUE class, in both panels.
            vs, vt = clase_f == clase, clase_b == clase
            ax.scatter(*fuente[vs].mean(0), marker="D", s=140,
                       facecolors="none", edgecolors=color, linewidths=2.0,
                       zorder=5)
            ax.scatter(*blanco[vt].mean(0), marker="D", s=95, c=[color],
                       edgecolors="k", linewidths=1.0, zorder=6)
        _ajustar(ax, marco)
    acierto_b = float((pred_b == clase_b).mean())
    a.set_title("color = clase REAL (sólo para mirar)\n"
                "x fuente (θs), o blanco (θt), ◇ centroides", fontsize=9)
    b.set_title(f"color = decisión de la cabeza · opacidad = confianza w\n"
                f"w media del blanco {w_b.mean():.3f} · acierto del blanco "
                f"{acierto_b:.3f}", fontsize=9)

    # Per-epoch curves, partial up to this frame, on fixed axes.
    hasta = epocas[:k + 1]
    w_medias = [float(np.mean(w)) for w in trayectoria["confianzaBlanco"]]
    aciertos = [float((np.asarray(p) == clase_b).mean())
                for p in trayectoria["prediccionBlanco"]]
    c.plot(hasta, w_medias[:k + 1], "o-", color="#9467bd", ms=3.5, lw=1.6,
           label="confianza media del blanco w (Ec. 24)")
    c.plot(hasta, aciertos[:k + 1], "s--", color="#8c564b", ms=3.0, lw=1.2,
           label="acierto del blanco (sólo evaluación)")
    c.set_ylim(-0.02, 1.02)
    c.set_xlim(-0.3, max(epocas[-1], 1) + 0.3)
    c.set_xlabel("época de la fase 2", fontsize=9)
    c.grid(alpha=0.25)
    c.tick_params(labelsize=7)
    objetivo = _por_epoca(fase_dos["historia"], fase_dos["pasosPorEpoca"],
                          "objetivo")
    d = c.twinx()
    xs = [e for e in hasta if e in objetivo]
    d.plot(xs, [objetivo[e] for e in xs], "-", color="0.35", lw=1.0,
           label="objetivo global + local (media por época)")
    tope = max(objetivo.values()) if objetivo else 1.0
    piso = min(0.0, min(objetivo.values())) if objetivo else 0.0
    d.set_ylim(piso, tope * 1.05 if tope > 0 else 1.0)
    d.tick_params(labelsize=7, colors="0.35")
    if epocas[k] >= fase_dos["mejorEpoca"] >= 0:
        c.axvline(fase_dos["mejorEpoca"], color="0.6", ls=":", lw=1.0)
    lineas = c.get_legend_handles_labels()
    otras = d.get_legend_handles_labels()
    c.legend(lineas[0] + otras[0], lineas[1] + otras[1], fontsize=7,
             loc="lower right", framealpha=0.85)
    c.set_title("por época, hasta este cuadro (línea punteada: época elegida)",
                fontsize=9)
    fig.suptitle(f"{nombre} · fase 2 · época {epocas[k]} de {epocas[-1]} · "
                 "fuente, atención y cabeza congeladas; sólo se mueve θt",
                 fontsize=11)
    fig.tight_layout()
    return fig


def figura_de_caminata(caminata: dict, nombre: str, k: int):
    """Sweep `k` of the confidence walk: latent paths and the partial w curve."""
    import matplotlib.pyplot as plt

    nube = np.asarray(caminata["nube"], dtype=float)
    clases = np.asarray(caminata["clasesDeLaNube"], dtype=int)
    pares = list(zip(caminata["arranques"],
                     [str(int(d)) for d in caminata["clasesDestino"]]))
    caminos = [np.asarray(caminata["caminos"][n][d]["camino"], dtype=float)
               for n, d in pares]
    fronteras = [np.asarray(f, dtype=float).reshape(1, -1)
                 for f in (caminata.get("fronteras") or {}).values()]
    marco = _marco(nube, caminos + fronteras)
    total = int(caminata["barridos"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    P = marco["proyectar"](nube)
    for clase in sorted(set(clases.tolist())):
        cual = clases == clase
        a.scatter(P[cual, 0], P[cual, 1], s=16, alpha=0.3,
                  color=COL_DURA.get(int(clase), "#777777"),
                  label=f"bolsas de la fuente, clase {clase}")
    C = marco["proyectar"](caminata["centros"])
    a.scatter(C[:, 0], C[:, 1], s=170, marker="X", c="white",
              edgecolors="black", linewidths=1.5, zorder=5)
    for clase in range(C.shape[0]):
        a.annotate(str(clase), (C[clase, 0], C[clase, 1]), fontsize=8,
                   ha="center", va="center", zorder=6, fontweight="bold")
    for (nombre_arranque, destino), camino in zip(pares, caminos):
        color = COL_DURA.get(int(destino), "#777777")
        trazo = "-" if nombre_arranque != "frontera-triple" else "--"
        frontera = (caminata.get("fronteras") or {}).get(nombre_arranque)
        if frontera is not None:
            f = marco["proyectar"](np.asarray(frontera).reshape(1, -1))
            a.plot([f[0, 0]], [f[0, 1]], "+", ms=10, mew=1.3, color=color,
                   alpha=0.75, zorder=5)
        Q = marco["proyectar"](camino)
        a.plot(Q[:k + 1, 0], Q[:k + 1, 1], color=color, ls=trazo, lw=2.0,
               zorder=4)
        a.plot([Q[0, 0]], [Q[0, 1]], "o", ms=7, color="white",
               markeredgecolor=color, markeredgewidth=1.8, zorder=6)
        a.plot([Q[k, 0]], [Q[k, 1]], "o", ms=9, color=color, zorder=7)
        w = np.asarray(caminata["caminos"][nombre_arranque][destino]["w"])
        b.plot(range(k + 1), w[:k + 1], color=color, ls=trazo, lw=2.0,
               marker="o", ms=3.2,
               label=f"bolsa del blanco más indecisa en {nombre_arranque} "
                     f"→ centro de la clase {destino}")
        b.plot([k], [w[k]], "o", ms=8, color=color)
    _ajustar(a, marco)
    a.set_title(f"{nombre} · representaciones de BOLSA (PCA de la fuente, "
                f"{100 * marco['explicada']:.0f}%)\n○ arranque · + frontera "
                "exacta de la cabeza · X centro de clase de la fuente",
                fontsize=9)
    a.legend(fontsize=7, loc="best", framealpha=0.85)
    b.set_xlim(-0.5, total - 0.5)
    b.set_ylim(-0.03, 1.03)
    b.axhline(0.0, color="#999999", lw=0.8, ls=":")
    b.axhline(1.0, color="#999999", lw=0.8, ls=":")
    b.set_xlabel("barrido del camino construido (distancia recorrida, "
                 "no tiempo de entrenamiento)", fontsize=9)
    b.set_ylabel(r"$w_j = 1 - \hat H_2(g_j)/\ln C$", fontsize=9)
    b.set_title(f"barrido {k} de {total - 1} · w leído con la cabeza "
                "entrenada y congelada", fontsize=9)
    b.legend(fontsize=7, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- GIFs


def _rasterizar(fig) -> np.ndarray:
    fig.set_dpi(DPI)
    return toy_run._rasterizar(fig)


def gif_en_memoria(cuadros: list[np.ndarray], ms: int = toy_run.MS_FRAME,
                   pausa: int = toy_run.MS_PAUSE) -> bytes:
    """Encode RGB frames as an animated GIF and read back the frame count."""
    from PIL import Image

    if len(cuadros) < 2:
        raise ValueError(f"una animación necesita al menos dos cuadros, "
                         f"hay {len(cuadros)}")
    imagenes = [Image.fromarray(c).convert("P", palette=Image.ADAPTIVE)
                for c in cuadros]
    duraciones = [ms] * (len(imagenes) - 1) + [pausa]
    salida = io.BytesIO()
    imagenes[0].save(salida, format="GIF", save_all=True,
                     append_images=imagenes[1:], duration=duraciones, loop=0)
    datos = salida.getvalue()
    with Image.open(io.BytesIO(datos)) as leido:
        if leido.n_frames != len(cuadros):
            raise ValueError(f"el GIF quedó con {leido.n_frames} cuadros y se "
                             f"le pasaron {len(cuadros)}")
    return datos


def gif_de_adaptacion(trayectoria: dict, fase_dos: dict, nombre: str) -> bytes:
    return gif_en_memoria([
        _rasterizar(figura_de_adaptacion(trayectoria, fase_dos, nombre, k))
        for k in range(len(trayectoria["epocas"]))])


def gif_de_caminata(caminata: dict, nombre: str) -> bytes:
    return gif_en_memoria([
        _rasterizar(figura_de_caminata(caminata, nombre, k))
        for k in range(int(caminata["barridos"]))])
