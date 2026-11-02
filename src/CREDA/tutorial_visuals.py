"""The two animations of the CREDA toyset tutorial, drawn in memory.

`CREDA/Notebooks/Toyset.ipynb` trains CREDA in front of the reader and then
shows two things a still figure cannot:

* **the adaptation GIF**: snapshots of the REAL training trajectory --- the
  model as it was at the start and after selected epochs --- projected on one
  fixed PCA basis, so what moves between frames is the latent and not the axes;
* **the confidence walk GIF**: Eq. 24's weight `w = 1 - H2(g)/ln C`, the one
  `CREDALoss` uses, read by the trained head along constructed segments that
  start on a decision boundary and end at a class centre of the source latent.

Nothing here trains, and nothing here writes a file: every GIF comes back as
`bytes`, for `IPython.display.Image(data=..., format="gif")`. The boundary
points and the walk are the production `puntos_de_frontera` and
`caminata_al_centro`; the confidence is `MIL_CREDA.confidence.confidences`,
the function `CREDALoss.forward` calls. Labels enter only to colour points and
to measure accuracy on the drawn sample; no function here is on a training path.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from CREDA.toy_experiment import CLASES, COL_DURA, caminata_al_centro, puntos_de_frontera
from MIL_CREDA.confidence import confidences


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        "in-memory observations of the tutorial's own training; every "
        "quantity drawn is computed by `CREDA.toy_experiment` or "
        "`MIL_CREDA.confidence`, so no method equation is implemented here"),
    "invariants": [
        "every_family_has_an_animated_adaptation_gif",
        "snapshots_run_from_the_start_to_the_final_epoch",
        "the_last_snapshot_is_the_final_model",
        "capturing_snapshots_does_not_change_training",
        "a_snapshot_restores_modes_and_leaves_buffers_and_rng_alone",
        "every_family_has_an_animated_confidence_walk",
        "the_walk_reads_w_through_the_trained_head_and_creda_confidence",
    ],
}

#: Milliseconds per frame and for the final pause of each GIF.
MS_FRAME, MS_WALK_FRAME, MS_PAUSE = 600, 140, 2500

#: The three walks of the original figure: two double boundaries and the
#: triple one, each heading to the centre of one class.
WALK_STARTS = (((0, 1), 0), ((1, 2), 2), ((0, 1, 2), 1))


# --------------------------------------------------------------------------
# Capturing the trajectory


def capture_snapshot(encoder_s, encoder_t, head, xs, xt, *, epoch):
    """Features, predictions and confidences of a fixed sample, without side effects.

    The modules go to `eval()` so BatchNorm reads its running statistics
    instead of updating them, everything runs under `no_grad`, nothing draws
    a random number, and every module gets back the mode it had. No labels are
    read: they are attached later, only to draw.
    """
    modules = (encoder_s, encoder_t, head)
    modes = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            f_s, f_t = encoder_s(xs), encoder_t(xt)
            g_s = F.softmax(head(f_s), dim=1)
            g_t = F.softmax(head(f_t), dim=1)
            w_s, w_t = confidences(g_s), confidences(g_t)
    finally:
        for module, mode in zip(modules, modes):
            module.train(mode)
    return dict(
        epoch=int(epoch),
        f_s=f_s.numpy().copy(), f_t=f_t.numpy().copy(),
        pred_s=g_s.argmax(1).numpy(), pred_t=g_t.argmax(1).numpy(),
        w_s=w_s.numpy().copy(), w_t=w_t.numpy().copy(),
    )


def snapshot_epochs(epochs, frames):
    """Which completed-epoch counts to keep: about `frames`, always the last."""
    if frames <= 0 or epochs <= 0:
        return set()
    kept = np.linspace(1, epochs, min(int(frames), int(epochs))).round().astype(int)
    return set(int(e) for e in kept) | {int(epochs)}


class TrajectoryRecorder:
    """Keeps bounded snapshots of a training run: the start and selected epochs.

    `start()` captures the model before the first update; calling the recorder
    with a zero-based `epoch` at the end of that epoch captures it when it is
    one of the kept ones. `snapshots[k]["epoch"]` counts completed epochs.
    With `frames=0` it keeps nothing at all.
    """

    def __init__(self, encoder_s, encoder_t, head, xs, xt, *, epochs, frames):
        self.modules = (encoder_s, encoder_t, head)
        self.xs, self.xt = xs, xt
        self.epochs = int(epochs)
        self.keep = snapshot_epochs(epochs, frames)
        self.snapshots = []

    def _capture(self, completed):
        self.snapshots.append(
            capture_snapshot(*self.modules, self.xs, self.xt, epoch=completed))

    def start(self):
        if self.keep:
            self._capture(0)

    def __call__(self, epoch):
        if epoch + 1 in self.keep:
            self._capture(epoch + 1)


# --------------------------------------------------------------------------
# Encoding


def figure_to_rgb(figure, dpi):
    figure.set_dpi(dpi)
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return pixels


def encode_gif(frames, ms_frame, ms_pause=MS_PAUSE):
    """RGB arrays to GIF bytes, read back to prove every frame survived.

    The pause goes on the last frame's duration and not as repeated frames:
    the encoder merges identical consecutive frames.
    """
    assert frames, "no frames to encode"
    images = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE)
              for f in frames]
    durations = [ms_frame] * (len(images) - 1) + [ms_pause]
    buffer = io.BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:],
                   duration=durations, loop=0)
    data = buffer.getvalue()
    with Image.open(io.BytesIO(data)) as check:
        assert check.n_frames == len(images), (check.n_frames, len(images))
    return data


# --------------------------------------------------------------------------
# The adaptation GIF


def _limits(points):
    low, high = np.percentile(points, [0.5, 99.5], axis=0)
    pad = 0.08 * (high - low) + 1e-6
    return (low[0] - pad[0], high[0] + pad[0]), (low[1] - pad[1], high[1] + pad[1])


def _adaptation_frame(name, snapshot, project, limits, ys, yt, curves, epochs,
                      version, dpi):
    xlim, ylim = limits
    ps, pt = project(snapshot["f_s"]), project(snapshot["f_t"])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    for c in range(CLASES):
        axes[0].scatter(*ps[ys == c].T, marker="x", s=10, lw=0.7, c=COL_DURA[c], alpha=0.5)
        axes[0].scatter(*pt[yt == c].T, marker="o", s=10, c=COL_DURA[c], alpha=0.5,
                        edgecolors="none")
        if (ys == c).any():
            axes[0].scatter(*ps[ys == c].mean(0), marker="D", s=110, facecolors="none",
                            edgecolors=COL_DURA[c], linewidths=2.2, zorder=5)
        if (yt == c).any():
            axes[0].scatter(*pt[yt == c].mean(0), marker="D", s=110, c=COL_DURA[c],
                            edgecolors="k", linewidths=0.8, zorder=5)
    axes[0].set_title("color = clase REAL · ◇ centro fuente · ◆ centro blanco",
                      fontsize=10)
    size = lambda w: 4 + 46 * np.clip(w, 0.0, 1.0)
    colours = np.array([COL_DURA[c] for c in range(CLASES)])
    axes[1].scatter(*ps.T, marker="x", s=size(snapshot["w_s"]), lw=0.7,
                    c=colours[snapshot["pred_s"]], alpha=0.5)
    axes[1].scatter(*pt.T, marker="o", s=size(snapshot["w_t"]),
                    c=colours[snapshot["pred_t"]], alpha=0.5, edgecolors="none")
    axes[1].set_title("color = PREDICCIÓN de la cabeza · tamaño = confianza w",
                      fontsize=10)
    for axis in axes[:2]:
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    shown = [e <= snapshot["epoch"] for e in curves["epoch"]]
    x = np.asarray(curves["epoch"])[shown]
    for key, label, style in (("acc_s", "acierto fuente", "-"),
                              ("acc_t", "acierto blanco", "-"),
                              ("w_t", "w medio del blanco", "--")):
        axes[2].plot(x, np.asarray(curves[key])[shown], style, marker="o", ms=3,
                     label=label)
    axes[2].axhline(1 / CLASES, color="0.6", lw=0.8, ls=":", label="azar")
    axes[2].set_xlim(0, epochs)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_xlabel("épocas completadas")
    axes[2].legend(loc="lower right", fontsize=8)
    axes[2].grid(alpha=0.25)
    axes[2].set_title("sobre los puntos dibujados (test)", fontsize=10)

    moves = ("fuente y cabeza QUIETAS: solo se mueve el encoder del blanco"
             if version == "v2" else
             "un solo encoder y una sola cabeza: se mueven los dos dominios y la cabeza")
    figure.suptitle(f"{name} · CREDA {version} · época {snapshot['epoch']} de {epochs}"
                    f" · {moves}", fontsize=11.5)
    figure.text(0.5, 0.01, "`x` fuente, `o` blanco · proyección fija: PCA del latente "
                "de la fuente en el primer cuadro · las etiquetas solo pintan y miden",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    return figure_to_rgb(figure, dpi)


def adaptation_gif(name, snapshots, ys, yt, *, version, epochs, dpi=64):
    """One frame per snapshot, all on the first snapshot's source PCA basis."""
    assert len(snapshots) > 1, "an animation needs at least two snapshots"
    ys, yt = np.asarray(ys), np.asarray(yt)
    reference = snapshots[0]["f_s"]
    mean = reference.mean(0)
    _, _, vt = np.linalg.svd(reference - mean, full_matrices=False)
    basis = vt[:2].T
    project = lambda z: (np.asarray(z) - mean) @ basis
    everything = np.concatenate(
        [project(s[key]) for s in snapshots for key in ("f_s", "f_t")])
    limits = _limits(everything)
    curves = dict(
        epoch=[s["epoch"] for s in snapshots],
        acc_s=[float((s["pred_s"] == ys).mean()) for s in snapshots],
        acc_t=[float((s["pred_t"] == yt).mean()) for s in snapshots],
        w_t=[float(s["w_t"].mean()) for s in snapshots],
    )
    frames = [_adaptation_frame(name, s, project, limits, ys, yt, curves, epochs,
                                version, dpi) for s in snapshots]
    return encode_gif(frames, MS_FRAME)


# --------------------------------------------------------------------------
# The confidence walk


def confidence_walks(latent, ys, head, steps):
    """The three constructed walks, read by the trained `head`.

    Each starts where the named classes tie and the others weigh nothing
    (`puntos_de_frontera`, solved exactly on the linear head) and goes in a
    straight line to the centre of one class of the source latent;
    `caminata_al_centro` reads `w` with the confidence `CREDALoss` uses.
    """
    latent = np.asarray(latent, dtype=float)
    ys = np.asarray(ys).ravel()
    centres = {c: latent[ys == c].mean(0) for c in range(CLASES)}
    paths = []
    for classes, destination in WALK_STARTS:
        start = puntos_de_frontera(head, classes, latente=latent.shape[1])
        t, w = caminata_al_centro(head, start, centres[destination], pasos=steps)
        z = start[None, :] + t[:, None] * (centres[destination] - start)[None, :]
        paths.append(dict(classes=classes, destination=destination, t=t, w=w, Z=z))
    return paths


def _walk_label(index, path, j):
    triple = "  (TRIPLE)" if len(path["classes"]) == CLASES else ""
    tie = "|".join(str(c) for c in path["classes"])
    return (f"{index + 1}. frontera {tie} → centro {path['destination']}{triple}"
            f" · w = {path['w'][j]:.3f}")


def confidence_walk_gif(name, latent, ys, paths, *, dpi=60):
    """One frame per step of the walk: the point moves and the `w` curve grows."""
    latent = np.asarray(latent, dtype=float)
    ys = np.asarray(ys).ravel()
    mean = latent.mean(0)
    _, _, vt = np.linalg.svd(latent - mean, full_matrices=False)
    project = lambda z: (np.atleast_2d(z) - mean) @ vt[:2].T
    cloud = project(latent)
    tracks = [project(p["Z"]) for p in paths]
    # The cloud's robust box, widened so every whole path stays in view.
    (x0, x1), (y0, y1) = _limits(cloud)
    ends = np.concatenate(tracks)
    xlim = (min(x0, ends[:, 0].min()), max(x1, ends[:, 0].max()))
    ylim = (min(y0, ends[:, 1].min()), max(y1, ends[:, 1].max()))
    steps = len(paths[0]["t"])
    frames = []
    for j in range(steps):
        figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
        for c in range(CLASES):
            axes[0].scatter(*cloud[ys == c].T, c=COL_DURA[c], s=6, alpha=0.3,
                            edgecolors="none", label=f"clase {c}")
        for i, (path, track) in enumerate(zip(paths, tracks)):
            axes[0].plot(*track.T, color="0.6", lw=1.0, ls=":", zorder=4)
            axes[0].plot(*track[:j + 1].T, color="k", lw=1.8, zorder=5)
            axes[0].scatter(*track[-1], s=120, marker="*",
                            c=COL_DURA[path["destination"]], edgecolors="k", zorder=6)
            axes[0].scatter(*track[j], s=150, facecolors="none", edgecolors="k",
                            linewidths=2.0, zorder=7)
            axes[0].annotate(f"{i + 1} · w={path['w'][j]:.3f}", tuple(track[j]),
                             fontsize=10, fontweight="bold", xytext=(8, 8),
                             textcoords="offset points", zorder=8)
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])
        axes[0].legend(loc="best", fontsize=9, markerscale=2.0)
        axes[0].set_title(f"{name} · el latente de la FUENTE (PCA)\n"
                          "el punto camina de la frontera al centro de una clase",
                          fontsize=10.5)
        for i, path in enumerate(paths):
            colour = COL_DURA[path["destination"]]
            axes[1].plot(path["t"][:j + 1], path["w"][:j + 1], lw=2.2, color=colour,
                         ls="-" if len(path["classes"]) < CLASES else "--",
                         label=_walk_label(i, path, j))
            axes[1].scatter([path["t"][j]], [path["w"][j]], s=50, color=colour,
                            edgecolors="k", zorder=5)
        axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(-0.03, 1.05)
        axes[1].set_xlabel("recorrido de la frontera al centro de clase (barrido, no época)")
        axes[1].set_ylabel("$w = 1 - \\hat{H}_2(g)/\\ln C$")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="lower right", fontsize=8.5)
        axes[1].set_title("el peso MIENTRAS el punto camina, leído por la cabeza "
                          "entrenada\nnadie mira la etiqueta del punto",
                          fontsize=10.5)
        figure.suptitle(f"{name.upper()} · qué hace el peso de la Ec. 24 · construido, "
                        f"no medido · barrido {j + 1} de {steps}", fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        frames.append(figure_to_rgb(figure, dpi))
    return encode_gif(frames, MS_WALK_FRAME)
