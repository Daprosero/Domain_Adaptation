#!/usr/bin/env python3
"""The three families, one row and three columns, in 3D, easiest to hardest."""
import sys, pathlib
sys.path.insert(0, "/Users/diego/.claude/jobs/9073f488/tmp/duros")
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # el proyector 3D
import familias as F

COLOR = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c'}
# lunas va mas cenital: su estructura vive en el plano (x,y) y la extrusion
# en z no lleva informacion de clase, asi que un angulo bajo superpone las
# dos laminas y la figura se lee como manchas en vez de dos parabolas.
VISTAS = {'lunas': (52, -89), 'toroide': (30, -62), 'rollo': (18, -72)}
NOTA = {'lunas':   'dos parábolas opuestas, extruidas · dos clases sobre la misma lámina',
        'toroide': 'un toro completo cortado en tres arcos de 120° del ángulo mayor',
        'rollo':   'un rollo suizo completo: estirado, cortado en tres y vuelto a enrollar'}

fig = plt.figure(figsize=(19.5, 8.0))
for i, (nom, fn) in enumerate(F.FAMILIAS):
    Xs, ys, Xt, yt = F.construir(nom, fn)
    a = fig.add_subplot(1, 3, i+1, projection='3d')
    a.set_box_aspect(None, zoom=1.18)
    for c in range(3):
        P = Xs[ys == c]
        a.scatter(P[:,0], P[:,1], P[:,2], c=COLOR[c], marker='x', s=11,
                  linewidths=.7, label=f'clase {c}' if i == 0 else None)
        Q = Xt[yt == c]
        a.scatter(Q[:,0], Q[:,1], Q[:,2], c=COLOR[c], marker='o', s=8,
                  alpha=1.0, edgecolors='none')
    todo = np.vstack([Xs, Xt]); p = 0.05*(todo.max(0) - todo.min(0))
    a.set_xlim(todo[:,0].min()-p[0], todo[:,0].max()+p[0])
    a.set_ylim(todo[:,1].min()-p[1], todo[:,1].max()+p[1])
    a.set_zlim(todo[:,2].min()-p[2], todo[:,2].max()+p[2])
    a.view_init(*VISTAS[nom])
    a.set_xticklabels([]); a.set_yticklabels([]); a.set_zticklabels([])
    a.set_title(f"{i+1}. {nom}\n{NOTA[nom]}\n"
                f"direcciones efectivas {F.dim_ef(Xs):.2f} de 3 · "
                f"separación entre clases {F.entre_dentro(Xs, ys):.0f}%",
                fontsize=10.5, linespacing=1.5)
    if i == 0: a.legend(loc='upper left', fontsize=9, markerscale=2.2)

fig.suptitle("Las tres familias, de la más fácil a la más difícil   ·   "
             "`x` fuente, `o` blanco (traslación + rotación 30° sobre un eje "
             "genérico + ruido del 2,5% de la dispersión propia)",
             fontsize=11.5, y=0.055)
# top=0.99 recortaba los titulos de tres renglones.
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.11, wspace=0.03)
salida = pathlib.Path("/Users/diego/.claude/jobs/9073f488/tmp/duros/familias.png")
fig.savefig(salida, dpi=105); plt.close(fig)
print(f"{salida}  ·  {salida.stat().st_size/1e3:.0f} kB")
