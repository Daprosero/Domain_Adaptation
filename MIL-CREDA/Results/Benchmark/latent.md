# Fase dos — research-concept-r16.md

## 1 · Razón entre distancias (más bajo es mejor, si la de entre clases no cayó)

razón: misma clase entre dominios / entre clases  ·  1 checkpoint(s) por celda  ·  research-concept-r16.md
el ± es la dispersión entre checkpoints guardados, no entre semillas: esto mide geometría y la fase uno mide exactitud
!! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.
!! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 0.598 ± 0.000 | 0.530 ± 0.000 | 1.205 ± 0.000 | 1.474 ± 0.000 | 1.205 ± 0.000 | 1.320 ± 0.000 | **1.055** |
| `CREDA*` | 0.586 ± 0.000 | 0.555 ± 0.000 | 1.230 ± 0.000 | 1.521 ± 0.000 | 1.195 ± 0.000 | 1.343 ± 0.000 | **1.071** |
| `CREDA` | 0.691 ± 0.000 | 0.672 ± 0.000 | 1.218 ± 0.000 | 1.780 ± 0.000 | 1.244 ± 0.000 | 1.451 ± 0.000 | **1.176** |
| `MIL-Baseline` | 0.961 ± 0.000 | 0.579 ± 0.000 | 1.175 ± 0.000 | 1.171 ± 0.000 | 1.138 ± 0.000 | 1.066 ± 0.000 | **1.015** |
| `MIL-CREDA**` | 0.821 ± 0.000 | 0.499 ± 0.000 | 1.140 ± 0.000 | 1.189 ± 0.000 | 1.181 ± 0.000 | 1.072 ± 0.000 | **0.984** |
| `MIL-CREDA*` | 0.815 ± 0.000 | 0.521 ± 0.000 | 1.134 ± 0.000 | 1.180 ± 0.000 | 1.223 ± 0.000 | 1.039 ± 0.000 | **0.985** |
| `MIL-CREDA` | 0.887 ± 0.000 | 0.455 ± 0.000 | 1.181 ± 0.000 | 1.192 ± 0.000 | 1.227 ± 0.000 | 1.104 ± 0.000 | **1.008** |
| `MIL-CREDA-U` | 1.160 ± 0.000 | 0.671 ± 0.000 | 1.347 ± 0.000 | 1.243 ± 0.000 | 1.246 ± 0.000 | 1.239 ± 0.000 | **1.151** |
| `MIL-CREDA-A` | 0.874 ± 0.000 | 0.752 ± 0.000 | 1.259 ± 0.000 | 1.740 ± 0.000 | 1.291 ± 0.000 | 1.150 ± 0.000 | **1.178** |
| `MIL-CREDA-K` | 0.835 ± 0.000 | 0.683 ± 0.000 | 1.191 ± 0.000 | 1.580 ± 0.000 | 1.188 ± 0.000 | 1.179 ± 0.000 | **1.109** |

Cada método contra su propio piso, transferencia por transferencia. «Alinea» = la razón bajó; «colapsa» = la razón no bajó y la distancia entre clases cayó más del 10%.

Método        piso            alinea  plano  empeora  colapsa
CREDA         Baseline             0      1        5        3
MIL-CREDA     MIL-Baseline         2      1        3        4
MIL-CREDA*    MIL-Baseline         4      1        1        2
MIL-CREDA**   MIL-Baseline         3      2        1        2
MIL-CREDA-U   MIL-Baseline         0      0        6        4
MIL-CREDA-A   MIL-Baseline         1      0        5        5
MIL-CREDA-K   MIL-Baseline         1      1        4        5
CREDA*        Baseline             0      2        4        1

Lo que carga peso no es el promedio sino que las transferencias coincidan: un método que alinea en una y empeora en otra no está diciendo nada todavía.
Piloto: son estimaciones puntuales, no resultados.

## 2 · Separabilidad de dominio (más cerca de 0,500 es mejor)

exactitud de un clasificador de dominio  ·  1 checkpoint(s) por celda  ·  research-concept-r16.md
el ± es la dispersión entre checkpoints guardados, no entre semillas: esto mide geometría y la fase uno mide exactitud
!! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.
!! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 0.930 ± 0.000 | 0.851 ± 0.000 | 0.947 ± 0.000 | 0.994 ± 0.000 | 0.894 ± 0.000 | 0.981 ± 0.000 | **0.933** |
| `CREDA*` | 0.921 ± 0.000 | 0.876 ± 0.000 | 0.954 ± 0.000 | 0.992 ± 0.000 | 0.887 ± 0.000 | 0.974 ± 0.000 | **0.934** |
| `CREDA` | 0.942 ± 0.000 | 0.901 ± 0.000 | 0.970 ± 0.000 | 0.994 ± 0.000 | 0.954 ± 0.000 | 0.986 ± 0.000 | **0.958** |
| `MIL-Baseline` | 0.890 ± 0.000 | 0.792 ± 0.000 | 0.847 ± 0.000 | 0.986 ± 0.000 | 0.890 ± 0.000 | 0.960 ± 0.000 | **0.894** |
| `MIL-CREDA**` | 0.890 ± 0.000 | 0.806 ± 0.000 | 0.819 ± 0.000 | 1.000 ± 0.000 | 0.903 ± 0.000 | 1.000 ± 0.000 | **0.903** |
| `MIL-CREDA*` | 0.863 ± 0.000 | 0.821 ± 0.000 | 0.848 ± 0.000 | 1.000 ± 0.000 | 0.903 ± 0.000 | 1.000 ± 0.000 | **0.906** |
| `MIL-CREDA` | 0.890 ± 0.000 | 0.848 ± 0.000 | 0.875 ± 0.000 | 1.000 ± 0.000 | 0.930 ± 0.000 | 1.000 ± 0.000 | **0.924** |
| `MIL-CREDA-U` | 0.931 ± 0.000 | 0.805 ± 0.000 | 0.918 ± 0.000 | 1.000 ± 0.000 | 0.960 ± 0.000 | 0.960 ± 0.000 | **0.929** |
| `MIL-CREDA-A` | 0.863 ± 0.000 | 0.874 ± 0.000 | 0.931 ± 0.000 | 0.986 ± 0.000 | 0.960 ± 0.000 | 0.971 ± 0.000 | **0.931** |
| `MIL-CREDA-K` | 0.819 ± 0.000 | 0.903 ± 0.000 | 0.890 ± 0.000 | 0.986 ± 0.000 | 0.931 ± 0.000 | 0.986 ± 0.000 | **0.919** |

## 3 · Masa en la clase verdadera (más alto es mejor, azar 0,100)

masa en la clase verdadera  ·  1 checkpoint(s) por celda  ·  research-concept-r16.md
el ± es la dispersión entre checkpoints guardados, no entre semillas: esto mide geometría y la fase uno mide exactitud
!! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.
!! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `MIL-CREDA` | 0.423 ± 0.000 | 0.851 ± 0.000 | 0.144 ± 0.000 | 0.268 ± 0.000 | 0.126 ± 0.000 | 0.201 ± 0.000 | **0.335** |
| `MIL-CREDA-U` | 0.236 ± 0.000 | 0.752 ± 0.000 | 0.100 ± 0.000 | 0.252 ± 0.000 | 0.171 ± 0.000 | 0.181 ± 0.000 | **0.282** |
| `MIL-CREDA-A` | 0.415 ± 0.000 | 0.612 ± 0.000 | 0.150 ± 0.000 | 0.168 ± 0.000 | 0.158 ± 0.000 | 0.184 ± 0.000 | **0.281** |
| `MIL-CREDA-K` | 0.508 ± 0.000 | 0.774 ± 0.000 | 0.174 ± 0.000 | 0.160 ± 0.000 | 0.177 ± 0.000 | 0.226 ± 0.000 | **0.337** |

Aciertos por transferencia (azar 0.100): MIL-Baseline U->M 7/10, M->U 2/10, S->M 2/10 · MIL-CREDA* U->M 9/10, M->U 3/10, S->M 2/10 · MIL-CREDA U->M 10/10, M->U 2/10, S->M 2/10. El término local, aislado como MIL-CREDA contra MIL-CREDA*: las transferencias no coinciden — 1 a favor, 1 en contra, 1 planas (M->U -0.078, S->M +0.003, U->M +0.060). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. Piloto: son estimaciones puntuales, no resultados.