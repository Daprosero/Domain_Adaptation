# Fase dos — research-concept-r16.md

> resnet18  ·  3 épocas  ·  1 repetición(es)  ·  research-concept-r16.md  
> la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación: nada por debajo de eso lo resuelve una transferencia sola  
> !! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.  
> !! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.  

## 1 · Razón entre distancias (más bajo es mejor, si la de entre clases no cayó)

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
Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

### 1b · Las dos distancias por separado (descriptivas, se leen juntas)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 19.011 ± 0.000 | 17.595 ± 0.000 | 27.487 ± 0.000 | 17.856 ± 0.000 | 31.137 ± 0.000 | 18.449 ± 0.000 | **21.922** |
| `CREDA*` | 18.060 ± 0.000 | 17.418 ± 0.000 | 25.121 ± 0.000 | 18.287 ± 0.000 | 28.046 ± 0.000 | 18.375 ± 0.000 | **20.885** |
| `CREDA` | 20.147 ± 0.000 | 19.653 ± 0.000 | 29.423 ± 0.000 | 17.541 ± 0.000 | 32.647 ± 0.000 | 15.486 ± 0.000 | **22.483** |
| `MIL-Baseline` | 36.729 ± 0.000 | 22.070 ± 0.000 | 46.951 ± 0.000 | 34.710 ± 0.000 | 34.522 ± 0.000 | 37.978 ± 0.000 | **35.493** |
| `MIL-CREDA**` | 27.433 ± 0.000 | 17.519 ± 0.000 | 33.541 ± 0.000 | 26.888 ± 0.000 | 32.890 ± 0.000 | 33.181 ± 0.000 | **28.575** |
| `MIL-CREDA*` | 29.132 ± 0.000 | 16.961 ± 0.000 | 34.971 ± 0.000 | 27.821 ± 0.000 | 32.938 ± 0.000 | 30.480 ± 0.000 | **28.717** |
| `MIL-CREDA` | 28.244 ± 0.000 | 14.845 ± 0.000 | 34.405 ± 0.000 | 27.028 ± 0.000 | 31.031 ± 0.000 | 29.627 ± 0.000 | **27.530** |
| `MIL-CREDA-U` | 29.386 ± 0.000 | 24.053 ± 0.000 | 26.726 ± 0.000 | 18.670 ± 0.000 | 36.375 ± 0.000 | 16.349 ± 0.000 | **25.260** |
| `MIL-CREDA-A` | 28.447 ± 0.000 | 22.947 ± 0.000 | 30.975 ± 0.000 | 15.355 ± 0.000 | 28.521 ± 0.000 | 15.846 ± 0.000 | **23.682** |
| `MIL-CREDA-K` | 31.415 ± 0.000 | 20.522 ± 0.000 | 28.805 ± 0.000 | 23.212 ± 0.000 | 24.193 ± 0.000 | 22.764 ± 0.000 | **25.152** |

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 31.803 ± 0.000 | 33.181 ± 0.000 | 22.816 ± 0.000 | 12.117 ± 0.000 | 25.847 ± 0.000 | 13.974 ± 0.000 | **23.290** |
| `CREDA*` | 30.844 ± 0.000 | 31.378 ± 0.000 | 20.431 ± 0.000 | 12.023 ± 0.000 | 23.474 ± 0.000 | 13.686 ± 0.000 | **21.973** |
| `CREDA` | 29.160 ± 0.000 | 29.241 ± 0.000 | 24.154 ± 0.000 | 9.853 ± 0.000 | 26.251 ± 0.000 | 10.673 ± 0.000 | **21.555** |
| `MIL-Baseline` | 38.205 ± 0.000 | 38.125 ± 0.000 | 39.948 ± 0.000 | 29.649 ± 0.000 | 30.323 ± 0.000 | 35.615 ± 0.000 | **35.311** |
| `MIL-CREDA**` | 33.405 ± 0.000 | 35.135 ± 0.000 | 29.434 ± 0.000 | 22.617 ± 0.000 | 27.851 ± 0.000 | 30.952 ± 0.000 | **29.899** |
| `MIL-CREDA*` | 35.737 ± 0.000 | 32.583 ± 0.000 | 30.828 ± 0.000 | 23.569 ± 0.000 | 26.926 ± 0.000 | 29.343 ± 0.000 | **29.831** |
| `MIL-CREDA` | 31.833 ± 0.000 | 32.645 ± 0.000 | 29.124 ± 0.000 | 22.680 ± 0.000 | 25.286 ± 0.000 | 26.841 ± 0.000 | **28.068** |
| `MIL-CREDA-U` | 25.335 ± 0.000 | 35.837 ± 0.000 | 19.846 ± 0.000 | 15.024 ± 0.000 | 29.196 ± 0.000 | 13.193 ± 0.000 | **23.072** |
| `MIL-CREDA-A` | 32.534 ± 0.000 | 30.522 ± 0.000 | 24.610 ± 0.000 | 8.824 ± 0.000 | 22.094 ± 0.000 | 13.778 ± 0.000 | **22.060** |
| `MIL-CREDA-K` | 37.635 ± 0.000 | 30.029 ± 0.000 | 24.177 ± 0.000 | 14.689 ± 0.000 | 20.371 ± 0.000 | 19.310 ± 0.000 | **24.368** |

CREDA contra Baseline: misma clase entre dominios +2.0%, clases distintas -9.2% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA contra MIL-Baseline: misma clase entre dominios -22.8%, clases distintas -20.5% sobre 6 transferencia(s) — la misma clase se juntó más de lo que se encogió el espacio. MIL-CREDA* contra MIL-Baseline: misma clase entre dominios -18.9%, clases distintas -15.5% sobre 6 transferencia(s) — la misma clase se juntó más de lo que se encogió el espacio. MIL-CREDA** contra MIL-Baseline: misma clase entre dominios -19.1%, clases distintas -15.3% sobre 6 transferencia(s) — la misma clase se juntó más de lo que se encogió el espacio. MIL-CREDA-U contra MIL-Baseline: misma clase entre dominios -25.3%, clases distintas -34.3% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA-A contra MIL-Baseline: misma clase entre dominios -30.7%, clases distintas -38.6% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA-K contra MIL-Baseline: misma clase entre dominios -27.2%, clases distintas -31.9% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. CREDA* contra Baseline: misma clase entre dominios -3.8%, clases distintas -5.2% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 2 · Separabilidad de dominio (más cerca de 0.500 es mejor)

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

Una diferencia negativa es mejor: la regla de dominio quedó más cerca del azar que en el mismo método sin adaptación. CREDA contra Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 0 en contra, 2 planas (M->S +0.023, M->U +0.012, S->M +0.001, S->U +0.005, U->M +0.050, U->S +0.060). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 5 a favor, 0 en contra, 1 planas (M->S +0.029, M->U +0.000, S->M +0.014, S->U +0.040, U->M +0.055, U->S +0.040). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA* contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 1 en contra, 1 planas (M->S +0.001, M->U -0.027, S->M +0.014, S->U +0.040, U->M +0.029, U->S +0.013). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA** contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 1 en contra, 1 planas (M->S -0.028, M->U +0.000, S->M +0.014, S->U +0.040, U->M +0.013, U->S +0.013). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-U contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 5 a favor, 0 en contra, 1 planas (M->S +0.071, M->U +0.042, S->M +0.014, S->U +0.000, U->M +0.012, U->S +0.070). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-A contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 1 en contra, 1 planas (M->S +0.085, M->U -0.027, S->M +0.000, S->U +0.011, U->M +0.082, U->S +0.070). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-K contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 1 en contra, 1 planas (M->S +0.044, M->U -0.070, S->M +0.000, S->U +0.026, U->M +0.110, U->S +0.042). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. CREDA* contra Baseline, distancia al azar (0.500): las transferencias no coinciden — 1 a favor, 0 en contra, 5 planas (M->S +0.007, M->U -0.008, S->M -0.001, S->U -0.007, U->M +0.025, U->S -0.007). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 3 · Masa en la clase verdadera (más alto es mejor, azar 0.100)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `MIL-CREDA` | 0.423 ± 0.000 | 0.851 ± 0.000 | 0.144 ± 0.000 | 0.268 ± 0.000 | 0.126 ± 0.000 | 0.201 ± 0.000 | **0.335** |
| `MIL-CREDA-U` | 0.236 ± 0.000 | 0.752 ± 0.000 | 0.100 ± 0.000 | 0.252 ± 0.000 | 0.171 ± 0.000 | 0.181 ± 0.000 | **0.282** |
| `MIL-CREDA-A` | 0.415 ± 0.000 | 0.612 ± 0.000 | 0.150 ± 0.000 | 0.168 ± 0.000 | 0.158 ± 0.000 | 0.184 ± 0.000 | **0.281** |
| `MIL-CREDA-K` | 0.508 ± 0.000 | 0.774 ± 0.000 | 0.174 ± 0.000 | 0.160 ± 0.000 | 0.177 ± 0.000 | 0.226 ± 0.000 | **0.337** |

La masa más alta es la de **MIL-CREDA-K**, contra un azar de 0.100. MIL-CREDA supera el azar en 6 de 6 transferencias MIL-CREDA-U supera el azar en 5 de 6 transferencias MIL-CREDA-A supera el azar en 6 de 6 transferencias MIL-CREDA-K supera el azar en 6 de 6 transferencias Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

### 3b · Cuánto reparte la atención (descriptivo)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `MIL-Baseline` | 0.710 ± 0.000 | 0.768 ± 0.000 | 0.609 ± 0.000 | 0.704 ± 0.000 | 0.785 ± 0.000 | 0.672 ± 0.000 | **0.708** |
| `MIL-CREDA**` | 0.695 ± 0.000 | 0.794 ± 0.000 | 0.638 ± 0.000 | 0.803 ± 0.000 | 0.824 ± 0.000 | 0.737 ± 0.000 | **0.749** |
| `MIL-CREDA*` | 0.679 ± 0.000 | 0.810 ± 0.000 | 0.586 ± 0.000 | 0.744 ± 0.000 | 0.822 ± 0.000 | 0.728 ± 0.000 | **0.728** |
| `MIL-CREDA` | 0.741 ± 0.000 | 0.849 ± 0.000 | 0.589 ± 0.000 | 0.777 ± 0.000 | 0.813 ± 0.000 | 0.758 ± 0.000 | **0.754** |
| `MIL-CREDA-U` | 0.691 ± 0.000 | 0.617 ± 0.000 | 0.714 ± 0.000 | 0.864 ± 0.000 | 0.616 ± 0.000 | 0.829 ± 0.000 | **0.722** |
| `MIL-CREDA-A` | 0.666 ± 0.000 | 0.684 ± 0.000 | 0.617 ± 0.000 | 0.766 ± 0.000 | 0.770 ± 0.000 | 0.761 ± 0.000 | **0.710** |
| `MIL-CREDA-K` | 0.814 ± 0.000 | 0.904 ± 0.000 | 0.775 ± 0.000 | 0.849 ± 0.000 | 0.902 ± 0.000 | 0.798 ± 0.000 | **0.840** |

La entropía está normalizada, así que su máximo es uno y ahí la atención no estaría haciendo nada. Cómo quedó cada uno: MIL-Baseline reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA** reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA* reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-U reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-A reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-K reparte de forma desigual sin apoyarse en unas pocas Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 4 · La correspondencia, sujeto por sujeto

| Método | Qué computa | U->M | M->U | S->M | Masa prom. |
|---|---|---|---|---|---|
| `MIL-Baseline` | sin adaptación: solo fuente | 7/10 | 2/10 | 2/10 | **0.500** |
| `MIL-CREDA*` | adaptación milcreda, ponderada, sin término local | 9/10 | 3/10 | 2/10 | **0.519** |
| `MIL-CREDA` | adaptación milcreda, ponderada, con término local | 10/10 | 2/10 | 2/10 | **0.514** |

Empatan sumando sus transferencias: MIL-CREDA*, MIL-CREDA. De acá no sale quién empareja mejor; el azar de acertar una clase es 0.100. El término local, aislado como MIL-CREDA contra MIL-CREDA*: las transferencias no coinciden — 1 a favor, 1 en contra, 1 planas (M->U -0.078, S->M +0.003, U->M +0.060). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.