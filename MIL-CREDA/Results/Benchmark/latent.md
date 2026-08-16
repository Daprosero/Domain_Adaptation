# Fase dos — research-concept-r17.md

> resnet18  ·  3 épocas  ·  1 repetición(es)  ·  research-concept-r17.md  
> la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación: nada por debajo de eso lo resuelve una transferencia sola  
> !! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.  
> !! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.  

## 1 · Razón entre distancias (más bajo es mejor, si la de entre clases no cayó)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 0.598 ± 0.000 | 0.530 ± 0.000 | 1.205 ± 0.000 | 1.474 ± 0.000 | 1.205 ± 0.000 | 1.320 ± 0.000 | **1.055** |
| `CREDA*` | 0.586 ± 0.000 | 0.555 ± 0.000 | 1.230 ± 0.000 | 1.521 ± 0.000 | 1.195 ± 0.000 | 1.343 ± 0.000 | **1.071** |
| `CREDA` | 0.691 ± 0.000 | 0.672 ± 0.000 | 1.218 ± 0.000 | 1.780 ± 0.000 | 1.244 ± 0.000 | 1.451 ± 0.000 | **1.176** |
| `MIL-Baseline` | 0.793 ± 0.000 | 0.572 ± 0.000 | 1.171 ± 0.000 | 1.338 ± 0.000 | 1.099 ± 0.000 | 1.175 ± 0.000 | **1.025** |
| `MIL-CREDA**` | 0.714 ± 0.000 | 0.562 ± 0.000 | 1.144 ± 0.000 | 1.415 ± 0.000 | 1.086 ± 0.000 | 1.152 ± 0.000 | **1.012** |
| `MIL-CREDA*` | 0.718 ± 0.000 | 0.584 ± 0.000 | 1.050 ± 0.000 | 1.300 ± 0.000 | 1.222 ± 0.000 | 1.086 ± 0.000 | **0.993** |
| `MIL-CREDA` | 0.740 ± 0.000 | 0.653 ± 0.000 | 1.144 ± 0.000 | 1.391 ± 0.000 | 1.324 ± 0.000 | 1.225 ± 0.000 | **1.080** |
| `MIL-CREDA-U` | 1.129 ± 0.000 | 0.780 ± 0.000 | 1.231 ± 0.000 | 1.807 ± 0.000 | 1.432 ± 0.000 | 1.331 ± 0.000 | **1.285** |
| `MIL-CREDA-A` | 1.001 ± 0.000 | 0.785 ± 0.000 | 1.098 ± 0.000 | 1.808 ± 0.000 | 1.377 ± 0.000 | 1.472 ± 0.000 | **1.257** |
| `MIL-CREDA-K` | 0.887 ± 0.000 | 0.707 ± 0.000 | 1.078 ± 0.000 | 1.776 ± 0.000 | 1.302 ± 0.000 | 1.583 ± 0.000 | **1.222** |

Cada método contra su propio piso, transferencia por transferencia. «Alinea» = la razón bajó; «colapsa» = la razón no bajó y la distancia entre clases cayó más del 10%.

Método        piso            alinea  plano  empeora  colapsa
CREDA         Baseline             0      1        5        3
MIL-CREDA     MIL-Baseline         2      0        4        4
MIL-CREDA*    MIL-Baseline         4      1        1        2
MIL-CREDA**   MIL-Baseline         3      2        1        3
MIL-CREDA-U   MIL-Baseline         0      0        6        6
MIL-CREDA-A   MIL-Baseline         1      0        5        5
MIL-CREDA-K   MIL-Baseline         1      0        5        5
CREDA*        Baseline             0      2        4        1

Lo que carga peso no es el promedio sino que las transferencias coincidan: un método que alinea en una y empeora en otra no está diciendo nada todavía.
Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

### 1b · Las dos distancias por separado (descriptivas, se leen juntas)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 19.011 ± 0.000 | 17.595 ± 0.000 | 27.487 ± 0.000 | 17.856 ± 0.000 | 31.137 ± 0.000 | 18.449 ± 0.000 | **21.922** |
| `CREDA*` | 18.060 ± 0.000 | 17.418 ± 0.000 | 25.121 ± 0.000 | 18.287 ± 0.000 | 28.046 ± 0.000 | 18.375 ± 0.000 | **20.885** |
| `CREDA` | 20.147 ± 0.000 | 19.653 ± 0.000 | 29.423 ± 0.000 | 17.541 ± 0.000 | 32.647 ± 0.000 | 15.486 ± 0.000 | **22.483** |
| `MIL-Baseline` | 30.844 ± 0.000 | 21.262 ± 0.000 | 45.067 ± 0.000 | 38.666 ± 0.000 | 35.521 ± 0.000 | 39.318 ± 0.000 | **35.113** |
| `MIL-CREDA**` | 22.040 ± 0.000 | 15.310 ± 0.000 | 24.691 ± 0.000 | 17.557 ± 0.000 | 18.968 ± 0.000 | 20.935 ± 0.000 | **19.917** |
| `MIL-CREDA*` | 20.096 ± 0.000 | 15.139 ± 0.000 | 27.291 ± 0.000 | 18.182 ± 0.000 | 23.191 ± 0.000 | 21.456 ± 0.000 | **20.893** |
| `MIL-CREDA` | 20.023 ± 0.000 | 14.880 ± 0.000 | 23.676 ± 0.000 | 20.407 ± 0.000 | 24.718 ± 0.000 | 24.114 ± 0.000 | **21.303** |
| `MIL-CREDA-U` | 24.678 ± 0.000 | 20.420 ± 0.000 | 13.587 ± 0.000 | 11.189 ± 0.000 | 29.737 ± 0.000 | 11.646 ± 0.000 | **18.543** |
| `MIL-CREDA-A` | 26.518 ± 0.000 | 19.433 ± 0.000 | 24.493 ± 0.000 | 8.394 ± 0.000 | 23.828 ± 0.000 | 8.971 ± 0.000 | **18.606** |
| `MIL-CREDA-K` | 22.670 ± 0.000 | 14.379 ± 0.000 | 21.485 ± 0.000 | 15.725 ± 0.000 | 19.652 ± 0.000 | 12.186 ± 0.000 | **17.683** |

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 31.803 ± 0.000 | 33.181 ± 0.000 | 22.816 ± 0.000 | 12.117 ± 0.000 | 25.847 ± 0.000 | 13.974 ± 0.000 | **23.290** |
| `CREDA*` | 30.844 ± 0.000 | 31.378 ± 0.000 | 20.431 ± 0.000 | 12.023 ± 0.000 | 23.474 ± 0.000 | 13.686 ± 0.000 | **21.973** |
| `CREDA` | 29.160 ± 0.000 | 29.241 ± 0.000 | 24.154 ± 0.000 | 9.853 ± 0.000 | 26.251 ± 0.000 | 10.673 ± 0.000 | **21.555** |
| `MIL-Baseline` | 38.890 ± 0.000 | 37.195 ± 0.000 | 38.479 ± 0.000 | 28.907 ± 0.000 | 32.317 ± 0.000 | 33.461 ± 0.000 | **34.875** |
| `MIL-CREDA**` | 30.854 ± 0.000 | 27.220 ± 0.000 | 21.579 ± 0.000 | 12.410 ± 0.000 | 17.464 ± 0.000 | 18.179 ± 0.000 | **21.284** |
| `MIL-CREDA*` | 28.003 ± 0.000 | 25.907 ± 0.000 | 25.997 ± 0.000 | 13.987 ± 0.000 | 18.977 ± 0.000 | 19.752 ± 0.000 | **22.104** |
| `MIL-CREDA` | 27.047 ± 0.000 | 22.774 ± 0.000 | 20.695 ± 0.000 | 14.668 ± 0.000 | 18.664 ± 0.000 | 19.678 ± 0.000 | **20.588** |
| `MIL-CREDA-U` | 21.857 ± 0.000 | 26.183 ± 0.000 | 11.036 ± 0.000 | 6.192 ± 0.000 | 20.771 ± 0.000 | 8.749 ± 0.000 | **15.798** |
| `MIL-CREDA-A` | 26.487 ± 0.000 | 24.743 ± 0.000 | 22.303 ± 0.000 | 4.643 ± 0.000 | 17.310 ± 0.000 | 6.095 ± 0.000 | **16.930** |
| `MIL-CREDA-K` | 25.545 ± 0.000 | 20.328 ± 0.000 | 19.930 ± 0.000 | 8.853 ± 0.000 | 15.088 ± 0.000 | 7.698 ± 0.000 | **16.241** |

CREDA contra Baseline: misma clase entre dominios +2.0%, clases distintas -9.2% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA contra MIL-Baseline: misma clase entre dominios -38.1%, clases distintas -41.4% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA* contra MIL-Baseline: misma clase entre dominios -39.4%, clases distintas -37.4% sobre 6 transferencia(s) — la misma clase se juntó más de lo que se encogió el espacio. MIL-CREDA** contra MIL-Baseline: misma clase entre dominios -41.6%, clases distintas -40.0% sobre 6 transferencia(s) — la misma clase se juntó más de lo que se encogió el espacio. MIL-CREDA-U contra MIL-Baseline: misma clase entre dominios -41.9%, clases distintas -55.5% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA-A contra MIL-Baseline: misma clase entre dominios -42.8%, clases distintas -53.3% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA-K contra MIL-Baseline: misma clase entre dominios -47.4%, clases distintas -54.6% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. CREDA* contra Baseline: misma clase entre dominios -3.8%, clases distintas -5.2% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 2 · Separabilidad de dominio (más cerca de 0.500 es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 0.930 ± 0.000 | 0.851 ± 0.000 | 0.947 ± 0.000 | 0.994 ± 0.000 | 0.894 ± 0.000 | 0.981 ± 0.000 | **0.933** |
| `CREDA*` | 0.921 ± 0.000 | 0.876 ± 0.000 | 0.954 ± 0.000 | 0.992 ± 0.000 | 0.887 ± 0.000 | 0.974 ± 0.000 | **0.934** |
| `CREDA` | 0.942 ± 0.000 | 0.901 ± 0.000 | 0.970 ± 0.000 | 0.994 ± 0.000 | 0.954 ± 0.000 | 0.986 ± 0.000 | **0.958** |
| `MIL-Baseline` | 0.850 ± 0.000 | 0.832 ± 0.000 | 0.793 ± 0.000 | 1.000 ± 0.000 | 0.863 ± 0.000 | 1.000 ± 0.000 | **0.890** |
| `MIL-CREDA**` | 0.836 ± 0.000 | 0.792 ± 0.000 | 0.861 ± 0.000 | 1.000 ± 0.000 | 0.903 ± 0.000 | 1.000 ± 0.000 | **0.899** |
| `MIL-CREDA*` | 0.878 ± 0.000 | 0.863 ± 0.000 | 0.904 ± 0.000 | 1.000 ± 0.000 | 0.903 ± 0.000 | 1.000 ± 0.000 | **0.925** |
| `MIL-CREDA` | 0.946 ± 0.000 | 0.876 ± 0.000 | 0.947 ± 0.000 | 1.000 ± 0.000 | 0.930 ± 0.000 | 1.000 ± 0.000 | **0.950** |
| `MIL-CREDA-U` | 0.947 ± 0.000 | 0.849 ± 0.000 | 0.960 ± 0.000 | 0.987 ± 0.000 | 0.905 ± 0.000 | 0.973 ± 0.000 | **0.937** |
| `MIL-CREDA-A` | 0.932 ± 0.000 | 0.960 ± 0.000 | 0.875 ± 0.000 | 1.000 ± 0.000 | 0.960 ± 0.000 | 1.000 ± 0.000 | **0.955** |
| `MIL-CREDA-K` | 0.836 ± 0.000 | 0.890 ± 0.000 | 0.861 ± 0.000 | 1.000 ± 0.000 | 0.960 ± 0.000 | 1.000 ± 0.000 | **0.924** |

Una diferencia negativa es mejor: la regla de dominio quedó más cerca del azar que en el mismo método sin adaptación. CREDA contra Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 0 en contra, 2 planas (M->S +0.023, M->U +0.012, S->M +0.001, S->U +0.005, U->M +0.050, U->S +0.060). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 0 en contra, 2 planas (M->S +0.153, M->U +0.096, S->M +0.000, S->U +0.000, U->M +0.044, U->S +0.067). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA* contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 0 en contra, 2 planas (M->S +0.110, M->U +0.029, S->M +0.000, S->U +0.000, U->M +0.030, U->S +0.040). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA** contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 2 a favor, 2 en contra, 2 planas (M->S +0.068, M->U -0.013, S->M +0.000, S->U +0.000, U->M -0.040, U->S +0.040). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-U contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 2 en contra, 0 planas (M->S +0.167, M->U +0.097, S->M -0.013, S->U -0.027, U->M +0.016, U->S +0.042). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-A contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 0 en contra, 2 planas (M->S +0.082, M->U +0.083, S->M +0.000, S->U +0.000, U->M +0.128, U->S +0.097). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-K contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 3 a favor, 1 en contra, 2 planas (M->S +0.068, M->U -0.013, S->M +0.000, S->U +0.000, U->M +0.057, U->S +0.097). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. CREDA* contra Baseline, distancia al azar (0.500): las transferencias no coinciden — 1 a favor, 0 en contra, 5 planas (M->S +0.007, M->U -0.008, S->M -0.001, S->U -0.007, U->M +0.025, U->S -0.007). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 3 · Masa en la clase verdadera (más alto es mejor, azar 0.100)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `MIL-CREDA` | 0.454 ± 0.000 | 0.569 ± 0.000 | 0.173 ± 0.000 | 0.230 ± 0.000 | 0.110 ± 0.000 | 0.152 ± 0.000 | **0.281** |
| `MIL-CREDA-U` | 0.196 ± 0.000 | 0.343 ± 0.000 | 0.097 ± 0.000 | 0.143 ± 0.000 | 0.120 ± 0.000 | 0.131 ± 0.000 | **0.172** |
| `MIL-CREDA-A` | 0.383 ± 0.000 | 0.451 ± 0.000 | 0.150 ± 0.000 | 0.126 ± 0.000 | 0.158 ± 0.000 | 0.146 ± 0.000 | **0.236** |
| `MIL-CREDA-K` | 0.347 ± 0.000 | 0.581 ± 0.000 | 0.148 ± 0.000 | 0.125 ± 0.000 | 0.131 ± 0.000 | 0.121 ± 0.000 | **0.242** |

La masa más alta es la de **MIL-CREDA**, contra un azar de 0.100. MIL-CREDA supera el azar en 6 de 6 transferencias MIL-CREDA-U supera el azar en 5 de 6 transferencias MIL-CREDA-A supera el azar en 6 de 6 transferencias MIL-CREDA-K supera el azar en 6 de 6 transferencias Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

### 3b · Cuánto reparte la atención (descriptivo)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `MIL-Baseline` | 0.730 ± 0.000 | 0.748 ± 0.000 | 0.596 ± 0.000 | 0.740 ± 0.000 | 0.714 ± 0.000 | 0.659 ± 0.000 | **0.698** |
| `MIL-CREDA**` | 0.812 ± 0.000 | 0.866 ± 0.000 | 0.661 ± 0.000 | 0.844 ± 0.000 | 0.832 ± 0.000 | 0.765 ± 0.000 | **0.797** |
| `MIL-CREDA*` | 0.846 ± 0.000 | 0.881 ± 0.000 | 0.641 ± 0.000 | 0.830 ± 0.000 | 0.789 ± 0.000 | 0.797 ± 0.000 | **0.798** |
| `MIL-CREDA` | 0.804 ± 0.000 | 0.905 ± 0.000 | 0.706 ± 0.000 | 0.882 ± 0.000 | 0.831 ± 0.000 | 0.834 ± 0.000 | **0.827** |
| `MIL-CREDA-U` | 0.820 ± 0.000 | 0.629 ± 0.000 | 0.677 ± 0.000 | 0.744 ± 0.000 | 0.706 ± 0.000 | 0.823 ± 0.000 | **0.733** |
| `MIL-CREDA-A` | 0.691 ± 0.000 | 0.781 ± 0.000 | 0.492 ± 0.000 | 0.704 ± 0.000 | 0.888 ± 0.000 | 0.588 ± 0.000 | **0.691** |
| `MIL-CREDA-K` | 0.903 ± 0.000 | 0.947 ± 0.000 | 0.773 ± 0.000 | 0.959 ± 0.000 | 0.927 ± 0.000 | 0.801 ± 0.000 | **0.885** |

La entropía está normalizada, así que su máximo es uno y ahí la atención no estaría haciendo nada. Cómo quedó cada uno: MIL-Baseline reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA** reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA* reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-U reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-A reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-K reparte de forma desigual sin apoyarse en unas pocas Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 4 · La correspondencia, sujeto por sujeto

| Método | Qué computa | U->M | M->U | S->M | Masa prom. |
|---|---|---|---|---|---|
| `MIL-Baseline` | sin adaptación: solo fuente | 6/10 | 2/10 | 1/10 | **0.535** |
| `MIL-CREDA*` | adaptación milcreda, ponderada, sin término local | 7/10 | 3/10 | 1/10 | **0.430** |
| `MIL-CREDA` | adaptación milcreda, ponderada, con término local | 6/10 | 3/10 | 1/10 | **0.418** |

Empareja mejor **MIL-CREDA***, sumando sus transferencias; el azar de acertar una clase es 0.100. El término local, aislado como MIL-CREDA contra MIL-CREDA*: las transferencias no coinciden — 1 a favor, 2 en contra, 0 planas (M->U -0.045, S->M +0.065, U->M -0.057). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.