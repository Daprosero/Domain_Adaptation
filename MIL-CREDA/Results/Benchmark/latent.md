# Fase dos — research-concept-r17.md

> resnet18  ·  3 épocas  ·  1 repetición(es)  ·  research-concept-r17.md  
> la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación: nada por debajo de eso lo resuelve una transferencia sola  
> !! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.  
> !! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.  

## 1 · Razón entre distancias (más bajo es mejor, si la de entre clases no cayó)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 0.622 ± 0.000 | 0.547 ± 0.000 | 1.231 ± 0.000 | 1.462 ± 0.000 | 1.238 ± 0.000 | 1.324 ± 0.000 | **1.071** |
| `CREDA*` | 0.622 ± 0.000 | 0.547 ± 0.000 | 1.231 ± 0.000 | 1.463 ± 0.000 | 1.238 ± 0.000 | 1.326 ± 0.000 | **1.071** |
| `CREDA` | 0.622 ± 0.000 | 0.547 ± 0.000 | 1.231 ± 0.000 | 1.460 ± 0.000 | 1.238 ± 0.000 | 1.325 ± 0.000 | **1.071** |
| `MIL-Baseline` | 0.821 ± 0.000 | 0.576 ± 0.000 | 1.200 ± 0.000 | 1.318 ± 0.000 | 1.119 ± 0.000 | 1.167 ± 0.000 | **1.034** |
| `MIL-CREDA**` | 0.801 ± 0.000 | 0.558 ± 0.000 | 1.178 ± 0.000 | 1.317 ± 0.000 | 1.146 ± 0.000 | 1.146 ± 0.000 | **1.024** |
| `MIL-CREDA*` | 0.789 ± 0.000 | 0.559 ± 0.000 | 1.202 ± 0.000 | 1.318 ± 0.000 | 1.139 ± 0.000 | 1.165 ± 0.000 | **1.029** |
| `MIL-CREDA` | 0.772 ± 0.000 | 0.555 ± 0.000 | 1.203 ± 0.000 | 1.318 ± 0.000 | 1.175 ± 0.000 | 1.159 ± 0.000 | **1.030** |
| `MIL-CREDA-U` | 1.182 ± 0.000 | 0.651 ± 0.000 | 1.397 ± 0.000 | 1.185 ± 0.000 | 1.306 ± 0.000 | 1.075 ± 0.000 | **1.133** |
| `MIL-CREDA-A` | 0.978 ± 0.000 | 0.678 ± 0.000 | 1.151 ± 0.000 | 1.524 ± 0.000 | 1.247 ± 0.000 | 1.109 ± 0.000 | **1.115** |
| `MIL-CREDA-K` | 0.901 ± 0.000 | 0.679 ± 0.000 | 1.163 ± 0.000 | 1.554 ± 0.000 | 1.212 ± 0.000 | 1.178 ± 0.000 | **1.115** |

Cada método contra su propio piso, transferencia por transferencia. «Alinea» = la razón bajó; «colapsa» = la razón no bajó y la distancia entre clases cayó más del 10%.

Método        piso            alinea  plano  empeora  colapsa
CREDA         Baseline             0      6        0        0
MIL-CREDA     MIL-Baseline         2      3        1        2
MIL-CREDA*    MIL-Baseline         1      4        1        0
MIL-CREDA**   MIL-Baseline         2      3        1        0
MIL-CREDA-U   MIL-Baseline         2      0        4        1
MIL-CREDA-A   MIL-Baseline         2      0        4        2
MIL-CREDA-K   MIL-Baseline         1      1        4        4
CREDA*        Baseline             0      6        0        0

Lo que carga peso no es el promedio sino que las transferencias coincidan: un método que alinea en una y empeora en otra no está diciendo nada todavía.
Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

### 1b · Las dos distancias por separado (descriptivas, se leen juntas)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 19.603 ± 0.000 | 18.069 ± 0.000 | 27.617 ± 0.000 | 18.078 ± 0.000 | 31.638 ± 0.000 | 18.946 ± 0.000 | **22.325** |
| `CREDA*` | 19.603 ± 0.000 | 18.069 ± 0.000 | 27.617 ± 0.000 | 18.085 ± 0.000 | 31.640 ± 0.000 | 18.949 ± 0.000 | **22.327** |
| `CREDA` | 19.603 ± 0.000 | 18.070 ± 0.000 | 27.616 ± 0.000 | 18.039 ± 0.000 | 31.636 ± 0.000 | 18.955 ± 0.000 | **22.320** |
| `MIL-Baseline` | 30.442 ± 0.000 | 21.618 ± 0.000 | 43.254 ± 0.000 | 35.821 ± 0.000 | 37.611 ± 0.000 | 39.401 ± 0.000 | **34.691** |
| `MIL-CREDA**` | 29.172 ± 0.000 | 21.012 ± 0.000 | 40.436 ± 0.000 | 35.776 ± 0.000 | 36.841 ± 0.000 | 38.013 ± 0.000 | **33.542** |
| `MIL-CREDA*` | 28.610 ± 0.000 | 20.624 ± 0.000 | 39.989 ± 0.000 | 35.791 ± 0.000 | 36.488 ± 0.000 | 38.863 ± 0.000 | **33.394** |
| `MIL-CREDA` | 27.798 ± 0.000 | 20.329 ± 0.000 | 38.885 ± 0.000 | 35.795 ± 0.000 | 35.290 ± 0.000 | 38.348 ± 0.000 | **32.741** |
| `MIL-CREDA-U` | 44.683 ± 0.000 | 24.912 ± 0.000 | 38.792 ± 0.000 | 20.586 ± 0.000 | 50.793 ± 0.000 | 23.379 ± 0.000 | **33.858** |
| `MIL-CREDA-A` | 37.463 ± 0.000 | 24.992 ± 0.000 | 40.098 ± 0.000 | 22.160 ± 0.000 | 31.811 ± 0.000 | 23.306 ± 0.000 | **29.972** |
| `MIL-CREDA-K` | 33.630 ± 0.000 | 22.928 ± 0.000 | 32.154 ± 0.000 | 28.687 ± 0.000 | 29.444 ± 0.000 | 28.644 ± 0.000 | **29.248** |

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 31.503 ± 0.000 | 33.035 ± 0.000 | 22.443 ± 0.000 | 12.364 ± 0.000 | 25.560 ± 0.000 | 14.304 ± 0.000 | **23.202** |
| `CREDA*` | 31.504 ± 0.000 | 33.034 ± 0.000 | 22.443 ± 0.000 | 12.358 ± 0.000 | 25.561 ± 0.000 | 14.290 ± 0.000 | **23.198** |
| `CREDA` | 31.504 ± 0.000 | 33.033 ± 0.000 | 22.442 ± 0.000 | 12.353 ± 0.000 | 25.558 ± 0.000 | 14.304 ± 0.000 | **23.199** |
| `MIL-Baseline` | 37.076 ± 0.000 | 37.522 ± 0.000 | 36.038 ± 0.000 | 27.172 ± 0.000 | 33.626 ± 0.000 | 33.759 ± 0.000 | **34.199** |
| `MIL-CREDA**` | 36.416 ± 0.000 | 37.646 ± 0.000 | 34.318 ± 0.000 | 27.158 ± 0.000 | 32.161 ± 0.000 | 33.181 ± 0.000 | **33.480** |
| `MIL-CREDA*` | 36.252 ± 0.000 | 36.862 ± 0.000 | 33.274 ± 0.000 | 27.153 ± 0.000 | 32.043 ± 0.000 | 33.361 ± 0.000 | **33.158** |
| `MIL-CREDA` | 36.009 ± 0.000 | 36.654 ± 0.000 | 32.320 ± 0.000 | 27.152 ± 0.000 | 30.043 ± 0.000 | 33.094 ± 0.000 | **32.545** |
| `MIL-CREDA-U` | 37.796 ± 0.000 | 38.278 ± 0.000 | 27.768 ± 0.000 | 17.373 ± 0.000 | 38.881 ± 0.000 | 21.746 ± 0.000 | **30.307** |
| `MIL-CREDA-A` | 38.289 ± 0.000 | 36.883 ± 0.000 | 34.835 ± 0.000 | 14.545 ± 0.000 | 25.505 ± 0.000 | 21.009 ± 0.000 | **28.511** |
| `MIL-CREDA-K` | 37.316 ± 0.000 | 33.749 ± 0.000 | 27.654 ± 0.000 | 18.455 ± 0.000 | 24.288 ± 0.000 | 24.309 ± 0.000 | **27.628** |

CREDA contra Baseline: misma clase entre dominios -0.0%, clases distintas -0.0% sobre 6 transferencia(s) — las dos se movieron parejo: el espacio cambió de escala y la razón no tendría por qué haberse movido. MIL-CREDA contra MIL-Baseline: misma clase entre dominios -5.6%, clases distintas -4.7% sobre 6 transferencia(s) — las dos se movieron parejo: el espacio cambió de escala y la razón no tendría por qué haberse movido. MIL-CREDA* contra MIL-Baseline: misma clase entre dominios -3.8%, clases distintas -2.9% sobre 6 transferencia(s) — las dos se movieron parejo: el espacio cambió de escala y la razón no tendría por qué haberse movido. MIL-CREDA** contra MIL-Baseline: misma clase entre dominios -3.2%, clases distintas -2.1% sobre 6 transferencia(s) — la misma clase se juntó más de lo que se encogió el espacio. MIL-CREDA-U contra MIL-Baseline: misma clase entre dominios +0.6%, clases distintas -12.5% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA-A contra MIL-Baseline: misma clase entre dominios -10.5%, clases distintas -18.4% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA-K contra MIL-Baseline: misma clase entre dominios -13.0%, clases distintas -20.1% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. CREDA* contra Baseline: misma clase entre dominios +0.0%, clases distintas -0.0% sobre 6 transferencia(s) — las dos se movieron parejo: el espacio cambió de escala y la razón no tendría por qué haberse movido. Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 2 · Separabilidad de dominio (más cerca de 0.500 es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 0.917 ± 0.000 | 0.848 ± 0.000 | 0.968 ± 0.000 | 0.993 ± 0.000 | 0.860 ± 0.000 | 0.987 ± 0.000 | **0.929** |
| `CREDA*` | 0.915 ± 0.000 | 0.847 ± 0.000 | 0.968 ± 0.000 | 0.993 ± 0.000 | 0.860 ± 0.000 | 0.988 ± 0.000 | **0.928** |
| `CREDA` | 0.917 ± 0.000 | 0.847 ± 0.000 | 0.968 ± 0.000 | 0.994 ± 0.000 | 0.858 ± 0.000 | 0.988 ± 0.000 | **0.928** |
| `MIL-Baseline` | 0.850 ± 0.000 | 0.831 ± 0.000 | 0.793 ± 0.000 | 1.000 ± 0.000 | 0.889 ± 0.000 | 1.000 ± 0.000 | **0.894** |
| `MIL-CREDA**` | 0.836 ± 0.000 | 0.832 ± 0.000 | 0.793 ± 0.000 | 1.000 ± 0.000 | 0.874 ± 0.000 | 1.000 ± 0.000 | **0.889** |
| `MIL-CREDA*` | 0.836 ± 0.000 | 0.775 ± 0.000 | 0.780 ± 0.000 | 1.000 ± 0.000 | 0.874 ± 0.000 | 1.000 ± 0.000 | **0.878** |
| `MIL-CREDA` | 0.836 ± 0.000 | 0.775 ± 0.000 | 0.780 ± 0.000 | 1.000 ± 0.000 | 0.860 ± 0.000 | 1.000 ± 0.000 | **0.875** |
| `MIL-CREDA-U` | 0.890 ± 0.000 | 0.824 ± 0.000 | 0.918 ± 0.000 | 1.000 ± 0.000 | 0.906 ± 0.000 | 0.960 ± 0.000 | **0.916** |
| `MIL-CREDA-A` | 0.866 ± 0.000 | 0.849 ± 0.000 | 0.903 ± 0.000 | 1.000 ± 0.000 | 0.946 ± 0.000 | 0.932 ± 0.000 | **0.916** |
| `MIL-CREDA-K` | 0.902 ± 0.000 | 0.902 ± 0.000 | 0.861 ± 0.000 | 1.000 ± 0.000 | 0.947 ± 0.000 | 1.000 ± 0.000 | **0.935** |

Una diferencia negativa es mejor: la regla de dominio quedó más cerca del azar que en el mismo método sin adaptación. CREDA contra Baseline, distancia al azar (0.500): plano en todas, dentro de ±0.010 (M->S +0.000, M->U +0.000, S->M +0.001, S->U +0.001, U->M -0.001, U->S -0.002). MIL-CREDA contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 0 a favor, 4 en contra, 2 planas (M->S -0.013, M->U -0.013, S->M +0.000, S->U +0.000, U->M -0.056, U->S -0.029). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA* contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 0 a favor, 4 en contra, 2 planas (M->S -0.013, M->U -0.013, S->M +0.000, S->U +0.000, U->M -0.056, U->S -0.014). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA** contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 0 a favor, 2 en contra, 4 planas (M->S +0.000, M->U -0.013, S->M +0.000, S->U +0.000, U->M +0.001, U->S -0.014). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-U contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 3 a favor, 1 en contra, 2 planas (M->S +0.125, M->U +0.040, S->M +0.000, S->U -0.040, U->M -0.008, U->S +0.017). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-A contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 1 en contra, 1 planas (M->S +0.110, M->U +0.016, S->M +0.000, S->U -0.068, U->M +0.017, U->S +0.057). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-K contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 0 en contra, 2 planas (M->S +0.068, M->U +0.052, S->M +0.000, S->U +0.000, U->M +0.070, U->S +0.058). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. CREDA* contra Baseline, distancia al azar (0.500): plano en todas, dentro de ±0.010 (M->S +0.000, M->U -0.001, S->M +0.000, S->U +0.000, U->M -0.001, U->S +0.000). Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 3 · Masa en la clase verdadera (más alto es mejor, azar 0.100)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `MIL-CREDA` | 0.612 ± 0.000 | 0.796 ± 0.000 | 0.173 ± 0.000 | 0.282 ± 0.000 | 0.231 ± 0.000 | 0.232 ± 0.000 | **0.388** |
| `MIL-CREDA-U` | 0.270 ± 0.000 | 0.793 ± 0.000 | 0.121 ± 0.000 | 0.248 ± 0.000 | 0.135 ± 0.000 | 0.244 ± 0.000 | **0.302** |
| `MIL-CREDA-A` | 0.383 ± 0.000 | 0.740 ± 0.000 | 0.188 ± 0.000 | 0.248 ± 0.000 | 0.161 ± 0.000 | 0.240 ± 0.000 | **0.327** |
| `MIL-CREDA-K` | 0.445 ± 0.000 | 0.685 ± 0.000 | 0.190 ± 0.000 | 0.246 ± 0.000 | 0.203 ± 0.000 | 0.322 ± 0.000 | **0.349** |

La masa más alta es la de **MIL-CREDA**, contra un azar de 0.100. MIL-CREDA supera el azar en 6 de 6 transferencias MIL-CREDA-U supera el azar en 6 de 6 transferencias MIL-CREDA-A supera el azar en 6 de 6 transferencias MIL-CREDA-K supera el azar en 6 de 6 transferencias Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

### 3b · Cuánto reparte la atención (descriptivo)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `MIL-Baseline` | 0.720 ± 0.000 | 0.767 ± 0.000 | 0.612 ± 0.000 | 0.729 ± 0.000 | 0.729 ± 0.000 | 0.644 ± 0.000 | **0.700** |
| `MIL-CREDA**` | 0.729 ± 0.000 | 0.775 ± 0.000 | 0.616 ± 0.000 | 0.729 ± 0.000 | 0.758 ± 0.000 | 0.651 ± 0.000 | **0.710** |
| `MIL-CREDA*` | 0.739 ± 0.000 | 0.787 ± 0.000 | 0.624 ± 0.000 | 0.729 ± 0.000 | 0.760 ± 0.000 | 0.650 ± 0.000 | **0.715** |
| `MIL-CREDA` | 0.744 ± 0.000 | 0.786 ± 0.000 | 0.624 ± 0.000 | 0.729 ± 0.000 | 0.801 ± 0.000 | 0.658 ± 0.000 | **0.724** |
| `MIL-CREDA-U` | 0.655 ± 0.000 | 0.525 ± 0.000 | 0.764 ± 0.000 | 0.793 ± 0.000 | 0.645 ± 0.000 | 0.808 ± 0.000 | **0.698** |
| `MIL-CREDA-A` | 0.618 ± 0.000 | 0.684 ± 0.000 | 0.457 ± 0.000 | 0.715 ± 0.000 | 0.686 ± 0.000 | 0.656 ± 0.000 | **0.636** |
| `MIL-CREDA-K` | 0.812 ± 0.000 | 0.891 ± 0.000 | 0.788 ± 0.000 | 0.837 ± 0.000 | 0.917 ± 0.000 | 0.840 ± 0.000 | **0.847** |

La entropía está normalizada, así que su máximo es uno y ahí la atención no estaría haciendo nada. Cómo quedó cada uno: MIL-Baseline reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA** reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA* reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-U reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-A reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-K reparte de forma desigual sin apoyarse en unas pocas Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 4 · La correspondencia, sujeto por sujeto

| Método | Qué computa | U->M | M->U | S->M | Masa prom. |
|---|---|---|---|---|---|
| `MIL-Baseline` | sin adaptación: solo fuente | 8/10 | 2/10 | 1/10 | **0.526** |
| `MIL-CREDA*` | adaptación milcreda, ponderada, sin término local | 8/10 | 2/10 | 1/10 | **0.555** |
| `MIL-CREDA` | adaptación milcreda, ponderada, con término local | 8/10 | 2/10 | 1/10 | **0.563** |

Empatan sumando sus transferencias: MIL-Baseline, MIL-CREDA*, MIL-CREDA. De acá no sale quién empareja mejor; el azar de acertar una clase es 0.100. El término local, aislado como MIL-CREDA contra MIL-CREDA*: las transferencias no coinciden — 1 a favor, 0 en contra, 2 planas (M->U +0.023, S->M +0.000, U->M +0.002). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.