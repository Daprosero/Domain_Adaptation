# Representación latente (v1) — research-concept-r17.md

> resnet18  ·  3 épocas  ·  1 repetición(es)  ·  research-concept-r17.md  
> la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación: nada por debajo de eso lo resuelve una transferencia sola  
> !! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.  
> !! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.  

## 1 · Razón entre distancias (más bajo es mejor, si la de entre clases no cayó)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 0.467 ± 0.000 | 0.342 ± 0.000 | 1.206 ± 0.000 | 1.913 ± 0.000 | 1.163 ± 0.000 | 1.557 ± 0.000 | **1.108** |
| `CREDA*` | 0.467 ± 0.000 | 0.342 ± 0.000 | 1.207 ± 0.000 | 1.913 ± 0.000 | 1.163 ± 0.000 | 1.561 ± 0.000 | **1.109** |
| `CREDA` | 0.467 ± 0.000 | 0.342 ± 0.000 | 1.207 ± 0.000 | 1.913 ± 0.000 | 1.163 ± 0.000 | 1.560 ± 0.000 | **1.109** |
| `MIL-Baseline` | 0.735 ± 0.000 | 0.432 ± 0.000 | 1.250 ± 0.000 | 1.535 ± 0.000 | 1.144 ± 0.000 | 1.253 ± 0.000 | **1.058** |
| `MIL-CREDA**` | 0.699 ± 0.000 | 0.429 ± 0.000 | 1.251 ± 0.000 | 1.535 ± 0.000 | 1.150 ± 0.000 | 1.251 ± 0.000 | **1.053** |
| `MIL-CREDA*` | 0.721 ± 0.000 | 0.429 ± 0.000 | 1.249 ± 0.000 | 1.536 ± 0.000 | 1.150 ± 0.000 | 1.252 ± 0.000 | **1.056** |
| `MIL-CREDA` | 0.724 ± 0.000 | 0.428 ± 0.000 | 1.249 ± 0.000 | 1.536 ± 0.000 | 1.154 ± 0.000 | 1.252 ± 0.000 | **1.057** |
| `MIL-CREDA-U` | 1.209 ± 0.000 | 0.517 ± 0.000 | 1.468 ± 0.000 | 1.406 ± 0.000 | 1.416 ± 0.000 | 1.224 ± 0.000 | **1.207** |
| `MIL-CREDA-A` | 1.000 ± 0.000 | 0.508 ± 0.000 | 1.165 ± 0.000 | 2.027 ± 0.000 | 1.343 ± 0.000 | 1.328 ± 0.000 | **1.229** |
| `MIL-CREDA-K` | 0.840 ± 0.000 | 0.595 ± 0.000 | 1.170 ± 0.000 | 1.918 ± 0.000 | 1.218 ± 0.000 | 1.245 ± 0.000 | **1.164** |

Cada método contra su propio piso, transferencia por transferencia. «Alinea» = la razón bajó; «colapsa» = la razón no bajó y la distancia entre clases cayó más del 10%.

Método        piso            alinea  plano  empeora  colapsa
CREDA         Baseline             0      6        0        0
MIL-CREDA     MIL-Baseline         0      6        0        0
MIL-CREDA*    MIL-Baseline         0      6        0        0
MIL-CREDA**   MIL-Baseline         1      5        0        0
MIL-CREDA-U   MIL-Baseline         2      0        4        2
MIL-CREDA-A   MIL-Baseline         1      0        5        2
MIL-CREDA-K   MIL-Baseline         1      1        4        0
CREDA*        Baseline             0      6        0        0

Lo que carga peso no es el promedio sino que las transferencias coincidan: un método que alinea en una y empeora en otra no está diciendo nada todavía.
Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

### 1b · Las dos distancias por separado (descriptivas, se leen juntas)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 0.142 ± 0.000 | 0.099 ± 0.000 | 0.288 ± 0.000 | 0.164 ± 0.000 | 0.279 ± 0.000 | 0.169 ± 0.000 | **0.190** |
| `CREDA*` | 0.142 ± 0.000 | 0.100 ± 0.000 | 0.288 ± 0.000 | 0.164 ± 0.000 | 0.279 ± 0.000 | 0.169 ± 0.000 | **0.190** |
| `CREDA` | 0.142 ± 0.000 | 0.100 ± 0.000 | 0.288 ± 0.000 | 0.164 ± 0.000 | 0.279 ± 0.000 | 0.169 ± 0.000 | **0.190** |
| `MIL-Baseline` | 0.262 ± 0.000 | 0.179 ± 0.000 | 0.423 ± 0.000 | 0.391 ± 0.000 | 0.392 ± 0.000 | 0.343 ± 0.000 | **0.331** |
| `MIL-CREDA**` | 0.254 ± 0.000 | 0.178 ± 0.000 | 0.423 ± 0.000 | 0.390 ± 0.000 | 0.393 ± 0.000 | 0.342 ± 0.000 | **0.330** |
| `MIL-CREDA*` | 0.261 ± 0.000 | 0.178 ± 0.000 | 0.421 ± 0.000 | 0.391 ± 0.000 | 0.394 ± 0.000 | 0.342 ± 0.000 | **0.331** |
| `MIL-CREDA` | 0.264 ± 0.000 | 0.177 ± 0.000 | 0.423 ± 0.000 | 0.391 ± 0.000 | 0.390 ± 0.000 | 0.342 ± 0.000 | **0.331** |
| `MIL-CREDA-U` | 0.399 ± 0.000 | 0.172 ± 0.000 | 0.482 ± 0.000 | 0.365 ± 0.000 | 0.389 ± 0.000 | 0.377 ± 0.000 | **0.364** |
| `MIL-CREDA-A` | 0.331 ± 0.000 | 0.191 ± 0.000 | 0.414 ± 0.000 | 0.386 ± 0.000 | 0.417 ± 0.000 | 0.302 ± 0.000 | **0.340** |
| `MIL-CREDA-K` | 0.301 ± 0.000 | 0.222 ± 0.000 | 0.381 ± 0.000 | 0.442 ± 0.000 | 0.388 ± 0.000 | 0.401 ± 0.000 | **0.356** |

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 0.304 ± 0.000 | 0.291 ± 0.000 | 0.239 ± 0.000 | 0.086 ± 0.000 | 0.240 ± 0.000 | 0.109 ± 0.000 | **0.211** |
| `CREDA*` | 0.304 ± 0.000 | 0.291 ± 0.000 | 0.239 ± 0.000 | 0.086 ± 0.000 | 0.240 ± 0.000 | 0.108 ± 0.000 | **0.211** |
| `CREDA` | 0.304 ± 0.000 | 0.291 ± 0.000 | 0.239 ± 0.000 | 0.085 ± 0.000 | 0.240 ± 0.000 | 0.108 ± 0.000 | **0.211** |
| `MIL-Baseline` | 0.356 ± 0.000 | 0.414 ± 0.000 | 0.339 ± 0.000 | 0.255 ± 0.000 | 0.342 ± 0.000 | 0.274 ± 0.000 | **0.330** |
| `MIL-CREDA**` | 0.364 ± 0.000 | 0.414 ± 0.000 | 0.338 ± 0.000 | 0.254 ± 0.000 | 0.342 ± 0.000 | 0.273 ± 0.000 | **0.331** |
| `MIL-CREDA*` | 0.362 ± 0.000 | 0.414 ± 0.000 | 0.337 ± 0.000 | 0.254 ± 0.000 | 0.342 ± 0.000 | 0.273 ± 0.000 | **0.330** |
| `MIL-CREDA` | 0.365 ± 0.000 | 0.414 ± 0.000 | 0.338 ± 0.000 | 0.254 ± 0.000 | 0.338 ± 0.000 | 0.274 ± 0.000 | **0.331** |
| `MIL-CREDA-U` | 0.330 ± 0.000 | 0.333 ± 0.000 | 0.328 ± 0.000 | 0.260 ± 0.000 | 0.275 ± 0.000 | 0.308 ± 0.000 | **0.306** |
| `MIL-CREDA-A` | 0.331 ± 0.000 | 0.375 ± 0.000 | 0.356 ± 0.000 | 0.190 ± 0.000 | 0.310 ± 0.000 | 0.228 ± 0.000 | **0.298** |
| `MIL-CREDA-K` | 0.358 ± 0.000 | 0.372 ± 0.000 | 0.325 ± 0.000 | 0.230 ± 0.000 | 0.319 ± 0.000 | 0.322 ± 0.000 | **0.321** |

CREDA contra Baseline: misma clase entre dominios -0.0%, clases distintas -0.0% sobre 6 transferencia(s) — las dos se movieron parejo: el espacio cambió de escala y la razón no tendría por qué haberse movido. MIL-CREDA contra MIL-Baseline: misma clase entre dominios -0.0%, clases distintas +0.2% sobre 6 transferencia(s) — las dos se movieron parejo: el espacio cambió de escala y la razón no tendría por qué haberse movido. MIL-CREDA* contra MIL-Baseline: misma clase entre dominios -0.2%, clases distintas +0.2% sobre 6 transferencia(s) — las dos se movieron parejo: el espacio cambió de escala y la razón no tendría por qué haberse movido. MIL-CREDA** contra MIL-Baseline: misma clase entre dominios -0.6%, clases distintas +0.3% sobre 6 transferencia(s) — las dos se movieron parejo: el espacio cambió de escala y la razón no tendría por qué haberse movido. MIL-CREDA-U contra MIL-Baseline: misma clase entre dominios +10.9%, clases distintas -5.8% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA-A contra MIL-Baseline: misma clase entre dominios +4.1%, clases distintas -10.4% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. MIL-CREDA-K contra MIL-Baseline: misma clase entre dominios +9.7%, clases distintas -2.0% sobre 6 transferencia(s) — las clases se juntaron entre sí más que la misma clase entre dominios. CREDA* contra Baseline: misma clase entre dominios +0.0%, clases distintas -0.0% sobre 6 transferencia(s) — las dos se movieron parejo: el espacio cambió de escala y la razón no tendría por qué haberse movido. Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 2 · Separabilidad de dominio (más cerca de 0.500 es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 0.917 ± 0.000 | 0.848 ± 0.000 | 0.968 ± 0.000 | 0.993 ± 0.000 | 0.860 ± 0.000 | 0.987 ± 0.000 | **0.929** |
| `CREDA*` | 0.916 ± 0.000 | 0.846 ± 0.000 | 0.967 ± 0.000 | 0.994 ± 0.000 | 0.862 ± 0.000 | 0.987 ± 0.000 | **0.929** |
| `CREDA` | 0.917 ± 0.000 | 0.850 ± 0.000 | 0.968 ± 0.000 | 0.994 ± 0.000 | 0.858 ± 0.000 | 0.986 ± 0.000 | **0.929** |
| `MIL-Baseline` | 0.850 ± 0.000 | 0.831 ± 0.000 | 0.793 ± 0.000 | 1.000 ± 0.000 | 0.889 ± 0.000 | 1.000 ± 0.000 | **0.894** |
| `MIL-CREDA**` | 0.836 ± 0.000 | 0.831 ± 0.000 | 0.780 ± 0.000 | 1.000 ± 0.000 | 0.889 ± 0.000 | 1.000 ± 0.000 | **0.889** |
| `MIL-CREDA*` | 0.836 ± 0.000 | 0.831 ± 0.000 | 0.793 ± 0.000 | 1.000 ± 0.000 | 0.874 ± 0.000 | 1.000 ± 0.000 | **0.889** |
| `MIL-CREDA` | 0.836 ± 0.000 | 0.831 ± 0.000 | 0.793 ± 0.000 | 1.000 ± 0.000 | 0.874 ± 0.000 | 1.000 ± 0.000 | **0.889** |
| `MIL-CREDA-U` | 0.890 ± 0.000 | 0.796 ± 0.000 | 0.918 ± 0.000 | 1.000 ± 0.000 | 0.906 ± 0.000 | 0.960 ± 0.000 | **0.912** |
| `MIL-CREDA-A` | 0.907 ± 0.000 | 0.876 ± 0.000 | 0.904 ± 0.000 | 1.000 ± 0.000 | 0.946 ± 0.000 | 0.919 ± 0.000 | **0.925** |
| `MIL-CREDA-K` | 0.915 ± 0.000 | 0.889 ± 0.000 | 0.861 ± 0.000 | 1.000 ± 0.000 | 0.947 ± 0.000 | 1.000 ± 0.000 | **0.935** |

Una diferencia negativa es mejor: la regla de dominio quedó más cerca del azar que en el mismo método sin adaptación. CREDA contra Baseline, distancia al azar (0.500): plano en todas, dentro de ±0.010 (M->S +0.000, M->U +0.000, S->M +0.001, S->U -0.001, U->M +0.001, U->S -0.002). MIL-CREDA contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 0 a favor, 2 en contra, 4 planas (M->S +0.000, M->U -0.013, S->M +0.000, S->U +0.000, U->M +0.000, U->S -0.014). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA* contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 0 a favor, 2 en contra, 4 planas (M->S +0.000, M->U -0.013, S->M +0.000, S->U +0.000, U->M +0.000, U->S -0.014). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA** contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 0 a favor, 2 en contra, 4 planas (M->S -0.013, M->U -0.013, S->M +0.000, S->U +0.000, U->M +0.000, U->S +0.000). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-U contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 3 a favor, 2 en contra, 1 planas (M->S +0.125, M->U +0.040, S->M +0.000, S->U -0.040, U->M -0.035, U->S +0.017). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-A contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 1 en contra, 1 planas (M->S +0.110, M->U +0.057, S->M +0.000, S->U -0.081, U->M +0.045, U->S +0.057). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. MIL-CREDA-K contra MIL-Baseline, distancia al azar (0.500): las transferencias no coinciden — 4 a favor, 0 en contra, 2 planas (M->S +0.068, M->U +0.066, S->M +0.000, S->U +0.000, U->M +0.057, U->S +0.058). Un promedio acá diría 'no hace nada' y estaría tapando que una transferencia sí se movió. CREDA* contra Baseline, distancia al azar (0.500): plano en todas, dentro de ±0.010 (M->S -0.001, M->U -0.001, S->M +0.002, S->U +0.000, U->M -0.002, U->S +0.002). Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 3 · Masa en la clase verdadera (más alto es mejor, azar 0.100)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `MIL-CREDA` | 0.566 ± 0.000 | 0.773 ± 0.000 | 0.165 ± 0.000 | 0.282 ± 0.000 | 0.223 ± 0.000 | 0.232 ± 0.000 | **0.373** |
| `MIL-CREDA-U` | 0.275 ± 0.000 | 0.723 ± 0.000 | 0.124 ± 0.000 | 0.248 ± 0.000 | 0.156 ± 0.000 | 0.250 ± 0.000 | **0.296** |
| `MIL-CREDA-A` | 0.407 ± 0.000 | 0.713 ± 0.000 | 0.189 ± 0.000 | 0.246 ± 0.000 | 0.168 ± 0.000 | 0.250 ± 0.000 | **0.329** |
| `MIL-CREDA-K` | 0.457 ± 0.000 | 0.627 ± 0.000 | 0.202 ± 0.000 | 0.236 ± 0.000 | 0.213 ± 0.000 | 0.342 ± 0.000 | **0.346** |

La masa más alta es la de **MIL-CREDA**, contra un azar de 0.100. MIL-CREDA supera el azar en 6 de 6 transferencias MIL-CREDA-U supera el azar en 6 de 6 transferencias MIL-CREDA-A supera el azar en 6 de 6 transferencias MIL-CREDA-K supera el azar en 6 de 6 transferencias Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

### 3b · Cuánto reparte la atención (descriptivo)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `MIL-Baseline` | 0.720 ± 0.000 | 0.767 ± 0.000 | 0.612 ± 0.000 | 0.729 ± 0.000 | 0.729 ± 0.000 | 0.644 ± 0.000 | **0.700** |
| `MIL-CREDA**` | 0.732 ± 0.000 | 0.766 ± 0.000 | 0.612 ± 0.000 | 0.731 ± 0.000 | 0.739 ± 0.000 | 0.645 ± 0.000 | **0.704** |
| `MIL-CREDA*` | 0.731 ± 0.000 | 0.767 ± 0.000 | 0.613 ± 0.000 | 0.729 ± 0.000 | 0.742 ± 0.000 | 0.644 ± 0.000 | **0.704** |
| `MIL-CREDA` | 0.733 ± 0.000 | 0.768 ± 0.000 | 0.614 ± 0.000 | 0.729 ± 0.000 | 0.754 ± 0.000 | 0.645 ± 0.000 | **0.707** |
| `MIL-CREDA-U` | 0.647 ± 0.000 | 0.528 ± 0.000 | 0.783 ± 0.000 | 0.793 ± 0.000 | 0.618 ± 0.000 | 0.811 ± 0.000 | **0.697** |
| `MIL-CREDA-A` | 0.611 ± 0.000 | 0.685 ± 0.000 | 0.439 ± 0.000 | 0.716 ± 0.000 | 0.661 ± 0.000 | 0.645 ± 0.000 | **0.626** |
| `MIL-CREDA-K` | 0.808 ± 0.000 | 0.887 ± 0.000 | 0.797 ± 0.000 | 0.838 ± 0.000 | 0.898 ± 0.000 | 0.818 ± 0.000 | **0.841** |

La entropía está normalizada, así que su máximo es uno y ahí la atención no estaría haciendo nada. Cómo quedó cada uno: MIL-Baseline reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA** reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA* reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-U reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-A reparte de forma desigual sin apoyarse en unas pocas MIL-CREDA-K reparte de forma desigual sin apoyarse en unas pocas Piloto de 1 repetición(es): estimación puntual, todavía no un veredicto.

## 4 · La correspondencia, sujeto por sujeto

(sin paneles medidos)

Sin paneles medidos: no hay nada que concluir.