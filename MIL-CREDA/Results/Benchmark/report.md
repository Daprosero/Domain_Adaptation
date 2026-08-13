# Fase uno — research-concept-r16.md

## 1 · Tiempo de entrenamiento (más bajo es mejor)

tiempo de entrenamiento (s)  ·  resnet18  ·  3 épocas  ·  1 semilla(s)  ·  research-concept-r16.md
!! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.
!! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | máx |
|---|---|---|---|---|---|---|---|---|
| `Baseline` | 5.86 ± 0.00 | 5.46 ± 0.00 | 5.56 ± 0.00 | 5.58 ± 0.00 | 5.92 ± 0.00 | 4.23 ± 0.00 | **5.43** | 5.43 |
| `CREDA*` | 12.20 ± 0.00 | 10.82 ± 0.00 | 10.13 ± 0.00 | 10.35 ± 0.00 | 10.92 ± 0.00 | 8.74 ± 0.00 | **10.53** | 10.53 |
| `CREDA` | 10.59 ± 0.00 | 10.05 ± 0.00 | 9.99 ± 0.00 | 10.60 ± 0.00 | 10.22 ± 0.00 | 8.01 ± 0.00 | **9.91** | 9.91 |
| `MIL-Baseline` | 6.21 ± 0.00 | 5.64 ± 0.00 | 5.75 ± 0.00 | 5.72 ± 0.00 | 5.68 ± 0.00 | 4.30 ± 0.00 | **5.55** | 5.55 |
| `MIL-CREDA**` | 13.01 ± 0.00 | 15.53 ± 0.00 | 14.15 ± 0.00 | 14.33 ± 0.00 | 14.02 ± 0.00 | 10.55 ± 0.00 | **13.60** | 13.60 |
| `MIL-CREDA*` | 13.87 ± 0.00 | 14.42 ± 0.00 | 14.26 ± 0.00 | 14.26 ± 0.00 | 14.01 ± 0.00 | 12.18 ± 0.00 | **13.83** | 13.83 |
| `MIL-CREDA` | 15.48 ± 0.00 | 15.50 ± 0.00 | 15.48 ± 0.00 | 16.00 ± 0.00 | 16.32 ± 0.00 | 11.95 ± 0.00 | **15.12** | 15.12 |
| `MIL-CREDA-U` | 15.52 ± 0.00 | 15.39 ± 0.00 | 14.89 ± 0.00 | 14.96 ± 0.00 | 14.92 ± 0.00 | 11.46 ± 0.00 | **14.52** | 14.52 |
| `MIL-CREDA-A` | 14.80 ± 0.00 | 15.05 ± 0.00 | 16.15 ± 0.00 | 15.38 ± 0.00 | 18.99 ± 0.00 | 11.60 ± 0.00 | **15.33** | 15.33 |
| `MIL-CREDA-K` | 14.71 ± 0.00 | 15.08 ± 0.00 | 15.15 ± 0.00 | 14.94 ± 0.00 | 18.48 ± 0.00 | 11.76 ± 0.00 | **15.02** | 15.02 |

Mejor promedio: **Baseline** con 5.43s; peor: MIL-CREDA-A con 15.33s. CREDA queda 4.48s por debajo de su piso Baseline. MIL-CREDA queda 9.57s por debajo de su piso MIL-Baseline. Con esta cantidad de repeticiones nada de lo anterior es un resultado: son estimaciones puntuales y la dispersión es cero por construcción.

## 2 · Exactitud en fuente (más alto es mejor)

exactitud en fuente (%)  ·  resnet18  ·  3 épocas  ·  1 semilla(s)  ·  research-concept-r16.md
!! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.
!! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.
la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | máx | peso |
|---|---|---|---|---|---|---|---|---|---|
| `Baseline` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 41.7 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.9** | 81.9 | 0.000 |
| `CREDA*` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 41.7 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.9** | 81.9 | 0.030 |
| `CREDA` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 38.9 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.5** | 81.5 | 0.273 |
| `MIL-Baseline` | 94.4 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 27.8 ± 0.0 | 94.4 ± 0.0 | 36.1 ± 0.0 | **75.5** | 75.5 | 0.000 |
| `MIL-CREDA**` | 97.2 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 27.8 ± 0.0 | 86.1 ± 0.0 | 33.3 ± 0.0 | **73.6** | 73.6 | 0.241 |
| `MIL-CREDA*` | 100.0 ± 0.0 | 100.0 ± 0.0 | 94.4 ± 0.0 | 25.0 ± 0.0 | 94.4 ± 0.0 | 36.1 ± 0.0 | **75.0** | 75.0 | 0.196 |
| `MIL-CREDA` | 97.2 ± 0.0 | 97.2 ± 0.0 | 97.2 ± 0.0 | 25.0 ± 0.0 | 91.7 ± 0.0 | 33.3 ± 0.0 | **73.6** | 73.6 | 0.247 |
| `MIL-CREDA-U` | 86.1 ± 0.0 | 88.9 ± 0.0 | 88.9 ± 0.0 | 30.6 ± 0.0 | 86.1 ± 0.0 | 25.0 ± 0.0 | **67.6** | 67.6 | 0.303 |
| `MIL-CREDA-A` | 77.8 ± 0.0 | 97.2 ± 0.0 | 66.7 ± 0.0 | 8.3 ± 0.0 | 86.1 ± 0.0 | 16.7 ± 0.0 | **58.8** | 58.8 | 0.326 |
| `MIL-CREDA-K` | 72.2 ± 0.0 | 94.4 ± 0.0 | 91.7 ± 0.0 | 30.6 ± 0.0 | 83.3 ± 0.0 | 19.4 ± 0.0 | **65.3** | 65.3 | 0.199 |

Mejor promedio: **Baseline** con 81.9%; peor: MIL-CREDA-A con 58.8%. CREDA queda 0.5% por debajo de su piso Baseline. MIL-CREDA queda 1.9% por debajo de su piso MIL-Baseline. Con esta cantidad de repeticiones nada de lo anterior es un resultado: son estimaciones puntuales y la dispersión es cero por construcción.

### 2b · Peldaños en fuente

peldaños · exactitud en fuente (%) · diferencia con signo hacia la derecha
!! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.
!! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.

| Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | a favor |
|---|---|---|---|---|---|---|---|---|
| Baseline → MIL-Baseline | -5.6 | +0.0 | +0.0 | -13.9 | -5.6 | -13.9 | **-6.5** | 0/6 |
| Baseline → CREDA* | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| CREDA* → CREDA | +0.0 | +0.0 | +0.0 | -2.8 | +0.0 | +0.0 | **-0.5** | 0/6 |
| MIL-Baseline → MIL-CREDA** | +2.8 | -2.8 | +0.0 | +0.0 | -8.3 | -2.8 | **-1.9** | 1/6 |
| MIL-CREDA** → MIL-CREDA* | +2.8 | +2.8 | -5.6 | -2.8 | +8.3 | +2.8 | **+1.4** | 4/6 |
| MIL-CREDA* → MIL-CREDA | -2.8 | -2.8 | +2.8 | +0.0 | -2.8 | -2.8 | **-1.4** | 1/6 |
| CREDA* → MIL-CREDA** | -2.8 | -2.8 | +0.0 | -13.9 | -13.9 | -16.7 | **-8.3** | 0/6 |
| CREDA → MIL-CREDA* | +0.0 | +0.0 | -5.6 | -13.9 | -5.6 | -13.9 | **-6.5** | 0/6 |
| CREDA → MIL-CREDA | -2.8 | -2.8 | -2.8 | -13.9 | -8.3 | -16.7 | **-7.9** | 0/6 |
| MIL-CREDA-U → MIL-CREDA-K | -13.9 | +5.6 | +2.8 | +0.0 | -2.8 | -5.6 | **-2.3** | 2/6 |
| MIL-CREDA-A → MIL-CREDA-K | -5.6 | -2.8 | +25.0 | +22.2 | -2.8 | +2.8 | **+6.5** | 3/6 |
| MIL-CREDA-K → MIL-CREDA | +25.0 | +2.8 | +5.6 | -5.6 | +8.3 | +13.9 | **+8.3** | 5/6 |

El peldaño que más se movió es **CREDA* → MIL-CREDA**** (-8.3), que lee the same rung, built two ways: unweighted. Coinciden en las 6 transferencias: Baseline → MIL-Baseline (-6.5), Baseline → CREDA* (+0.0), CREDA* → CREDA (-0.5), CREDA* → MIL-CREDA** (-8.3), CREDA → MIL-CREDA* (-6.5), CREDA → MIL-CREDA (-7.9). Con una sola repetición la coincidencia entre transferencias es lo único que carga peso, y no reemplaza a las repeticiones.

## 3 · Exactitud en destino (más alto es mejor)

exactitud en destino (%)  ·  resnet18  ·  3 épocas  ·  1 semilla(s)  ·  research-concept-r16.md
!! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.
!! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.
la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | máx | peso |
|---|---|---|---|---|---|---|---|---|---|
| `Baseline` | 77.8 ± 0.0 | 97.2 ± 0.0 | 8.3 ± 0.0 | 72.2 ± 0.0 | 19.4 ± 0.0 | 47.2 ± 0.0 | **53.7** | 53.7 | 0.000 |
| `CREDA*` | 83.3 ± 0.0 | 100.0 ± 0.0 | 8.3 ± 0.0 | 72.2 ± 0.0 | 22.2 ± 0.0 | 47.2 ± 0.0 | **55.6** | 55.6 | 0.030 |
| `CREDA` | 72.2 ± 0.0 | 97.2 ± 0.0 | 13.9 ± 0.0 | 52.8 ± 0.0 | 19.4 ± 0.0 | 36.1 ± 0.0 | **48.6** | 48.6 | 0.273 |
| `MIL-Baseline` | 47.2 ± 0.0 | 83.3 ± 0.0 | 22.2 ± 0.0 | 38.9 ± 0.0 | 11.1 ± 0.0 | 33.3 ± 0.0 | **39.4** | 39.4 | 0.000 |
| `MIL-CREDA**` | 58.3 ± 0.0 | 86.1 ± 0.0 | 25.0 ± 0.0 | 44.4 ± 0.0 | 13.9 ± 0.0 | 33.3 ± 0.0 | **43.5** | 43.5 | 0.241 |
| `MIL-CREDA*` | 52.8 ± 0.0 | 86.1 ± 0.0 | 25.0 ± 0.0 | 33.3 ± 0.0 | 8.3 ± 0.0 | 30.6 ± 0.0 | **39.4** | 39.4 | 0.196 |
| `MIL-CREDA` | 41.7 ± 0.0 | 94.4 ± 0.0 | 13.9 ± 0.0 | 36.1 ± 0.0 | 8.3 ± 0.0 | 25.0 ± 0.0 | **36.6** | 36.6 | 0.247 |
| `MIL-CREDA-U` | 19.4 ± 0.0 | 80.6 ± 0.0 | 8.3 ± 0.0 | 38.9 ± 0.0 | 19.4 ± 0.0 | 38.9 ± 0.0 | **34.3** | 34.3 | 0.303 |
| `MIL-CREDA-A` | 38.9 ± 0.0 | 63.9 ± 0.0 | 19.4 ± 0.0 | 27.8 ± 0.0 | 13.9 ± 0.0 | 19.4 ± 0.0 | **30.6** | 30.6 | 0.326 |
| `MIL-CREDA-K` | 52.8 ± 0.0 | 86.1 ± 0.0 | 19.4 ± 0.0 | 22.2 ± 0.0 | 22.2 ± 0.0 | 25.0 ± 0.0 | **38.0** | 38.0 | 0.199 |

Mejor promedio: **CREDA*** con 55.6%; peor: MIL-CREDA-A con 30.6%. CREDA queda 5.1% por debajo de su piso Baseline. MIL-CREDA queda 2.8% por debajo de su piso MIL-Baseline. Con esta cantidad de repeticiones nada de lo anterior es un resultado: son estimaciones puntuales y la dispersión es cero por construcción.

### 3b · Peldaños en destino

peldaños · exactitud en destino (%) · diferencia con signo hacia la derecha
!! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.
!! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.

| Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | a favor |
|---|---|---|---|---|---|---|---|---|
| Baseline → MIL-Baseline | -30.6 | -13.9 | +13.9 | -33.3 | -8.3 | -13.9 | **-14.4** | 1/6 |
| Baseline → CREDA* | +5.6 | +2.8 | +0.0 | +0.0 | +2.8 | +0.0 | **+1.9** | 3/6 |
| CREDA* → CREDA | -11.1 | -2.8 | +5.6 | -19.4 | -2.8 | -11.1 | **-6.9** | 1/6 |
| MIL-Baseline → MIL-CREDA** | +11.1 | +2.8 | +2.8 | +5.6 | +2.8 | +0.0 | **+4.2** | 5/6 |
| MIL-CREDA** → MIL-CREDA* | -5.6 | +0.0 | +0.0 | -11.1 | -5.6 | -2.8 | **-4.2** | 0/6 |
| MIL-CREDA* → MIL-CREDA | -11.1 | +8.3 | -11.1 | +2.8 | +0.0 | -5.6 | **-2.8** | 2/6 |
| CREDA* → MIL-CREDA** | -25.0 | -13.9 | +16.7 | -27.8 | -8.3 | -13.9 | **-12.0** | 1/6 |
| CREDA → MIL-CREDA* | -19.4 | -11.1 | +11.1 | -19.4 | -11.1 | -5.6 | **-9.3** | 1/6 |
| CREDA → MIL-CREDA | -30.6 | -2.8 | +0.0 | -16.7 | -11.1 | -11.1 | **-12.0** | 0/6 |
| MIL-CREDA-U → MIL-CREDA-K | +33.3 | +5.6 | +11.1 | -16.7 | +2.8 | -13.9 | **+3.7** | 4/6 |
| MIL-CREDA-A → MIL-CREDA-K | +13.9 | +22.2 | +0.0 | -5.6 | +8.3 | +5.6 | **+7.4** | 4/6 |
| MIL-CREDA-K → MIL-CREDA | -11.1 | +8.3 | -5.6 | +13.9 | -13.9 | +0.0 | **-1.4** | 2/6 |

El peldaño que más se movió es **Baseline → MIL-Baseline** (-14.4), que lee what the bag representation buys, with adaptation off. Coinciden en las 6 transferencias: MIL-CREDA** → MIL-CREDA* (-4.2), CREDA → MIL-CREDA (-12.0). Con una sola repetición la coincidencia entre transferencias es lo único que carga peso, y no reemplaza a las repeticiones.