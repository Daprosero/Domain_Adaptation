# Fase uno — research-concept-r16.md

> resnet18  ·  3 épocas  ·  1 repetición(es)  ·  research-concept-r16.md  
> la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación: nada por debajo de eso lo resuelve una transferencia sola  
> !! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.  
> !! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.  

## 1 · Tiempo de entrenamiento (más bajo es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 4.98 ± 0.00 | 4.81 ± 0.00 | 4.69 ± 0.00 | 4.72 ± 0.00 | 4.73 ± 0.00 | 4.72 ± 0.00 | **4.78** |
| `CREDA*` | 10.12 ± 0.00 | 8.71 ± 0.00 | 8.30 ± 0.00 | 8.98 ± 0.00 | 8.66 ± 0.00 | 8.91 ± 0.00 | **8.94** |
| `CREDA` | 8.92 ± 0.00 | 8.82 ± 0.00 | 8.20 ± 0.00 | 9.43 ± 0.00 | 8.77 ± 0.00 | 9.16 ± 0.00 | **8.88** |
| `MIL-Baseline` | 4.85 ± 0.00 | 4.82 ± 0.00 | 4.79 ± 0.00 | 4.86 ± 0.00 | 4.80 ± 0.00 | 4.84 ± 0.00 | **4.83** |
| `MIL-CREDA**` | 12.69 ± 0.00 | 13.18 ± 0.00 | 11.78 ± 0.00 | 12.42 ± 0.00 | 12.34 ± 0.00 | 12.59 ± 0.00 | **12.50** |
| `MIL-CREDA*` | 12.20 ± 0.00 | 12.47 ± 0.00 | 12.45 ± 0.00 | 12.38 ± 0.00 | 12.35 ± 0.00 | 12.47 ± 0.00 | **12.39** |
| `MIL-CREDA` | 13.02 ± 0.00 | 13.48 ± 0.00 | 13.02 ± 0.00 | 13.22 ± 0.00 | 13.39 ± 0.00 | 13.61 ± 0.00 | **13.29** |
| `MIL-CREDA-U` | 12.80 ± 0.00 | 13.66 ± 0.00 | 13.14 ± 0.00 | 12.95 ± 0.00 | 13.25 ± 0.00 | 13.25 ± 0.00 | **13.17** |
| `MIL-CREDA-A` | 12.79 ± 0.00 | 13.28 ± 0.00 | 13.20 ± 0.00 | 13.15 ± 0.00 | 13.28 ± 0.00 | 13.17 ± 0.00 | **13.15** |
| `MIL-CREDA-K` | 12.69 ± 0.00 | 13.24 ± 0.00 | 13.79 ± 0.00 | 13.24 ± 0.00 | 12.94 ± 0.00 | 13.17 ± 0.00 | **13.18** |

Mejor promedio: **Baseline** con 4.78s; peor: MIL-CREDA con 13.29s. CREDA queda 4.11s por debajo de su piso Baseline. MIL-CREDA queda 8.46s por debajo de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

## 2 · Exactitud en fuente (más alto es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 41.7 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.9** |
| `CREDA*` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 41.7 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.9** |
| `CREDA` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 38.9 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.5** |
| `MIL-Baseline` | 94.4 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 27.8 ± 0.0 | 94.4 ± 0.0 | 36.1 ± 0.0 | **75.5** |
| `MIL-CREDA**` | 97.2 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 27.8 ± 0.0 | 86.1 ± 0.0 | 33.3 ± 0.0 | **73.6** |
| `MIL-CREDA*` | 100.0 ± 0.0 | 100.0 ± 0.0 | 94.4 ± 0.0 | 25.0 ± 0.0 | 94.4 ± 0.0 | 36.1 ± 0.0 | **75.0** |
| `MIL-CREDA` | 97.2 ± 0.0 | 97.2 ± 0.0 | 97.2 ± 0.0 | 25.0 ± 0.0 | 91.7 ± 0.0 | 33.3 ± 0.0 | **73.6** |
| `MIL-CREDA-U` | 86.1 ± 0.0 | 88.9 ± 0.0 | 88.9 ± 0.0 | 30.6 ± 0.0 | 86.1 ± 0.0 | 25.0 ± 0.0 | **67.6** |
| `MIL-CREDA-A` | 77.8 ± 0.0 | 97.2 ± 0.0 | 66.7 ± 0.0 | 8.3 ± 0.0 | 86.1 ± 0.0 | 16.7 ± 0.0 | **58.8** |
| `MIL-CREDA-K` | 72.2 ± 0.0 | 94.4 ± 0.0 | 91.7 ± 0.0 | 30.6 ± 0.0 | 83.3 ± 0.0 | 19.4 ± 0.0 | **65.3** |

Mejor promedio: **Baseline** con 81.9%; peor: MIL-CREDA-A con 58.8%. CREDA queda 0.5% por debajo de su piso Baseline. MIL-CREDA queda 1.9% por debajo de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

### 2b · Peldaños en fuente

| Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | gana izq. |
|---|---|---|---|---|---|---|---|---|
| Baseline → MIL-Baseline | +5.6 | +0.0 | +0.0 | +13.9 | +5.6 | +13.9 | **+6.5** | 4/6 |
| Baseline → CREDA* | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| CREDA* → CREDA | +0.0 | +0.0 | +0.0 | +2.8 | +0.0 | +0.0 | **+0.5** | 1/6 |
| MIL-Baseline → MIL-CREDA** | -2.8 | +2.8 | +0.0 | +0.0 | +8.3 | +2.8 | **+1.9** | 3/6 |
| MIL-CREDA** → MIL-CREDA* | -2.8 | -2.8 | +5.6 | +2.8 | -8.3 | -2.8 | **-1.4** | 2/6 |
| MIL-CREDA* → MIL-CREDA | +2.8 | +2.8 | -2.8 | +0.0 | +2.8 | +2.8 | **+1.4** | 4/6 |
| CREDA* → MIL-CREDA** | +2.8 | +2.8 | +0.0 | +13.9 | +13.9 | +16.7 | **+8.3** | 5/6 |
| CREDA → MIL-CREDA* | +0.0 | +0.0 | +5.6 | +13.9 | +5.6 | +13.9 | **+6.5** | 4/6 |
| CREDA → MIL-CREDA | +2.8 | +2.8 | +2.8 | +13.9 | +8.3 | +16.7 | **+7.9** | 6/6 |
| MIL-CREDA-U → MIL-CREDA-K | +13.9 | -5.6 | -2.8 | +0.0 | +2.8 | +5.6 | **+2.3** | 3/6 |
| MIL-CREDA-A → MIL-CREDA-K | +5.6 | +2.8 | -25.0 | -22.2 | +2.8 | -2.8 | **-6.5** | 3/6 |
| MIL-CREDA-K → MIL-CREDA | -25.0 | -2.8 | -5.6 | +5.6 | -8.3 | -13.9 | **-8.3** | 1/6 |

El peldaño que más separa es **CREDA* → MIL-CREDA****: **CREDA*** queda 8.3 por encima de MIL-CREDA**, y eso lee el mismo peldaño, construido de dos maneras: sin ponderar. Se inclinan igual en las 6 transferencias: Baseline → CREDA*, CREDA → MIL-CREDA. Con esta cantidad de repeticiones lo que carga peso es la coincidencia entre transferencias, no la magnitud: seis acuerdos y tres contra tres promedian parecido y dicen cosas distintas.

## 3 · Exactitud en destino (más alto es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 77.8 ± 0.0 | 97.2 ± 0.0 | 8.3 ± 0.0 | 72.2 ± 0.0 | 19.4 ± 0.0 | 47.2 ± 0.0 | **53.7** |
| `CREDA*` | 83.3 ± 0.0 | 100.0 ± 0.0 | 8.3 ± 0.0 | 72.2 ± 0.0 | 22.2 ± 0.0 | 47.2 ± 0.0 | **55.6** |
| `CREDA` | 72.2 ± 0.0 | 97.2 ± 0.0 | 13.9 ± 0.0 | 52.8 ± 0.0 | 19.4 ± 0.0 | 36.1 ± 0.0 | **48.6** |
| `MIL-Baseline` | 47.2 ± 0.0 | 83.3 ± 0.0 | 22.2 ± 0.0 | 38.9 ± 0.0 | 11.1 ± 0.0 | 33.3 ± 0.0 | **39.4** |
| `MIL-CREDA**` | 58.3 ± 0.0 | 86.1 ± 0.0 | 25.0 ± 0.0 | 44.4 ± 0.0 | 13.9 ± 0.0 | 33.3 ± 0.0 | **43.5** |
| `MIL-CREDA*` | 52.8 ± 0.0 | 86.1 ± 0.0 | 25.0 ± 0.0 | 33.3 ± 0.0 | 8.3 ± 0.0 | 30.6 ± 0.0 | **39.4** |
| `MIL-CREDA` | 41.7 ± 0.0 | 94.4 ± 0.0 | 13.9 ± 0.0 | 36.1 ± 0.0 | 8.3 ± 0.0 | 25.0 ± 0.0 | **36.6** |
| `MIL-CREDA-U` | 19.4 ± 0.0 | 80.6 ± 0.0 | 8.3 ± 0.0 | 38.9 ± 0.0 | 19.4 ± 0.0 | 38.9 ± 0.0 | **34.3** |
| `MIL-CREDA-A` | 38.9 ± 0.0 | 63.9 ± 0.0 | 19.4 ± 0.0 | 27.8 ± 0.0 | 13.9 ± 0.0 | 19.4 ± 0.0 | **30.6** |
| `MIL-CREDA-K` | 52.8 ± 0.0 | 86.1 ± 0.0 | 19.4 ± 0.0 | 22.2 ± 0.0 | 22.2 ± 0.0 | 25.0 ± 0.0 | **38.0** |

Mejor promedio: **CREDA*** con 55.6%; peor: MIL-CREDA-A con 30.6%. CREDA queda 5.1% por debajo de su piso Baseline. MIL-CREDA queda 2.8% por debajo de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

### 3b · Peldaños en destino

| Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | gana izq. |
|---|---|---|---|---|---|---|---|---|
| Baseline → MIL-Baseline | +30.6 | +13.9 | -13.9 | +33.3 | +8.3 | +13.9 | **+14.4** | 5/6 |
| Baseline → CREDA* | -5.6 | -2.8 | +0.0 | +0.0 | -2.8 | +0.0 | **-1.9** | 0/6 |
| CREDA* → CREDA | +11.1 | +2.8 | -5.6 | +19.4 | +2.8 | +11.1 | **+6.9** | 5/6 |
| MIL-Baseline → MIL-CREDA** | -11.1 | -2.8 | -2.8 | -5.6 | -2.8 | +0.0 | **-4.2** | 0/6 |
| MIL-CREDA** → MIL-CREDA* | +5.6 | +0.0 | +0.0 | +11.1 | +5.6 | +2.8 | **+4.2** | 4/6 |
| MIL-CREDA* → MIL-CREDA | +11.1 | -8.3 | +11.1 | -2.8 | +0.0 | +5.6 | **+2.8** | 3/6 |
| CREDA* → MIL-CREDA** | +25.0 | +13.9 | -16.7 | +27.8 | +8.3 | +13.9 | **+12.0** | 5/6 |
| CREDA → MIL-CREDA* | +19.4 | +11.1 | -11.1 | +19.4 | +11.1 | +5.6 | **+9.3** | 5/6 |
| CREDA → MIL-CREDA | +30.6 | +2.8 | +0.0 | +16.7 | +11.1 | +11.1 | **+12.0** | 5/6 |
| MIL-CREDA-U → MIL-CREDA-K | -33.3 | -5.6 | -11.1 | +16.7 | -2.8 | +13.9 | **-3.7** | 2/6 |
| MIL-CREDA-A → MIL-CREDA-K | -13.9 | -22.2 | +0.0 | +5.6 | -8.3 | -5.6 | **-7.4** | 1/6 |
| MIL-CREDA-K → MIL-CREDA | +11.1 | -8.3 | +5.6 | -13.9 | +13.9 | +0.0 | **+1.4** | 3/6 |

El peldaño que más separa es **Baseline → MIL-Baseline**: **Baseline** queda 14.4 por encima de MIL-Baseline, y eso lee qué compra la representación por bolsas, con la adaptación apagada. Se inclinan igual en las 6 transferencias: Baseline → CREDA*, MIL-Baseline → MIL-CREDA**. Con esta cantidad de repeticiones lo que carga peso es la coincidencia entre transferencias, no la magnitud: seis acuerdos y tres contra tres promedian parecido y dicen cosas distintas.