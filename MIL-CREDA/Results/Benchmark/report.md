# Fase uno — research-concept-r17.md

> resnet18  ·  3 épocas  ·  1 repetición(es)  ·  research-concept-r17.md  
> la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación: nada por debajo de eso lo resuelve una transferencia sola  
> !! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.  
> !! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.  

## 1 · Tiempo de entrenamiento (más bajo es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 4.82 ± 0.00 | 4.82 ± 0.00 | 4.68 ± 0.00 | 4.68 ± 0.00 | 4.67 ± 0.00 | 4.69 ± 0.00 | **4.73** |
| `CREDA*` | 10.68 ± 0.00 | 8.98 ± 0.00 | 8.73 ± 0.00 | 8.86 ± 0.00 | 8.66 ± 0.00 | 8.79 ± 0.00 | **9.12** |
| `CREDA` | 8.98 ± 0.00 | 8.80 ± 0.00 | 8.69 ± 0.00 | 9.38 ± 0.00 | 8.74 ± 0.00 | 9.24 ± 0.00 | **8.97** |
| `MIL-Baseline` | 4.86 ± 0.00 | 4.79 ± 0.00 | 4.77 ± 0.00 | 4.80 ± 0.00 | 4.76 ± 0.00 | 4.77 ± 0.00 | **4.79** |
| `MIL-CREDA**` | 12.40 ± 0.00 | 12.38 ± 0.00 | 12.60 ± 0.00 | 12.53 ± 0.00 | 12.28 ± 0.00 | 12.40 ± 0.00 | **12.43** |
| `MIL-CREDA*` | 12.07 ± 0.00 | 12.58 ± 0.00 | 12.46 ± 0.00 | 12.81 ± 0.00 | 12.42 ± 0.00 | 12.13 ± 0.00 | **12.41** |
| `MIL-CREDA` | 13.49 ± 0.00 | 13.42 ± 0.00 | 13.44 ± 0.00 | 13.61 ± 0.00 | 13.18 ± 0.00 | 13.46 ± 0.00 | **13.43** |
| `MIL-CREDA-U` | 13.41 ± 0.00 | 13.17 ± 0.00 | 13.04 ± 0.00 | 13.05 ± 0.00 | 12.80 ± 0.00 | 12.87 ± 0.00 | **13.06** |
| `MIL-CREDA-A` | 12.93 ± 0.00 | 13.21 ± 0.00 | 13.20 ± 0.00 | 12.93 ± 0.00 | 13.09 ± 0.00 | 13.11 ± 0.00 | **13.08** |
| `MIL-CREDA-K` | 13.42 ± 0.00 | 13.17 ± 0.00 | 13.09 ± 0.00 | 12.99 ± 0.00 | 12.78 ± 0.00 | 12.46 ± 0.00 | **12.99** |

Mejor promedio: **Baseline** con 4.73s; peor: MIL-CREDA con 13.43s. CREDA queda 4.25s por debajo de su piso Baseline. MIL-CREDA queda 8.64s por debajo de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

## 2 · Exactitud en fuente (más alto es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 41.7 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.9** |
| `CREDA*` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 41.7 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.9** |
| `CREDA` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 38.9 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.5** |
| `MIL-Baseline` | 100.0 ± 0.0 | 97.2 ± 0.0 | 97.2 ± 0.0 | 19.4 ± 0.0 | 91.7 ± 0.0 | 38.9 ± 0.0 | **74.1** |
| `MIL-CREDA**` | 100.0 ± 0.0 | 100.0 ± 0.0 | 91.7 ± 0.0 | 22.2 ± 0.0 | 69.4 ± 0.0 | 38.9 ± 0.0 | **70.4** |
| `MIL-CREDA*` | 91.7 ± 0.0 | 77.8 ± 0.0 | 91.7 ± 0.0 | 22.2 ± 0.0 | 77.8 ± 0.0 | 30.6 ± 0.0 | **65.3** |
| `MIL-CREDA` | 86.1 ± 0.0 | 100.0 ± 0.0 | 66.7 ± 0.0 | 27.8 ± 0.0 | 88.9 ± 0.0 | 33.3 ± 0.0 | **67.1** |
| `MIL-CREDA-U` | 80.6 ± 0.0 | 69.4 ± 0.0 | 77.8 ± 0.0 | 8.3 ± 0.0 | 66.7 ± 0.0 | 25.0 ± 0.0 | **54.6** |
| `MIL-CREDA-A` | 66.7 ± 0.0 | 94.4 ± 0.0 | 86.1 ± 0.0 | 8.3 ± 0.0 | 83.3 ± 0.0 | 5.6 ± 0.0 | **57.4** |
| `MIL-CREDA-K` | 86.1 ± 0.0 | 80.6 ± 0.0 | 83.3 ± 0.0 | 19.4 ± 0.0 | 75.0 ± 0.0 | 22.2 ± 0.0 | **61.1** |

Mejor promedio: **Baseline** con 81.9%; peor: MIL-CREDA-U con 54.6%. CREDA queda 0.5% por debajo de su piso Baseline. MIL-CREDA queda 6.9% por debajo de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

### 2b · Peldaños en fuente

| Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | gana izq. |
|---|---|---|---|---|---|---|---|---|
| Baseline → MIL-Baseline | +0.0 | +2.8 | +2.8 | +22.2 | +8.3 | +11.1 | **+7.9** | 5/6 |
| Baseline → CREDA* | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| CREDA* → CREDA | +0.0 | +0.0 | +0.0 | +2.8 | +0.0 | +0.0 | **+0.5** | 1/6 |
| MIL-Baseline → MIL-CREDA** | +0.0 | -2.8 | +5.6 | -2.8 | +22.2 | +0.0 | **+3.7** | 2/6 |
| MIL-CREDA** → MIL-CREDA* | +8.3 | +22.2 | +0.0 | +0.0 | -8.3 | +8.3 | **+5.1** | 3/6 |
| MIL-CREDA* → MIL-CREDA | +5.6 | -22.2 | +25.0 | -5.6 | -11.1 | -2.8 | **-1.9** | 2/6 |
| CREDA* → MIL-CREDA** | +0.0 | +0.0 | +8.3 | +19.4 | +30.6 | +11.1 | **+11.6** | 4/6 |
| CREDA → MIL-CREDA* | +8.3 | +22.2 | +8.3 | +16.7 | +22.2 | +19.4 | **+16.2** | 6/6 |
| CREDA → MIL-CREDA | +13.9 | +0.0 | +33.3 | +11.1 | +11.1 | +16.7 | **+14.4** | 5/6 |
| MIL-CREDA-U → MIL-CREDA-K | -5.6 | -11.1 | -5.6 | -11.1 | -8.3 | +2.8 | **-6.5** | 1/6 |
| MIL-CREDA-A → MIL-CREDA-K | -19.4 | +13.9 | +2.8 | -11.1 | +8.3 | -16.7 | **-3.7** | 3/6 |
| MIL-CREDA-K → MIL-CREDA | +0.0 | -19.4 | +16.7 | -8.3 | -13.9 | -11.1 | **-6.0** | 1/6 |

El peldaño que más separa es **CREDA → MIL-CREDA***: **CREDA** queda 16.2 por encima de MIL-CREDA*, y eso lee el mismo peldaño, construido de dos maneras: ponderado. Se inclinan igual en las 6 transferencias: Baseline → CREDA*, CREDA → MIL-CREDA*. Con esta cantidad de repeticiones lo que carga peso es la coincidencia entre transferencias, no la magnitud: seis acuerdos y tres contra tres promedian parecido y dicen cosas distintas.

## 3 · Exactitud en destino (más alto es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 77.8 ± 0.0 | 97.2 ± 0.0 | 8.3 ± 0.0 | 72.2 ± 0.0 | 19.4 ± 0.0 | 47.2 ± 0.0 | **53.7** |
| `CREDA*` | 83.3 ± 0.0 | 100.0 ± 0.0 | 8.3 ± 0.0 | 72.2 ± 0.0 | 22.2 ± 0.0 | 47.2 ± 0.0 | **55.6** |
| `CREDA` | 72.2 ± 0.0 | 97.2 ± 0.0 | 13.9 ± 0.0 | 52.8 ± 0.0 | 19.4 ± 0.0 | 36.1 ± 0.0 | **48.6** |
| `MIL-Baseline` | 63.9 ± 0.0 | 80.6 ± 0.0 | 19.4 ± 0.0 | 36.1 ± 0.0 | 13.9 ± 0.0 | 30.6 ± 0.0 | **40.7** |
| `MIL-CREDA**` | 66.7 ± 0.0 | 72.2 ± 0.0 | 16.7 ± 0.0 | 25.0 ± 0.0 | 16.7 ± 0.0 | 30.6 ± 0.0 | **38.0** |
| `MIL-CREDA*` | 58.3 ± 0.0 | 69.4 ± 0.0 | 27.8 ± 0.0 | 25.0 ± 0.0 | 8.3 ± 0.0 | 13.9 ± 0.0 | **33.8** |
| `MIL-CREDA` | 44.4 ± 0.0 | 66.7 ± 0.0 | 22.2 ± 0.0 | 27.8 ± 0.0 | 8.3 ± 0.0 | 19.4 ± 0.0 | **31.5** |
| `MIL-CREDA-U` | 19.4 ± 0.0 | 41.7 ± 0.0 | 13.9 ± 0.0 | 19.4 ± 0.0 | 11.1 ± 0.0 | 25.0 ± 0.0 | **21.8** |
| `MIL-CREDA-A` | 44.4 ± 0.0 | 50.0 ± 0.0 | 16.7 ± 0.0 | 19.4 ± 0.0 | 19.4 ± 0.0 | 22.2 ± 0.0 | **28.7** |
| `MIL-CREDA-K` | 38.9 ± 0.0 | 75.0 ± 0.0 | 16.7 ± 0.0 | 19.4 ± 0.0 | 11.1 ± 0.0 | 16.7 ± 0.0 | **29.6** |

Mejor promedio: **CREDA*** con 55.6%; peor: MIL-CREDA-U con 21.8%. CREDA queda 5.1% por debajo de su piso Baseline. MIL-CREDA queda 9.3% por debajo de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

### 3b · Peldaños en destino

| Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | gana izq. |
|---|---|---|---|---|---|---|---|---|
| Baseline → MIL-Baseline | +13.9 | +16.7 | -11.1 | +36.1 | +5.6 | +16.7 | **+13.0** | 5/6 |
| Baseline → CREDA* | -5.6 | -2.8 | +0.0 | +0.0 | -2.8 | +0.0 | **-1.9** | 0/6 |
| CREDA* → CREDA | +11.1 | +2.8 | -5.6 | +19.4 | +2.8 | +11.1 | **+6.9** | 5/6 |
| MIL-Baseline → MIL-CREDA** | -2.8 | +8.3 | +2.8 | +11.1 | -2.8 | +0.0 | **+2.8** | 3/6 |
| MIL-CREDA** → MIL-CREDA* | +8.3 | +2.8 | -11.1 | +0.0 | +8.3 | +16.7 | **+4.2** | 4/6 |
| MIL-CREDA* → MIL-CREDA | +13.9 | +2.8 | +5.6 | -2.8 | +0.0 | -5.6 | **+2.3** | 3/6 |
| CREDA* → MIL-CREDA** | +16.7 | +27.8 | -8.3 | +47.2 | +5.6 | +16.7 | **+17.6** | 5/6 |
| CREDA → MIL-CREDA* | +13.9 | +27.8 | -13.9 | +27.8 | +11.1 | +22.2 | **+14.8** | 5/6 |
| CREDA → MIL-CREDA | +27.8 | +30.6 | -8.3 | +25.0 | +11.1 | +16.7 | **+17.1** | 5/6 |
| MIL-CREDA-U → MIL-CREDA-K | -19.4 | -33.3 | -2.8 | +0.0 | +0.0 | +8.3 | **-7.9** | 1/6 |
| MIL-CREDA-A → MIL-CREDA-K | +5.6 | -25.0 | +0.0 | +0.0 | +8.3 | +5.6 | **-0.9** | 3/6 |
| MIL-CREDA-K → MIL-CREDA | -5.6 | +8.3 | -5.6 | -8.3 | +2.8 | -2.8 | **-1.9** | 2/6 |

El peldaño que más separa es **CREDA* → MIL-CREDA****: **CREDA*** queda 17.6 por encima de MIL-CREDA**, y eso lee el mismo peldaño, construido de dos maneras: sin ponderar. Se inclinan igual en las 6 transferencias: Baseline → CREDA*. Con esta cantidad de repeticiones lo que carga peso es la coincidencia entre transferencias, no la magnitud: seis acuerdos y tres contra tres promedian parecido y dicen cosas distintas.