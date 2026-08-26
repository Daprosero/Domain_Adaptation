# Fase uno — research-concept-r17.md

> resnet18  ·  3 épocas  ·  1 repetición(es)  ·  research-concept-r17.md  
> la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación: nada por debajo de eso lo resuelve una transferencia sola  
> !! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.  
> !! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.  

## 0 · El techo de cada familia

| Familia | Brazo | 0.0001 | 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|---|---|
| `creda` | `D` | **85.8** | 85.8 | 85.8 | 59.2 | 50.0 |
| `milcreda` | `G` | 45.8 | 47.5 | **60.0** | 55.0 | 43.3 |

**creda** se queda en 0.0001, elegido por desempate entre 3 techos empatados sobre el rol `valid` con 3 repetición(es) de 20 épocas y las semillas **no** coinciden entre sí. **milcreda** se queda en 0.01, elegido por una diferencia en el criterio sobre el rol `valid` con 3 repetición(es) de 20 épocas y las semillas coinciden. La rejilla no se inclinó para creda: el techo lo puso la regla de desempate y no el criterio, así que sostiene menos de lo que un número elegido parece sostener.

### 0b · Qué techo rige en cada transferencia

| Familia | M->U | U->M | M->S | S->M | U->S | S->U |
|---|---|---|---|---|---|---|
| `creda` | **0.0001** | 0.0001 | 0.0001 | **0.0001** | 0.0001 | 0.0001 |
| `milcreda` | **0.01** | 0.01 | 0.01 | **0.0001** | 0.01 | 0.01 |

En las transferencias que la búsqueda midió rige el ganador de esa transferencia, por la misma lectura apareada y el mismo desempate. En las restantes rige el ganador de las medidas tomadas juntas: es una aplicación fuera de muestra y se declara como tal, porque ese escalar no se eligió mirándolas. **creda**: 2 medida(s) y 4 heredada(s), todas en 0.0001. Ninguna transferencia medida se aparta del ganador agrupado, así que separar las dos lecturas no cambió ningún techo de esta familia. **milcreda**: 2 medida(s), 4 heredada(s) a 0.01, y 1 de las medidas elige otro techo — `S->M` a 0.0001. Ahí la familia deja de correr a un coeficiente único, así que su promedio entre transferencias mezcla dos escalares; dentro de cada transferencia todos los brazos siguen compartiendo el techo, que es lo que mantiene atribuible cada peldaño.

## 1 · Tiempo de entrenamiento (más bajo es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 4.72 ± 0.00 | 4.31 ± 0.00 | 4.27 ± 0.00 | 4.26 ± 0.00 | 4.26 ± 0.00 | 4.27 ± 0.00 | **4.35** |
| `CREDA*` | 9.12 ± 0.00 | 7.89 ± 0.00 | 7.58 ± 0.00 | 7.91 ± 0.00 | 7.94 ± 0.00 | 7.73 ± 0.00 | **8.03** |
| `CREDA` | 7.71 ± 0.00 | 7.96 ± 0.00 | 7.43 ± 0.00 | 7.72 ± 0.00 | 7.73 ± 0.00 | 7.86 ± 0.00 | **7.74** |
| `MIL-Baseline` | 4.51 ± 0.00 | 4.37 ± 0.00 | 4.34 ± 0.00 | 4.65 ± 0.00 | 4.35 ± 0.00 | 4.32 ± 0.00 | **4.42** |
| `MIL-CREDA**` | 11.37 ± 0.00 | 11.16 ± 0.00 | 11.14 ± 0.00 | 11.67 ± 0.00 | 11.17 ± 0.00 | 11.21 ± 0.00 | **11.28** |
| `MIL-CREDA*` | 11.29 ± 0.00 | 11.13 ± 0.00 | 11.19 ± 0.00 | 11.47 ± 0.00 | 11.10 ± 0.00 | 11.28 ± 0.00 | **11.24** |
| `MIL-CREDA` | 12.32 ± 0.00 | 12.17 ± 0.00 | 12.14 ± 0.00 | 12.17 ± 0.00 | 12.09 ± 0.00 | 12.31 ± 0.00 | **12.20** |
| `MIL-CREDA-U` | 11.76 ± 0.00 | 11.75 ± 0.00 | 11.91 ± 0.00 | 12.17 ± 0.00 | 11.84 ± 0.00 | 11.84 ± 0.00 | **11.88** |
| `MIL-CREDA-A` | 11.66 ± 0.00 | 11.59 ± 0.00 | 12.06 ± 0.00 | 12.13 ± 0.00 | 12.21 ± 0.00 | 11.85 ± 0.00 | **11.91** |
| `MIL-CREDA-K` | 12.00 ± 0.00 | 12.00 ± 0.00 | 12.24 ± 0.00 | 11.92 ± 0.00 | 11.97 ± 0.00 | 11.85 ± 0.00 | **12.00** |

Mejor promedio: **Baseline**; peor: MIL-CREDA, a 7.85s de distancia. CREDA queda 3.39s por debajo de su piso Baseline. MIL-CREDA queda 7.78s por debajo de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

## 2 · Exactitud en fuente (más alto es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 38.9 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.5** |
| `CREDA*` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 38.9 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.5** |
| `CREDA` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 38.9 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.5** |
| `MIL-Baseline` | 100.0 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 86.1 ± 0.0 | 30.6 ± 0.0 | **72.7** |
| `MIL-CREDA**` | 100.0 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 83.3 ± 0.0 | 30.6 ± 0.0 | **72.2** |
| `MIL-CREDA*` | 100.0 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 91.7 ± 0.0 | 33.3 ± 0.0 | **74.1** |
| `MIL-CREDA` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 91.7 ± 0.0 | 33.3 ± 0.0 | **74.5** |
| `MIL-CREDA-U` | 100.0 ± 0.0 | 86.1 ± 0.0 | 94.4 ± 0.0 | 19.4 ± 0.0 | 83.3 ± 0.0 | 36.1 ± 0.0 | **69.9** |
| `MIL-CREDA-A` | 80.6 ± 0.0 | 97.2 ± 0.0 | 91.7 ± 0.0 | 16.7 ± 0.0 | 83.3 ± 0.0 | 30.6 ± 0.0 | **66.7** |
| `MIL-CREDA-K` | 97.2 ± 0.0 | 91.7 ± 0.0 | 91.7 ± 0.0 | 19.4 ± 0.0 | 94.4 ± 0.0 | 27.8 ± 0.0 | **70.4** |

Mejor promedio: **Baseline**; peor: MIL-CREDA-A, a 14.8% de distancia. CREDA no se separa de su piso Baseline. MIL-CREDA queda 1.9% por encima de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

### 2b · Peldaños en fuente

| Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | gana izq. |
|---|---|---|---|---|---|---|---|---|
| Baseline → MIL-Baseline | +0.0 | +2.8 | +0.0 | +16.7 | +13.9 | +19.4 | **+8.8** | 4/6 |
| Baseline → CREDA* | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| CREDA* → CREDA | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| MIL-Baseline → MIL-CREDA** | +0.0 | +0.0 | +0.0 | +0.0 | +2.8 | +0.0 | **+0.5** | 1/6 |
| MIL-CREDA** → MIL-CREDA* | +0.0 | +0.0 | +0.0 | +0.0 | -8.3 | -2.8 | **-1.9** | 0/6 |
| MIL-CREDA* → MIL-CREDA | +0.0 | -2.8 | +0.0 | +0.0 | +0.0 | +0.0 | **-0.5** | 0/6 |
| CREDA* → MIL-CREDA** | +0.0 | +2.8 | +0.0 | +16.7 | +16.7 | +19.4 | **+9.3** | 4/6 |
| CREDA → MIL-CREDA* | +0.0 | +2.8 | +0.0 | +16.7 | +8.3 | +16.7 | **+7.4** | 4/6 |
| CREDA → MIL-CREDA | +0.0 | +0.0 | +0.0 | +16.7 | +8.3 | +16.7 | **+6.9** | 3/6 |
| MIL-CREDA-U → MIL-CREDA-K | +2.8 | -5.6 | +2.8 | +0.0 | -11.1 | +8.3 | **-0.5** | 3/6 |
| MIL-CREDA-A → MIL-CREDA-K | -16.7 | +5.6 | +0.0 | -2.8 | -11.1 | +2.8 | **-3.7** | 2/6 |
| MIL-CREDA-K → MIL-CREDA | -2.8 | -8.3 | -8.3 | -2.8 | +2.8 | -5.6 | **-4.2** | 1/6 |

El peldaño que más separa es **CREDA* → MIL-CREDA****: **CREDA*** queda 9.3 por encima de MIL-CREDA**, y eso lee el mismo peldaño, construido de dos maneras: sin ponderar. Se inclinan igual en las 6 transferencias: Baseline → CREDA*, CREDA* → CREDA, MIL-CREDA** → MIL-CREDA*, MIL-CREDA* → MIL-CREDA. Con esta cantidad de repeticiones lo que carga peso es la coincidencia entre transferencias, no la magnitud: seis acuerdos y tres contra tres promedian parecido y dicen cosas distintas.

## 3 · Exactitud en destino (más alto es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 77.8 ± 0.0 | 94.4 ± 0.0 | 8.3 ± 0.0 | 69.4 ± 0.0 | 16.7 ± 0.0 | 47.2 ± 0.0 | **52.3** |
| `CREDA*` | 77.8 ± 0.0 | 94.4 ± 0.0 | 8.3 ± 0.0 | 69.4 ± 0.0 | 16.7 ± 0.0 | 50.0 ± 0.0 | **52.8** |
| `CREDA` | 77.8 ± 0.0 | 94.4 ± 0.0 | 8.3 ± 0.0 | 69.4 ± 0.0 | 16.7 ± 0.0 | 50.0 ± 0.0 | **52.8** |
| `MIL-Baseline` | 61.1 ± 0.0 | 80.6 ± 0.0 | 16.7 ± 0.0 | 38.9 ± 0.0 | 19.4 ± 0.0 | 27.8 ± 0.0 | **40.7** |
| `MIL-CREDA**` | 69.4 ± 0.0 | 83.3 ± 0.0 | 13.9 ± 0.0 | 38.9 ± 0.0 | 25.0 ± 0.0 | 30.6 ± 0.0 | **43.5** |
| `MIL-CREDA*` | 72.2 ± 0.0 | 83.3 ± 0.0 | 19.4 ± 0.0 | 38.9 ± 0.0 | 25.0 ± 0.0 | 30.6 ± 0.0 | **44.9** |
| `MIL-CREDA` | 72.2 ± 0.0 | 83.3 ± 0.0 | 19.4 ± 0.0 | 38.9 ± 0.0 | 25.0 ± 0.0 | 27.8 ± 0.0 | **44.4** |
| `MIL-CREDA-U` | 25.0 ± 0.0 | 88.9 ± 0.0 | 11.1 ± 0.0 | 27.8 ± 0.0 | 11.1 ± 0.0 | 33.3 ± 0.0 | **32.9** |
| `MIL-CREDA-A` | 38.9 ± 0.0 | 75.0 ± 0.0 | 22.2 ± 0.0 | 30.6 ± 0.0 | 19.4 ± 0.0 | 27.8 ± 0.0 | **35.6** |
| `MIL-CREDA-K` | 52.8 ± 0.0 | 77.8 ± 0.0 | 22.2 ± 0.0 | 41.7 ± 0.0 | 22.2 ± 0.0 | 36.1 ± 0.0 | **42.1** |

Mejor promedio: **CREDA***; peor: MIL-CREDA-U, a 19.9% de distancia. CREDA queda 0.5% por encima de su piso Baseline. MIL-CREDA queda 3.7% por encima de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

### 3b · Peldaños en destino

| Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | gana izq. |
|---|---|---|---|---|---|---|---|---|
| Baseline → MIL-Baseline | +16.7 | +13.9 | -8.3 | +30.6 | -2.8 | +19.4 | **+11.6** | 4/6 |
| Baseline → CREDA* | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -2.8 | **-0.5** | 0/6 |
| CREDA* → CREDA | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| MIL-Baseline → MIL-CREDA** | -8.3 | -2.8 | +2.8 | +0.0 | -5.6 | -2.8 | **-2.8** | 1/6 |
| MIL-CREDA** → MIL-CREDA* | -2.8 | +0.0 | -5.6 | +0.0 | +0.0 | +0.0 | **-1.4** | 0/6 |
| MIL-CREDA* → MIL-CREDA | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +2.8 | **+0.5** | 1/6 |
| CREDA* → MIL-CREDA** | +8.3 | +11.1 | -5.6 | +30.6 | -8.3 | +19.4 | **+9.3** | 4/6 |
| CREDA → MIL-CREDA* | +5.6 | +11.1 | -11.1 | +30.6 | -8.3 | +19.4 | **+7.9** | 4/6 |
| CREDA → MIL-CREDA | +5.6 | +11.1 | -11.1 | +30.6 | -8.3 | +22.2 | **+8.3** | 4/6 |
| MIL-CREDA-U → MIL-CREDA-K | -27.8 | +11.1 | -11.1 | -13.9 | -11.1 | -2.8 | **-9.3** | 1/6 |
| MIL-CREDA-A → MIL-CREDA-K | -13.9 | -2.8 | +0.0 | -11.1 | -2.8 | -8.3 | **-6.5** | 0/6 |
| MIL-CREDA-K → MIL-CREDA | -19.4 | -5.6 | +2.8 | +2.8 | -2.8 | +8.3 | **-2.3** | 3/6 |

El peldaño que más separa es **Baseline → MIL-Baseline**: **Baseline** queda 11.6 por encima de MIL-Baseline, y eso lee qué compra la representación por bolsas, con la adaptación apagada. Se inclinan igual en las 6 transferencias: Baseline → CREDA*, CREDA* → CREDA, MIL-CREDA** → MIL-CREDA*, MIL-CREDA-A → MIL-CREDA-K. Con esta cantidad de repeticiones lo que carga peso es la coincidencia entre transferencias, no la magnitud: seis acuerdos y tres contra tres promedian parecido y dicen cosas distintas.