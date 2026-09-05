# Informe de campaña (v1) — research-concept-r17.md

> resnet18  ·  3 épocas  ·  1 repetición(es)  ·  research-concept-r17.md  
> la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación: nada por debajo de eso lo resuelve una transferencia sola  
> !! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.  
> !! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.  

## 0 · El techo de cada familia

Sin búsqueda de techos: no hay rejilla que mostrar. La campaña se niega a correr hasta que exista.

Sin búsqueda: ningún techo está elegido y nada de abajo puede correr.

### 0b · Qué techo rige en cada transferencia

Sin búsqueda de techos: ninguna transferencia tiene techo elegido.

Sin búsqueda: no hay regla que aplicar todavía.

## 1 · Tiempo de entrenamiento (más bajo es mejor)

(sin corridas medidas)

Sin conclusión: tiempo de entrenamiento no se promedia entre máquinas — cada corrida es la lectura de su propio entorno, no una propiedad del método ni de la máquina que la corrió. Ver la tabla de arriba, corrida por corrida.

## 2 · Exactitud en fuente (más alto es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 38.9 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.5** |
| `CREDA*` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 38.9 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.5** |
| `CREDA` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 38.9 ± 0.0 | 100.0 ± 0.0 | 50.0 ± 0.0 | **81.5** |
| `MIL-Baseline` | 100.0 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 86.1 ± 0.0 | 30.6 ± 0.0 | **72.7** |
| `MIL-CREDA**` | 100.0 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 86.1 ± 0.0 | 30.6 ± 0.0 | **72.7** |
| `MIL-CREDA*` | 100.0 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 86.1 ± 0.0 | 30.6 ± 0.0 | **72.7** |
| `MIL-CREDA` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 86.1 ± 0.0 | 30.6 ± 0.0 | **73.1** |
| `MIL-CREDA-U` | 88.9 ± 0.0 | 83.3 ± 0.0 | 88.9 ± 0.0 | 19.4 ± 0.0 | 83.3 ± 0.0 | 33.3 ± 0.0 | **66.2** |
| `MIL-CREDA-A` | 80.6 ± 0.0 | 97.2 ± 0.0 | 88.9 ± 0.0 | 19.4 ± 0.0 | 83.3 ± 0.0 | 36.1 ± 0.0 | **67.6** |
| `MIL-CREDA-K` | 94.4 ± 0.0 | 91.7 ± 0.0 | 94.4 ± 0.0 | 19.4 ± 0.0 | 97.2 ± 0.0 | 25.0 ± 0.0 | **70.4** |

Mejor promedio: **Baseline**; peor: MIL-CREDA-U, a 15.3% de distancia. CREDA no se separa de su piso Baseline. MIL-CREDA queda 0.5% por encima de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

### 2b · Peldaños en fuente

| Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | gana izq. |
|---|---|---|---|---|---|---|---|---|
| Baseline → MIL-Baseline | +0.0 | +2.8 | +0.0 | +16.7 | +13.9 | +19.4 | **+8.8** | 4/6 |
| Baseline → CREDA* | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| CREDA* → CREDA | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| MIL-Baseline → MIL-CREDA** | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| MIL-CREDA** → MIL-CREDA* | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| MIL-CREDA* → MIL-CREDA | +0.0 | -2.8 | +0.0 | +0.0 | +0.0 | +0.0 | **-0.5** | 0/6 |
| CREDA* → MIL-CREDA** | +0.0 | +2.8 | +0.0 | +16.7 | +13.9 | +19.4 | **+8.8** | 4/6 |
| CREDA → MIL-CREDA* | +0.0 | +2.8 | +0.0 | +16.7 | +13.9 | +19.4 | **+8.8** | 4/6 |
| CREDA → MIL-CREDA | +0.0 | +0.0 | +0.0 | +16.7 | +13.9 | +19.4 | **+8.3** | 3/6 |
| MIL-CREDA-U → MIL-CREDA-K | -5.6 | -8.3 | -5.6 | +0.0 | -13.9 | +8.3 | **-4.2** | 1/6 |
| MIL-CREDA-A → MIL-CREDA-K | -13.9 | +5.6 | -5.6 | +0.0 | -13.9 | +11.1 | **-2.8** | 2/6 |
| MIL-CREDA-K → MIL-CREDA | -5.6 | -8.3 | -5.6 | -2.8 | +11.1 | -5.6 | **-2.8** | 1/6 |

El peldaño que más separa es **Baseline → MIL-Baseline**: **Baseline** queda 8.8 por encima de MIL-Baseline, y eso lee qué compra la representación por bolsas, con la adaptación apagada. Se inclinan igual en las 6 transferencias: Baseline → CREDA*, CREDA* → CREDA, MIL-Baseline → MIL-CREDA**, MIL-CREDA** → MIL-CREDA*, MIL-CREDA* → MIL-CREDA. Con esta cantidad de repeticiones lo que carga peso es la coincidencia entre transferencias, no la magnitud: seis acuerdos y tres contra tres promedian parecido y dicen cosas distintas.

## 3 · Exactitud en destino (más alto es mejor)

| Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|
| `Baseline` | 77.8 ± 0.0 | 94.4 ± 0.0 | 8.3 ± 0.0 | 69.4 ± 0.0 | 16.7 ± 0.0 | 47.2 ± 0.0 | **52.3** |
| `CREDA*` | 77.8 ± 0.0 | 94.4 ± 0.0 | 8.3 ± 0.0 | 69.4 ± 0.0 | 16.7 ± 0.0 | 47.2 ± 0.0 | **52.3** |
| `CREDA` | 77.8 ± 0.0 | 94.4 ± 0.0 | 8.3 ± 0.0 | 69.4 ± 0.0 | 16.7 ± 0.0 | 47.2 ± 0.0 | **52.3** |
| `MIL-Baseline` | 61.1 ± 0.0 | 80.6 ± 0.0 | 16.7 ± 0.0 | 38.9 ± 0.0 | 19.4 ± 0.0 | 27.8 ± 0.0 | **40.7** |
| `MIL-CREDA**` | 72.2 ± 0.0 | 80.6 ± 0.0 | 16.7 ± 0.0 | 38.9 ± 0.0 | 22.2 ± 0.0 | 27.8 ± 0.0 | **43.1** |
| `MIL-CREDA*` | 63.9 ± 0.0 | 80.6 ± 0.0 | 16.7 ± 0.0 | 41.7 ± 0.0 | 19.4 ± 0.0 | 27.8 ± 0.0 | **41.7** |
| `MIL-CREDA` | 66.7 ± 0.0 | 80.6 ± 0.0 | 16.7 ± 0.0 | 41.7 ± 0.0 | 19.4 ± 0.0 | 27.8 ± 0.0 | **42.1** |
| `MIL-CREDA-U` | 25.0 ± 0.0 | 77.8 ± 0.0 | 11.1 ± 0.0 | 30.6 ± 0.0 | 13.9 ± 0.0 | 38.9 ± 0.0 | **32.9** |
| `MIL-CREDA-A` | 41.7 ± 0.0 | 75.0 ± 0.0 | 22.2 ± 0.0 | 27.8 ± 0.0 | 19.4 ± 0.0 | 25.0 ± 0.0 | **35.2** |
| `MIL-CREDA-K` | 55.6 ± 0.0 | 66.7 ± 0.0 | 27.8 ± 0.0 | 41.7 ± 0.0 | 22.2 ± 0.0 | 44.4 ± 0.0 | **43.1** |

Mejor promedio: **Baseline**; peor: MIL-CREDA-U, a 19.4% de distancia. CREDA no se separa de su piso Baseline. MIL-CREDA queda 1.4% por encima de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

### 3b · Peldaños en destino

| Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | gana izq. |
|---|---|---|---|---|---|---|---|---|
| Baseline → MIL-Baseline | +16.7 | +13.9 | -8.3 | +30.6 | -2.8 | +19.4 | **+11.6** | 4/6 |
| Baseline → CREDA* | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| CREDA* → CREDA | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| MIL-Baseline → MIL-CREDA** | -11.1 | +0.0 | +0.0 | +0.0 | -2.8 | +0.0 | **-2.3** | 0/6 |
| MIL-CREDA** → MIL-CREDA* | +8.3 | +0.0 | +0.0 | -2.8 | +2.8 | +0.0 | **+1.4** | 2/6 |
| MIL-CREDA* → MIL-CREDA | -2.8 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **-0.5** | 0/6 |
| CREDA* → MIL-CREDA** | +5.6 | +13.9 | -8.3 | +30.6 | -5.6 | +19.4 | **+9.3** | 4/6 |
| CREDA → MIL-CREDA* | +13.9 | +13.9 | -8.3 | +27.8 | -2.8 | +19.4 | **+10.6** | 4/6 |
| CREDA → MIL-CREDA | +11.1 | +13.9 | -8.3 | +27.8 | -2.8 | +19.4 | **+10.2** | 4/6 |
| MIL-CREDA-U → MIL-CREDA-K | -30.6 | +11.1 | -16.7 | -11.1 | -8.3 | -5.6 | **-10.2** | 1/6 |
| MIL-CREDA-A → MIL-CREDA-K | -13.9 | +8.3 | -5.6 | -13.9 | -2.8 | -19.4 | **-7.9** | 1/6 |
| MIL-CREDA-K → MIL-CREDA | -11.1 | -13.9 | +11.1 | +0.0 | +2.8 | +16.7 | **+0.9** | 3/6 |

El peldaño que más separa es **Baseline → MIL-Baseline**: **Baseline** queda 11.6 por encima de MIL-Baseline, y eso lee qué compra la representación por bolsas, con la adaptación apagada. Se inclinan igual en las 6 transferencias: Baseline → CREDA*, CREDA* → CREDA, MIL-Baseline → MIL-CREDA**, MIL-CREDA* → MIL-CREDA. Con esta cantidad de repeticiones lo que carga peso es la coincidencia entre transferencias, no la magnitud: seis acuerdos y tres contra tres promedian parecido y dicen cosas distintas.