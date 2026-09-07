# Informe de campaña (v1) — research-concept-r17.md

> resnet18  ·  3 épocas  ·  1 repetición(es)  ·  research-concept-r17.md  
> la exactitud se mueve de a 2.78 puntos sobre 36 bolsas de evaluación: nada por debajo de eso lo resuelve una transferencia sola  
> !! 1 repetición(es): el ± de abajo es cero por construcción, no por acuerdo. Son estimaciones puntuales, no resultados.  
> !! piloto: el protocolo declara 30 repeticiones y 20 épocas. Nada de esto es un resultado.  

## 0 · El techo de cada familia

**Estos techos son de un ENSAYO** (3 épocas), porque no hay búsqueda completa. El protocolo pide 20 épocas: no se citan como resultados, ni en el informe, ni en el resumen, ni en conversación.

| Familia | Brazo | Transferencia | Techo | Criterio | Meseta | Trials |
|---|---|---|---|---|---|---|
| `creda` | `D` | M->S | **0.00282167** | 10.0 | 4 | 4 |
| `creda` | `D` | M->U | **0.000135666** | 80.0 | 4 | 4 |
| `creda` | `D` | S->M | **0.000165025** | 75.0 | 3 | 4 |
| `creda` | `D` | S->U | **0.00121795** | 50.0 | 4 | 4 |
| `creda` | `D` | U->M | **0.000302765** | 100.0 | 4 | 4 |
| `creda` | `D` | U->S | **0.000105352** | 20.0 | 4 | 4 |
| `milcreda` | `G` | M->S | **0.000643961** | 20.0 | 4 | 4 |
| `milcreda` | `G` | M->U | **0.00515442** | 80.0 | 2 | 4 |
| `milcreda` | `G` | S->M | **0.000558888** | 35.0 | 4 | 4 |
| `milcreda` | `G` | S->U | **0.000308476** | 35.0 | 4 | 4 |
| `milcreda` | `G` | U->M | **0.000582196** | 85.0 | 3 | 4 |
| `milcreda` | `G` | U->S | **0.0035048** | 15.0 | 4 | 4 |

En negrita el techo que puso la regla de meseta y no el criterio. **Meseta** es cuántos techos el ruido estimado no distinguió del mejor: uno significa que el criterio decidió.

**creda** se queda en 0.000302765, elegido por una diferencia en el criterio sobre el rol `valid` con 4 trial(s) de 3 épocas y la resolución del criterio es 0.05, **por debajo de la escala que su respuesta necesita**. **milcreda** se queda en 0.000582196, elegido por una diferencia en el criterio sobre el rol `valid` con 4 trial(s) de 3 épocas y la resolución del criterio es 0.05, **por debajo de la escala que su respuesta necesita**.

### 0b · Qué techo rige en cada transferencia

**Estos techos son de un ENSAYO** (3 épocas), porque no hay búsqueda completa. El protocolo pide 20 épocas: no se citan como resultados, ni en el informe, ni en el resumen, ni en conversación.

| Familia | M->U | U->M | M->S | S->M | U->S | S->U |
|---|---|---|---|---|---|---|
| `creda` | **0.000135666** | **0.000302765** | **0.00282167** | **0.000165025** | **0.000105352** | **0.00121795** |
| `milcreda` | **0.00515442** | **0.000582196** | **0.000643961** | **0.000558888** | **0.0035048** | **0.000308476** |

En las transferencias que la búsqueda midió rige el ganador de esa transferencia, por la misma lectura apareada y el mismo desempate. En las restantes rige el ganador de las medidas tomadas juntas: es una aplicación fuera de muestra y se declara como tal, porque ese escalar no se eligió mirándolas. **creda**: 6 medida(s), 0 heredada(s), y 5 de las medidas elige otro techo — `M->S`, `M->U`, `S->M`, `S->U`, `U->S`. Entre el techo más alto y el más bajo de la fila hay un factor 26.8. Ahí la familia deja de correr a un coeficiente único, así que su promedio entre transferencias mezcla escalares distintos; dentro de cada transferencia todos los brazos siguen compartiendo el techo, que es lo que mantiene atribuible cada peldaño. **milcreda**: 6 medida(s), 0 heredada(s), y 5 de las medidas elige otro techo — `M->S`, `M->U`, `S->M`, `S->U`, `U->S`. Entre el techo más alto y el más bajo de la fila hay un factor 16.7. Ahí la familia deja de correr a un coeficiente único, así que su promedio entre transferencias mezcla escalares distintos; dentro de cada transferencia todos los brazos siguen compartiendo el techo, que es lo que mantiene atribuible cada peldaño.

## 1 · Tiempo de entrenamiento (más bajo es mejor)

> cada fila es una corrida en su propia máquina, nunca promediada con otra: no hay una columna de método que hable por todas.  

| Método | Transferencia | Entorno | Semilla | tiempo de entrenamiento (s) |
|---|---|---|---|---|
| `Baseline` | M->U | `7c284fdfafc7` | 0 | 4.98 |
| `CREDA*` | M->U | `7c284fdfafc7` | 0 | 10.14 |
| `CREDA` | M->U | `7c284fdfafc7` | 0 | 8.42 |
| `MIL-Baseline` | M->U | `7c284fdfafc7` | 0 | 4.89 |
| `MIL-CREDA**` | M->U | `7c284fdfafc7` | 0 | 12.56 |
| `MIL-CREDA*` | M->U | `7c284fdfafc7` | 0 | 12.00 |
| `MIL-CREDA` | M->U | `7c284fdfafc7` | 0 | 13.83 |
| `MIL-CREDA-U` | M->U | `7c284fdfafc7` | 0 | 13.69 |
| `MIL-CREDA-A` | M->U | `7c284fdfafc7` | 0 | 17.18 |
| `MIL-CREDA-K` | M->U | `7c284fdfafc7` | 0 | 25.22 |
| `Baseline` | U->M | `7c284fdfafc7` | 0 | 5.12 |
| `CREDA*` | U->M | `7c284fdfafc7` | 0 | 10.74 |
| `CREDA` | U->M | `7c284fdfafc7` | 0 | 8.42 |
| `MIL-Baseline` | U->M | `7c284fdfafc7` | 0 | 4.84 |
| `MIL-CREDA**` | U->M | `7c284fdfafc7` | 0 | 12.10 |
| `MIL-CREDA*` | U->M | `7c284fdfafc7` | 0 | 11.87 |
| `MIL-CREDA` | U->M | `7c284fdfafc7` | 0 | 12.95 |
| `MIL-CREDA-U` | U->M | `7c284fdfafc7` | 0 | 15.69 |
| `MIL-CREDA-A` | U->M | `7c284fdfafc7` | 0 | 24.63 |
| `MIL-CREDA-K` | U->M | `7c284fdfafc7` | 0 | 13.17 |
| `Baseline` | M->S | `7c284fdfafc7` | 0 | 5.09 |
| `CREDA*` | M->S | `7c284fdfafc7` | 0 | 12.92 |
| `CREDA` | M->S | `7c284fdfafc7` | 0 | 11.83 |
| `MIL-Baseline` | M->S | `7c284fdfafc7` | 0 | 4.99 |
| `MIL-CREDA**` | M->S | `7c284fdfafc7` | 0 | 11.78 |
| `MIL-CREDA*` | M->S | `7c284fdfafc7` | 0 | 12.41 |
| `MIL-CREDA` | M->S | `7c284fdfafc7` | 0 | 13.38 |
| `MIL-CREDA-U` | M->S | `7c284fdfafc7` | 0 | 13.44 |
| `MIL-CREDA-A` | M->S | `7c284fdfafc7` | 0 | 12.19 |
| `MIL-CREDA-K` | M->S | `7c284fdfafc7` | 0 | 12.23 |
| `Baseline` | S->M | `7c284fdfafc7` | 0 | 4.68 |
| `CREDA*` | S->M | `7c284fdfafc7` | 0 | 9.19 |
| `CREDA` | S->M | `7c284fdfafc7` | 0 | 8.08 |
| `MIL-Baseline` | S->M | `7c284fdfafc7` | 0 | 4.68 |
| `MIL-CREDA**` | S->M | `7c284fdfafc7` | 0 | 11.84 |
| `MIL-CREDA*` | S->M | `7c284fdfafc7` | 0 | 11.70 |
| `MIL-CREDA` | S->M | `7c284fdfafc7` | 0 | 13.05 |
| `MIL-CREDA-U` | S->M | `7c284fdfafc7` | 0 | 12.33 |
| `MIL-CREDA-A` | S->M | `7c284fdfafc7` | 0 | 13.58 |
| `MIL-CREDA-K` | S->M | `7c284fdfafc7` | 0 | 20.03 |
| `Baseline` | U->S | `7c284fdfafc7` | 0 | 5.20 |
| `CREDA*` | U->S | `7c284fdfafc7` | 0 | 10.87 |
| `CREDA` | U->S | `7c284fdfafc7` | 0 | 7.80 |
| `MIL-Baseline` | U->S | `7c284fdfafc7` | 0 | 4.82 |
| `MIL-CREDA**` | U->S | `7c284fdfafc7` | 0 | 11.11 |
| `MIL-CREDA*` | U->S | `7c284fdfafc7` | 0 | 11.20 |
| `MIL-CREDA` | U->S | `7c284fdfafc7` | 0 | 12.47 |
| `MIL-CREDA-U` | U->S | `7c284fdfafc7` | 0 | 12.55 |
| `MIL-CREDA-A` | U->S | `7c284fdfafc7` | 0 | 12.01 |
| `MIL-CREDA-K` | U->S | `7c284fdfafc7` | 0 | 12.17 |
| `Baseline` | S->U | `7c284fdfafc7` | 0 | 4.55 |
| `CREDA*` | S->U | `7c284fdfafc7` | 0 | 9.67 |
| `CREDA` | S->U | `7c284fdfafc7` | 0 | 11.70 |
| `MIL-Baseline` | S->U | `7c284fdfafc7` | 0 | 5.31 |
| `MIL-CREDA**` | S->U | `7c284fdfafc7` | 0 | 19.19 |
| `MIL-CREDA*` | S->U | `7c284fdfafc7` | 0 | 12.56 |
| `MIL-CREDA` | S->U | `7c284fdfafc7` | 0 | 12.45 |
| `MIL-CREDA-U` | S->U | `7c284fdfafc7` | 0 | 12.32 |
| `MIL-CREDA-A` | S->U | `7c284fdfafc7` | 0 | 12.23 |
| `MIL-CREDA-K` | S->U | `7c284fdfafc7` | 0 | 12.32 |

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