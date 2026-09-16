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
| `MIL-Baseline` | M->U | `4bef9265ee57` | 0 | 5.12 |
| `MIL-CREDA**` | M->U | `4bef9265ee57` | 0 | 11.94 |
| `MIL-CREDA*` | M->U | `4bef9265ee57` | 0 | 12.32 |
| `MIL-CREDA` | M->U | `4bef9265ee57` | 0 | 13.11 |
| `MIL-CREDA-U` | M->U | `4bef9265ee57` | 0 | 12.91 |
| `MIL-CREDA-A` | M->U | `4bef9265ee57` | 0 | 13.88 |
| `MIL-CREDA-K` | M->U | `4bef9265ee57` | 0 | 13.08 |
| `MIL-Baseline` | U->M | `4bef9265ee57` | 0 | 4.89 |
| `MIL-CREDA**` | U->M | `4bef9265ee57` | 0 | 12.20 |
| `MIL-CREDA*` | U->M | `4bef9265ee57` | 0 | 12.07 |
| `MIL-CREDA` | U->M | `4bef9265ee57` | 0 | 13.19 |
| `MIL-CREDA-U` | U->M | `4bef9265ee57` | 0 | 12.97 |
| `MIL-CREDA-A` | U->M | `4bef9265ee57` | 0 | 13.04 |
| `MIL-CREDA-K` | U->M | `4bef9265ee57` | 0 | 13.01 |
| `MIL-Baseline` | M->S | `4bef9265ee57` | 0 | 5.18 |
| `MIL-CREDA**` | M->S | `4bef9265ee57` | 0 | 11.97 |
| `MIL-CREDA*` | M->S | `4bef9265ee57` | 0 | 11.84 |
| `MIL-CREDA` | M->S | `4bef9265ee57` | 0 | 13.39 |
| `MIL-CREDA-U` | M->S | `4bef9265ee57` | 0 | 12.92 |
| `MIL-CREDA-A` | M->S | `4bef9265ee57` | 0 | 12.81 |
| `MIL-CREDA-K` | M->S | `4bef9265ee57` | 0 | 12.99 |
| `MIL-Baseline` | S->M | `4bef9265ee57` | 0 | 4.94 |
| `MIL-CREDA**` | S->M | `4bef9265ee57` | 0 | 11.93 |
| `MIL-CREDA*` | S->M | `4bef9265ee57` | 0 | 12.15 |
| `MIL-CREDA` | S->M | `4bef9265ee57` | 0 | 13.13 |
| `MIL-CREDA-U` | S->M | `4bef9265ee57` | 0 | 12.86 |
| `MIL-CREDA-A` | S->M | `4bef9265ee57` | 0 | 13.07 |
| `MIL-CREDA-K` | S->M | `4bef9265ee57` | 0 | 13.13 |
| `MIL-Baseline` | U->S | `4bef9265ee57` | 0 | 4.98 |
| `MIL-CREDA**` | U->S | `4bef9265ee57` | 0 | 11.80 |
| `MIL-CREDA*` | U->S | `4bef9265ee57` | 0 | 11.96 |
| `MIL-CREDA` | U->S | `4bef9265ee57` | 0 | 13.01 |
| `MIL-CREDA-U` | U->S | `4bef9265ee57` | 0 | 12.88 |
| `MIL-CREDA-A` | U->S | `4bef9265ee57` | 0 | 12.93 |
| `MIL-CREDA-K` | U->S | `4bef9265ee57` | 0 | 12.87 |
| `MIL-Baseline` | S->U | `4bef9265ee57` | 0 | 4.94 |
| `MIL-CREDA**` | S->U | `4bef9265ee57` | 0 | 12.05 |
| `MIL-CREDA*` | S->U | `4bef9265ee57` | 0 | 11.95 |
| `MIL-CREDA` | S->U | `4bef9265ee57` | 0 | 13.67 |
| `MIL-CREDA-U` | S->U | `4bef9265ee57` | 0 | 12.79 |
| `MIL-CREDA-A` | S->U | `4bef9265ee57` | 0 | 12.85 |
| `MIL-CREDA-K` | S->U | `4bef9265ee57` | 0 | 13.14 |

Sin conclusión: tiempo de entrenamiento no se promedia entre máquinas — cada corrida es la lectura de su propio entorno, no una propiedad del método ni de la máquina que la corrió. Ver la tabla de arriba, corrida por corrida.

## 2 · Exactitud en fuente (más alto es mejor)

| Ruido | Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|---|
| sin | `MIL-Baseline` | 100.0 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 86.1 ± 0.0 | 30.6 ± 0.0 | **72.7** |
| sin | `MIL-CREDA**` | 100.0 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 86.1 ± 0.0 | 30.6 ± 0.0 | **72.7** |
| sin | `MIL-CREDA*` | 100.0 ± 0.0 | 97.2 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 86.1 ± 0.0 | 30.6 ± 0.0 | **72.7** |
| sin | `MIL-CREDA` | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 22.2 ± 0.0 | 86.1 ± 0.0 | 30.6 ± 0.0 | **73.1** |
| sin | `MIL-CREDA-U` | 88.9 ± 0.0 | 83.3 ± 0.0 | 88.9 ± 0.0 | 19.4 ± 0.0 | 83.3 ± 0.0 | 33.3 ± 0.0 | **66.2** |
| sin | `MIL-CREDA-A` | 80.6 ± 0.0 | 97.2 ± 0.0 | 88.9 ± 0.0 | 19.4 ± 0.0 | 83.3 ± 0.0 | 36.1 ± 0.0 | **67.6** |
| sin | `MIL-CREDA-K` | 94.4 ± 0.0 | 91.7 ± 0.0 | 94.4 ± 0.0 | 19.4 ± 0.0 | 97.2 ± 0.0 | 25.0 ± 0.0 | **70.4** |

Mejor promedio: **MIL-CREDA**; peor: MIL-CREDA-U, a 6.9% de distancia. MIL-CREDA queda 0.5% por encima de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

### 2b · Peldaños en fuente

| Ruido | Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | gana izq. |
|---|---|---|---|---|---|---|---|---|---|
| sin | MIL-Baseline → MIL-CREDA** | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| sin | MIL-CREDA** → MIL-CREDA* | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **+0.0** | 0/6 |
| sin | MIL-CREDA* → MIL-CREDA | +0.0 | -2.8 | +0.0 | +0.0 | +0.0 | +0.0 | **-0.5** | 0/6 |
| sin | MIL-CREDA-U → MIL-CREDA-K | -5.6 | -8.3 | -5.6 | +0.0 | -13.9 | +8.3 | **-4.2** | 1/6 |
| sin | MIL-CREDA-A → MIL-CREDA-K | -13.9 | +5.6 | -5.6 | +0.0 | -13.9 | +11.1 | **-2.8** | 2/6 |
| sin | MIL-CREDA-K → MIL-CREDA | -5.6 | -8.3 | -5.6 | -2.8 | +11.1 | -5.6 | **-2.8** | 1/6 |

El peldaño que más separa es **MIL-CREDA-U → MIL-CREDA-K**: **MIL-CREDA-K** queda 4.2 por encima de MIL-CREDA-U, y eso lee qué compra la selección por atención frente a una regular. Se inclinan igual en las 6 transferencias: MIL-Baseline → MIL-CREDA**, MIL-CREDA** → MIL-CREDA*, MIL-CREDA* → MIL-CREDA. Con esta cantidad de repeticiones lo que carga peso es la coincidencia entre transferencias, no la magnitud: seis acuerdos y tres contra tres promedian parecido y dicen cosas distintas.

## 3 · Exactitud en destino (más alto es mejor)

| Ruido | Método | M->U | U->M | M->S | S->M | U->S | S->U | Prom. |
|---|---|---|---|---|---|---|---|---|
| sin | `MIL-Baseline` | 61.1 ± 0.0 | 80.6 ± 0.0 | 16.7 ± 0.0 | 38.9 ± 0.0 | 19.4 ± 0.0 | 27.8 ± 0.0 | **40.7** |
| sin | `MIL-CREDA**` | 72.2 ± 0.0 | 80.6 ± 0.0 | 16.7 ± 0.0 | 38.9 ± 0.0 | 22.2 ± 0.0 | 27.8 ± 0.0 | **43.1** |
| sin | `MIL-CREDA*` | 63.9 ± 0.0 | 80.6 ± 0.0 | 16.7 ± 0.0 | 41.7 ± 0.0 | 19.4 ± 0.0 | 27.8 ± 0.0 | **41.7** |
| sin | `MIL-CREDA` | 66.7 ± 0.0 | 80.6 ± 0.0 | 16.7 ± 0.0 | 41.7 ± 0.0 | 19.4 ± 0.0 | 27.8 ± 0.0 | **42.1** |
| sin | `MIL-CREDA-U` | 25.0 ± 0.0 | 77.8 ± 0.0 | 11.1 ± 0.0 | 30.6 ± 0.0 | 13.9 ± 0.0 | 38.9 ± 0.0 | **32.9** |
| sin | `MIL-CREDA-A` | 41.7 ± 0.0 | 75.0 ± 0.0 | 22.2 ± 0.0 | 27.8 ± 0.0 | 19.4 ± 0.0 | 25.0 ± 0.0 | **35.2** |
| sin | `MIL-CREDA-K` | 55.6 ± 0.0 | 66.7 ± 0.0 | 27.8 ± 0.0 | 41.7 ± 0.0 | 22.2 ± 0.0 | 44.4 ± 0.0 | **43.1** |

Mejor promedio: **MIL-CREDA****; peor: MIL-CREDA-U, a 10.2% de distancia. MIL-CREDA queda 1.4% por encima de su piso MIL-Baseline. Con 1 repetición(es) esto es una estimación puntual y no un veredicto: la dispersión es cero por construcción, no por acuerdo. Más repeticiones lo refuerzan o lo cambian.

### 3b · Peldaños en destino

| Ruido | Peldaño | M->U | U->M | M->S | S->M | U->S | S->U | Prom. | gana izq. |
|---|---|---|---|---|---|---|---|---|---|
| sin | MIL-Baseline → MIL-CREDA** | -11.1 | +0.0 | +0.0 | +0.0 | -2.8 | +0.0 | **-2.3** | 0/6 |
| sin | MIL-CREDA** → MIL-CREDA* | +8.3 | +0.0 | +0.0 | -2.8 | +2.8 | +0.0 | **+1.4** | 2/6 |
| sin | MIL-CREDA* → MIL-CREDA | -2.8 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | **-0.5** | 0/6 |
| sin | MIL-CREDA-U → MIL-CREDA-K | -30.6 | +11.1 | -16.7 | -11.1 | -8.3 | -5.6 | **-10.2** | 1/6 |
| sin | MIL-CREDA-A → MIL-CREDA-K | -13.9 | +8.3 | -5.6 | -13.9 | -2.8 | -19.4 | **-7.9** | 1/6 |
| sin | MIL-CREDA-K → MIL-CREDA | -11.1 | -13.9 | +11.1 | +0.0 | +2.8 | +16.7 | **+0.9** | 3/6 |

El peldaño que más separa es **MIL-CREDA-U → MIL-CREDA-K**: **MIL-CREDA-K** queda 10.2 por encima de MIL-CREDA-U, y eso lee qué compra la selección por atención frente a una regular. Se inclinan igual en las 6 transferencias: MIL-Baseline → MIL-CREDA**, MIL-CREDA* → MIL-CREDA. Con esta cantidad de repeticiones lo que carga peso es la coincidencia entre transferencias, no la magnitud: seis acuerdos y tres contra tres promedian parecido y dicen cosas distintas.