# Sin BatchNorm: dónde está el cuello, y cuánto de él se puede cerrar

**Barrido autónomo, ~1.700 corridas, 10 semillas por celda, todo sin
BatchNorm**, sobre el arreglo de dos fases de `src/CREDA/two_phase.py` y a
través del `CREDALoss` que shipea `src/CREDA/models.py`.

Documento temporal. No es una revisión gestionada y no modifica
`research-concept-r25.md`.

---

## 1. La conclusión, arriba

**El cuello no es la alineación. Es la lectura. Y tiene forma exacta: una afín
por dimensión, 32 números, entre el codificador objetivo y la cabeza congelada.**

Eso está demostrado por localización, no por argumento. Ajustando el mejor mapa
de cada familia con las etiquetas verdaderas del blanco, a través de la cabeza
congelada:

| columna | nada | corrim. **16** | diagonal **16** | **afín diag · 32** | ortogonal **120** | libre **272** |
|---|---|---|---|---|---|---|
| rollo | 0,7004 | 0,7228 | 0,7374 | **0,7939** | 0,8316 | 0,8749 |
| toroides | 0,4679 | 0,6399 | 0,7416 | **0,8538** | 0,8814 | 0,9097 |
| lunas | 0,1400 | 0,2174 | 0,4364 | **0,6466** | 0,6379 | 0,8217 |

**32 números cierran el 54 % al 88 % de la brecha, y en lunas le ganan a un mapa
ortogonal de 120 parámetros.** La corrección que falta es **escalado y
corrimiento por eje** — que es, en esencia, lo que BatchNorm hace— y **no** una
rotación.

Y el estado de la solución, honesto:

| columna | lo declarado | **mejor sin etiquetas** | oráculo de 32 | techo libre | % del techo |
|---|---|---|---|---|---|
| **rollo** | 0,6215 | **0,8008** | 0,8578 | 0,8939 | **89,6 %** |
| **toroides** | 0,2367 | **0,5238** | 0,8621 | 0,9049 | 57,9 % |
| **lunas** | 0,1221 | **0,2794** | 0,7685 | 0,8057 | 34,7 % |

**rollo está esencialmente resuelto** — 89,6 % de lo que un mapa lineal libre
alcanza, y el oráculo de 32 solo agrega 0,057 más. En las otras dos la familia
de 32 números **alcanza** (0,86 y 0,77) y lo que no alcanza es **el estimador
sin etiquetas**: se recupera entre un tercio y un 11 % de ese potencial.

**Eso reformula el problema abierto.** No es *«la formulación no funciona»*. Es
*«existe una corrección de 32 números, está probado que basta, y estimarla sin
etiquetas es lo que falta»*. Mucho más accionable.

---

## 2. La configuración que lo logra

```python
# Fase 1: UN codificador y la cabeza sobre la fuente, y se congelan los dos.
train_source(enc, cab, Xs, ys, epochs=60, batch_size=256, generator=gen)

# Una copia única de esos pesos siembra el codificador objetivo, buffers incluidos.
enc_t = clone_for_target(enc)

# El término, con tres banderas y una rampa:
cr = CREDALoss(
    sigma              = 'auto',
    sigma_from         = 'batch-source',   # 'class-source' gana rollo; ver §5
    normalize_by_bounds= True,             # Ecs. (35)-(36) PRENDIDAS
    normalize_kernel_embedding = False,    # sin L2
    weight_cross_block = False,            # ver §6: conviene prenderlo
    use_entropy_weighting = True,
)

# Y la afín de 32 números EN LÍNEA, con dos objetivos sobre parámetros DISJUNTOS
# -- así que no hay λ que balancear:
d = nn.Parameter(torch.ones(16)); b = nn.Parameter(torch.zeros(16))
for paso in range(800):
    cr.class_mixing = min(1.0, paso / 800)              # la rampa
    g_t = softmax(cab(enc_t(x_t) * d + b))              # la lectura CORREGIDA
    # (1) el término mueve enc_t:      alinea las features
    paso_de(cr(f_s, f_t, y_s, g_t), enc_t)
    # (2) info mutua mueve d, b:       aprende a leerlas, sin etiquetas
    paso_de(info_max(cab(enc_t(x_t).detach() * d + b)), [d, b])
```

`info_max(z) = H(p|x) − H(E[p])`: predicciones seguras **y** clases balanceadas.
La entropía **sola** colapsa (0,6645 en rollo contra 0,7004); el término de
diversidad es lo que la vuelve segura, y sobre 32 parámetros el óptimo degenerado
casi no tiene dónde esconderse.

---

## 3. La familia mixta tiene, por fin, un ganador interior

Esto es lo que la propuesta necesitaba y antes no aparecía. Con σ arreglado, la
rampa `α = min(1, paso/K)` **le gana a los dos extremos**:

| | α≡1 *(declarado)* | K=200 | K=400 | **K=800** | α≡0 *(marginal)* |
|---|---|---|---|---|---|
| rollo · `class-source` | 0,6513 | 0,7620 | 0,7861 | **0,7999** | 0,7389 |
| toroides · `batch-source` | 0,2637 | 0,3765 | 0,4504 | **0,4679** | 0,4479 |
| lunas · `batch-source` | 0,1394 | 0,1374 | 0,1410 | 0,1403 | **0,2450** |

Monótono en K, dos columnas de tres. `K` va desacoplado del largo de la corrida
(`paso/K`, no `paso/total`) — con `paso/total`, cambiar el largo cambiaba el
**remedio** y no la ventana, y eso contaminó una comparación entera antes de
detectarlo.

Debilidad honesta: `K=800` sobre 800 pasos hace que α llegue a 1 justo al final,
así que es *«marginal con entrega tardía»* más que *«el condicional toma el
control»*.

---

## 4. Lo refutado, con cada predicción escrita antes de medir

**La norma L2 sobre el embedding.** Predije que rescataría al σ por clase.
Pasó lo contrario: daña en las tres columnas y **en toroides destruye la
representación** — acierto 0,1688 con techo del probe **0,1633**, contra 0,4383
y 0,8530 sin ella. Y la razón de σ entre clases **explota** a 16,60 en vez de
igualarse, porque la L2 acota las distancias a `[0,2]` y un cociente de medianas
chicas es grande.

**Procrustes ortogonal.** Predije que si lo que faltaba era una rotación, el mapa
ortogonal óptimo la encontraría. Da `0,6943 / 0,4132 / 0,1319` contra
`0,7004 / 0,4679 / 0,1400` sin él. Y **con las etiquetas verdaderas** el mejor
ortogonal llega solo a `0,7216 / 0,4509 / 0,1731`. **No es una rotación.**

**La KL de las features a `N(0,I)`.** Su gradiente es **424.000× a 10.000.000×**
el del término, así que todo `λ=1` anterior era **KL sola**. Con λ balanceado
(`1,5e-06`): en los dos codificadores es catastrófica (−0,4123 `t=−13,20`), solo
en el blanco no hace nada donde el método funciona (+0,0062 `t=0,44`) y los
momentos quedan intactos. Ayuda solo en lunas. **No hay λ intermedio donde le
agregue al término.**

**Auto-entrenamiento sobre los 32 números** — pseudo-rótulos, con filtro de
confianza, iterado 5 rondas, y combinado con info-max. Las cinco variantes dan
entre 0,2142 y 0,5238, **ninguna mejor que info-max sola**. Una vez que la afín
se entrenó en línea, refinarla después no agrega nada.

**La composición en línea → post-hoc.** Peor que sus dos componentes: 0,4085 en
toroides contra 0,5152 en línea, 0,2261 en lunas contra 0,2794 post-hoc.

**La correa L2 en el espacio de parámetros**: −0,4452 (`t=−43,79`). Congela el
codificador cerca de la fuente.

---

## 5. El σ por clase: tres defectos medidos

**(A) No es monótono en la desalineación.** Desplazando el blanco de una clase:
el `creda_c` sube a 0,5048 al desplazamiento 10 y **baja** a 0,4365 y 0,3972 en
30 y 100. Una clase peor alineada reporta **menos** discrepancia. Con un σ que no
lee el blanco, el mismo barrido sube monótono y satura en `ln 2 = 0,69315`. La
causa: la mediana se toma sobre fuente ∪ blanco, así que **la unidad de medida
del kernel se mueve con la cosa que mide**.

**(B) Mide las clases a resoluciones distintas.** `max σ_c / min σ_c` llega a
**3,31** en corridas reales, y después esos `creda_c` se promedian en la Ec. (37)
como si fueran comparables. La cota de la Ec. (36) mitiga eso **8×** (la caída
pasa de 0,10768 a 0,01334) pero no lo cierra.

**(C) Y deriva ×2,65 a ×3,36 durante la fase 2**, porque incluye `f_t`, que es lo
que el optimizador mueve. El σ de la fuente sola deriva ×1,03. En rollo la
agrupada salta a 249 al paso 800, justo donde marginal se derrumba — la dirección
causal **no** está establecida.

**(D) En lunas destruye la representación.** Techos del probe: `batch-source`
0,77–0,85; `class-pooled` 0,48–0,52; `class-source` **0,31**; con L2 0,11.

---

## 6. Qué se puede implementar sin romper nada matemático

**Limpia — la cota de las Ecs. (35)-(36) prendida.** Lo que shipea es el default
**apagado**, o sea que el `src` devuelve el `creda_c` crudo promediado y no el
`ℓ_glob,c` que la formulación define. Prenderla **acerca** el código a la
revisión. Mide levemente mejor en las tres columnas y mitiga el defecto (A) ocho
veces.

**Con demostración — un solo σ.** r25 línea 732: *«un único ancho de banda σ
gobierna los tres bloques, ya que los tres derivan del mismo kernel de
instancia»*, y la 826 reutiliza ese kernel en el término local para *«no
introducir un segundo kernel ni otro ancho de banda σ»*.

Y es **demostrable**, no preferencia: con σ distinto por bloque `K_mix` deja de
ser PSD — `λ_min` de −0,90 a −21,64 en cuatro combinaciones. Un kernel gaussiano
con un σ es definido positivo, y ensamblar los tres bloques de *ese mismo* kernel
sobre la unión de los puntos da la matriz de Gram de esa unión. Mezclá σ y
`H₂(K̆_mix)` **no es la entropía de nada**.

- `batch-source` → **un** σ, compatible con la 732. De dónde se calcula r25 no lo
  declara, y `AGREED.md:143` registra exactamente eso. **Recomendado.**
- `class-source` → **C** anchos de banda, o sea C kernels de instancia.
  Contradice *«el mismo kernel de instancia»*. Gana rollo por 0,052 y destruye el
  techo de lunas de 0,84 a 0,31.

**En contra de la formulación — el bloque cruzado apagado.** El ganador medido
tiene `weight_cross_block=False`, pero la Ec. (26) declara
`K̃_c^{st} = K_c^{st} D`. La diferencia **no es medible** (0,7999 contra 0,7860,
0,4679 contra 0,4579), así que **prenderlo cumple la ecuación sin costo
detectable**. Eso es lo que yo haría.

**Notación nueva — la rampa α(t) y la afín de 32 números.** Las dos piden
deliberación. La familia convexa en sí es barata de justificar: `α=1` recupera la
Ec. (37) exactamente, `α=0` es el límite marginal, la cota `[0,1]` se **hereda**
de ser combinación convexa de dos cantidades acotadas, y el bloque marginal son
las mismas Ecs. (23)–(36) con un bloque. Lo que hay que deliberar es α(t) como
coeficiente que corre por conteo de pasos, y la afín como módulo que se interpone
entre el codificador y la cabeza.

---

## 7. El problema abierto, nombrado con precisión

**Estimar los 32 números sin etiquetas.** La familia basta — el oráculo da
`0,8578 / 0,8621 / 0,7685`. Cinco estimadores label-free, todos medidos:

| estimador | rollo | toroides | lunas |
|---|---|---|---|
| estandarizar al momento de la fuente | 0,7051 | 0,5322 | 0,2998 |
| información mutua | 0,7019 | 0,5686 | 0,1875 |
| estandarizar → info mutua | 0,7060 | **0,6207** | **0,3075** |
| auto-entrenamiento (5 variantes) | ≤0,7950 | ≤0,5238 | ≤0,2332 |
| **la afín en línea** | **0,7999** | 0,5152 | 0,2591 |
| *oráculo de 32* | *0,8578* | *0,8621* | *0,7685* |

Ninguno pasa de recuperar un tercio. **Eso es el problema abierto**, y es de
estimación, no de formulación.

**Por qué el término no puede ayudar ahí.** Rotando el blanco adaptado alrededor
de su media —lo que preserva μ exacto y el espectro de Σ— el acierto cae de
0,7620 a 0,0450 y el término se mueve 0,012. Entre 40 rotaciones, la que el
término elegiría tiene acierto **0,0000 / 0,0009 / 0,0556** contra 0,2009 /
0,2546 / 0,2222 de la mejor disponible. **Si lo dejás elegir, elige la peor.**

> La sensibilidad del objetivo a la legibilidad es ~1,2 % de su propia escala.
> La de la tarea es total.

Cualquier objetivo construido solo con distancias par a par y entropías de kernel
es casi ciego a lo que decide la lectura. El remedio no vive ahí.

**Sin señal de corte no supervisada.** El mejor paso cambia por columna — rollo
pica en 400, toroides en 200, lunas en 800 — y no hay etiqueta con la que parar.
Excepción medida: los brazos anclados en el espacio de la cabeza **no se
derrumban** (rollo plano en 0,72 a lo largo de 800 pasos contra 0,33 de
marginal), y una meseta no necesita regla de parada.

---

## Apéndice — lo corrido

| tanda | ejes | brazos | corridas | tiempo |
|---|---|---|---|---|
| etapa 1 | origen de σ × norma L2 × cota | 15 | 450 | 3044 s |
| etapa 2 | largo de rampa K × bloque cruzado, en dos escalas | 20 | 600 | 4156 s |
| etapa 3 | colocación de la KL × λ balanceado, en dos escalas | 10 | 300 | 2576 s |
| Procrustes | Q de centroides, Q al azar, Q oráculo | 4 | 30 | 175 s |
| escalera | seis familias de mapas, todas con etiquetas | 7 | 30 | 232 s |
| afín sin etiquetas | estandarizar, info-max, oráculo | 6 | 30 | 207 s |
| afín compuesta | ocho composiciones label-free | 9 | 30 | 250 s |
| afín en línea | en línea × dos inicializaciones | 3 | 30 | 984 s |
| composición final | en línea × post-hoc × oráculos | 6 | 30 | 589 s |
| auto-entrenamiento | 5 variantes de pseudo-rótulos | 7 | 30 | 263 s |

Reglas de decisión fijadas **antes** de empezar: gana el mejor acierto en su
propio mejor paso con `t` de Welch; `|t| < 2` es empate y gana el más simple;
ante empate decide la menor brecha al techo; una tanda sin mejora medible se
reporta así y no se le inventa un ganador. Toda tabla lleva techo del probe y
trayectoria, nunca un solo paso.

---

## 8. Dónde se rompe, paso a paso — y el término no es el culpable

Última tanda: **una muestra, tres ruteos, misma semilla y mismo orden de lote**,
así que lo único distinto es de dónde salen las etiquetas con las que el término
rutea. `verdad` usa las etiquetas reales del blanco: hace trampa a propósito y da
**la cota del término con ruteo perfecto**.

**toroides**, muestra #0, clase real 0:

| paso | `pseudo` · rut / cos→real / acc | `verdad` · rut / cos→real / acc |
|---|---|---|
| 0 | **0 ✓** / 0,2341 / 0,2546 | 0 ✓ / 0,2341 / 0,2546 |
| 25 | **0 ✓** / 0,2661 / 0,2991 | 0 ✓ / 0,2396 / 0,3056 |
| 50 | **0 ✓** / 0,3926 / 0,2815 | 0 ✓ / 0,3976 / 0,2824 |
| 100 | **8 ✗** / 0,4288 / 0,4815 | 0 ✓ / 0,4378 / 0,4130 |
| 200 | 1 ✗ / 0,5559 / 0,5556 | 0 ✓ / 0,5141 / 0,4806 |
| 400 | 1 ✗ / 0,4806 / 0,5593 | 0 ✓ / **0,7896** / 0,5593 |
| 800 | 1 ✗ / **−0,7121** / 0,5750 | 0 ✓ / **0,9946** / **0,7880** |

**Con ruteo perfecto el descenso del término llega a `cos→real = 0,9946`.**
Apunta casi exactamente al centroide fuente de la clase correcta, y toroides
sube de `0,5750` a `0,7880` contra un techo de ~0,90.

**El término no es el problema.** Todo el encuadre anterior —*«el término apunta
al lugar equivocado»*— era cierto como descripción y engañoso como atribución:
apunta donde el ruteo le dice, y con el ruteo correcto apunta casi perfecto.

**Y el punto de quiebre es específico.** La muestra tenía el ruteo **correcto
hasta el paso 50** y lo pierde en el 100 — exactamente cuando el `%ok` del lote
salta de 25,4 % a 45,1 %. **A medida que el ruteo global mejora, esa muestra se
sacrifica.** El término optimiza el agregado y entrega individuos.

En **rollo** la muestra elegida nunca tuvo el ruteo correcto: `cos→real` arranca
en −0,05 y se queda negativo los 800 pasos, mientras el lote llega a 85,8 % y el
acierto a 0,7926. Con `verdad`, `cos→real` va a **0,9423**, la distancia al
centroide cae de 57,9 a **4,70** y el acierto llega a **0,8454**.

### La relación que esto establece

| columna | % de ruteo correcto | acierto con pseudo | acierto con ruteo perfecto | techo |
|---|---|---|---|---|
| rollo | 85,8 % | 0,7926 | **0,8454** | ~0,89 |
| toroides | 54,1 % | 0,5750 | **0,7880** | ~0,90 |

**El acierto del ruteo es el cuello, y mapea casi linealmente al acierto final.**
Y la afín de 32 números mueve el `%ok` apenas: 85,8 % → 86,9 % en rollo. Explica
por qué su ganancia es modesta ahí y por qué el oráculo de 32 números queda lejos
sin etiquetas: la afín arregla la **lectura final**, no el **ruteo** que el
término consume en cada paso.

### Lo que esto cambia en la dirección del remedio

Hay dos cuellos, no uno, y son distintos:

1. **El ruteo durante el entrenamiento** — qué etiqueta usa el término en cada
   paso. Con 54 % correcto toroides llega a 0,58; con 100 % llega a 0,79. Esa
   diferencia de 0,21 es lo que un mejor ruteo compra, y mapea casi lineal.
2. **La lectura al final** — qué mapa lee las features ya alineadas. Una afín de
   32 números cierra 54–88 % de esa brecha, y sin etiquetas se alcanza un tercio.

Son **independientes**: la afín mejora (2) y casi no toca (1). Un remedio que
solo arregle la lectura final deja el ruteo igual de malo en cada paso, y el
término seguirá alineando hacia donde un ruteo al 54 % le indique.

**El modo de falla medido, para nombrarlo:** una muestra con ruteo correcto
puede perderlo mientras el ruteo agregado mejora. Cualquier remedio al ruteo
tiene que ser **por muestra y con memoria** —no un agregado por lote— porque es
exactamente el agregado el que sacrifica individuos.
