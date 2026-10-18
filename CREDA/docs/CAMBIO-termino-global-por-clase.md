# El enrutamiento por clase del término global, y la propuesta de uso mixto

**Documento temporal.** No es una revisión gestionada y no modifica
`research-concept-r25.md`. Toda cita de ecuación remite a r25 tal como está.

Reemplaza a `CAMBIO-enrutamiento-blando.md`, cuyo nombre quedó mintiendo: el
enrutamiento blando que ese documento proponía **está refutado**, y la medición
que lo refuta está en el §4 de acá.

---

## 1. La matemática actual

### 1.1 La cadena declarada

La Ec. (22) le asigna a cada bolsa objetivo una pseudoetiqueta dura mediante una
regla determinista $\mathfrak a$ que elige una clase de probabilidad máxima:

$$
\widetilde y_j^{t} = \mathfrak a \left( \mathbf g_j^{t} \right) \in \mathcal C .
\tag{22}
$$

Esa pseudoetiqueta define el **enrutamiento** — qué bolsas entran al bloque de
qué clase — junto con las cardinalidades que después leen las cotas:

$$
S_c = \left\{ i : c_i^{s} = c \right\},
\qquad
T_c = \left\{ j : \widetilde y_j^{t} = c \right\},
\qquad
n_c^{s} = \left| S_c \right|,
\qquad
n_c^{t} = \left| T_c \right| .
$$

Por separado, la Ec. (24) cuantifica la **confianza** de cada pseudoetiqueta
como el complemento normalizado de su entropía cuadrática de Rényi:

$$
w_j^{t} = 1 - \frac{\widehat H_2 \left( \mathbf g_j^{t} \right)}{\ln C}
\quad \in [0,1] .
\tag{24}
$$

La Ec. (25) la restringe al enrutamiento y la vuelve diagonal, y la Ec. (26) la
aplica a los bloques que involucran al objetivo:

$$
\widetilde{\mathbf w}_c^{t} = \left( w_j^{t} \right)_{j \in T_c} \in \mathbb R^{n_c^{t}},
\qquad
\mathbf D_c^{t} = \operatorname{diag} \left( \widetilde{\mathbf w}_c^{t} \right),
\tag{25}
$$

$$
\widetilde{\mathbf K}_c^{t} = \mathbf D_c^{t} \mathbf K_c^{t} \mathbf D_c^{t},
\qquad
\widetilde{\mathbf K}_c^{st} = \mathbf K_c^{st} \mathbf D_c^{t} .
\tag{26}
$$

De ahí en adelante nada vuelve a mirar las etiquetas: la Ec. (32) normaliza por
traza, la Ec. (33) arma la discrepancia

$$
J_c = \tfrac{1}{2} \left[ H_2 ( \breve{\mathbf K}_c^{s} ) + H_2 ( \breve{\widetilde{\mathbf K}}_c^{t} ) \right] - H_2 ( \breve{\widetilde{\mathbf K}}_c^{\mathrm{mix}} ),
\tag{33}
$$

la Ec. (35) fija las cotas a partir de las cardinalidades del enrutamiento

$$
L_c = - \ln ( n_c^{s} + n_c^{t} ),
\qquad
U_c = \tfrac{1}{2} \left( \ln n_c^{s} + \ln n_c^{t} \right),
\tag{35}
$$

y la Ec. (36) la lleva al intervalo unitario:

$$
\ell_{\mathrm{glob},c} = \frac{U_c - J_c}{U_c - L_c}, \qquad 0 \leq \ell_{\mathrm{glob},c} \leq 1 .
\tag{36}
$$

El término global promedia sobre las clases activas $\mathcal C_{\mathrm{act}}$
(Ec. 37).

### 1.2 La distribución se usa dos veces, de dos maneras

| Uso | Ecuación | Qué decide | Naturaleza |
|---|---|---|---|
| **Confianza** | (24) → (25) → (26) | *cuánto* pesa la bolsa dentro de su bloque | **blanda**, continua en `[0,1]`, diferenciable |
| **Enrutamiento** | (22) → `T_c` | *en qué bloque* entra la bolsa | **dura**, binaria, no diferenciable |

r25 declara la segunda explícitamente:

> «La asignación dura de pseudoetiqueta $\widetilde y_j^{t} = \mathfrak a ( \mathbf g_j^{t} )$
> (Ec. 22), que determina ese enrutamiento, **no es diferenciable**; los
> gradientes fluyen a través de las magnitudes continuas —entradas de las
> matrices, confianzas y pesos—, **no a través de la decisión discreta de clase**.»

Correcto y bien puesto. Lo que no dice es qué ocurre cuando esa decisión
discreta, que se fija en cada paso hacia adelante y no recibe gradiente, está
**equivocada**.

---

## 2. El problema, medido

Tres conjuntos sintéticos (`rollo`, `toroides`, `lunas`), diez clases, azar
`0.1000`, dos backbones. Arreglo de dos codificadores fijado en
`src/CREDA/two_phase.py`: la fase 1 entrena el codificador fuente y el
clasificador y los **congela**; la fase 2 copia los pesos finales **una sola
vez** al codificador objetivo y lo deja entrenar libre bajo el término. La
fuente pasa por su codificador, el objetivo por el suyo. Sin sincronización.
Diez semillas, `t` de Welch.

### 2.1 El término apunta bien, al lugar equivocado

Una muestra de entrenamiento del objetivo, clase real `0`, seguida los 200
pasos, con el lote completo pasando por detrás. Coseno entre el **descenso** del
término sobre su representación y dos direcciones: hacia el centroide fuente de
su clase **real** y hacia el de su **pseudo-clase**. La etiqueta real sólo mide.

| | paso 0 | 10 | 50 | 200 |
|---|---|---|---|---|
| **con BN** · pseudoetiqueta | 1 | **0** | 0 | 0 |
| · `cos(descenso → real)` | 0.94 | 0.99 | 0.94 | 0.98 |
| **sin BN** · pseudoetiqueta | 1 | 1 | 1 | **1** |
| · `cos(descenso → pseudo)` | 0.94 | **0.98** | **0.95** | **0.92** |
| · `cos(descenso → real)` | 0.89 | 0.83 | −0.38 | **−0.92** |

Sin BatchNorm el término empuja esa muestra hacia el centroide de su
**pseudo-clase** con coseno entre `0.92` y `0.98` **en los doscientos pasos**, y
la pseudoetiqueta está mal desde el paso cero y no se corrige nunca. Con
BatchNorm se corrige en el paso 10 y el mismo término la lleva a su lugar.

**El término no es débil ni ciego: apunta con precisión, y apunta donde el
enrutamiento le dijo.**

### 2.2 El colapso de clases es el mecanismo, en un escalar

Cuántas clases **distintas** elige el argmax por paso, de 10 posibles,
registrado sobre 43 261 bloques:

| columna | red | clases/paso | blanco |
|---|---|---|---|
| rollo | con BN | **10.00** | 0.8990 |
| rollo | sin BN | **8.40** | 0.5943 |
| toroides | con BN | **10.00** | 0.9351 |
| toroides | sin BN | **3.00** | 0.2906 |
| lunas | con BN | **10.00** | 0.8382 |
| lunas | sin BN | **1.86** | 0.1150 |

Con BatchNorm el enrutamiento usa las diez clases en cada paso, en las tres
columnas. Sin BatchNorm el conteo colapsa, y **ordena los resultados exacto**.

r25 declara este modo y es el régimen de operación (línea 1087):

> «La pseudoetiqueta dura se vuelve además la misma para toda bolsa objetivo, y
> $\mathcal C_{\mathrm{act}}$ en Ec. 37 se reduce a una sola clase, con lo que el
> término global exige menos.»

En `lunas` sin BatchNorm el promedio se toma sobre **1.86 clases**.

### 2.3 El sopesado es inocente, y está medido

El brazo sin `c` lleva $\mathbf D_c^{t}$ **entero** —`K̃^t = D K^t D` y
`K̃^{st} = K^{st} D`, completos— y **nunca evalúa la Ec. (22)**. Es el mejor
brazo sin BatchNorm. El sopesado es blando y no se compromete; el enrutamiento
es duro y sí.

La diferencia es de naturaleza, no de grado. Un peso blando equivocado cuesta
poco: la bolsa aporta de más o de menos a un bloque al que de todos modos
pertenece. Un enrutamiento duro equivocado la mete en el **bloque errado**, y
desde ahí el término la alinea contra la clase equivocada con coseno `0.95`.

Y el sopesado **amplifica** el error en vez de contenerlo, porque mide confianza
y no corrección: una cabeza segura y equivocada le sube el fader.

### 2.4 BatchNorm no se arregla normalizando después

El coseno agregado sobre las 1920 muestras entre el descenso y la dirección que
cierra el corrimiento $\mu_t - \mu_s$ está entre **0.02 y 0.09 en todos los
pasos**, con y sin BatchNorm. **El término es casi ortogonal al corrimiento**:
BatchNorm lo cierra estructuralmente en cada paso hacia adelante; el término
nunca apunta ahí.

Y la reparación obvia está refutada: estandarizar las representaciones del
objetivo a los momentos de la fuente —exactamente lo que BatchNorm calcula— da
**0.598 sin BatchNorm contra 0.910 con él**, y en el paso 0 las dos son casi
iguales (`0.485` contra `0.506`). La brecha se abre **durante** la adaptación.
Lo que BatchNorm compra es **el codificador que produce**, no la operación.

### 2.5 La asimetría ya está en el documento

r25 **ya tomó la decisión blanda, una vez**, al construir la referencia fuente
personalizada del término **local**, Ec. (29):

> «**Conservando la distribución completa $\mathbf g_j^{t}$** —en lugar de usar
> sólo la pseudoetiqueta dura, **que descartaría la incertidumbre del
> clasificador**—, la masa que el objetivo asigna a cada clase se reparte entre
> sus sujetos fuente mediante $\pi_{ji \mid c}$.»

El término local conserva $\mathbf g_j^{t}$ completa y **rechaza explícitamente**
la pseudoetiqueta dura, por el motivo exacto que el §2.1 exhibe. El término
global usa la dura, en `T_c`, y el documento **no da ninguna razón**: `T_c` se
introduce junto a `S_c` como definición.

---

## 3. El bloque marginal

Todo lo que sigue necesita un objeto: la **misma construcción con el índice de
clase quitado**. No hay ecuación nueva — es las Ecs. (23)–(36) con un solo
bloque.

Con $S = \{1, \ldots, N_s\}$ y $T = \{1, \ldots, N_t\}$, es decir sin partir:

$$
\mathbf D^{t} = \operatorname{diag} \left( \left( w_j^{t} \right)_{j=1}^{N_t} \right),
\qquad
\widetilde{\mathbf K}^{t} = \mathbf D^{t} \mathbf K^{t} \mathbf D^{t},
\qquad
\widetilde{\mathbf K}^{st} = \mathbf K^{st} \mathbf D^{t},
$$

$$
J = \tfrac{1}{2} \left[ H_2 ( \breve{\mathbf K}^{s} ) + H_2 ( \breve{\widetilde{\mathbf K}}^{t} ) \right] - H_2 ( \breve{\widetilde{\mathbf K}}^{\mathrm{mix}} ),
$$

$$
L = - \ln ( N_s + N_t ),
\qquad
U = \tfrac{1}{2} \left( \ln N_s + \ln N_t \right),
\qquad
\ell_{\mathrm{marg}} = \frac{U - J}{U - L} \in [0,1] .
$$

**Conserva el sopesado entero** y **nunca evalúa la Ec. (22)**. Es la Ec. (26)
en su límite agnóstico de clase, con $C = 1$.

---

## 4. La propuesta de uso mixto

### 4.1 La forma

Una combinación convexa entre el bloque marginal y el término declarado:

$$
\boxed{
\ell_{\mathrm{glob}}^{\mathrm{mix}}
=
(1 - \alpha) \, \ell_{\mathrm{marg}}
+
\alpha \, \frac{1}{\left| \mathcal C_{\mathrm{act}} \right|}
\sum_{c \in \mathcal C_{\mathrm{act}}} \ell_{\mathrm{glob},c},
\qquad
\alpha \in [0,1]
}
$$

> `ℓ_glob^mix = (1−α)·ℓ_marg + α·⟨ℓ_glob,c⟩` — una familia de un parámetro que
> va del bloque único a la partición por clase.

### 4.2 Las cuatro propiedades que la hacen admisible

1. **$\alpha = 1$ recupera la Ec. (37) exactamente.** La formulación declarada
   no se contradice: es el extremo de esta familia, no una alternativa a ella.
2. **$\alpha = 0$ es el límite marginal**, el brazo que las mediciones eligen.
3. **La cota de la Ec. (36) se hereda sin demostrar nada.** Una combinación
   convexa de dos cantidades en `[0,1]` está en `[0,1]`. No hay cota nueva que
   probar ni `L` y `U` nuevos que derivar.
4. **No hay objeto nuevo.** El bloque marginal son las mismas Ecs. (23)–(36)
   con un bloque, y $\alpha$ es el único símbolo que se agrega.

### 4.3 Los tres $\alpha$ medidos, y qué dio cada uno

Cada brazo en su mejor paso, contra la formulación declarada **en el mismo
paso**, 10 semillas:

| $\alpha$ | rollo sin BN | toroides sin BN | lunas sin BN | rollo con BN | toroides con BN | lunas con BN |
|---|---|---|---|---|---|---|
| **$0$** (marginal) | **+0.1493** `t=5.15` | **+0.2091** `t=8.25` | +0.0442 `t=2.16` | −0.0044 `t=−0.31` | +0.0141 `t=1.80` | +0.0367 `t=2.19` |
| **$t/T$** (reloj) | +0.1419 `t=4.65` | +0.1112 `t=5.63` | +0.0176 `t=1.14` | +0.0020 `t=0.13` | +0.0106 `t=1.33` | +0.0154 `t=1.39` |
| **$\left\|\{\widetilde y_j^{t}\}\right\| / C$** | +0.0093 `t=0.30` | +0.1421 `t=4.68` | +0.0211 `t=1.54` | **+0.0000** `t=0.00` | **+0.0000** `t=0.00` | **+0.0000** `t=0.00` |
| **$1$** (declarado) | — | — | — | — | — | — |

Un remedio debe cumplir **dos polos**: ganarle a lo declarado sin BatchNorm, y
no romper lo que ya funciona con BatchNorm.

- **$\alpha = 0$ cumple los dos.** Gana en las tres columnas sin BatchNorm y con
  BatchNorm no es mediblemente peor en ninguna y es **mejor en dos**.
- **$\alpha = t/T$ cumple los dos** pero gana menos, y paga con un coeficiente
  que corre por reloj y no por ninguna propiedad de los rótulos.
- **$\alpha = \left|\{\widetilde y_j^{t}\}\right| / C$ cumple el polo de
  BatchNorm de forma EXACTA** —lee `1.000` en las tres columnas, la formulación
  se reduce a la declarada y el resultado es **bit-idéntico**— pero **falla el
  otro**: `rollo` lee `0.889` y gana `+0.0093`, nada.

  El motivo: **el conteo de clases no es la corrección de las clases.** `rollo`
  sin BatchNorm usa 8.4 de 10 clases **estando 46–52 % mal ruteado**, así que un
  coeficiente basado en el conteo le entrega la autoridad a una partición que
  está mitad equivocada.

### 4.4 El enrutamiento blando, refutado

La propuesta del documento anterior era conservar `c` y volver la pertenencia
fraccionaria, $\mathbf D_c^{t} = \operatorname{diag}(g_{\cdot,c}^{t} \odot \mathbf w^{t})$.
**Está refutada**: `0.5646` contra `0.5885` de lo declarado, `t = −0.54`.

El motivo, medido. El reparto blando atenúa una bolsa mal ruteada y le da
presencia en su bloque correcto, pero en proporción al **mismo posterior
equivocado**. La razón volumen-equivocado : volumen-correcto queda:

| $p$ en la clase equivocada | razón |
|---|---|
| 0.40 | 1.3 : 1 |
| 0.70 | 4.7 : 1 |
| 0.90 | 18 : 1 |
| 0.98 | **98 : 1** |

Y entre las bolsas **realmente** mal ruteadas de una corrida sin BatchNorm,
**entre 81 % y 91 % tienen $p_{\max} \geq 0.95$**; la banda donde el reparto
blando trabaja de verdad ($p < 0.6$) contiene **entre 0.1 % y 3 %** de ellas.
Con BatchNorm el mal ruteo es raro (3.7 %–5.6 %) y las que quedan son de
confianza media, repartidas parejo, razón ~5 : 1.

**El reparto blando sirve justo donde no se necesita y no sirve justo donde se
necesita.**

Arregla un defecto real, eso sí: bajo ruteo duro un bloque de una sola bolsa
tiene $\widetilde{\mathbf K}_c^{t}$ de $1 \times 1$, que normalizado por su traza
queda `[1]` y cuya entropía es **0 para cualquier dato** — esa mitad de la
Ec. (33) aporta una constante. Ocurre en 50 de 8405 bloques en `rollo` sin
BatchNorm. El reparto blando lo elimina. **No era el defecto que costaba el
acierto.**

---

## 5. Qué queda en pie, y qué cuesta

**$\alpha = 0$ es el remedio medido.** Los dos polos cumplidos, el mayor margen
sin BatchNorm, cero notación nueva.

**Y su precio es conceptual, no numérico: borra la `c` de la Ec. (26)**, que es
la contribución del trabajo. Alinea marginales con un re-sopesado por confianza
—que es un método real y no es CORAL— pero no es alineación condicional por
clase, y la alineación de marginales está bien cubierta por el trabajo previo.

**La familia mixta es el marco honesto para decidirlo**, porque contiene a r25
en $\alpha = 1$ y al ganador empírico en $\alpha = 0$, con la cota heredada y un
solo símbolo nuevo. Lo que las mediciones dicen hoy es que **ningún $\alpha$
intermedio medido le gana al extremo**, y que el único $\alpha$ derivado de una
cantidad observable falla por confundir cuántas clases se usan con cuán bien se
usan.

**Lo que haría falta para que $\alpha$ intermedio valga la pena** es una señal de
**calidad** del enrutamiento, no de su diversidad, computable sin etiquetas. No
la tenemos medida, y no hay que inventarla en prosa.

---

## 6. Lo que esto no arregla

`lunas`, donde el **oráculo también falla** (`+0.017` con las etiquetas
verdaderas en la mano). Ahí el obstáculo no es el enrutamiento. Es la columna
cuyo bloque cruzado se midió más plano (`desvío/medio ≈ 0.04`, contra
`0.66–0.81` en `rollo`). Queda abierto y nombrado.

`marginal` ahí da `+0.0442` con `t = 2.16` sobre un absoluto de `0.1553` contra
un azar de `0.1000`: **medible y sin significado**.

---

## Apéndice — una fragilidad latente que NO es un hallazgo

El `+1e-6` estabilizador de `_renyi_entropy_order_2` rompe la invariancia de
escala de la normalización por traza cuando $\operatorname{tr}(\widetilde{\mathbf K}_c^{t})$
se le acerca. Sintéticamente, con todas las confianzas en cero,
$H_2 = 13.8155 = -\ln(10^{-6})$ y $\ell_{\mathrm{glob},c} = -0.2916$, **fuera del
`[0,1]` que la Ec. (36) declara**.

**En una corrida real no se dispara nunca**: 0 de 43 261 bloques salieron de
`[0,1]`, y `w_min < 0.01` ocurrió cero veces. Se deja escrito como fragilidad,
no como defecto observado.
