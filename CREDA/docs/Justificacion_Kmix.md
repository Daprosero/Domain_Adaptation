# La normalización de `K_mix`, derivada

## 1. El estimador y la condición ideal

$$
\tilde{I}_2 \;=\; \tfrac{1}{2}\,H_2\!\left(\mathbf{K}_c^{s}\right)
\;+\; \tfrac{1}{2}\,H_2\!\left(\tilde{\mathbf{K}}_c^{t}\right)
\;-\; H_2\!\left(\mathbf{K}_c^{st}\right)
$$

donde, por la Ec. 9 en su forma estable de Frobenius,

$$
H_2(\mathbf{K}) \;=\; -\log\!\left(\operatorname{tr}\!\left(\breve{\mathbf{A}}^{\top}\breve{\mathbf{A}}\right)\right)
\;=\; -\log\!\left(\left\lVert \breve{\mathbf{A}} \right\rVert_F^{2}\right),
\qquad
\breve{\mathbf{A}} \;=\; \frac{\mathbf{K}}{\operatorname{tr}(\mathbf{K})}
$$

**Los kernels.**

Para la clase $c$, el lote aporta

$$
\text{fuente:}\quad \mathbf{f}_s^{1},\dots,\mathbf{f}_s^{\,n_c^{s}}
\qquad\qquad
\text{blanco:}\quad \mathbf{f}_t^{1},\dots,\mathbf{f}_t^{\,n_c^{t}}
$$

con el kernel gaussiano

$$
\kappa(\mathbf{a},\mathbf{b}) \;=\; \exp\!\left(-\frac{\lVert \mathbf{a}-\mathbf{b}\rVert^{2}}{2\sigma^{2}}\right),
\qquad\qquad
\kappa(\mathbf{a},\mathbf{a}) \;=\; 1
$$

y de ahí las tres matrices:

$$
\left[\mathbf{K}_c^{s}\right]_{ij} = \kappa\!\left(\mathbf{f}_s^{i},\mathbf{f}_s^{j}\right)
\;\in\; \mathbb{R}^{\,n_c^{s}\times n_c^{s}}
$$

$$
\left[\tilde{\mathbf{K}}_c^{t}\right]_{ij} = \kappa\!\left(\mathbf{f}_t^{i},\mathbf{f}_t^{j}\right)
\;\in\; \mathbb{R}^{\,n_c^{t}\times n_c^{t}}
$$

$$
\left[\mathbf{K}_c^{st}\right]_{ij} = \kappa\!\left(\mathbf{f}_s^{i},\mathbf{f}_t^{j}\right)
\;\in\; \mathbb{R}^{\,n_c^{s}\times n_c^{t}}
$$

Las dos primeras son cuadradas y simétricas, con unos en la diagonal. La tercera
es rectangular.

**La condición ideal.**

$$
n_c^{s} \;=\; n_c^{t} \;=\; n
\qquad\text{y}\qquad
\mathbf{f}_s^{i} \;=\; \mathbf{f}_t^{i}\quad \forall\, i
$$

Las nubes están concentradas y traslapadas, muestra contra muestra. Bajo esa
condición las tres matrices coinciden:

$$
\left[\mathbf{K}_c^{s}\right]_{ij}
= \left[\tilde{\mathbf{K}}_c^{t}\right]_{ij}
= \left[\mathbf{K}_c^{st}\right]_{ij}
= \kappa\!\left(\mathbf{f}_s^{i},\mathbf{f}_s^{j}\right)
\;=\!:\; \left[\mathbf{K}\right]_{ij}
$$

$$
\boxed{\;\mathbf{K}_c^{s} \;=\; \tilde{\mathbf{K}}_c^{t} \;=\; \mathbf{K}_c^{st} \;=\; \mathbf{K} \;\in\; \mathbb{R}^{\,n\times n}\;}
$$

En particular $\mathbf{K}_c^{st}$ es acá cuadrada y simétrica, de modo que
$H_2(\mathbf{K}_c^{st})$ está definida.

---

## 2. Las normas de Frobenius en el caso ideal

Como $\kappa(\mathbf{a},\mathbf{a})=1$, la diagonal de $\mathbf{K}$ son unos:

$$
\operatorname{tr}(\mathbf{K}) \;=\; \sum_{i=1}^{n} \left[\mathbf{K}\right]_{ii} \;=\; n
$$

Entonces

$$
\breve{\mathbf{A}} \;=\; \frac{\mathbf{K}}{\operatorname{tr}(\mathbf{K})} \;=\; \frac{\mathbf{K}}{n}
\qquad\Longrightarrow\qquad
\left\lVert \breve{\mathbf{A}} \right\rVert_F^{2}
\;=\; \frac{\left\lVert \mathbf{K} \right\rVert_F^{2}}{n^{2}}
$$

Las tres normas del estimador son por lo tanto **la misma**:

$$
\left\lVert \breve{\mathbf{A}}\!\left(\mathbf{K}_c^{s}\right) \right\rVert_F^{2}
= \left\lVert \breve{\mathbf{A}}\!\left(\tilde{\mathbf{K}}_c^{t}\right) \right\rVert_F^{2}
= \left\lVert \breve{\mathbf{A}}\!\left(\mathbf{K}_c^{st}\right) \right\rVert_F^{2}
= \frac{\left\lVert \mathbf{K} \right\rVert_F^{2}}{n^{2}}
$$

y en consecuencia las tres entropías coinciden:

$$
H_2\!\left(\mathbf{K}_c^{s}\right)
= H_2\!\left(\tilde{\mathbf{K}}_c^{t}\right)
= H_2\!\left(\mathbf{K}_c^{st}\right)
= -\log\!\left(\frac{\left\lVert \mathbf{K} \right\rVert_F^{2}}{n^{2}}\right)
\;=\!:\; h
$$

Sustituyendo en el estimador:

$$
\tilde{I}_2 \;=\; \tfrac{1}{2}\,h \;+\; \tfrac{1}{2}\,h \;-\; h \;=\; 0
$$

$$
\boxed{\;\tilde{I}_2 \;=\; 0 \quad \text{en la condición ideal}\;}
$$

El $\tfrac{1}{2}$ es lo que produce ese cero: con los coeficientes unitarios de
la Ec. 11 quedaría $h + h - h = h \neq 0$.

---

## 3. El $\mathbf{K}_c^{\text{mix}}$ del artículo

La sección 2 evaluó $H_2(\mathbf{K}_c^{st})$ apoyada enteramente en la condición
ideal, que la volvía cuadrada y simétrica. Fuera de ahí no lo es, y entonces no
hay nada que evaluar:

- **no es cuadrada** cuando $n_c^{s}\neq n_c^{t}$, de modo que ni siquiera tiene
  traza con la que normalizar;
- **no es simétrica** aun cuando lo sea, porque en general
  $\kappa(\mathbf{f}_s^{i},\mathbf{f}_t^{j})\neq\kappa(\mathbf{f}_s^{j},\mathbf{f}_t^{i})$,
  de modo que su espectro no es real ni no negativo y no se puede leer como una
  pdf.

$H_2$ es la entropía de Rényi de un espectro normalizado (Ec. 9), y para eso
pide una Gram. Por eso el artículo no la evalúa sobre $\mathbf{K}_c^{st}$: lo
embebe en un bloque cuadrado y simétrico (Ec. 22).

$$
\mathbf{K}_c^{\text{mix}} \;=\;
\begin{pmatrix}
\mathbf{K}_c^{s} & \mathbf{K}_c^{st}\\[2pt]
\left(\mathbf{K}_c^{st}\right)^{\top} & \tilde{\mathbf{K}}_c^{t}
\end{pmatrix}
\;\in\; \mathbb{R}^{\,N\times N},
\qquad N = n_c^{s}+n_c^{t}
$$

$$
\tilde{I}_2 \;=\; \tfrac{1}{2}\,H_2\!\left(\mathbf{K}_c^{s}\right)
\;+\; \tfrac{1}{2}\,H_2\!\left(\tilde{\mathbf{K}}_c^{t}\right)
\;-\; H_2\!\left(\mathbf{K}_c^{\text{mix}}\right)
$$

Bajo la condición ideal de la sección 1 los tres bloques son $\mathbf{K}$, y
con $n_c^{s}=n_c^{t}=n$ queda $N=2n$:

$$
\mathbf{K}_c^{\text{mix}}
\;=\;
\begin{pmatrix}\mathbf{K} & \mathbf{K}\\ \mathbf{K} & \mathbf{K}\end{pmatrix}
\;\in\; \mathbb{R}^{\,2n\times 2n}
$$

Su diagonal son $2n$ unos, así que

$$
\operatorname{tr}\!\left(\mathbf{K}_c^{\text{mix}}\right) \;=\; 2n \;=\; 2\operatorname{tr}(\mathbf{K})
$$

La norma de Frobenius al cuadrado, que es una suma sobre todas las entradas,
se parte con los bloques: partir en bloques es partir el conjunto de índices. Si las filas y columnas se
parten en $S$ (las $n_c^{s}$ de la fuente) y $T$ (las $n_c^{t}$ del blanco), los
pares quedan en cuatro grupos disjuntos que cubren todo:

$$
\{1,\dots,N\}\times\{1,\dots,N\}
\;=\;
(S\times S)\ \sqcup\ (S\times T)\ \sqcup\ (T\times S)\ \sqcup\ (T\times T)
$$

y una suma sobre un conjunto es la suma de las sumas sobre cualquier partición
suya. Como además transponer sólo reordena las entradas,
$\lVert \mathbf{A}^{\top}\rVert_F^{2}=\lVert \mathbf{A}\rVert_F^{2}$, los dos
bloques cruzados aportan lo mismo:

$$
\boxed{\;
\left\lVert \mathbf{K}_c^{\text{mix}} \right\rVert_F^{2}
\;=\;
\left\lVert \mathbf{K}_c^{s}\right\rVert_F^{2}
\;+\;2\left\lVert \mathbf{K}_c^{st}\right\rVert_F^{2}
\;+\;\left\lVert \tilde{\mathbf{K}}_c^{t}\right\rVert_F^{2}
\;}
$$

Esto vale para cualquier par de tamaños. En la condición ideal los tres son
$\mathbf{K}$, de modo que $1+2+1=4$:

$$
\left\lVert \mathbf{K}_c^{\text{mix}} \right\rVert_F^{2}
\;=\;
4\left\lVert \mathbf{K} \right\rVert_F^{2}
$$

y, de la traza de arriba,

$$
\operatorname{tr}\!\left(\mathbf{K}_c^{\text{mix}}\right)^{2} \;=\; (2n)^{2} \;=\; 4n^{2}
$$

$$
\left\lVert \breve{\mathbf{A}}\!\left(\mathbf{K}_c^{\text{mix}}\right) \right\rVert_F^{2}
\;=\;
\frac{4\left\lVert \mathbf{K} \right\rVert_F^{2}}{4n^{2}}
\;=\;
\frac{\left\lVert \mathbf{K} \right\rVert_F^{2}}{n^{2}}
\;=\;
\left\lVert \breve{\mathbf{A}}(\mathbf{K}) \right\rVert_F^{2}
$$

El $4$ de los cuatro bloques se cancela contra el $4$ de la traza al cuadrado, y
queda la misma norma de la sección 2. Por lo tanto

$$
\boxed{\;H_2\!\left(\mathbf{K}_c^{\text{mix}}\right) \;=\; H_2(\mathbf{K}) \;=\; h\;}
$$

$$
\tilde{I}_2 \;=\; \tfrac{1}{2}h \;+\; \tfrac{1}{2}h \;-\; h \;=\; 0
$$

El bloque del artículo reproduce el caso de la sección 2 exactamente.

---

## 4. Con $n_c^{s}=n_c^{t}$ el factor de escala se cancela

Acá no se supone nada sobre las nubes: $\mathbf{K}_c^{s}$,
$\mathbf{K}_c^{st}$ y $\tilde{\mathbf{K}}_c^{t}$ son arbitrarias. Lo único que
se impone es $n_c^{s}=n_c^{t}=n$.

La normalización por bloques divide cada bloque por el factor que le
corresponde: el propio de la fuente por $n_c^{s}$, el propio del blanco por
$n_c^{t}$, y el cruzado por $\sqrt{n_c^{s}n_c^{t}}$. Con $n_c^{s}=n_c^{t}=n$
**los tres factores coinciden**, porque $\sqrt{n\cdot n}=n$. Las dos matrices,
en paralelo:

$$
\mathbf{K}_c^{\text{mix}} =
\begin{pmatrix}
\mathbf{K}_c^{s} & \mathbf{K}_c^{st}\\[6pt]
\left(\mathbf{K}_c^{st}\right)^{\top} & \tilde{\mathbf{K}}_c^{t}
\end{pmatrix}
\qquad\qquad
\hat{\mathbf{K}}_c^{\text{mix}} =
\begin{pmatrix}
\dfrac{\mathbf{K}_c^{s}}{n} & \dfrac{\mathbf{K}_c^{st}}{n}\\[10pt]
\dfrac{\left(\mathbf{K}_c^{st}\right)^{\top}}{n} & \dfrac{\tilde{\mathbf{K}}_c^{t}}{n}
\end{pmatrix}
= \frac{1}{n}\,\mathbf{K}_c^{\text{mix}}
$$

**Las trazas**, sumando la diagonal de cada una. En la izquierda son unos; en la
derecha, unos divididos por $n$:

$$
\operatorname{tr}\!\left(\mathbf{K}_c^{\text{mix}}\right)
=\underbrace{1+\dots+1}_{n\ \text{fuente}}+\underbrace{1+\dots+1}_{n\ \text{blanco}}
= 2n
$$

$$
\operatorname{tr}\!\left(\hat{\mathbf{K}}_c^{\text{mix}}\right)
=\underbrace{\tfrac{1}{n}+\dots+\tfrac{1}{n}}_{n\ \text{fuente}}+\underbrace{\tfrac{1}{n}+\dots+\tfrac{1}{n}}_{n\ \text{blanco}}
= 1+1 = 2
$$

**Las normas.** La de la derecha sale de la de la izquierda, porque el escalar
$1/n$ sale al cuadrado:

$$
\left\lVert \hat{\mathbf{K}}_c^{\text{mix}}\right\rVert_F^{2}
= \left\lVert \tfrac{1}{n}\,\mathbf{K}_c^{\text{mix}}\right\rVert_F^{2}
= \left(\tfrac{1}{n}\right)^{2}\left\lVert \mathbf{K}_c^{\text{mix}}\right\rVert_F^{2}
= \frac{\left\lVert \mathbf{K}_c^{\text{mix}}\right\rVert_F^{2}}{n^{2}}
$$

Y ahora cada una dividida por **su propia traza** al cuadrado, alineadas para
poder compararlas:

$$
\begin{aligned}
\left\lVert \breve{\mathbf{A}}\!\left(\mathbf{K}_c^{\text{mix}}\right) \right\rVert_F^{2}
&= \frac{\left\lVert \mathbf{K}_c^{\text{mix}}\right\rVert_F^{2}}
        {\operatorname{tr}\!\left(\mathbf{K}_c^{\text{mix}}\right)^{2}}
= \frac{\left\lVert \mathbf{K}_c^{\text{mix}}\right\rVert_F^{2}}{\left(2n\right)^{2}}
= \frac{\left\lVert \mathbf{K}_c^{\text{mix}}\right\rVert_F^{2}}{4n^{2}}
\\[10pt]
\left\lVert \breve{\mathbf{A}}\!\left(\hat{\mathbf{K}}_c^{\text{mix}}\right) \right\rVert_F^{2}
&= \frac{\left\lVert \hat{\mathbf{K}}_c^{\text{mix}}\right\rVert_F^{2}}
        {\operatorname{tr}\!\left(\hat{\mathbf{K}}_c^{\text{mix}}\right)^{2}}
= \frac{\left\lVert \mathbf{K}_c^{\text{mix}}\right\rVert_F^{2}/n^{2}}{2^{2}}
= \frac{\left\lVert \mathbf{K}_c^{\text{mix}}\right\rVert_F^{2}}{4n^{2}}
\end{aligned}
$$

Las dos dan lo mismo: el $n^{2}$ que aparece en la norma se compensa con el
$n^{2}$ que desaparece de la traza al cuadrado.

$$
\boxed{\;
n_c^{s}=n_c^{t}
\;\Longrightarrow\;
\left\lVert \breve{\mathbf{A}}\!\left(\hat{\mathbf{K}}_c^{\text{mix}}\right) \right\rVert_F^{2}
=
\left\lVert \breve{\mathbf{A}}\!\left(\mathbf{K}_c^{\text{mix}}\right) \right\rVert_F^{2}
\;\Longrightarrow\;
H_2\!\left(\hat{\mathbf{K}}_c^{\text{mix}}\right) = H_2\!\left(\mathbf{K}_c^{\text{mix}}\right)
\;}
$$

**La entropía lee el espectro, no el conteo.** Con tamaños iguales la
normalización no puede cambiar nada: es la identidad, cualesquiera sean las
nubes.

---

## 5. Con $n_c^{s}\neq n_c^{t}$ deja de cancelar

Ahora sin imponer nada sobre los tamaños. Los tres factores ya no coinciden, de
modo que $\hat{\mathbf{K}}_c^{\text{mix}}$ deja de ser un múltiplo de
$\mathbf{K}_c^{\text{mix}}$. Las dos matrices, en paralelo:

$$
\mathbf{K}_c^{\text{mix}} =
\begin{pmatrix}
\mathbf{K}_c^{s} & \mathbf{K}_c^{st}\\[6pt]
\left(\mathbf{K}_c^{st}\right)^{\top} & \tilde{\mathbf{K}}_c^{t}
\end{pmatrix}
\qquad\qquad
\hat{\mathbf{K}}_c^{\text{mix}} =
\begin{pmatrix}
\dfrac{\mathbf{K}_c^{s}}{n_c^{s}} & \dfrac{\mathbf{K}_c^{st}}{\sqrt{n_c^{s}n_c^{t}}}\\[10pt]
\dfrac{\left(\mathbf{K}_c^{st}\right)^{\top}}{\sqrt{n_c^{s}n_c^{t}}} & \dfrac{\tilde{\mathbf{K}}_c^{t}}{n_c^{t}}
\end{pmatrix}
$$

**Las trazas**, sumando la diagonal de cada una. A la izquierda son unos; a la
derecha, cada dominio lleva su propio divisor:

$$
\operatorname{tr}\!\left(\mathbf{K}_c^{\text{mix}}\right)
=\underbrace{1+\dots+1}_{n_c^{s}\ \text{fuente}}+\underbrace{1+\dots+1}_{n_c^{t}\ \text{blanco}}
= n_c^{s}+n_c^{t}
$$

$$
\operatorname{tr}\!\left(\hat{\mathbf{K}}_c^{\text{mix}}\right)
=\underbrace{\tfrac{1}{n_c^{s}}+\dots+\tfrac{1}{n_c^{s}}}_{n_c^{s}\ \text{fuente}}
+\underbrace{\tfrac{1}{n_c^{t}}+\dots+\tfrac{1}{n_c^{t}}}_{n_c^{t}\ \text{blanco}}
= 1+1 = 2
$$

La de la izquierda **es el conteo de muestras**. La de la derecha vale $2$
cualesquiera sean los tamaños: cada dominio aporta exactamente $1$.

**Las normas**, también en paralelo. A la izquierda las tres sobre un mismo
divisor; a la derecha cada una sobre el de su bloque:

$$
\left\lVert \breve{\mathbf{A}}\!\left(\mathbf{K}_c^{\text{mix}}\right) \right\rVert_F^{2}
=
\frac{\left\lVert \mathbf{K}_c^{s}\right\rVert_F^{2}
+2\left\lVert \mathbf{K}_c^{st}\right\rVert_F^{2}
+\left\lVert \tilde{\mathbf{K}}_c^{t}\right\rVert_F^{2}}
{\left(n_c^{s}+n_c^{t}\right)^{2}}
$$

$$
\left\lVert \breve{\mathbf{A}}\!\left(\hat{\mathbf{K}}_c^{\text{mix}}\right) \right\rVert_F^{2}
=
\frac{1}{4}\left[
\frac{\left\lVert \mathbf{K}_c^{s}\right\rVert_F^{2}}{\left(n_c^{s}\right)^{2}}
+\frac{2\left\lVert \mathbf{K}_c^{st}\right\rVert_F^{2}}{n_c^{s}\,n_c^{t}}
+\frac{\left\lVert \tilde{\mathbf{K}}_c^{t}\right\rVert_F^{2}}{\left(n_c^{t}\right)^{2}}
\right]
$$

Las dos llevan **las mismas tres normas**; lo que cambia es con qué se divide
cada una:

| norma | sin normalizar | normalizada |
|---|---|---|
| $\left\lVert \mathbf{K}_c^{s}\right\rVert_F^{2}$ | $\left(n_c^{s}+n_c^{t}\right)^{2}$ | $4\left(n_c^{s}\right)^{2}$ |
| $2\left\lVert \mathbf{K}_c^{st}\right\rVert_F^{2}$ | $\left(n_c^{s}+n_c^{t}\right)^{2}$ | $4\,n_c^{s}n_c^{t}$ |
| $\left\lVert \tilde{\mathbf{K}}_c^{t}\right\rVert_F^{2}$ | $\left(n_c^{s}+n_c^{t}\right)^{2}$ | $4\left(n_c^{t}\right)^{2}$ |

Sin normalizar las tres comparten **un mismo divisor**; normalizando, cada una
lleva el de su bloque. Los tres divisores de la derecha coinciden entre sí — y
con el de la izquierda — exactamente cuando $n_c^{s}=n_c^{t}$, que es el caso de
la sección 4.

$$
\boxed{\;
\left\lVert \breve{\mathbf{A}}\!\left(\hat{\mathbf{K}}_c^{\text{mix}}\right) \right\rVert_F^{2}
=
\left\lVert \breve{\mathbf{A}}\!\left(\mathbf{K}_c^{\text{mix}}\right) \right\rVert_F^{2}
\iff
n_c^{s}=n_c^{t}
\;}
$$

Y una diferencia estructural entre las dos trazas:

$$
\operatorname{tr}\!\left(\mathbf{K}_c^{\text{mix}}\right) = n_c^{s}+n_c^{t}
\qquad\text{contra}\qquad
\operatorname{tr}\!\left(\hat{\mathbf{K}}_c^{\text{mix}}\right) = 2
$$

La traza de la versión normalizada **no depende de los tamaños**. La de la
versión sin normalizar es el conteo de muestras.

Las dos expresiones llevan **las mismas tres cantidades**. Sacando factor común
en cada término, con

$$
p \;=\; \frac{n_c^{s}}{n_c^{s}+n_c^{t}}
\qquad\qquad
q \;=\; \frac{n_c^{t}}{n_c^{s}+n_c^{t}}
\qquad\qquad
p+q \;=\; 1
$$

quedan una al lado de la otra:

$$
\left\lVert \breve{\mathbf{A}}\!\left(\mathbf{K}_c^{\text{mix}}\right) \right\rVert_F^{2}
=
p^{2}\,\frac{\left\lVert \mathbf{K}_c^{s}\right\rVert_F^{2}}{\left(n_c^{s}\right)^{2}}
\;+\;
2pq\,\frac{\left\lVert \mathbf{K}_c^{st}\right\rVert_F^{2}}{n_c^{s}n_c^{t}}
\;+\;
q^{2}\,\frac{\left\lVert \tilde{\mathbf{K}}_c^{t}\right\rVert_F^{2}}{\left(n_c^{t}\right)^{2}}
$$

$$
\left\lVert \breve{\mathbf{A}}\!\left(\hat{\mathbf{K}}_c^{\text{mix}}\right) \right\rVert_F^{2}
=
\tfrac{1}{4}\,\frac{\left\lVert \mathbf{K}_c^{s}\right\rVert_F^{2}}{\left(n_c^{s}\right)^{2}}
\;+\;
\tfrac{1}{2}\,\frac{\left\lVert \mathbf{K}_c^{st}\right\rVert_F^{2}}{n_c^{s}n_c^{t}}
\;+\;
\tfrac{1}{4}\,\frac{\left\lVert \tilde{\mathbf{K}}_c^{t}\right\rVert_F^{2}}{\left(n_c^{t}\right)^{2}}
$$

Los dos son promedios ponderados de lo mismo —$p^{2}+2pq+q^{2}=(p+q)^{2}=1$ y
$\tfrac14+\tfrac12+\tfrac14=1$— y lo único que los separa son los pesos: a la
izquierda **puro conteo**, a la derecha constantes. Con $n_c^{s}=n_c^{t}$ se
tiene $p=q=\tfrac12$, de donde $p^{2}=\tfrac14$, $2pq=\tfrac12$,
$q^{2}=\tfrac14$: **los mismos pesos**, que es el resultado de la sección 4.

**Y eso es lo que justifica normalizar.** Con tamaños distintos
$\hat{\mathbf{K}}_c^{\text{mix}}$ no es la misma matriz —su espectro cambia—
pero sigue siendo PSD: dividir cada bloque por su factor equivale a reescalar
cada tramo del vector de prueba por la raíz de su propio conteo, de modo que con
$\mathbf{v}=(\mathbf{x},\mathbf{y})$ y
$\mathbf{w}=\left(\mathbf{x}/\sqrt{n_c^{s}},\,\mathbf{y}/\sqrt{n_c^{t}}\right)$
sale $\mathbf{v}^{\top}\hat{\mathbf{K}}_c^{\text{mix}}\mathbf{v}
=\mathbf{w}^{\top}\mathbf{K}_c^{\text{mix}}\mathbf{w}\geq 0$. El estimador de
Rényi no se altera: la Ec. 21 se evalúa igual sobre ella. Y el fenómeno que mide
es exactamente el que $\mathbf{K}_c^{\text{mix}}$ sin normalizar mide cuando
$n_c^{s}=n_c^{t}$ —los pesos $\left(\tfrac14,\tfrac12,\tfrac14\right)$— sólo que
ahora en toda proporción, y no sólo en la balanceada. **Esa es la lectura que no
depende de cuántos datos aportó cada dominio.**

---

---

## 6. Con el bloque del blanco SOPESADO, la igualdad SE MANTIENE

Las secciones 1 a 5 trabajan sobre bloques cuya diagonal es de unos. El término
usa el bloque del blanco **ponderado por confianza**,
$\tilde{\mathbf{K}}_c^{t}=\mathbf{K}_c^{t}\circ\left(\mathbf{w}\mathbf{w}^{\top}\right)$,
y el cruzado también, $\tilde{\mathbf{K}}_c^{st}=\mathbf{K}_c^{st}\operatorname{diag}(\mathbf{w})$.
Esta sección dice qué sobrevive de la sección 4 y qué no.

### 6.1 La condición cambia, el teorema no

La diagonal de $\tilde{\mathbf{K}}_c^{t}$ es $w_j^{2}$ y no unos, de modo que

$$
\operatorname{tr}\!\left(\tilde{\mathbf{K}}_c^{t}\right)=\sum_{j} w_j^{2}
\;\neq\; n_c^{t}
\qquad\text{salvo que }\mathbf{w}\equiv\mathbf{1}.
$$

El argumento de la sección 4 **no pide cardinalidades iguales: pide que los
tres factores coincidan**, y los tres factores son
$\operatorname{tr}(\mathbf{K}_c^{s})$, $\operatorname{tr}(\tilde{\mathbf{K}}_c^{t})$
y su media geométrica. La media geométrica iguala a las otras dos exactamente
cuando éstas son iguales entre sí, así que la condición es

$$
\boxed{\;\operatorname{tr}\!\left(\mathbf{K}_c^{s}\right)
=\operatorname{tr}\!\left(\tilde{\mathbf{K}}_c^{t}\right)
\;\Longleftrightarrow\;
n_c^{s}=\sum_j w_j^{2}\;}
$$

que con $\mathbf{w}\equiv\mathbf{1}$ se reduce a $n_c^{s}=n_c^{t}$, la de la
sección 4. **Bajo esa condición la normalización vuelve a ser un múltiplo
escalar y toda la sección 4 corre palabra por palabra.**

### 6.2 Verificado: la igualdad se mantiene

Reescalando $\mathbf{w}\mapsto\mathbf{w}\sqrt{n_c^{s}/\sum_j w_j^{2}}$ se impone
la condición. Comparando $H_2$ de la mezcla sopesada **sin normalizar** contra
la **normalizada** —los dos lados con los mismos bloques, que es lo que la
sección 4 afirma—:

| pesos | $\operatorname{tr}(\mathbf{K}^{s})$ | $\operatorname{tr}(\tilde{\mathbf{K}}^{t})$ | $H_2$ cruda | $H_2$ norm. | diferencia |
|---|---|---|---|---|---|
| sin pesos | 60.00 | 60.00 | 3.36495 | 3.36495 | $2\times10^{-7}$ |
| típicos, sin reescalar | 60.00 | 27.46 | 3.27752 | 3.25905 | $1.9\times10^{-2}$ |
| típicos, **reescalados** | 60.00 | 60.00 | 3.25905 | 3.25905 | **0** |
| mitad uno / mitad cero | 60.00 | 30.00 | 3.24081 | 3.18682 | $5.4\times10^{-2}$ |
| idem, **reescalados** | 60.00 | 60.00 | 3.18682 | 3.18682 | **0** |

**Cero exacto.** La normalización es inocente: cuando las masas coinciden no
toca nada, y cuando no coinciden es justamente lo que vuelve comparables las
dos lecturas. La sección 5 ya decía eso para el caso de cardinalidades
distintas; acá vale igual con pesos.

### 6.3 Lo que SÍ cambia, y no es la normalización

Hay un efecto real del sopesado, y conviene no atribuírselo a la normalización
porque está presente **con y sin ella**. Comparando ahora sopesado contra sin
sopesar, con la **geometría idéntica** —el kernel no se toca—:

| pesos | $\sum_j w_j^{2}$ | ESS | $H_2(\tilde{\mathbf{K}}_c^{t})$ | $\mathrm{creda}_c$ |
|---|---|---|---|---|
| sin pesos | 60.00 | 60.0 | 3.1678 | 0.20525 |
| típicos | 27.46 | 38.9 | 2.9194 | 0.22355 |
| típicos, reescalados | 60.00 | 38.9 | 2.9194 | 0.22355 |
| mitad uno / mitad cero | 30.00 | 30.0 | 2.7519 | 0.23504 |
| idem, reescalados | 60.00 | 30.0 | 2.7519 | 0.23504 |

El efecto **sigue al tamaño efectivo de muestra y es indiferente a
$\sum_j w_j^{2}$**: las filas reescaladas repiten las originales bit a bit. No
es desajuste de masas, que es lo que la normalización arregla.

La razón es que $\operatorname{diag}(\mathbf{w})$ es una **congruencia** y no un
escalar. Para un bloque cuyo kernel se acerca a la identidad,
$\tilde{\mathbf{K}}_c^{t}\approx\operatorname{diag}(w_j^{2})$ y

$$
\frac{\operatorname{tr}\left(\tilde{\mathbf{K}}_c^{t\,2}\right)}
     {\operatorname{tr}\left(\tilde{\mathbf{K}}_c^{t}\right)^{2}}
= \frac{\sum_j w_j^{4}}{\left(\sum_j w_j^{2}\right)^{2}}
= \frac{1}{\mathrm{ESS}(\mathbf{w})},
\qquad
H_2\!\left(\tilde{\mathbf{K}}_c^{t}\right)\approx\log \mathrm{ESS}(\mathbf{w}).
$$

**$H_2$ del bloque sopesado lee cuántas muestras quedan con peso, y eso es una
cantidad del vector de pesos y no de las nubes.**

### 6.4 La consecuencia sobre la puntuación

Con $\mathrm{creda}_c=H_2(\text{mezcla})-\tfrac12\left(H_2^{s}+H_2^{t}\right)$,
desparejar los pesos baja $H_2^{t}$ en $-0.194$ y baja la mezcla sólo en
$-0.075$, de donde

$$
\Delta\,\mathrm{creda}_c \;=\; -0.075-\tfrac12(-0.194)\;=\;+0.022,
$$

el valor medido. La mezcla no sigue a $H_2^{t}$ y el factor $\tfrac12$ no
compensa la diferencia. Barriendo el coeficiente de variación de $\mathbf{w}$
con la geometría fija, el efecto crece monótono y satura: $0$ a $cv=0$,
$0.0247$ a $cv=0.30$, $0.0528$ a $cv=0.60$, $0.0639$ a $cv=1$, con $r=+0.971$
y $p=6.1\times10^{-5}$.

**Lo paga cualquier vector de pesos desparejo, incluido uno perfecto**, y está
antes de normalizar. Es una propiedad del sopesado dentro de $H_2$, no de la
proporción de mezcla.

---

> **Alcance.** Las identidades de 6.1 y 6.3 son algebraicas. Los números de
> 6.2, 6.3 y 6.4 salen de un montaje sintético por instancias —no bolsas—, con
> una cabeza de un solo `Linear`, tres clases, la fuente congelada y sólo el
> término condicional por clase.
>
> **Corrección de una versión anterior de esta sección**, que afirmaba que la
> sección 4 dejaba de valer. No deja de valer: la condición cambia de
> $n_c^{s}=n_c^{t}$ a $n_c^{s}=\sum_j w_j^{2}$ y el teorema corre igual, con
> diferencia exactamente nula. Lo que aquella versión medía era sopesado contra
> sin sopesar, que es la afirmación de 6.3 y no la de la sección 4.

