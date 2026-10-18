# Lo que dejó `Toy_Datasets.ipynb`, retirado el 2026-10-12

El cuaderno que produjo estos artefactos fue **borrado** ese día, por decisión del
operador, para rehacerlo con la estructura de dos secciones. **Su código se
perdió**: estaba sin rastrear en git y el único respaldo vivía en un directorio
temporal de sesión que se limpió. Lo que sigue acá son sus SALIDAS, que nunca se
tocaron, y la descripción de qué eran — para que un lector no encuentre cinco
archivos huérfanos sin saber de qué hablan.

## Qué midió

Corría sobre el toy propio de MIL-CREDA (`src/MIL_CREDA_Benchmark/toy.py`): nueve
códigos de dominio sobre tres familias (rollo suizo, toroides, medias lunas), 10
clases, 12 bolsas por clase, 30 instancias de 45 puntos por sujeto. **Ese toy fue
descartado** a favor del de CREDA — su `TOY_NOISE_SIGMA = 0.3` es absoluto
mientras las tres familias viven en escalas 0.013 / 0.496 / 5.137, así que el
mismo ruido nominal infla un sujeto 29x, 1.34x y 1.01x: tres experimentos
distintos bajo una misma etiqueta.

## Los archivos

- `toy_search.json` — la sintonización sobre `valid`: rejilla de 4 multiplicadores
  de sigma x 3 tau x 3 tasas, 1 semilla, 20 épocas, brazo `G`, sobre 6 celdas
  (3 familias x 2 backbones). Lleva el `metodo` y la `tasa` elegidos por celda.
- `toy_datasets.json` — las filas medidas sobre `eval`: 20 épocas, 3 semillas,
  3 tasas de contaminación, 4 brazos, 2 redes = **144 filas**.
- `toy_figures/toy_dominios.png` — los dos dominios por columna y por régimen.
- `toy_figures/toy_alineacion.png` — una línea de cada bolsa `eval` del blanco a
  la bolsa `train` de fuente donde pi pone más masa; verde si era el par
  verdadero por construcción, roja si no.
- `toy_figures/toy_sopesado.png` — las instancias contaminadas como cruces, con
  el tamaño del marcador igual al peso de ATENCIÓN (beta, intra-bolsa; no la
  confianza `w` de la Ec. 24). Medido al empate, 15 instancias del donante
  contra 15 propias.

## Qué NO vale de acá

Todas estas filas se midieron bajo **r25**, antes de que el término global pasara
a r29 (Ec. 32 el balance por traza, Ecs. 34-35 las cotas entropy-cosh, Ec. 36 el
span sin gradiente, Ec. 37 la compuerta de confianza). No son comparables con
nada que se mida de ahora en adelante.
