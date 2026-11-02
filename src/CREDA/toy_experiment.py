"""El toyset de CREDA: las tres familias duras, sus particiones, el codificador
y las piezas de calculo que el cuaderno tutorial importa. Nada de aca entrena
ni dibuja.

El cuaderno `CREDA/Notebooks/Toyset.ipynb` escribe sus propios bucles de
entrenamiento, su construccion de la perdida y sus figuras. Lo que toma de
aca es el toyset mismo --- `FAMILIAS`, `datos_semilla`, `MLP` ---, el termino
supervisado normalizado de la fase 1 (`perdida_fuente`), el ancho de banda
congelado sobre la fuente (`sigma_de_la_fuente`), el monitor sin etiquetas
(`clases_efectivas`) y, a traves de `CREDA.tutorial_visuals`, las fronteras
exactas del cabezal y la caminata de confianza (`puntos_de_frontera`,
`caminata_al_centro`). El toyset de MIL-CREDA (`MIL_CREDA_Benchmark.toy_bags`
y `toy_run`) lee de aca las mismas familias (`DURAS`, `perturbar`), el ancho
de banda congelado y `ASPECTO_MAX`.

El banco experimental anterior --- el corredor de dos fases, las busquedas de
hiperparametros, los barridos, el artefacto en disco, la contaminacion, los
techos de oraculo, las ablaciones y el modulo de dibujo `toy_plots` --- se
retiro cuando el cuaderno tutorial lo reemplazo: ningun cuaderno vivo lo
alcanzaba.

**This module implements no equation, and the provenance below says so.** It
holds the toyset and the helpers the tutorial reads; every term it computes
with comes from somewhere else -- `CREDA.bounds.source_loss` for Eq. (21),
`MIL_CREDA.confidence.confidences` for Eq. (24) and `CREDALoss` for the
bandwidth heuristic -- so stamping a revision here would empty the check for
the modules that carry one.

What it does declare is a CONFIGURATION: which bounds map, which balanced mass,
and whether the per-class score is normalized at all. The tutorial's loss is
built from those three declarations (`tests/test_creda_tutorial.py`).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CREDA.bounds import source_loss
from CREDA.models import CREDALoss
from MIL_CREDA.confidence import confidences


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        "the toyset and the helpers the tutorial reads, not a term; see the "
        "module docstring"),
    "invariants": [
        "the_toy_now_normalizes_by_bounds",
        "the_toy_declares_the_two_new_values",
        "the_bandwidth_is_the_median_over_the_whole_source_train",
        "moving_the_target_train_cannot_move_the_bandwidth",
        "the_batch_size_cannot_move_the_bandwidth",
        "the_supervised_term_is_equation_twenty_ones_normalized_form",
        "the_supervised_term_matches_a_hand_written_equation_twenty_one",
        "the_supervised_term_lives_in_the_unit_interval",
        "the_toy_never_imports_the_method_it_is_the_baseline_for",
        "the_effective_classes_stay_within_one_and_the_class_count",
    ],
}

RUIDO_REL, TRASLADO_REL = 0.025, 3.0
# El angulo de la rotacion que preserva Sigma, en grados. Es el boton que decide
# cuanta de la perturbacion NO es de segundo orden; ver `perturbar` abajo.
ANGULO_SIGMA = 10.0

def _rotacion_generica(angulo, eje=(1.0, 0.7, 0.4)):
    # Rodrigues: una rotacion de verdad en 3D. Una alineada a un eje deja una
    # coordenada intacta, y entonces el problema que plantea es mas facil de lo
    # que su angulo sugiere.
    k = np.array(eje, dtype=float); k = k/np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angulo)*K + (1 - np.cos(angulo))*(K @ K)

def _raiz(S):
    # Sigma^(1/2) y Sigma^(-1/2) por descomposicion espectral.
    w, V = np.linalg.eigh(S); w = np.clip(w, 1e-12, None)
    return V @ np.diag(np.sqrt(w)) @ V.T, V @ np.diag(1/np.sqrt(w)) @ V.T

def perturbar(Xs, Xt, rng, grados=None):
    """El corrimiento de dominio, en DOS partes con papeles distintos.

    Antes era una rotacion generica de 30 grados mas una traslacion de 3 sigma.
    El problema era que una rotacion generica MUEVE la covarianza --- medido,
    0.269 relativo --- asi que AdaBN arregla parte de ella y el residuo no es
    demostrablemente no-estadistico. Lo que CREDA ganaba quedaba confundido con
    exactamente lo que AdaBN arregla, que es lo que habia que excluir.

    Las dos partes de ahora:

      1. una rotacion que preserva Sigma EXACTAMENTE, con
             A = Sigma_s^(1/2) R(theta) Sigma_t^(-1/2)
         que cumple A Sigma_t A^T = Sigma_s para cualquier R ortogonal. Centrada
         en la media y recentrada a mu_s, el blanco queda con la media y la
         covarianza ENTERAS de la fuente: medido, 1e-15 sin ruido y 1e-3 con el
         ruido al 2.5%, que es unas 30 veces por DEBAJO del piso de dos sorteos
         independientes de la misma distribucion (3e-2 a 4.6e-2). AdaBN corrige
         media y varianza por canal, y CORAL la covarianza entera: los dos son
         no-ops DEMOSTRABLES sobre esta parte, no empiricamente debiles.

      2. una traslacion de 3 sigma, que AdaBN si cierra. No esta de adorno: con
         la rotacion sola, `.predict` queda en el azar y entonces el ruteo por
         clase de CREDA --- que sale de softmax(cab(f_t)) --- es ruido y CREDA
         no tiene de donde arrancar. Medido en ese regimen: AdaBN 0.4675 contra
         un .predict de 0.4675 (no-op exacto) y CREDA entre -0.006 y -0.046.
         Con ruteo de ORACULO en cambio sube +0.058/+0.163/+0.130, asi que el
         kernel TIENE la informacion y lo que falta es el arranque.

    Por eso la atribucion limpia es CREDA menos AdaBN y no CREDA menos
    `.predict`: AdaBN cierra lo que el segundo orden puede cerrar, y lo que
    quede encima tiene brecha de covarianza cero.

    `grados` en 10 por medicion: con la traslacion puesta, CREDA menos AdaBN da
    +0.105 (0 grados), +0.133 (10), +0.089 (20), +0.062 (30), +0.031 (45),
    -0.009 (60) sobre rollo, y AdaBN cae monotono 0.765 -> 0.594, que es la
    confirmacion medida de que la rotacion no es corregible por segundo orden.
    Sobre 3 semillas: la ventana 10-20 es solida, 10 contra 20 no es resoluble.
    """
    g = ANGULO_SIGMA if grados is None else grados
    esc = float(Xs.std(0).mean())
    Hs, _ = _raiz(np.cov(Xs.T))
    _, Hti = _raiz(np.cov(Xt.T))
    A = Hs @ _rotacion_generica(np.radians(g)) @ Hti
    Y = (Xt - Xt.mean(0)) @ A.T + Xs.mean(0)
    return (Y + np.full(3, TRASLADO_REL*esc/np.sqrt(3))
            + rng.normal(0, RUIDO_REL*esc, Xt.shape))

def dura_lunas(n, rng):
    # Dos parabolas opuestas, extruidas. El brazo llega a 1.6 y la extrusion a
    # 0.5: la parabola tiene que dominar la lamina o la figura se lee como dos
    # planos. Las tres clases con n/3 cada una -- un 1:1:2 seria un confundidor.
    k = n//3; BRAZO, ALTO = 1.6, 0.5
    t1 = rng.uniform(-BRAZO, 0.0, k)          # clase 0: mitad izquierda
    t2 = rng.uniform(0.0, BRAZO, k)           # clase 1: mitad derecha, MISMA curva
    t3 = rng.uniform(-BRAZO, BRAZO, n - 2*k)  # clase 2: la lamina opuesta
    z = lambda m: rng.uniform(-ALTO, ALTO, m)
    X = np.vstack([np.column_stack([t1, t1**2, z(k)]),
                   np.column_stack([t2, t2**2, z(k)]),
                   np.column_stack([t3 + 1.5, -t3**2 + 1.4, z(n - 2*k)])])
    y = np.concatenate([np.zeros(k), np.ones(k), np.full(n - 2*k, 2)])
    return X, y.astype(np.int64)

def dura_toroide(n, rng):
    # Un toro completo cortado en tres arcos de 120 grados del angulo mayor: las
    # clases quedan sobre un anillo, asi que ninguna esta entre las otras dos.
    R, r = 3.0, 1.1
    phi = rng.uniform(0, 2*np.pi, n); th = rng.uniform(0, 2*np.pi, n)
    X = np.column_stack([(R + r*np.cos(th))*np.cos(phi),
                         (R + r*np.cos(th))*np.sin(phi), r*np.sin(th)])
    return X, (np.floor(phi/(2*np.pi/3)).astype(np.int64) % 3)

def dura_rollo(n, rng):
    # Estirarlo, cortarlo en tres segmentos consecutivos y volver a enrollarlo:
    # los segmentos se entrelazan en el espacio aunque esten lejos en la variedad.
    t = rng.uniform(1.5*np.pi, 4.5*np.pi, n); h = rng.uniform(0, 12.0, n)
    X = np.column_stack([t*np.cos(t), h, t*np.sin(t)])
    cortes = np.linspace(1.5*np.pi, 4.5*np.pi, 4)
    return X, np.digitize(t, cortes[1:-1]).astype(np.int64)

DURAS = [('lunas', dura_lunas), ('toroide', dura_toroide), ('rollo', dura_rollo)]
VISTA_DURA = {'lunas': (52, -89), 'toroide': (30, -62), 'rollo': (18, -72)}
COL_DURA = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c'}

#: Los puntos por dominio que el toyset sortea. `datos_semilla` sortea con
#: ella, asi que es UNA constante y no un literal repetido.
N_POR_DOMINIO = 1800

SEMILLA, CLASES, DIM = 0, 3, 16

# UNA sola capa oculta en vez de dos. El cuaderno de una capa existe para
# preguntar si el termino de fuente puede suplir la capacidad que la red pierde,
# y para eso la red tiene que perderla de verdad. La normalizacion sigue yendo
# DESPUES de la primera oculta, que es el mismo lugar que en la red completa:
# con cero ocultas no habria donde ponerla y el eje que se compara desapareceria.
UNA_CAPA = True

#: Las dos banderas de la fase 2 que todavia se leen: el cuaderno construye su
#: perdida con estos valores, y `tests/test_milcreda_pinned_laws.py` fija los
#: defaults de `MIL_CREDA` contra ellos. Las demas llaves alimentaban al
#: corredor retirado y su busqueda.
HP_2 = dict(
            # Si el peso entra tambien en el bloque CRUZADO. `True` es lo
            # publicado y lo que la r29 describe. `False` es la version
            # original de CREDA, y la diferencia no es cosmetica: con el
            # cruzado sopesado, una muestra de peso CERO recibe gradiente
            # cero del termino y queda CONGELADA donde esta; sin sopesar,
            # sigue recibiendolo por el cruzado y el termino todavia puede
            # repararla. Es el eje que separa 'quien define la geometria'
            # de 'a quien repara el termino', que hoy decide UN solo
            # numero y conviene lo contrario en cada una.
            CRUZADO_SOPESADO=True,
            # Instrumento: `h_t` sin gradiente dentro de la
            # puntuacion por clase. Ver `CREDALoss`.
            RHO_T_SIN_GRAD=False)

#: El unico trabajo de esta instancia es prestar su heuristico de ancho de
#: banda. Existe para que el sigma que se congela sea EL MISMO estadistico que
#: el kernel habria calculado, y no un segundo que se le parece:
#: `_median_pairwise` es la mediana de las distancias pareadas sobre UN
#: conjunto, y reimplementarla aca seria dos definiciones de la misma cosa que
#: nada obliga a coincidir. `tests/test_toy_sigma_fijo.py` las fija iguales.
#:
#: **No nombra ninguna bandera y no le hace falta: esto no calcula una
#: perdida.** Solo se le llama `_median_pairwise`, un estadistico del ancho de
#: banda que no mira cotas, ni entropias, ni bloques de clase.
_HEURISTICO_DE_SIGMA = CREDALoss()


def sigma_de_la_fuente(H):
    """El ancho de banda: la mediana pareada del train de la FUENTE, un float.

    **Solo la fuente, y por eso es inmune.** Nada que se inyecte en el blanco
    --- intrusos, reubicaciones, contaminacion --- puede moverlo, que es lo que
    hace medible un experimento de sopesado: el daño que entra por el bloque de
    clase deja de confundirse con el que entra por la unidad de medida.

    **Y un solo numero para todos los lotes y todas las clases.** El alcance
    por lote estimaba de nuevo en cada paso, de modo que las lecturas del
    termino no eran comparables ENTRE pasos. Medido en `models.SIGMA_SCOPES`:
    el alcance agrupado deriva 2.65x a 3.36x en 800 pasos del regimen libre
    mientras el de solo-fuente deriva 1.03 a 1.06, porque el encoder de la
    fuente esta congelado y su mediana varia solo con el muestreo del lote.
    Congelarlo saca tambien ese 3-6%.

    **Lo que se paga, y se reporta en vez de adaptarse.** Queda calibrado en el
    espacio de la fase 1 mientras `enc_t` se mueve. Para `K_s` es exacto para
    siempre: el encoder de la fuente no se mueve. Para `K_t` y `K_st` no, y la
    direccion importa --- una contraccion la castiga el termino (el colapso
    mide 0.4110 sobre un maximo de 0.4231), pero una expansion grande lleva el
    exponencial a ~0, mata el gradiente y frena la adaptacion con un loss de
    aspecto sano.
    """
    return float(_HEURISTICO_DE_SIGMA._median_pairwise(torch.as_tensor(H)))

#: Si el puntaje por clase se mapea a [0,1] por la Ec. 36. ACA SI, desde que
#: r28 cambio las cotas; el divisor es el de r31 (ver el cierre del bloque).
#:
#: El mapa de r25 estuvo APAGADO aca, y las mediciones que lo apagaron siguen
#: siendo ciertas. Son la evidencia sobre la que r28 reemplazo la ecuacion, asi
#: que se conservan; lo que ya no se sostiene es la conclusion:
#:
#:   Ec. 35 (r25)   L = -ln(n_s+n_t)   U = (ln n_s + ln n_t)/2
#:   con n=43 por clase: L=-4.4543, U=3.7612, ancho 8.2155
#:   y `creda_c` vive, medido, en [0, 0.4231] --- el 5.1% de ese ancho
#:
#: Los dos extremos eran INALCANZABLES por construccion: l=0 exigiria
#: creda_c = -3.7612 (negativo, imposible) y l=1 exigiria +4.4543, diez veces el
#: maximo real. Todo caia entre 0.457 y 0.509. Y empeoraba con n: el ancho crece
#: como ln n mientras el rango real no. Con 1078 por clase ocupaba el 2.9%.
#:
#: La causa la decia el propio docstring de `conservative_bounds`: las cotas
#: salen de combinar los extremos de cada entropia POR SEPARADO, "conservative
#: and possibly not tight: the three matrices are not independent and their
#: extremes need not be reachable at once".
#:
#: La Ec. 35 de r28 toma los extremos de la PROPIA cota del puntaje (Ec. 34),
#: evaluada sobre las mismas matrices normalizadas con las que se construye
#: J_c: L = -ln 2 y U = 2 ln cosh((H2(K~_t) - H2(K_s))/4). Los dos son
#: alcanzables, y eso es la diferencia entera. Medido aca con n=43 por clase,
#: r25 contra r28 sobre las mismas cinco configuraciones:
#:
#:   configuracion                  J_c      r25      r28
#:   identicas (alineacion)     -0.0000   0.4578   0.0000
#:   ruido puro                 -0.0429   0.4630   0.2816
#:   clases permutadas          -0.4362   0.5109   0.6293
#:   blanco colapsado           -0.6922   0.5421   0.9986
#:   separadas 40 sigma         -0.6931   0.5422   0.99998
#:   rango                                0.0844   0.99998
#:
#: El tope no se alcanza exactamente y no puede: el kernel gaussiano es
#: estrictamente positivo, asi que el bloque cruzado nunca se anula y L_c se
#: aproxima como limite. r28 lo dice en la seccion 5.
#:
#: Y el cambio no es solo de escala: r25 es una afin de extremos CONSTANTES, asi
#: que conserva el orden de lo que mapea, y el puntaje crudo no alcanza su
#: maximo en la alineacion cuando las entropias marginales difieren. El
#: contraejemplo esta en r28 seccion 5 y se mide igual aca: con K_s = I/n_s y el
#: blanco colapsado lejos, J_c = +0.5173 contra el 0 de la alineacion perfecta,
#: y r25 lee 0.3949 para el colapso contra 0.4578 para la alineacion --- o sea
#: PREFIERE el colapso. r28 lee 0.1773 contra 0.0000.
#:
#: **2026-10-17: la columna `r28` de arriba es historia fechada y NO es lo que
#: esta bandera computa hoy.** Las dos columnas estan rotuladas por la revision
#: bajo la que se midieron, asi que ninguna es falsa; lo que faltaba es que
#: r31 reemplazo el divisor de la Ec. 36. r28 y r29 dividian por el span por
#: clase `U_c - L_c`; r31 divide por el literal `ln 2`, sobre el teorema de r30
#: `0 <= U_c - J_c <= ln 2`. La relacion entre las dos lecturas es exacta y
#: sale de las dos ecuaciones, no de una medicion:
#:
#:   r31 = r28 * (U_c + ln 2) / ln 2
#:
#: Para el colapso del ultimo parrafo, medido: `U_c = 0.7782`, factor 2.1227, y
#: el 0.1773 de r28 son 0.3764 bajo r31 --- el 0.0000 de la alineacion no se
#: mueve, porque ahi las dos marginales son la misma matriz, `U_c = 0`, y los
#: dos divisores coinciden. El ORDEN, que es lo que el parrafo argumenta, no
#: cambia: el colapso sigue leyendo peor que la alineacion.
#:
#: Las otras cuatro filas de la tabla NO se recalcularon y no se pueden
#: derivar de lo escrito: el factor necesita el `U_c` de cada fila y la tabla
#: no lo guardo. Quedan como lo que son, cinco lecturas de r28.
NORMALIZAR_POR_COTAS = True

#: Cual de las dos Ecs. 35 usa `NORMALIZAR_POR_COTAS`, y cual masa iguala el
#: balance de la Ec. 32. Declarados aca y no clavados en la construccion de la
#: perdida, porque son las dos elecciones que deciden en que escala viven todos
#: los numeros de la fase 2 y un lector tiene que poder verlas sin leer el
#: bucle del cuaderno.
FORMA_DE_COTAS = 'entropy-cosh'
BALANCE_DE_MASA = 'trace'

# Hasta donde la caja se mantiene cuadrada. 3 a 1 deja a toroide (1.9) cuadrada
# y suelta a lunas (44.5 y 14.0), que es justo el reparto que hace falta.
ASPECTO_MAX = 3.0

# Las familias de entrada son las DURAS de la primera celda. Las anteriores
# tomaban 3 de 10 etiquetas de un generador cuyas etiquetas no siguen su
# geometria, y llegaban al codificador con 1,04 a 1,63 direcciones efectivas: el
# kernel no puede separar tres clases sobre una recta, por bien sintonizado que
# este. Acá la clase es un pedazo de la figura.
FAMILIAS = [
    dict(nombre='lunas',   gen=dura_lunas,   amb=3,
         nota='dos parábolas opuestas · dos clases sobre la misma lámina'),
    dict(nombre='toroide', gen=dura_toroide, amb=3,
         nota='un toro completo en tres arcos de 120°'),
    dict(nombre='rollo',   gen=dura_rollo,   amb=3,
         nota='un rollo suizo estirado, cortado en tres y vuelto a enrollar'),
]

#: Como se parte cada dominio: train, valid, test. El resto es test.
FRAC_TRAIN, FRAC_VALID = 0.6, 0.2


def partir(y, fracs=None, semilla=None):
    """TRES splits estratificados por clase: train, valid, test.

    Eran DOS, y el de evaluacion hacia doble funcion: `train_source_ce` elegia
    la epoca con el, y la tabla reportaba el acierto sobre el MISMO split. El
    numero reportado estaba optimistamente sesgado por construccion, porque la
    epoca se habia elegido para maximizar exactamente esa cantidad.

    Ahora: train ajusta, valid elige (parada temprana y busqueda de
    hiperparametros), y test se lee UNA vez, al final, solo para las tablas.
    """
    if semilla is None: semilla = SEMILLA        # en la llamada, no al definir
    f_tr, f_va = (FRAC_TRAIN, FRAC_VALID) if fracs is None else fracs
    assert 0 < f_tr and 0 < f_va and f_tr + f_va < 1, (f_tr, f_va)
    rng = np.random.default_rng(semilla); tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]; rng.shuffle(idx)
        a = int(f_tr*len(idx)); b = a + int(f_va*len(idx))
        tr.append(idx[:a]); va.append(idx[a:b]); te.append(idx[b:])
    t, v, e = (np.concatenate(tr), np.concatenate(va), np.concatenate(te))
    # Que de verdad sean disjuntos y cubran todo: un solapamiento aca es fuga
    # silenciosa de test a la busqueda, y no da ningun error.
    assert len(set(t) | set(v) | set(e)) == len(t) + len(v) + len(e) == len(y), (
        len(t), len(v), len(e), len(y), 'los splits se solapan o pierden puntos')
    return t, v, e

NORMAS = ('ninguna', 'batchnorm', 'layernorm')

class MLP(nn.Module):
    # El eje del paso 4 del plan: ninguna, BatchNorm, LayerNorm. Las tres en el
    # MISMO lugar -- despues de la primera oculta -- porque si cada una fuera a
    # una capa distinta la comparacion mediria el lugar y no el tipo. Ese lugar
    # salio de medir el BatchNorm sobre 5 semillas: solo en la primera lleva
    # toroides de 0.3651 a 0.7222 y lunas de 0.3306 a 0.6534, por encima de
    # ponerlo en las dos ocultas, y en la salida es el PEOR sitio (0.4590 en
    # rollo, debajo de no poner ninguno).
    #
    # Y las dos normalizan cosas distintas, que es el punto de probar las dos:
    # BatchNorm resta la media por CANAL sobre el lote, asi que lo que le hace a
    # una muestra depende de con quien le toco viajar -- y en fase 2 reestima sus
    # corridas con lotes del dominio nuevo. LayerNorm normaliza por FEATURES
    # dentro de cada muestra: no tiene estado, no mira el lote, y por eso no
    # puede traer la ventaja que BatchNorm trae por reestimar.
    def __init__(self, entrada, pos, dim=None):
        super().__init__()
        assert pos in NORMAS, pos
        DIM = globals()['DIM'] if dim is None else dim
        if UNA_CAPA:
            # UNA capa, literal: un solo Linear de la entrada al latente, la
            # ReLU, y la normalizacion AL FINAL --- sobre el embedding mismo y
            # no sobre una oculta intermedia, porque no hay ninguna.
            #
            #     entrada -> Linear -> ReLU -> [ BatchNorm | LayerNorm | nada ]
            #
            # Es otro lugar para la norma que en la red completa, y eso es
            # deliberado: con una sola capa el unico sitio posible es ese.
            capas = [nn.Linear(entrada, DIM), nn.ReLU()]
            if pos == 'batchnorm': capas.append(nn.BatchNorm1d(DIM))
            if pos == 'layernorm': capas.append(nn.LayerNorm(DIM))
        else:
            capas = [nn.Linear(entrada, 32)]
            if pos == 'batchnorm': capas.append(nn.BatchNorm1d(32))
            if pos == 'layernorm': capas.append(nn.LayerNorm(32))
            capas += [nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, DIM)]
        self.net = nn.Sequential(*capas)
    def forward(self, x): return self.net(x)


def datos_semilla(fam, semilla, grados=None):
    # La semilla mueve los DATOS y no solo la inicializacion de pesos: con un
    # unico sorteo, la dispersion entre corridas seria solo la de los pesos.
    #
    # `grados` expone el angulo de rotacion que `perturbar` ya aceptaba y que
    # nadie podia pedir desde aca. Es el EJE DE CALIBRACION: con el angulo
    # chico los errores de pseudoetiqueta caen cerca de las fronteras --- el
    # posterior queda repartido y `w` los puede ver --- y con el angulo grande
    # grupos enteros se asignan con confianza a la clase equivocada, que es el
    # regimen donde `w` pesa igual a error y a acierto.
    #
    # `None` es el valor declarado `ANGULO_SIGMA` y la corrida sale BIT A BIT
    # como la de hoy; un test lo fija contra esta misma llamada.
    rng = np.random.default_rng(semilla)
    Xs, ys = fam['gen'](N_POR_DOMINIO, rng)
    Xt, yt = fam['gen'](N_POR_DOMINIO, np.random.default_rng(semilla + 1000))
    Xt = perturbar(Xs, Xt, rng, grados=grados)
    Xs, Xt = Xs.astype(np.float32), Xt.astype(np.float32)
    return _empacar(Xs, ys, Xt, yt, semilla)


def _empacar(Xs, ys, Xt, yt, semilla):
    """Los seis splits, con nombres que dicen cual es cual.

    `_tr` entrena, `_va` elige, `_te` SOLO reporta. Los nombres viejos --- `Xs`
    para train y `Xse` para 'evaluacion' --- no distinguian valid de test porque
    no habia valid, y asi es como el mismo split acabo eligiendo y reportando.
    """
    T = torch.tensor
    i_s, v_s, e_s = partir(ys, semilla=semilla)
    i_t, v_t, e_t = partir(yt, semilla=semilla)
    return dict(
        Xs_tr=T(Xs[i_s]), ys_tr=T(ys[i_s]),
        Xs_va=T(Xs[v_s]), ys_va=T(ys[v_s]),
        Xs_te=T(Xs[e_s]), ys_te=T(ys[e_s]),
        Xt_tr=T(Xt[i_t]), yt_tr=T(yt[i_t]),
        Xt_va=T(Xt[v_t]), yt_va=T(yt[v_t]),
        Xt_te=T(Xt[e_t]), yt_te=T(yt[e_t]))


def incertidumbre_declarada(logits):
    """`U(g) = H2(g)/ln C` de la Ec. 24: el complemento de la confianza.

    **No calcula: delega en `MIL_CREDA.confidence.confidences`, que es la
    UNICA implementacion de la Ec. 24.** Hasta el 2026-10-18 esta funcion
    calculaba `-log(sum g^2 + 1e-6) / ln C` por su cuenta, y las dos copias de
    una sola ecuacion llegaron a discrepar sin que nada fallara.

    La lee `pesos_declarados`, que es la que `caminata_al_centro` dibuja.
    """
    # `confidences` normaliza por `ln(G.shape[1])`, y este modulo por `CLASES`.
    # Son el mismo numero mientras el cabezal tenga una salida por clase, que
    # es lo que esto afirma: si alguna vez dejaran de serlo, el normalizador
    # cambiaria sin que ninguna lectura se viera distinta.
    assert logits.shape[1] == CLASES, (logits.shape[1], CLASES)
    return 1.0 - confidences(F.softmax(logits, 1))


def clases_efectivas(rut):
    """`cef`: el numero EFECTIVO de clases en el RUTEO del blanco.

    `exp(H(p))`, con `p` el histograma normalizado del `argmax` sobre el blanco
    y `H` natural. Vale `1` cuando todo el blanco cae en UNA clase --- el
    colapso --- y `CLASES` cuando se reparte parejo, asi que vive en
    `[1, CLASES]`.

    El histograma se toma sobre `CLASES` casilleros FIJOS y no sobre las clases
    presentes. Contar solo las presentes haria que un colapso a una clase y un
    reparto perfecto entre las dos que quedaron dieran numeros que no se pueden
    comparar entre pasos, que es justamente la comparacion que el criterio
    hace.

    NO USA `y_t`, y eso es lo que la habilita como monitor: es el histograma
    de las predicciones del PROPIO modelo. El cuaderno la imprime por epoca
    sin leer ninguna etiqueta del blanco.
    """
    r = (rut.detach().cpu().numpy() if isinstance(rut, torch.Tensor)
         else np.asarray(rut))
    r = r.reshape(-1).astype(np.int64)
    assert r.size > 0, 'un ruteo vacio no tiene numero efectivo de clases'
    cuenta = np.bincount(r, minlength=CLASES)
    assert len(cuenta) == CLASES, (
        len(cuenta), CLASES, 'el ruteo nombra una clase fuera de rango')
    p = cuenta / float(r.size)
    nz = p[p > 0.0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def puntos_de_frontera(cab, clases, latente=None):
    """Un punto del LATENTE donde las `clases` nombradas empatan y ganan.

    Las fronteras son del CABEZAL, que es un `Linear`, asi que son exactas y
    se resuelven en vez de buscarse: el empate `w_i.z + b_i = w_j.z + b_j` es
    un sistema lineal, y con `k` clases empatadas son `k-1` ecuaciones.

    Dos condiciones, y la segunda es la que hace que sea una frontera:

      EMPATE   las `k` nombradas con el mismo logit --- resuelto, no estimado.
      MAYORES  ese logit es el maximo. Un empate en el fondo, donde una clase
               que nadie nombro gana, no es una frontera: es un punto
               cualquiera, y `w` ahi no vale lo que la figura promete.

    Con `k-1` ecuaciones en `d` incognitas el sistema esta subdeterminado para
    `d > k-1`, asi que se toma la solucion de NORMA MINIMA --- `lstsq` --- y
    despues se empuja a lo largo del nucleo hasta que el empate sea el maximo.
    Devuelve el punto como `ndarray` de dimension `d`.
    """
    clases = tuple(int(c) for c in clases)
    assert len(clases) >= 2, clases
    W = cab.weight.detach().numpy().astype(float)
    b = cab.bias.detach().numpy().astype(float)
    d = W.shape[1] if latente is None else int(latente)
    # `k-1` ecuaciones de empate contra la primera nombrada.
    A = np.stack([W[clases[0]] - W[c] for c in clases[1:]])
    y = np.array([b[c] - b[clases[0]] for c in clases[1:]])
    z, *_ = np.linalg.lstsq(A, y, rcond=None)
    otras = [c for c in range(W.shape[0]) if c not in clases]
    if not otras:
        # Frontera de TODAS las clases: la solucion de norma minima ya es el
        # punto, no hay nadie a quien superar.
        return z.astype(float)
    # Y el empate tiene que GANAR ESTRICTAMENTE a las no nombradas. Sin el
    # `estrictamente`, con sesgo cero la frontera DOBLE devuelve el origen ---
    # donde las tres empatan --- que es el punto TRIPLE con otro nombre, y las
    # dos curvas de la figura arrancarian del mismo lugar.
    _, _, Vt = np.linalg.svd(A)
    nucleo = Vt[len(clases)-1:]
    assert len(nucleo), (
        clases, 'el empate no deja ninguna direccion libre: con este cabezal '
        'esa frontera es un punto solo y no se le puede exigir que gane')
    # La direccion del nucleo que mas separa al empatado de las otras.
    sep = nucleo @ (W[clases[0]] - W[otras].mean(0))
    j = int(np.argmax(np.abs(sep)))
    dir_ = nucleo[j] * float(np.sign(sep[j]))
    # El criterio NO es "gana por un pelo" sino "las demas no pesan". Con un
    # margen minimo el posterior sale (0.373, 0.373, 0.254) --- la tercera
    # clase sigue adentro --- y `w` da 0.026 en vez del 0.369 que una frontera
    # de DOS clases tiene por definicion. Se empuja hasta que la masa de las
    # no nombradas sea despreciable, que es lo que hace que el punto sea la
    # frontera de ESAS dos y no un punto cualquiera entre tres.
    for t in np.concatenate([np.linspace(0.25, 8.0, 32), np.array([16., 32., 64., 128.])]):
        cand = z + t * dir_
        lg = W @ cand + b
        e = np.exp(lg - lg.max()); g = e / e.sum()
        if g[otras].sum() < 1e-4:
            return cand.astype(float)
    raise ValueError(
        f'no se encontro un punto donde {clases} empaten y las demas no '
        'pesen; con este cabezal esa frontera no es separable de las otras')


def caminata_al_centro(cab, inicio, centro, pasos=60):
    """`w` a lo largo del segmento del punto de frontera al centro de clase.

    Es la figura entera en una linea: el punto arranca donde el modelo NO sabe
    --- `w` casi cero en la frontera triple, 0.369 en una doble --- y camina
    al centro, donde el posterior satura y `w` llega a 1.

    Lo que demuestra, y por eso no necesita semillas ni test estadistico: `w`
    es una funcion del posterior y nada mas, asi que esta curva es una
    propiedad de la Ec. 24 y del cabezal, EXACTA. Y la premisa del operador se
    ve literal: el punto puede no ser de esa clase --- nadie le pregunto su
    etiqueta --- y aun asi `w` sube al acercarse, porque lo que mide es
    confianza y no acierto.

    Devuelve `(t, w)` con `t` en [0, 1] a lo largo del segmento.
    """
    inicio = np.asarray(inicio, dtype=float).ravel()
    centro = np.asarray(centro, dtype=float).ravel()
    assert inicio.shape == centro.shape, (inicio.shape, centro.shape)
    t = np.linspace(0.0, 1.0, int(pasos))
    Z = inicio[None, :] + t[:, None] * (centro - inicio)[None, :]
    with torch.no_grad():
        lg = cab(torch.as_tensor(Z, dtype=torch.float32))
    return t, pesos_declarados(lg).astype(float)


def pesos_declarados(logits):
    """`w_j = 1 - H2(g_j)/ln C`: el peso de la formulacion (Ecs. 24 y 26).

    Mide INCERTIDUMBRE, no error. Un punto que cae bien adentro de la clase
    equivocada esta confiadamente equivocado y tiene `w` ALTO; solo los que
    caen entre clases se achican.
    """
    return (1.0 - incertidumbre_declarada(logits)).numpy().astype(np.float32)


def perdida_fuente(logits, y):
    """El termino supervisado de la Ec. 21, por la copia que tiene CREDA.

    **La copia de `CREDA.bounds`, nunca el original de `MIL_CREDA`, y la razon
    es la linea del import.** CREDA es la LINEA BASE contra la que se mide
    MIL-CREDA, asi que un modulo de CREDA que importara de `MIL_CREDA` pondria
    la dependencia al reves --- el control dependiendo de lo que controla --- y
    la comparacion quedaria contaminada en el import, antes de producir un solo
    numero. `src/CREDA/bounds.py` lo explica en su propio docstring y
    `tests/test_creda_sample.py` sostiene, funcion por funcion, que la copia y
    el original dan lo mismo. Misma funcion; otra procedencia.

    **La Ec. 24 es la excepcion, y la regla dice por que.** El peso de
    confianza SI se importa de `MIL_CREDA.confidence`, porque lo que el
    parrafo de arriba protege es una LINEA BASE y la Ec. 24 no es parte de
    ninguna: es el TRATAMIENTO, y un tratamiento no tiene independencia que
    perder respecto del metodo que lo propone. La Ec. 21 de aqui abajo si es
    aritmetica de la linea base, y se queda copiada.

    **Que cambia respecto de la cross-entropy cruda.** La Ec. 21 divide por
    `B_src`, el supremo exacto de la perdida de una unidad, de modo que el
    termino vive en [0, 1]: cero para una prediccion correcta y certera. La CE
    cruda arrancaba en `ln C = 1.0986` y no tenia techo. `B_src` es una
    constante fijada por `eps`, asi que dividir por ella no mueve el
    minimizador.

    `source_loss` toma PROBABILIDADES y etiquetas en one-hot --- una fila por
    unidad etiquetada, que aca es una muestra --- asi que el softmax y el
    one-hot se construyen aca y no se le pasan logits.
    """
    assert logits.dim() == 2, logits.shape
    assert logits.shape[1] == CLASES, (logits.shape, CLASES)
    G = F.softmax(logits, 1)
    Y = F.one_hot(y.reshape(-1), CLASES).to(G.dtype)
    return source_loss(G, Y)
