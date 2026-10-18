"""El experimento del toyset: las tres familias duras, la red de UNA capa, las
dos fases y el barrido de tablas. Todo lo que CALCULA vive aca; nada dibuja.

Este modulo es nuevo y no toca nada de lo que ya existia: `training_pipeline`
sigue siendo el de los experimentos del articulo, con sus defaults intactos, y
`Results_Generator` / `Plots_Generator` no importan nada de aca.

La fase 1 entrena la fuente con cross-entropy y NADA mas. Llevo tres terminos
de latente a lo largo del estudio --- la divergencia de Hellinger global contra
N(0,I), el par de entropias de Renyi sobre las clases, y la KL --- y los tres
se retiraron: medido sobre 18 celdas, cambiar el termino abre 0.1322 de acierto
en el blanco y cambiar la normalizacion abre 0.2438. Lo que decide es la
arquitectura.

**This module implements no equation, and the provenance below says so.** It is
the toy's experiment harness: it builds the three families, splits them, trains
the two phases, sweeps the table and writes the artifact. Every term it
computes with comes from somewhere else -- `CREDALoss` for the adaptation
term, which is the ONLY term with a gradient here -- so there is no
equation of the revision written here, and stamping a revision on a harness
would empty the check for the modules that carry one.

What it does declare is a CONFIGURATION: which bounds map, which balanced mass,
and whether the per-class score is normalized at all. Those three are what the
invariants below are about, together with `verificar_coherencia`, which refuses
an incoherent combination of them at import rather than twelve minutes into a
run. Every one of those ids is a test that already existed in
`tests/test_creda_r28_bounds.py` when this declaration was written.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CREDA.bounds import source_bound, source_loss
from CREDA.models import CREDALoss
from CREDA.two_phase import clone_for_target


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        "the toy's experiment harness, not a term; see the module docstring"),
    "invariants": [
        "the_toy_now_normalizes_by_bounds",
        "the_toy_declares_the_two_new_values",
        "the_loss_is_built_from_those_declarations",
        # Renombrado 2026-10-10: el reclamo se fortalecio. Antes fijaba que
        # la busqueda no podia elegir un alcance POR CLASE; ahora, que no
        # elige ancho de banda en absoluto --- lo calcula
        # `sigma_de_la_fuente` una vez sobre el train de la fuente y queda
        # congelado. Lo que lo saco fue una medicion y no r28: las tres
        # familias habian ELEGIDO `batch-pooled`, el alcance cuya mediana
        # lee el blanco y por eso baja cuando el blanco se desalinea.
        "the_phase_two_search_cannot_pick_a_bandwidth_at_all",
        "the_run_reports_the_one_bandwidth_it_used",
        "the_bandwidth_is_the_median_over_the_whole_source_train",
        "moving_the_target_train_cannot_move_the_bandwidth",
        "the_batch_size_cannot_move_the_bandwidth",
        "the_scope_key_is_gone_from_the_defaults_and_the_search",
        "a_stale_scope_key_is_refused_rather_than_ignored",
        "the_run_reports_the_scale_drift_it_paid_for",
        "every_arm_records_the_routing_indicator_not_only_the_ceiling",
        "the_routing_indicator_never_reaches_a_decision",
        "the_live_declaration_is_coherent",
        "a_row_asking_for_no_balance_is_refused",
        "the_cardinality_mass_is_refused",
        "the_table_is_checked_and_not_only_the_frames",
        # LA PASADA DE LA FASE 2 y su criterio. La fase 2 sorteaba
        # `randperm(n)[:LOTE_2]` de cero en cada paso --- muestreo con
        # reemplazo entre pasos, sin pasada y por lo tanto sin epoca --- y la
        # parada temprana decidia POR PASO, sobre un modelo que habia visto un
        # lote arbitrario. Las cuatro viven en
        # `tests/test_creda_fase2_por_epoca.py`.
        "the_phase_two_epoch_draws_every_training_point_exactly_once",
        "the_phase_two_criterion_is_read_at_the_end_of_an_epoch",
        "the_phase_two_patience_is_declared_in_epochs_and_not_derived_from_the_lot",
        "the_target_defines_the_epoch_and_the_source_must_agree",
        # El termino supervisado y el arnes de contaminacion. Cada uno tiene
        # una forma de salir mal que deja todos los numeros con buena cara, y
        # las siete viven en `tests/test_toy_contamination.py`.
        "the_supervised_term_is_equation_twenty_ones_normalized_form",
        "the_supervised_term_matches_a_hand_written_equation_twenty_one",
        "the_supervised_term_lives_in_the_unit_interval",
        "the_toy_never_imports_the_method_it_is_the_baseline_for",
        "contamination_reaches_the_target_train_split_only",
        "a_zero_contamination_fraction_is_a_no_op",
        "the_contaminated_indices_are_recorded_with_their_declared_count",
        "the_same_seed_and_fraction_contaminate_identically",
        "both_weighting_arms_of_a_scenario_share_one_phase_two_search",
        # La reconstruccion de la seccion: la perturbacion de FRONTERA, que es
        # la que el peso puede ver, y el techo de ORACULO, que no es un metodo.
        # Las trece viven en `tests/test_toy_frontera.py`, y la primera es la
        # premisa de la que depende todo lo demas: si el peso de los puntos
        # movidos no cae por debajo del de los limpios, el experimento es nulo.
        "the_boundary_perturbation_drops_the_weight_of_the_points_it_moved",
        "the_adversarial_perturbation_does_not_drop_the_weight",
        "the_two_perturbations_are_separated_by_the_declared_margin",
        "both_perturbations_leave_the_target_test_split_byte_identical",
        "both_perturbations_are_deterministic_from_the_seed",
        "the_perturbed_index_count_is_the_declared_rounding",
        "a_zero_fraction_is_a_no_op_for_both_modes",
        "the_boundary_mode_refuses_to_run_without_the_head",
        "the_oracle_weight_is_exactly_the_correctness_indicator",
        "no_hyperparameter_search_can_select_the_oracle_arm",
        "the_oracle_arm_is_labelled_a_ceiling_in_the_artifact_it_writes",
        "all_three_arms_of_a_scenario_receive_the_same_hp2",
        "the_sweep_records_the_premise_it_rests_on",
        "the_oracle_arm_weights_what_it_says_it_weights",
        # La TERCERA construccion: intrusos lejos en el latente. Las dos
        # anteriores volvieron nulas y los dos nulos estan explicados por
        # medicion --- ver `MODO_INTRUSOS` --- de modo que esta no es una
        # variante mas sino el unico regimen en el que hay algo para que un
        # sopesado quite. Las siete viven en `tests/test_toy_intrusos.py`, y la
        # primera es la licencia: sin distancia latente no hay dano y nada de
        # lo que se mida despues significa algo.
        "the_intruders_land_materially_farther_from_their_class_centroid",
        "the_intrusion_leaves_the_target_test_split_byte_identical",
        "a_zero_intrusion_is_a_no_op_and_so_is_its_oracle_arm",
        "the_intruder_count_per_class_is_the_declared_rounding",
        "the_intruder_oracle_weight_is_exactly_the_intruder_indicator",
        "all_three_intrusion_arms_of_a_scenario_receive_the_same_hp2",
        "no_search_can_select_the_intruder_ceiling_and_it_says_it_is_one",
        # EL CRITERIO DE LA FASE 2, desde el 2026-10-16: el MAXIMO de `cef`.
        # Sus tests estan en `tests/test_creda_criterio_cef.py`, y el que
        # importa es el de la DIRECCION: un `<` por un `>` no rompe nada que
        # se vea.
        "the_phase_two_criterion_is_the_maximum_of_cef",
        "the_phase_two_patience_counts_epochs_without_a_new_maximum",
        "the_effective_classes_stay_within_one_and_the_class_count",
        "the_watchman_is_recorded_and_decides_nothing",
        "the_checkpoint_cef_is_derived_from_the_recorded_routing",
        "the_criterion_signal_and_its_direction_are_readable_from_the_flags",
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
NOTA_DURA = {
    'lunas':   'dos parábolas opuestas, extruidas · dos clases sobre la misma lámina',
    'toroide': 'un toro completo cortado en tres arcos de 120° del ángulo mayor',
    'rollo':   'un rollo suizo completo: estirado, cortado en tres y vuelto a enrollar'}
COL_DURA = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c'}

#: Los puntos por dominio que el toyset sortea. Era un literal `1800` repetido
#: en CINCO sitios --- el default de aca abajo, los dos sorteos de `preparar`,
#: los dos de `datos_semilla` y la llave que `banderas` publica --- y pasa a ser
#: UNA constante porque la conversion de pasos a epocas se deriva de ella
#: (`BLOQUES_AL_LOTE_DECLARADO`): una derivacion que leyera uno de los cinco
#: mientras otro se movia no se pondria roja. El VALOR no cambia.
N_POR_DOMINIO = 1800

def construir_dura(fn, n=N_POR_DOMINIO, semilla=0):
    rng = np.random.default_rng(semilla)
    Xs, ys = fn(n, rng)
    Xt, yt = fn(n, np.random.default_rng(semilla + 1000))
    return Xs, ys, perturbar(Xs, Xt, rng), yt

def dim_ef_dura(X):
    w = np.clip(np.linalg.eigvalsh(np.cov(X.T)), 1e-15, None)
    return float(w.sum()**2/(w**2).sum())

def entre_dura(X, y):
    mu = X.mean(0); e = d = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        e += len(Xc)*float(((Xc.mean(0)-mu)**2).sum())/len(X)
        d += float(((Xc-Xc.mean(0))**2).sum())/len(X)
    return 100*e/(e+d)

COLOR = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c'}      # color = clase
MARCA = {'fuente': 'x', 'blanco': 'o'}                   # forma = dominio
SEMILLA, CLASES, DIM, LOTE = 0, 3, 16, 128
# EPOCAS_1 es un TECHO, no un largo: la fase 1 para en su mejor epoca de
# validacion de la fuente, y 60 estaba midiendo una fuente sin converger --
# medido, eso movia el blanco hasta +0.26.
#
# `PASOS_2` es un TECHO EN PASOS, y eso es una decision y no una herencia: la
# rejilla de puntos de control (`CUADROS`, `CTRL`) y la de gradiente
# (`PASOS_GRAD`) son indices de PASO, y `tests/test_toy_run.py` fija
# `toy_run.FULL.steps_2 == PASOS_2`. Leerlo como epocas serian 1600+ pasos al
# lote declarado, que es otro experimento.
#
# Lo que esta linea decia antes y era FALSO: "la fase 2 si corre sus 200 pasos
# enteros, porque para ELLA no hay criterio legitimo". La fase 2 tiene parada
# temprana --- hoy `MIN_EPOCAS_2`/`PACIENCIA_2_EPOCAS`, antes `MIN_PASOS_2`/
# `PACIENCIA_2` --- y el artefacto publicado ya lo desmentia: sobre las 90
# corridas de las nueve celdas, `paso_fin` tiene piso 70 y llega a 200 en una
# minoria. Era una frase que sobrevivio a su mecanismo.
EPOCAS_1, LR_1, LR_2, PASOS_2, K = 200, 1e-2, 1e-3, 200, 400

# UNA sola capa oculta en vez de dos. El cuaderno de una capa existe para
# preguntar si el termino de fuente puede suplir la capacidad que la red pierde,
# y para eso la red tiene que perderla de verdad. La normalizacion sigue yendo
# DESPUES de la primera oculta, que es el mismo lugar que en la red completa:
# con cero ocultas no habria donde ponerla y el eje que se compara desapareceria.
UNA_CAPA = True
CUADROS = list(range(0, PASOS_2 + 1, 10))
# Los campos del registro por punto de control, en orden:
#   paso, proyeccion del blanco, acc VALID, acc full, prediccion, acc fuente,
#   U, acc TEST
#
# El de test se agrego AL FINAL a proposito, no insertado: asi ningun indice de
# los que ya existian se corre, que es el defecto que esta aridad declarada
# existe para atrapar y que ya aparecio tres veces.
# Declarado aca y comprobado donde se construye y donde se desempaca. Agregar un
# campo sin tocar al consumidor ya costo una corrida completa: el error salia
# dentro de panel_embedding, 200 lineas mas abajo y despues del barrido.
#: Los campos del punto de control, EN ORDEN y por nombre. Que existan como
#: nombres y no solo como aridad es el arreglo de un defecto que aparecio CINCO
#: veces: cada campo nuevo se agregaba al final para no correr los indices, y un
#: unpack posicional --- `paso, Pt, acc, ... = cu[k]` --- fallaba igual. Con
#: `punto()` abajo, agregar un campo no rompe a ningun lector.
CAMPOS_CU_NOMBRES = ('paso', 'proy', 'acc_va', 'pok', 'rut', 'acc_s', 'u_t',
                     'acc_te')
CAMPOS_CU = len(CAMPOS_CU_NOMBRES)


def punto(cu, k):
    """El punto de control `k` como diccionario, por nombre."""
    r = cu[k]
    assert len(r) == CAMPOS_CU, (len(r), CAMPOS_CU)
    return dict(zip(CAMPOS_CU_NOMBRES, r))
#: Las cuatro cantidades de la Ec. 33 por clase, EN ORDEN DE FILA. Las tres
#: primeras son las entropias que `CREDALoss.forward` appendea en ese orden ---
#: fuente, blanco SOPESADO, mezcla BALANCEADA --- y la cuarta es su
#: combinacion:
#:
#:     J_c = 0.5 * ( H2(K_s,c) + H2(K~_t,c) ) - H2(K~_mix,c)
#:
#: El orden existe como NOMBRES y no como aridad sola por la razon de
#: `CAMPOS_CU_NOMBRES`, y aqui muerde mas: las tres entropias son numeros del
#: mismo tipo y la misma escala, asi que dos filas intercambiadas no se ven en
#: ningun grafico y solo cambian la aritmetica de `J`.
CAMPOS_EC33_NOMBRES = ('h_s', 'h_t', 'h_mix', 'J')
CAMPOS_EC33 = len(CAMPOS_EC33_NOMBRES)


def ec33(serie, k):
    """El punto de control `k` de la serie de la Ec. 33, por nombre.

    Devuelve las cuatro filas como arreglos de una entrada por clase VIVA ---
    las clases que `CREDALoss.forward` no salteo en ese punto --- asi que su
    largo puede cambiar entre puntos de control y no es `CLASES`.
    """
    A = np.asarray(serie[k])
    assert A.ndim == 2 and A.shape[0] == CAMPOS_EC33, (A.shape, CAMPOS_EC33)
    return dict(zip(CAMPOS_EC33_NOMBRES, A))
# Y los de las dos historias, por la misma razon y a raiz del mismo defecto: al
# salir el termino de fuente, la fase 1 bajo de 5 campos a 3 y la fase 2 corrio
# los suyos --- lo que era el termino de fuente en r[2] hoy es el Renyi. Un
# panel que siga leyendo las posiciones viejas dibuja una columna por otra,
# sin error ninguno. Y la aridad volvio a 6 el 2026-10-11, al retirarse el
# warmup: la columna de su factor se fue y las dos de despues se corrieron.
#   fase 1: epoca, CE de train, CE de VALID, acierto de VALID de la fuente
#   fase 2: paso, CE fuente, CREDA, total, CREDA de VALID, vigia
CAMPOS_HS = 4
CAMPOS_HT = 6
#: Epocas/pasos sin mejorar antes de cortar. El criterio de parada es un
#: early-stopping DECLARADO por adelantado, no elegido comparando contra el
#: oraculo: eso ultimo seria seleccion sobre y_t un nivel mas arriba.
#:
#: `PACIENCIA_1` es de la fase 1 y es EN EPOCAS desde siempre: la fase 1
#: siempre recorrio una pasada.
#:
#: `PACIENCIA_2` y `MIN_PASOS_2` eran el criterio de la fase 2 EN PASOS y YA NO
#: LO SON --- lo es el par en epocas, declarado junto a `FRAC_TRAIN` mas abajo
#: porque se deriva de el. Conservan su valor y eso es deliberado, no un olvido:
#: `tests/test_toy_run.py` fija `toy_run.FULL.patience_2 == PACIENCIA_2` y
#: `min_steps_2 == MIN_PASOS_2`, y el criterio de `MIL_CREDA_Benchmark` todavia
#: lee PASOS (`toy_run.py`, `if step + 1 >= scale.min_steps_2`) y es de otro
#: tramo. Mover cualquiera de los dos valores desde aca restataria ese banco en
#: silencio. Siguen leidos por tres sitios: `banderas()`, esa escala, y la
#: derivacion de `MIN_EPOCAS_2`.
PACIENCIA_1, PACIENCIA_2, MIN_PASOS_2 = 30, 30, 40

#: Los hiperparametros que la busqueda de cada fase puede mover, con el valor de
#: hoy. `corrida` y `train_source_ce` los reciben; nada los lee de un global, o
#: un trial de Optuna no podria cambiarlos.
#:
#: `lambda_creda` NO esta aca a proposito: va clavado en 1.0 y el techo de la
#: rampa es el unico coeficiente. Los dos se MULTIPLICAN, asi que buscar los dos
#: es gastar dos dimensiones en el mismo numero.
HP_1 = dict(LR_1=1e-2, WD_1=0.0, DIM=16, LOTE_1=128)
#: `SIGMA_FROM` NO esta aca y su ausencia es la decision, no un olvido. El
#: ancho de banda dejo de ser un hiperparametro: `corrida` lo calcula UNA vez
#: tras la fase 1 sobre el train completo de la fuente y lo congela --- ver
#: `sigma_de_la_fuente`. Pasar la llave retirada en `hp2` se refuta, porque un
#: `barrido.json` viejo la trae y un llamador que la manda cree que decide algo.
#: `Q_COMPUERTA` es la tasa que `PESO_COMPUERTA` cierra, y solo ese brazo la
#: lee. Vive en `HP_2` y no en un global para que un trial pueda barrerla:
#: nada en este modulo lee un HP de un global, o Optuna no podria cambiarlo.
HP_2 = dict(LR_2=1e-3, WD_2=0.0, LOTE_2=128,
            Q_COMPUERTA=0.25,
            # La precision que `PESO_ORACULO_RUIDOSO` finge. Solo ese brazo
            # la lee, y es un techo: no se busca nunca.
            PRECISION_TECHO=1.0,
            # La inyeccion en el embedding: tasa y radio en unidades del
            # radio rms del lote. `RHO_EMB=0` la apaga, y a cero la
            # corrida tiene que ser la limpia bit a bit.
            RHO_EMB=0.0, RADIO_EMB=4.0,
            # La proyeccion al posterior uniforme: la tasa. A 0 la corrida
            # es la limpia bit a bit.
            # La tasa y el DESPLAZAMIENTO dentro del subespacio uniforme.
            # El segundo es el eje que separa 'el peso no sirve' de 'no
            # habia daño que reparar'.
            RHO_UNI=0.0, DESPL_UNI=0.0,
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
            RHO_T_SIN_GRAD=False,
            # Purezas como estadistico U. Ver `CREDALoss`.
            PUREZA_U=False,
            # El paso en que el SOPESADO empieza a actuar. Antes de el,
            # `w = 1` para todos y el brazo declarado ES el apagado.
            #
            # Ya no hay nada mas que entre de a poco: el warmup que
            # difería el TERMINO se retiro el 2026-10-11 por no estar en la
            # formulacion. Hoy el termino entra entero desde el paso 0 y
            # esta llave decide unicamente cuando entra el PESO.
            #
            # Por que: en el paso 0 `enc_t` es un CLON de `enc`, asi que el
            # ruteo del blanco es el clasificador de la FUENTE aplicado al
            # blanco --- exactamente el brazo `.predict`, medido en 0.3703
            # / 0.4157 / 0.3528 sobre tres clases. Y el ruteo del arranque
            # es medible peor: los bloques de clase tienen mediana 2 a 6
            # con el cabezal de fase 1 y 4 a 10 durante la corrida. Si `w`
            # sale de esa posterior, al principio apaga muestras por una
            # incertidumbre que es del MODELO y no de la muestra.
            #
            # Retardo y no rampa a proposito: una rampa mezcla CUANDO
            # empezar con CON QUE PENDIENTE, y la que se prueba es la
            # primera. Si el retardo no compra nada, no hay pendiente que
            # lo salve.
            PASO_W=0,
            # LA ENTREGA: `class_mixing = 0` (marginal puro) antes de este
            # paso, `1.0` (condicional pura) desde el. Nunca una mezcla: los
            # dos extremos de la familia convexa y nada en el medio.
            #
            # El marginal no recorre clases, asi que NO PUEDE mal rutear --- y
            # la fusion que medimos es justamente de la condicional al
            # arranque, cuando las pseudoetiquetas son basura (`tr_e` -4.6%,
            # p<0.001, con el barajado en cero exacto). Primero las nubes se
            # superponen con un termino que no puede equivocarse de clase, y
            # recien entonces entra la que las necesita.
            #
            # `0` es el default y es lo de hoy bit a bit: el modulo tenia
            # `cr.class_mixing = 1.0` clavado, asi que el marginal no se usaba
            # nunca. Un `PASO_COND` que la corrida no alcanza es marginal puro
            # de punta a punta, y eso es declarable a proposito.
            PASO_COND=0)

#: El unico trabajo de esta instancia es prestar su heuristico de ancho de
#: banda. Existe para que el sigma que `corrida` congela sea EL MISMO
#: estadistico que el kernel habria calculado, y no un segundo que se le
#: parece: `_median_pairwise` es la mediana de las distancias pareadas sobre UN
#: conjunto, y reimplementarla aca seria dos definiciones de la misma cosa que
#: nada obliga a coincidir. `tests/test_toy_sigma_fijo.py` las fija iguales.
#:
#: **No nombra ninguna bandera y no le hace falta: esto no calcula una
#: perdida.** Es la unica construccion de `CREDALoss` en este modulo que no
#: declara `normalize_by_bounds`, y la razon es que solo se le llama
#: `_median_pairwise`, un estadistico del ancho de banda que no mira cotas, ni
#: entropias, ni bloques de clase. Las otras dos construcciones del modulo si
#: la nombran, por `NORMALIZAR_POR_COTAS` directa o a traves de `cfg`.
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
    aspecto sano. Nada en la busqueda compensa la escala: `DELTA` es el
    parametro de la RAMPA de lambda y no un factor de sigma. De ahi
    `sigma_deriva` y `k_t_saturada` en el registro de cada corrida.
    """
    return float(_HEURISTICO_DE_SIGMA._median_pairwise(torch.as_tensor(H)))
#: El WARMUP y el COEFICIENTE se retiraron el 2026-10-11, por decision del
#: operador y verificado contra la revision: `research-concept-r29.md` tiene
#: CERO menciones de rampa, ramp, schedule o warmup. No estaban en la
#: formulacion.
#:
#: Que era cada uno, y por que ninguno se defiende:
#:
#:   el COEFICIENTE (`TECHO` x `lambda_creda`). Un lambda existe para
#:   balancear DOS terminos. Con `mover_fuente=False` la CE de la fuente llega
#:   DETACHADA, asi que el unico termino con gradiente es CREDA: un escalar
#:   sobre el unico termino es una tasa de aprendizaje con otro nombre,
#:   redundante con `LR_2`, con el que ademas se multiplica. Medido, entre
#:   `1e-3` y `10` --- cuatro ordenes --- el acierto no se mueve (0.69-0.85),
#:   porque Adam actualiza con `lr*m/(sqrt(v)+eps)` y escalar la perdida por
#:   `c` escala `m` y `v` por igual.
#:
#:   el WARMUP, `(1-e^-delta*p)/(1+e^-delta*p)`: 0.000 en el paso 0, 0.104 en
#:   el 20, 0.780 en el 200. No es un lambda, es un arranque diferido.
#:
#: Lo que su retiro cuesta, medido el 2026-10-09 sobre 10 semillas: -0.0102 de
#: media, CONCENTRADO en las tres celdas de BatchNorm (-0.081, -0.076, -0.033),
#: porque le compraba pasos a BatchNorm para reestimar sus corridas antes de
#: que CREDA empujara. Eso es una necesidad de AdaBN y no de CREDA. Donde no
#: hay buffers que reestimar --- `rollo` sin normalizacion --- retirarlo GANA
#: +0.148.

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
#: los numeros de la fase 2 y un lector tiene que poder verlas sin leer
#: `corrida`. `banderas()` las guarda con cada corrida por la misma razon.
FORMA_DE_COTAS = 'entropy-cosh'
BALANCE_DE_MASA = 'trace'

# La FUENTE se entrena con cross-entropy y nada mas. Hubo tres variantes --- las
# dos entropias de Renyi sobre sus clases, la divergencia de Hellinger contra
# N(0,I), y nada --- y se retiraron: medido sobre las tres familias, el termino
# de fuente abria 0.1322 contra los 0.2438 que abre elegir la normalizacion, y
# solo-CE empataba con el mejor de ellos. Lo que decide es la arquitectura.

VISTA = (22, -62)                                        # el angulo 3D, fijo
# Las filas del cuadro: BatchNorm al INICIO contra BatchNorm PROFUNDO, que es
# la comparacion que el barrido de posiciones dejo abierta.
# UNA sola declaracion de las configuraciones, de la que salen TANTO los GIF
# como las columnas de la tabla.
#
# Antes eran dos listas escritas aparte y divergieron: las figuras mostraban un
# brazo que la tabla no tenia, y el brazo que la tabla reportaba como mejor no
# tenia figura. Quien mirara el 0,94 de la tabla no podia ver de donde salia, y
# quien mirara el GIF veia un numero mucho peor para lo que parecia lo mismo.
# Mientras las dos listas se mantengan a mano, vuelven a separarse.
NOMBRE_FILA = {'ninguna': 'sin normalización', 'batchnorm': 'BatchNorm',
               'layernorm': 'LayerNorm'}
CONFIGS = [
    ('ninguna', 'sinnorm', 'acc target', 'sin normalizacion'),
    ('batchnorm', 'batch', 'acc target BN', 'BatchNorm'),
    ('layernorm', 'layer', 'acc target LN', 'LayerNorm'),
]
BRAZOS    = [q for q, _, _, _ in CONFIGS]
# El coeficiente y el warmup se retiraron el 2026-10-11: no estan en
# `research-concept-r29.md` --- cero menciones de rampa, ramp, schedule o
# warmup --- y sobre un unico termino con gradiente un lambda no balancea nada.
# Ver el bloque de `CAMPOS_HT` y el test `test_toy_sin_rampa.py`.
POS_FILAS = ['ninguna', 'batchnorm', 'layernorm']
# Hasta donde la caja se mantiene cuadrada. 3 a 1 deja a toroide (1.9) cuadrada
# y suelta a lunas (44.5 y 14.0), que es justo el reparto que hace falta.
ASPECTO_MAX = 3.0
# Por debajo de esta participacion el panel deja de dibujar un scatter 2-D y pasa
# a la vista 1-D. 1.25 es el umbral: con 1.005 a 1.99 medidos, deja en 2-D a las
# que tienen plano y manda a 1-D a las que son una recta.
ASPECTO_1D = 1.25
CLAVE     = {q: c for q, c, _, _ in CONFIGS}
COL_TABLA = {q: t for q, _, t, _ in CONFIGS}
ETQ_BRAZO = {q: e for q, _, _, e in CONFIGS}
# El regimen de la fuente es ahora el eje entre las dos figuras del dataset, asi
# que necesita su propia clave de archivo. Las dos salen de FILAS y no de una
# lista aparte: un nombre que no estuviera en FILAS seria un GIF que nadie pide.
CLAVE_REG = {True: 'movil', False: 'quieta'}

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

#: Los lotes de fase 2 que la busqueda puede elegir. Era un literal dentro de
#: `buscar_hp2` y sube a constante porque el criterio tiene que poder afirmar
#: algo sobre TODA la rejilla: la razon de declarar la paciencia en epocas es
#: justamente que este conjunto existe.
LOTES_2_BUSCADOS = (32, 64, 128, 256)

#: Los bloques que una pasada tiene al lote DECLARADO del modulo, sobre el
#: train declarado. Es el ancla de la conversion de pasos a epocas, y es una
#: constante del MODULO y no del run a proposito: derivarla del lote que cada
#: corrida uso devolveria el criterio a depender de la busqueda, que es
#: exactamente lo que la decision de declararlo en epocas retira.
BLOQUES_AL_LOTE_DECLARADO = -(-int(FRAC_TRAIN * N_POR_DOMINIO) // LOTE)

#: EL CRITERIO DE LA FASE 2, EN EPOCAS. Las dos mitades tienen procedencia
#: distinta y se dice cual es cual, porque redondear una hacia la otra seria
#: inventar un numero y presentarlo como derivado.
#:
#: `MIN_EPOCAS_2` SE DERIVA de lo que `MIN_PASOS_2 = 40` compraba: una guarda
#: antes de que el criterio pueda elegir. Cuarenta pasos al lote declarado son
#: `ceil(40 / 9) = 5` epocas. Medido, esa guarda MUERDE: en el artefacto
#: publicado `mejor_paso == MIN_PASOS_2` en 5 de las 9 celdas, o sea que el
#: minimo del vigia cae a menudo en el primer punto legal.
MIN_EPOCAS_2 = max(1, -(-MIN_PASOS_2 // BLOQUES_AL_LOTE_DECLARADO))

#: `PACIENCIA_2_EPOCAS` esta DECLARADA y NO derivada, y es la decision del
#: operador: «que mire las 5 epocas y ya». El motivo es que el lote SE BUSCA
#: (`LOTES_2_BUSCADOS`): una paciencia derivada de `PACIENCIA_2` pasos daria 1,
#: 2, 4 o 6 epocas segun el lote, asi que la RESOLUCION del criterio quedaria
#: decidida por un hiperparametro que no sabe que la esta decidiendo.
#: Declararla la saca de esa dependencia del todo.
#:
#: Y la asimetria que esto deja es INTENCIONAL, no un defecto a tapar: el techo
#: esta en pasos y el criterio en epocas, asi que a `LOTE_2=32` hay 5 epocas
#: enteras en el techo y una paciencia de 5 las corre todas, mientras a
#: `LOTE_2=256` hay 40 y puede cortar temprano. Es la consecuencia honesta de
#: que las dos unidades se encuentren, y queda dicha en vez de maquillada.
PACIENCIA_2_EPOCAS = 5


def pasos_de_una_epoca(n_fuente, n_blanco, lote):
    """Cuantos pasos tiene UNA pasada de la fase 2, y quien la define.

    **El BLANCO la define.** La fase 2 existe para adaptar `enc_t`, y la fuente
    ya tuvo su pasada completa en la fase 1; el blanco no tuvo ninguna. La
    fuente tiene que COINCIDIR en bloques, y si no coincide esto se NIEGA en
    vez de envolver la permutacion corta: envolverla dejaria la propiedad que
    todo el cambio existe para tener --- cada punto sale exactamente una vez
    por epoca --- falsa sin que nada fallara.

    Los dos dominios NO tienen el mismo numero de puntos, y eso esta medido y
    no supuesto: `lunas` sortea 1080/1080, y `toroide` y `rollo` 1078/1079,
    porque el piso por clase del split estratificado de `partir` pierde un
    punto. Lo que SI coincide, a los cuatro lotes de `LOTES_2_BUSCADOS`, es el
    numero de BLOQUES --- `ceil(1078/128) = ceil(1079/128) = 9` --- y es eso lo
    que una pasada necesita.

    La negativa es ALCANZABLE y no un adorno: 1024 y 1025 a lote 128 dan 8
    bloques contra 9.
    """
    lote = int(lote)
    assert lote >= 1, lote
    b_s = -(-int(n_fuente) // lote)
    b_t = -(-int(n_blanco) // lote)
    assert b_s == b_t, (
        n_fuente, n_blanco, lote, b_s, b_t,
        'los dos dominios no coinciden en bloques, asi que una pasada no '
        'puede cubrir a los dos exactamente una vez; decidir cual define la '
        'epoca es una decision sobre el experimento y no se toma aca')
    return b_t


def pasada_en_lotes(n, lote, generator):
    """Los bloques de UNA pasada: una permutacion de `n` cortada en lotes.

    La MISMA forma que `train_source_ce` ya usa en la fase 1 --- `randperm` una
    vez y despues `range(0, n, lote)` --- y es todo el arreglo. La fase 2
    sorteaba `randperm(n)[:lote]` DE CERO en cada paso, que es muestreo con
    reemplazo ENTRE pasos: un punto podia salir dos veces seguidas y otro
    quedarse veinte pasos sin salir, de modo que no habia pasada y por lo tanto
    no habia epoca.

    El ultimo bloque es PARCIAL y se devuelve igual --- 56 puntos de 1080 a
    lote 128 --- porque dejarlo afuera rompe la propiedad: esos puntos no
    saldrian en la epoca. `train_source_ce` tambien lo conserva.
    """
    n, lote = int(n), int(lote)
    orden = torch.randperm(n, generator=generator)
    bloques = [orden[ini:ini + lote] for ini in range(0, n, lote)]
    assert sum(len(b) for b in bloques) == n, (n, lote)
    return bloques


#: Los campos de la bitacora de pasadas de la fase 2, EN ORDEN y por nombre.
#: La misma forma que `CAMPOS_CU_NOMBRES`, y por la misma razon: un campo
#: agregado al final rompe cualquier desempaque posicional sin dar error.
#: `cef` entro AL FINAL el 2026-10-16, no insertada: es la senal que el
#: criterio lee, y la bitacora por epoca es el unico sitio donde se puede
#: RELEER la eleccion sin volver a correr.
CAMPOS_P2_NOMBRES = ('epoca', 'pasos', 'distintos_s', 'distintos_t',
                     'suma_s', 'suma_t', 'cef')
CAMPOS_P2 = len(CAMPOS_P2_NOMBRES)


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

def preparar(fam):
    rng = np.random.default_rng(SEMILLA)
    Xs_np, ys_np = fam['gen'](N_POR_DOMINIO, rng)
    Xt_np, yt_np = fam['gen'](N_POR_DOMINIO, np.random.default_rng(SEMILLA + 1000))
    Xt_np = perturbar(Xs_np, Xt_np, rng)
    Xs_np = Xs_np.astype(np.float32); Xt_np = Xt_np.astype(np.float32)
    # La particion de evaluacion de la FUENTE se descartaba: con la fuente
    # congelada su acierto era un numero fijo y daba igual medirlo en train. En
    # el paso 3 la fuente se mueve, asi que su acierto es un resultado -- y
    # medirlo sobre lo que acaba de entrenar no distingue 'sigue sabiendo' de
    # 'se lo aprendio'.
    return dict(Xs_np=Xs_np, ys_np=ys_np, Xt_np=Xt_np, yt_np=yt_np,
                **_empacar(Xs_np, ys_np, Xt_np, yt_np, SEMILLA))


def datos_semilla(fam, semilla, grados=None):
    # preparar() fija SEMILLA; la tabla y la rejilla necesitan variar la semilla
    # de los DATOS tambien, o las cinco corridas comparten un unico sorteo y la
    # dispersion que reportan es solo la de la inicializacion de pesos.
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


#: La escalera DECLARADA de contaminacion, el cero incluido y primero. Fijada
#: en `odd/tasks/toyset-tests-estadisticos-y-ruido.md` antes de implementar, y
#: no se revisa despues de ver un resultado: mover la escalera al ver la curva
#: es elegir el nivel que da el numero que se queria.
#:
#: El cero no es un adorno. Es el nivel que dice que el arnes de perturbacion
#: no movio nada por su cuenta --- aunque esta vez NO reproduce el artefacto
#: que hay en disco, porque el termino supervisado cambio con la Ec. 21 y ese
#: artefacto quedo superado, no contradicho.
RHOS = (0.00, 0.10, 0.20, 0.30)

#: El NIVEL unico al que corren las dos perturbaciones, decidido por el
#: operador el 2026-10-10: el mas alto de los ya declarados. La escalera se
#: dejo de recorrer --- lo que la seccion compara hoy son los TRES pesos entre
#: si a contaminacion fija, no la curva contra `rho` --- y `RHOS` se conserva
#: porque es la escalera declarada y porque las pruebas leen sus cuatro niveles
#: para comprobar el conteo de indices.
RHO_UNICO = max(RHOS)

#: Una escalera de UN escalon, para el ENSAYO a escala chica.
#:
#: Fue el default de produccion entre el 2026-10-10 y el 2026-10-11, cuando lo
#: que la seccion comparaba eran los tres pesos entre si a contaminacion fija.
#: El 2026-10-11 el operador redisenio el cuaderno y pidio la curva contra
#: `rho` --- filas de nivel de ruido en la figura y en el gif, y una tabla
#: "ampliada a los diferentes niveles" --- asi que el default volvio a `RHOS` y
#: esto quedo para el ensayo, que es donde un solo escalon sigue sirviendo.
#:
#: Mientras fue el default, el cuaderno IMPRIMIA `RHOS` y corria un escalon: el
#: rotulo decia cuatro niveles sobre datos de uno. Es el mismo defecto que ya
#: aparecio con "5 semillas" sobre numeros de 10.
ESCALERA_UNICA = (RHO_UNICO,)

#: El corrimiento del flujo de azar de la contaminacion. Los datos usan
#: `semilla` y `semilla + 1000` (el blanco) y el lote fijo de valid usa
#: `semilla + 777`; este flujo tiene que ser DISJUNTO de los tres o la eleccion
#: de los puntos envenenados quedaria correlacionada con el sorteo que los
#: genero, y la contaminacion dejaria de ser independiente de los datos.
SEMILLA_CONTAMINACION = 50_000

#: Las DOS perturbaciones, por nombre. Cada una le hace al peso
#: `w_j = 1 - H2(g_j)/ln C` algo distinto, y esa diferencia es el experimento:
#:
#: `reubicado`  reubica el punto DENTRO de la region de otra clase. Queda
#:               confiadamente equivocado, asi que `w` NO baja --- medido a
#:               rho=0.30 sobre las tres familias y dos semillas, la media de
#:               los movidos contra la de los limpios dio -0.0297, +0.0753,
#:               +0.0087, +0.0150, +0.0502, +0.0213: cinco de seis ARRIBA.
#:               Pedirle al sopesado que los descuente es pedirle que ordene
#:               por error con una senal ciega al error, y por eso el primer
#:               diseno de esta seccion volvio nulo. Se conserva porque ese es
#:               un HALLAZGO reportable --- el regimen donde el peso se rompe
#:               --- y no codigo muerto.
#: `frontera`    pone el punto donde el cabezal de la fuente esta INDECISO, de
#:               modo que `g` se aplana de verdad y `w` cae. Ahi descontar es lo
#:               correcto y no descontar deberia costar acierto, que es una
#:               pregunta que el sopesado puede aprobar o reprobar por sus
#:               meritos. Medido en la misma rejilla: -0.4830, -0.4737, -0.5813,
#:               -0.6208, -0.5766, -0.6078.
MODO_REUBICADO = 'reubicado'
MODO_FRONTERA = 'frontera'
MODOS_CONTAMINACION = (MODO_REUBICADO, MODO_FRONTERA)

#: Las posiciones sobre el segmento que `frontera` prueba, declaradas: 40 pasos
#: iguales, el cero EXCLUIDO porque dejar el punto donde estaba no es moverlo.
#: Es una rejilla y no una biseccion a proposito: `w` sobre el segmento no es
#: unimodal --- el cabezal es una red, no un lineal --- y una biseccion que
#: asume monotonia encontraria un minimo local distinto segun por donde empiece.
#: Una rejilla fija se evalua en UN forward por punto y es la misma para todos.
ALFAS_FRONTERA = np.linspace(0.0, 1.0, 41)[1:]

#: La separacion DECLARADA entre las dos premisas, en unidades de `w`.
#:
#: No es una tolerancia elegida despues de ver los numeros: se fija entre las
#: dos poblaciones medidas arriba --- la peor de `frontera` da -0.4737 y la peor
#: de `reubicado` -0.0297 --- con mas de un margen de aire a cada lado, y el
#: test que lo lee comprueba exactamente esa separacion. Mover este numero para
#: que una medicion entre es elegir el umbral que da el resultado que se queria.
MARGEN_FRONTERA = 0.20

#: La TERCERA construccion, y la unica que puede dar un no-nulo. No esta en
#: `MODOS_CONTAMINACION` y eso es una decision, no un olvido: los dos modos de
#: ahi comparten el mecanismo de `contaminar` --- un conteo GLOBAL de indices,
#: `round(rho * n)`, sorteado sobre el train entero --- y esta construccion se
#: declara POR CLASE, de modo que su conteo es la suma de los redondeos por
#: clase y NO coincide con el global: medido en rollo a rho=0.10, por clase da
#: 36 + 38 + 33 = 107 y el global da 108. Meterla en esa tupla haria fallar
#: `tests/test_toy_frontera.py::test_the_perturbed_index_count_is_the_declared_rounding`
#: con razon, porque ese test afirma el redondeo global de las dos
#: perturbaciones que lo tienen. Vive en `intrusar`, con su propio conteo, su
#: propio techo y su propio barrido.
#:
#: **Por que esta existe, medido y no supuesto.** Lo que hace dano es la
#: distancia GEOMETRICA al bloque en el que el punto entra:
#:
#: - `reubicado` deja el punto a ~1 radio rms del centro de la clase a la que
#:   rutea, y en 4 de 6 celdas MAS CERCA de ese centro que los limpios
#:   (0.918/0.984/1.581/1.585/1.097/1.125 contra 1.009/1.040/1.669/1.659/
#:   1.165/1.145). Un punto indistinguible de un miembro nativo del bloque al
#:   que se suma no distorsiona la estructura de segundo orden de ese bloque,
#:   asi que no hay nada que ningun sopesado pueda quitar --- ni el estimador de
#:   entropia ni un oraculo perfecto. ESO, y no "`w` mide incertidumbre y no
#:   error", es la razon del techo nulo de esa construccion.
#: - empujar el punto por un rayo generico vuelve el posterior MAS confiado y no
#:   mas plano: `w` llega a 0.998 y la mediana sobre 512 direcciones es 1.0000
#:   para todo radio >= 4. Con `UNA_CAPA` el cabezal es
#:   `Linear -> ReLU -> BatchNorm -> Linear`, afin por region, asi que a lo
#:   largo de un rayo el patron de la ReLU se congela, los logits crecen lineal
#:   y la softmax satura.
#:
#: Un intruso lejos en el latente distorsiona la estructura de su bloque POR
#: CONSTRUCCION, de modo que hay algo para quitar. Si `w` lo ve es otra
#: pregunta, separada y medible, y es el HALLAZGO --- no el objetivo.
MODO_INTRUSOS = 'intrusos'
assert MODO_INTRUSOS not in MODOS_CONTAMINACION, (
    MODO_INTRUSOS, MODOS_CONTAMINACION,
    'el conteo de esta construccion es por clase y el de esas dos es global')

#: Los radios que la calibracion prueba, en unidades del radio rms del train
#: del blanco alrededor de su media. Es una REJILLA declarada y no una
#: biseccion, por la misma razon que `ALFAS_FRONTERA`: se evalua en UN forward
#: por radio y es la misma para todos.
#:
#: Geometrica y no lineal porque lo que se busca cruza tres ordenes: medido, el
#: radio rms del train del blanco vale 1.41 en lunas, 3.18 en toroide y 10.23 en
#: rollo, y la distancia latente que un mismo radio de entrada produce no es la
#: misma en las tres --- a 4 radios la razon contra los limpios dio 29.4 / 6.2 /
#: 5.4. Un radio FIJO mediria una intrusion distinta en cada familia y la tabla
#: las promediaria como si fueran la misma.
RADIOS_INTRUSO = tuple(float(r) for r in np.geomspace(0.25, 64.0, 25))

#: La LICENCIA del experimento, en razon de distancias latentes, en el latente
#: del codificador congelado de la fase 1. UN factor y UNA referencia --- la
#: distancia media de los limpios al centroide de SU clase --- contra DOS
#: lecturas, y las dos tienen que pasar:
#:
#:   `intruso`          la distancia al centroide de su propia clase. Es la que
#:                      dice que la estructura de segundo orden de ese bloque
#:                      quedo distorsionada, que es lo que un peso podria quitar.
#:   `cercano_intruso`  la distancia al centroide MAS CERCANO de los tres. Es la
#:                      que dice que el punto es un INTRUSO y no un miembro de
#:                      otra clase. `reubicado` pasaba la primera por
#:                      construccion y reprobaba esta: caia a ~1 radio del centro
#:                      ajeno, en 4 de 6 celdas mas cerca que los limpios.
#:
#: Las dos contra la MISMA referencia y el MISMO factor a proposito. Un segundo
#: umbral, contra un segundo denominador, seria un numero que nadie derivo --- y
#: la primera version de este archivo lo tenia, con `cercano_limpio` de
#: referencia, hasta que una medicion lo reprobo: eso se arregla haciendo que la
#: calibracion cumpla la licencia que el test ya pedia, nunca bajando el umbral
#: para que la medicion entre.
#:
#: Declarado ANTES de medir ningun dano.
#:
#: `intrusar` elige el MENOR radio de la rejilla que lo alcanza, y ese criterio
#: de desempate tambien esta declarado: la premisa es una licencia y no un dial,
#: y pasarse de largo compra dano por un canal que NINGUN peso del termino puede
#: quitar. El BatchNorm de `enc_t` corre en `train()` durante la fase 2, asi que
#: un intruso de magnitud enorme contamina la media y la varianza del lote y
#: aplasta a los limpios --- con 38 de 128 puntos a 29 veces la magnitud el
#: desvio del lote se infla ~16 veces. Eso degradaria los tres brazos por igual
#: y el techo no podria recuperarlo: un dano irrecuperable por construccion no
#: mide el sopesado, mide el BatchNorm.
#:
#: Medido con este valor: la calibracion elige 2.00-2.52 radios y la razon cae
#: entre 3.02 y 3.42 en las tres familias y dos semillas, de modo que la
#: intrusion es comparable entre familias en las unidades que importan.
FACTOR_INTRUSO = 3.0


def incertidumbre_declarada(logits):
    """`U(g) = H2(g)/ln C` de la Ec. 24, en UN solo lugar del modulo.

    `H2(g) = -log(sum g^2 + 1e-6)`, con el mismo epsilon que
    `_renyi_entropy_order_2`. La que `CREDALoss.forward` usa es identica pero
    esta escrita EN LINEA dentro de `forward` y no se puede importar; que no sea
    importable es un defecto de `models.py` y no algo que este modulo pueda
    arreglar sin tocar la linea base.

    Existe como funcion porque la expresion ya se leia en dos sitios --- el
    `u_t` del punto de control y el peso por punto --- y ahora en cuatro, con la
    perturbacion de frontera y su premisa. Cuatro copias de una formula son
    cuatro oportunidades de que una se quede atras sin que nada falle.
    """
    g = F.softmax(logits, 1)
    return -torch.log((g**2).sum(1) + 1e-6) / float(np.log(CLASES))


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

    NO USA `y_t`, y eso es lo que la habilita como criterio: es el histograma
    de las predicciones del PROPIO modelo. `pok` y `acc_va` miran la etiqueta
    del blanco, y una parada que las leyera fugaria lo que la tabla publica.

    En UN SOLO lugar del modulo a proposito. La leen dos consumidores --- el
    criterio de la fase 2 al final de cada epoca, y el artefacto por punto de
    control, derivada del `rut` que `cu` ya guarda --- y dos copias de la
    formula serian dos oportunidades de que una se quede atras sin que nada
    falle, que es el defecto que `incertidumbre_declarada` ya existe para
    evitar.
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


def ec33_por_clase(cr, f_s, y_s, f_t, g_t):
    """Las cuatro cantidades de la Ec. 33 por clase, LEIDAS de `CREDALoss`.

    Devuelve un arreglo `(CAMPOS_EC33, clases vivas)` con las filas en el
    orden de `CAMPOS_EC33_NOMBRES`.

    **No reimplementa nada, y eso es la decision entera.** `CREDALoss.forward`
    ya calcula `h_s`, `h_t` y `h_mix` por clase en cada evaluacion, porque las
    necesita para `J_c`; lo que falta no es el calculo, es que salgan. Asi que
    esto corre `forward` y OBSERVA sus tres entropias, parchando
    `_renyi_entropy_order_2` en la instancia mientras la llamada dura. Rehacer
    la cuenta aqui --- los kernels, el sopesado de la Ec. 24, el balance por
    traza de la Ec. 32 --- seria una SEGUNDA via para el mismo numero, y la
    que se quedara atras no daria error: es el defecto que `cef_cu` ya existe
    para no cometer, derivada del `rut` que el registro guarda en vez de
    recalculada del modelo.

    El espia tiene que ser ACOTADO. Uno puesto una vez y dejado disparar lo
    hace 54.861 veces en una corrida de 200 epocas --- MEDIDO: el termino se
    evalua en cada paso de entrenamiento y otra vez sobre el lote fijo de
    valid --- asi que se pone y se saca dentro de esta funcion, y
    `tests/test_creda_ec33_series.py` cuenta las lecturas de una corrida
    entera y exige una por punto de control.

    **`class_mixing` se fuerza a 1 y se restaura.** La Ec. 33 es el termino
    POR CLASE; con la mezcla por debajo de 1 `forward` llama ademas a
    `_marginal_score`, que suma tres entropias mas que no son de ninguna
    clase, y la captura dejaria de ser tres por clase. Se restaura en el
    `finally` porque el bucle de la fase 2 lo reescribe en cada paso a partir
    de `PASO_COND`: un valor que quedara pegado cambiaria el TERMINO que el
    paso siguiente optimiza, o sea la medicion perturbaria lo que mide.

    **Lo que mide del sopesado es lo que el brazo configuro.** `target_weights`
    no se pasa, asi que `forward` calcula los pesos de la Ec. 24 cuando el
    brazo los tiene prendidos y `h_t` es la entropia del bloque SOPESADO; con
    `use_entropy_weighting=False` es la del bloque crudo, que es lo que ese
    brazo le da al termino. Lo que NO replica es el RETARDO por paso
    (`PASO_W`), que vive en el bucle: antes de esa marca el termino multiplica
    por unos y esta lectura no, porque es una serie de diagnostico de la
    ecuacion y no una reconstruccion del paso.

    Sin gradiente y sin azar, que son los dos canales por los que una medicion
    perturbaria la corrida: `no_grad()` aqui, y nada en el camino
    --- `_gaussian_kernel`, `_mix_kernel_concat`, `_renyi_entropy_order_2` ---
    sortea nada. Los MODOS son del llamador: este no los toca porque el bloque
    del punto de control ya corre en `eval()`, y `no_grad()` solo no alcanzaria
    --- apaga el gradiente, no BatchNorm.
    """
    cap = []
    propio = type(cr)._renyi_entropy_order_2
    mezcla = cr.class_mixing

    def espia(K):
        h = propio(cr, K)
        cap.append(float(h))
        return h

    cr._renyi_entropy_order_2 = espia
    cr.class_mixing = 1.0
    try:
        with torch.no_grad():
            cr(f_s, f_t, y_s, g_t)
    finally:
        del cr._renyi_entropy_order_2
        cr.class_mixing = mezcla
    # Tres por clase viva, en el orden en que `forward` las appendea. La
    # aridad se afirma en vez de suponerse: si `forward` agregara una cuarta
    # entropia por clase, o llamara al bloque marginal, el reshape de abajo
    # repartiria los numeros entre filas equivocadas sin que nada falle.
    assert len(cap) % 3 == 0, (
        len(cap), 'la captura no trae tres entropias por clase')
    h_s, h_t, h_mix = np.asarray(cap, dtype=np.float64).reshape(-1, 3).T
    return np.stack([h_s, h_t, h_mix, 0.5 * (h_s + h_t) - h_mix])


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
    caen entre clases se achican. Esa es la propiedad entera que separa las dos
    perturbaciones de arriba.
    """
    return (1.0 - incertidumbre_declarada(logits)).numpy().astype(np.float32)


def pesos_compuerta(logits, q):
    """El peso declarado aplicado como COMPUERTA: `0` en la fraccion `q` menos
    confiada del lote, `1` en el resto.

    **La misma señal que `pesos_declarados`, la forma de `pesos_oraculo`.** Y
    existe porque medido, en el dato limpio y con el ancho de banda congelado,
    ninguna de las dos cosas que uno culparia primero es la que falla:

    * la señal NO es la equivocada --- el AUC del peso declarado contra el
      indicador de mal ruteo da 0.820 en toroide y 0.713 en rollo; y
    * la masa quitada NO es el mecanismo --- el declarado y el oraculo le dejan
      al bloque del blanco masa casi igual (0.797 contra 0.760 en toroide,
      0.796 contra 0.761 en rollo) con desenlaces opuestos.

    Lo que difiere es la FORMA del descuento. El declarado lleva un mal ruteado
    de 0.906 a 0.735 --- conserva el 73% de un punto que habria que borrar ---
    y el oraculo lo lleva de 1.0 a 0.0. La hipotesis que esto instrumenta es
    que un peso continuo RE-ESCALA todas las filas y por eso deforma la
    geometria que la entropia mide, mientras un {0,1} QUITA filas: limpiar y
    deformar no son la misma operacion.

    **Es un METODO y no un techo**, y la diferencia esta en la firma: no recibe
    `y`. El umbral sale del cuantil `q` del propio peso declarado DENTRO del
    lote, asi que no hay nada que leer de las etiquetas del blanco y la
    busqueda puede elegir `q` con el vigia, igual que cualquier otro HP.

    **Su prediccion es asimetrica a proposito**, y esa asimetria es lo que la
    hace refutable en vez de una pesca: tiene que recuperar parte del techo
    donde el orden informa, y NO puede ayudar en lunas, donde el AUC es 0.491
    --- el azar exacto, con un Delta-w de +0.001. Una compuerta que ayude en
    lunas FALSIFICA la hipotesis: lo que compro ahi no fue selectividad.

    `q = 0` es exactamente el brazo apagado, y la escalera tiene que contener
    su propio piso o `q` no tiene origen.
    """
    w = pesos_declarados(logits)
    if q <= 0.0:
        return np.ones(len(w), dtype=np.float32)
    # El cuantil del propio lote y no un umbral absoluto: la escala de `w`
    # depende de cuan confiada esta la cabeza, que se mueve con el paso, de
    # modo que un corte fijo cerraria una fraccion distinta en cada paso y `q`
    # dejaria de ser la tasa que la escalera barre.
    corte = float(np.quantile(w, q))
    # `>` y no `>=`: con empates en el corte --- que los hay cuando la cabeza
    # satura --- `>=` dejaria pasar el cuantil entero y la tasa no seria `q`.
    return (w > corte).astype(np.float32)


def pesos_oraculo(logits, y):
    """`w_j = 1` donde la pseudoetiqueta ACIERTA, `0` donde se equivoca.

    **Lee `y_t`, asi que NO es un metodo.** Es un TECHO, del mismo tipo que la
    columna del pico que este proyecto ya reporta: dice cuanto habria para
    ganar si el sopesado identificara perfectamente los puntos cuyo ruteo esta
    mal. `oraculo - apagado` acota lo que el sopesado puede comprar y
    `declarado - apagado` dice cuanto de eso captura el estimador de entropia,
    de modo que cualquiera de los tres desenlaces es un resultado.

    Que sea inalcanzable desde la busqueda no se deja a la prosa: `PESOS_BUSCABLES`
    no lo contiene, `buscar_hp2` lo rechaza por nombre, y el brazo que lo usa se
    rotula con `PALABRA_TECHO` en el nombre Y en una clave del registro.
    """
    pred = torch.as_tensor(logits).argmax(1)
    return (pred == torch.as_tensor(y)).numpy().astype(np.float32)


def intrusar_el_embedding(f, rho, radio, sorteo):
    """REEMPLAZA una fraccion `rho` de las filas de `f` por intrusos que caen a
    `radio` radios rms de la media del lote. Devuelve `(f_nuevo, mascara)`.

    **Despues de `enc_t`, y ahi vive todo el valor.** Tres construcciones
    anteriores inyectaron por la ENTRADA y ninguna le dejo a un sopesado algo
    para quitar, porque el daño entraba por cantidades calculadas ANTES de que
    exista cualquier peso por punto. Dos de esos canales estan cerrados ahora:

    * las estadisticas de lote del BatchNorm del encoder --- cerrado por
      inyectar DESPUES de `enc_t`, que ya corrio sobre los puntos limpios; y
    * el ancho de banda agrupado --- ya cerrado, porque sigma es una constante
      calibrada en el train de la FUENTE y congelada (`sigma_de_la_fuente`), de
      modo que nada inyectado en el blanco lo mueve. Con la mediana por lote un
      intruso movia la unidad de medida misma.

    Queda el bloque de clase sopesado, que es el unico canal que `w`
    multiplica. Es la primera construccion donde un sopesado funciona o el
    aporte esta muerto en este regimen.

    **REEMPLAZA y no agrega**: el tamaño del bloque y la composicion por clase
    quedan como estaban, asi que lo unico que cambio es DONDE caen `rho` filas.

    **Direcciones en pares antipodales.** Una direccion fija es un reescalado de
    features disfrazado de intrusion: moveria la media del lote, y entonces lo
    que se midiera despues no distinguiria 'hay intrusos' de 'el lote se
    corrio'. Con pares, la media no se mueve, y un test lo fija.

    **Lo que esta construccion NO es, y se dice porque cambia lo que mide:** un
    embedding escrito directamente es una CONSTANTE, asi que a diferencia de
    `intrusar` no tiene camino de gradiente de vuelta al encoder. Puede
    envenenar el termino igual --- ocupa una fila del bloque del blanco y, con
    el balance por traza de r28, suma masa a `tr(K~_t) = suma de w_j^2` sin
    aportar similitud, lo que diluye el bloque para todos los demas --- y eso
    SI es alcanzable por `w`. Pero es otra intervencion que una que fluye por el
    encoder, y las dos no se leen como el mismo experimento.
    """
    n = len(f)
    k = int(round(float(rho) * n))
    mal = torch.zeros(n, dtype=torch.bool)
    if k <= 0:
        return f, mal
    rng = np.random.default_rng(int(sorteo))
    idx = torch.as_tensor(rng.choice(n, k, replace=False).copy())
    mal[idx] = True
    mu = f.mean(0, keepdim=True).detach()
    # El radio rms del lote alrededor de su media, que es la unidad en la que
    # `radio` esta declarado. Un radio absoluto mediria una intrusion distinta
    # en cada familia --- medido, el radio rms del train del blanco vale 1.41
    # en lunas, 3.18 en toroide y 10.23 en rollo.
    rms = float((f.detach() - mu).norm(dim=1).pow(2).mean().sqrt()) + 1e-12
    # Pares antipodales: la mitad de las direcciones es la negacion de la otra,
    # asi que la media de los intrusos cae en `mu` y no la corre.
    media = (k + 1) // 2
    dirs = torch.as_tensor(rng.standard_normal((media, f.shape[1])),
                           dtype=f.dtype)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    pares = torch.cat([dirs, -dirs], 0)[:k]
    nuevo = f.clone()
    nuevo[idx] = mu + pares * (radio * rms)
    return nuevo, mal


def uniformar_el_embedding(f, cab, rho, sorteo, desplazamiento=0.0):
    """PROYECTA una fraccion `rho` de las filas al posterior UNIFORME, que es
    donde `w` vale CERO. Devuelve `(f_nuevo, mascara)`.

    **La perturbacion que el peso SI descuenta, y la unica que faltaba.** Con
    la cabeza afin, dos muestras con el argmax igual de inestable reciben pesos
    muy distintos: logits `(0, 0, -50)` --- empate exacto entre dos clases ---
    da `w = 0.369`, y `(0, 0, 0)` --- empate entre TODAS --- da `w = 0.000`.
    `Sum_c g_c^2` no mide el gap entre las dos primeras: mide cuantas clases
    estan en juego.

    De modo que el punto que lleva al peso a su MINIMO no es el que cae entre
    dos clases --- ese es `MODO_FRONTERA`, y recibe peso MEDIO --- sino el que
    cae donde se cruzan TODAS las fronteras. Ninguna de las cuatro
    construcciones anteriores lo produce: `reubicado` aterriza dentro de otra
    clase (confiado), el empuje por un rayo satura (`w -> 1`), `frontera`
    empata dos (peso medio) y los intrusos caen lejos (peso alto). El regimen
    donde el peso puede descontar al maximo nunca se midio.

    **Se resuelve, no se busca.** La cabeza es un solo `Linear`, asi que
    `{f : W f + b tiene todas sus componentes iguales}` es un subespacio AFIN
    de codimension C-1: `A f = c` con `A = [w_1 - w_2; ...; w_{C-1} - w_C]` y
    `c` las diferencias de sesgos. La proyeccion por pseudoinversa da el punto
    MAS CERCANO con posterior uniforme.

    **Y que el desplazamiento sea minimo es la mitad del diseño**: separa «el
    peso descuenta puntos inciertos» de «el peso descuenta puntos lejanos»,
    que es exactamente la confusion que costo cuatro construcciones.

    **`desplazamiento` resuelve la tension que eso deja.** Medido con la
    proyeccion sola, la perturbacion resulto casi inocua --- MEJORO el acierto
    en 4 de 6 celdas --- y es coherente: desplazamiento minimo e incertidumbre
    maxima implican daño minimo, porque el punto se queda donde estaba y solo
    recibe una etiqueta arbitraria. Y ahi «el peso no reparo» no prueba nada,
    porque no habia que reparar.

    Los dos requisitos NO se oponen. El subespacio tiene codimension C-1, asi
    que le quedan D-C+1 direcciones LIBRES: moverse dentro de el aleja el punto
    cuanto se quiera y deja el posterior EXACTAMENTE uniforme. `w = 0` a
    cualquier distancia.

    Eso vuelve al desplazamiento un EJE barrible con el peso clavado en su
    minimo, que es la unica forma de separar «el peso no sirve» de «no habia
    daño que reparar». La unidad es el radio rms del lote, por la misma razon
    que en `intrusar_el_embedding`: un radio absoluto mide una intrusion
    distinta en cada familia.
    """
    n = len(f)
    k = int(round(float(rho) * n))
    mal = torch.zeros(n, dtype=torch.bool)
    if k <= 0:
        return f, mal
    rng = np.random.default_rng(int(sorteo))
    idx = torch.as_tensor(rng.choice(n, k, replace=False).copy())
    mal[idx] = True
    W = cab.weight.detach(); b = cab.bias.detach()
    A = W[:-1] - W[1:]                      # (C-1, D)
    c = (b[1:] - b[:-1])                    # (C-1,)
    sel = f[idx].detach()
    # f* = f - A^+ (A f - c): el punto mas cercano del subespacio afin.
    resid = sel @ A.T - c
    paso = torch.linalg.lstsq(A, resid.T).solution.T
    proy = sel - paso
    if float(desplazamiento) > 0.0:
        # Direcciones en el NUCLEO de A: moverse ahi no cambia ningun logit
        # relativo, asi que el posterior sigue uniforme y `w` sigue en cero.
        # `svd` da una base ortonormal del nucleo en las ultimas D-(C-1) filas
        # de Vh; un gradiente numerico ahi seria indistinguible de salir del
        # subespacio, y un test fija que el posterior no deriva.
        Vh = torch.linalg.svd(A, full_matrices=True)[2]
        nulo = Vh[A.shape[0]:]                       # (D-C+1, D)
        dirs = torch.as_tensor(rng.standard_normal((k, nulo.shape[0])),
                               dtype=f.dtype) @ nulo
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        # Pares antipodales, por la misma razon que en `intrusar_el_embedding`:
        # una direccion neta corre la media del lote y eso es un reescalado
        # disfrazado de perturbacion.
        dirs[k // 2:] = -dirs[:k - k // 2][:k - k // 2]
        mu = f.mean(0, keepdim=True).detach()
        rms = float((f.detach() - mu).norm(dim=1).pow(2).mean().sqrt()) + 1e-12
        proy = proy + dirs * (float(desplazamiento) * rms)
    nuevo = f.clone()
    nuevo[idx] = proy
    return nuevo, mal


def pesos_oraculo_emb(mal):
    """`w_j = 0` en los intrusos del embedding EXACTAMENTE, `1` en el resto.

    **No es `pesos_oraculo` y no puede serlo.** Ese otro techo clava el peso en
    la correctitud de la PSEUDOETIQUETA, asi que a inyeccion cero ya pone ceros
    --- los limpios mal ruteados --- y su brazo no es el piso ni ahi. Este
    contesta 'cuanto habria para ganar quitando lo que inyectamos', que es otra
    pregunta.

    **LEE quienes se inyectaron, asi que es un TECHO y no un metodo.** Lo puede
    ser con exactitud porque los pusimos nosotros: no hay estimacion de por
    medio.
    """
    return (~torch.as_tensor(mal)).numpy().astype(np.float32)


def pesos_oraculo_ruidoso(logits, y, precision, sorteo):
    """El oraculo CORROMPIDO a una `precision` declarada, al MISMO conteo.

    Contesta la unica pregunta que la compuerta refutada dejo en pie: **cuanta
    precision necesita una seleccion antes de que quitar puntos pague**.

    La compuerta --- la señal declarada umbralada en {0,1} --- perdio en los
    cuatro cortes de toroide, la familia de MEJOR orden (AUC 0.820),
    monotonamente en la tasa, con p de 0.0039 a 0.0137 y ganando UNA semilla de
    diez en todos. Y el oraculo, que quita con precision 1.0 por construccion,
    GANA (+0.0110 en toroide, +0.0788 en rollo). Entre esos dos extremos esta
    todo, y la monotonia dice de que lado esta el costo: **borrar un bien
    ruteado cuesta mas que conservar un mal ruteado.**

    **El conteo se mantiene igual al del oraculo, y eso es el diseño.** Quita
    `k = |mal ruteados|` puntos, de los cuales `round(precision * k)` lo son de
    verdad y el resto son buenos sacrificados. Si el conteo bajara con la
    precision serian DOS cambios por peldaño y la lectura no perteneceria a
    ninguno de los dos.

    **Lee `y_t`, asi que es un TECHO y nunca un metodo**: su nombre lleva
    `PALABRA_TECHO`, `PESOS_BUSCABLES` no lo contiene y `buscar_hp2` lo rechaza
    por nombre. No es un brazo que alguien pueda correr: es una COTA sobre lo
    que cualquier señal podria comprar. Si el equilibrio queda en 0.95,
    ninguna señal sin etiquetas lo alcanza, y eso deja de ser una conclusion
    sobre nuestro estimador para ser una sobre la formulacion.

    **Ojo con la tasa base al leer un peldaño.** Una seleccion AL AZAR tiene
    precision igual a la fraccion mal ruteada, que esta medida y es distinta en
    cada familia: 0.240 en toroide y rollo, 0.603 en lunas. Un peldaño de 0.60
    es casi perfecto en toroide y es exactamente el azar en lunas, asi que
    comparar el mismo numero entre familias sin eso dice lo contrario de lo que
    pasa.

    `sorteo` siembra el sorteo: la misma entrada da el mismo vector, o el delta
    de un peldaño lleva adentro el ruido del sorteo en vez de medir precision.
    """
    mal = pesos_oraculo(logits, y) < 0.5
    idx_mal = np.flatnonzero(mal)
    idx_bien = np.flatnonzero(~mal)
    k = len(idx_mal)
    w = np.ones(len(mal), dtype=np.float32)
    if k == 0:
        return w
    # `round` y no `int`: truncar sesgaria la escalera entera hacia abajo y el
    # peldaño de 1.0 dejaria de ser el oraculo exacto, que es su origen.
    n_mal = min(int(round(float(precision) * k)), k)
    n_bien = min(k - n_mal, len(idx_bien))
    rng = np.random.default_rng(int(sorteo))
    quita = np.concatenate([
        rng.choice(idx_mal, n_mal, replace=False) if n_mal else np.empty(0, int),
        rng.choice(idx_bien, n_bien, replace=False) if n_bien else np.empty(0, int)])
    w[quita] = 0.0
    return w


def pesador_del_cabezal(enc, cab):
    """Un lector de `(w, prediccion)` sobre el cabezal CONGELADO de la fase 1.

    Devuelve una clausura y no los tensores porque `frontera` evalua la rejilla
    de `ALFAS_FRONTERA` punto por punto, y lo que necesita es poder PREGUNTARLE
    al cabezal por posiciones que todavia no existen.

    Restaura el modo train/eval que encontro: `corrida` administra esos modos y
    un `eval()` que se quedara puesto cambiaria lo que BatchNorm hace mas
    adelante sin que nada lo diga.
    """
    def pesar(X):
        modos = (enc.training, cab.training)
        enc.eval(); cab.eval()
        try:
            with torch.no_grad():
                lg = cab(enc(torch.as_tensor(X)))
                return pesos_declarados(lg), lg.argmax(1).numpy()
        finally:
            enc.train(modos[0]); cab.train(modos[1])
    return pesar


def fase1_congelada(fam, D, pos, semilla, hp1):
    """La fase 1 sola, con la MISMA receta y el mismo orden de azar que `corrida`.

    Existe porque `frontera` necesita el cabezal ANTES de contaminar, y lo que
    tiene que leer es el cabezal que la corrida va a usar de verdad --- medir la
    premisa sobre otro cabezal seria medir una premisa sobre nada. Se le pasa
    despues a `corrida` por `fase1=`, asi que no es una reproduccion: es el
    mismo objeto.

    Entrenar sobre los datos LIMPIOS es legitimo y no un atajo: la fase 1 solo
    mira `Xs_*`/`ys_*` y las dos perturbaciones solo tocan `Xt_tr`, asi que el
    cabezal es el mismo con y sin contaminacion. Los seis splits que recibe son
    los MISMOS objetos en los dos casos, porque `contaminar` no copia nada que
    no mueva.

    Devuelve la tupla que `corrida` espera en `fase1=`: `(enc, cab, hist, ep)`.
    """
    gen = torch.Generator().manual_seed(semilla)
    torch.manual_seed(semilla); np.random.seed(semilla)
    enc = MLP(fam['amb'], pos, dim=hp1['DIM'])
    cab = nn.Linear(hp1['DIM'], CLASES)
    hist, ep = train_source_ce(
        enc, cab, D['Xs_tr'], D['ys_tr'], D['Xs_va'], D['ys_va'],
        epochs=EPOCAS_1, batch_size=hp1['LOTE_1'], generator=gen,
        learning_rate=hp1['LR_1'], weight_decay=hp1['WD_1'])
    return enc, cab, hist, ep


def contaminar(D, rho, semilla, modo=MODO_REUBICADO, pesar=None):
    """Perturba una fraccion `rho` del TRAIN del blanco, de UNA de dos maneras.

    **Solo el train del blanco, las dos.** El termino se envenena donde entrena
    y la metrica se queda sobre datos que nadie movio, que es la unica razon por
    la que el acierto es comparable entre escenarios: lo que se mide es cuanto
    dano hacen los puntos malos, no cuanto se corrio el conjunto de evaluacion.
    Un mecanismo que tocara tambien el test daria una curva monotona igual,
    midiendo su propia perturbacion.

    **`reubicado`: se cambia la clase de region, no la etiqueta.** El punto se
    traslada restandole el centroide de SU clase y sumandole el de otra,

        x_j <- x_j - mu_{y_j} + mu_{c'},    c' != y_j

    de modo que conserva su desviacion dentro de la clase --- no se inventa
    geometria nueva, se reubica la que hay --- y su etiqueta verdadera se
    mantiene. Eso es lo que lo hace un OUTLIER y no un error de etiqueta: para
    la metrica sigue siendo de su clase, y para el ruteo del termino cae donde
    cae. Queda CONFIADAMENTE EQUIVOCADO, asi que su `w` no baja; ver
    `MODOS_CONTAMINACION` para los seis numeros que lo miden.

    **`frontera`: se lleva el punto a donde el cabezal esta indeciso.** Se
    interpola x_j hacia un punto del train del blanco que el cabezal RUTEA A
    OTRA CLASE,

        x_j(alpha) = x_j + alpha (x_d - x_j),    alpha en ALFAS_FRONTERA

    y se queda la posicion de la rejilla cuyo `w` es MINIMO, es decir aquella
    donde `g` esta mas plano. Que el segmento una dos puntos que el cabezal
    rutea distinto es lo que garantiza que lo cruza: el ruteo tiene que cambiar
    en algun lado del camino, y ahi los logits estan empatados.

    **`frontera` lee el cabezal y no las etiquetas.** `pesar` sale de
    `pesador_del_cabezal`, que se construye con la fase 1 --- entrenada con
    `y_s` y nada mas --- de modo que la eleccion de donde poner cada punto no
    toca `y_t` en ningun paso. El punto conserva su etiqueta verdadera, igual
    que en `reubicado`.

    Un `modo='frontera'` sin `pesar` se RECHAZA en vez de caer a un punto medio
    entre centroides: ese punto medio esta medido y NO aplana `g` --- los seis
    deltas dieron -0.0790, +0.0840, +0.0131, +0.0281, +0.0085, +0.0032 --- asi
    que una caida silenciosa a el mediria una perturbacion que nadie declaro y
    pasaria todos los demas tests de este archivo.

    **Deterministico desde la semilla**: los indices, la clase donante y el
    punto donante salen de un `default_rng(semilla + SEMILLA_CONTAMINACION)`,
    asi que dos llamadas con la misma semilla y el mismo `rho` dan el mismo
    dato bit a bit, y dos semillas distintas no. Los indices se sortean ANTES
    de ramificar por modo, asi que las dos perturbaciones mueven LOS MISMOS
    puntos y la comparacion entre ellas es apareada.

    No muta lo que recibe: devuelve un diccionario nuevo cuyos demas splits son
    LOS MISMOS objetos, de modo que "el test no se toco" es estructural y no
    una comparacion que haya que acordarse de hacer.

    Devuelve `(D2, indices)`, con los indices ORDENADOS: la figura necesita
    saber cuales son los puntos, y lo sabemos porque los inyectamos nosotros.
    """
    assert 0.0 <= rho < 1.0, rho
    assert modo in MODOS_CONTAMINACION, (modo, MODOS_CONTAMINACION)
    X = D['Xt_tr']
    y = D['yt_tr']
    n = len(X)
    k = int(round(rho * n))
    D2 = dict(D)
    if k == 0:
        # El cero es un no-op EXACTO, y por eso devuelve el mismo tensor: una
        # copia seria igual en bytes y distinta en identidad, y entonces el
        # nivel cero dejaria de poder afirmarse bit a bit.
        return D2, np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(semilla + SEMILLA_CONTAMINACION)
    idx = np.sort(rng.permutation(n)[:k])
    yn = y.numpy()
    clases = np.unique(yn)
    Xn = X.numpy().copy()
    if modo == MODO_REUBICADO:
        centros = {int(c): X.numpy()[yn == c].mean(0) for c in clases}
        assert len(centros) >= 2, (
            'una sola clase en el train del blanco: no hay region ajena a la '
            'que reubicar nada')
        for j in idx:
            propia = int(yn[j])
            ajenas = [int(c) for c in clases if int(c) != propia]
            donante = ajenas[int(rng.integers(len(ajenas)))]
            Xn[j] = Xn[j] - centros[propia] + centros[donante]
    else:
        assert pesar is not None, (
            modo, 'la perturbacion de frontera necesita el cabezal congelado '
            'de la fase 1 para saber donde esta la frontera; sin el no hay '
            'caida silenciosa a un punto medio, que esta medido y no aplana g')
        _, ruta = pesar(X)
        assert len(np.unique(ruta)) >= 2, (
            np.unique(ruta), 'el cabezal rutea todo a una clase, asi que no '
            'hay frontera entre dos a la que llevar nada')
        for j in idx:
            otras = np.nonzero(ruta != ruta[j])[0]
            assert len(otras) > 0, (int(j), int(ruta[j]))
            d = Xn[int(otras[int(rng.integers(len(otras)))])]
            cand = Xn[j][None, :] + ALFAS_FRONTERA[:, None] * (d - Xn[j])[None, :]
            w_cand, _ = pesar(torch.tensor(cand, dtype=X.dtype))
            Xn[j] = cand[int(np.argmin(w_cand))]
    D2['Xt_tr'] = torch.tensor(Xn, dtype=X.dtype)
    # Que de verdad se movieron los `k` que se dicen, comprobado al salir y
    # para los DOS modos: un `rho` que reubicara un punto sobre si mismo ---
    # centroides iguales, una clase donante mal elegida, un `alpha` que resulto
    # cero --- devolveria la celda entera plana y la figura mostraria que el
    # sopesado no compra nada.
    movidas = int((~np.isclose(Xn, X.numpy()).all(axis=1)).sum())
    assert movidas == k, (modo, movidas, k,
                          'la perturbacion no movio los k puntos')
    return D2, idx


def latente_del_codificador(enc):
    """Un lector del LATENTE sobre el codificador CONGELADO de la fase 1.

    El gemelo de `pesador_del_cabezal` para la otra cantidad: ese devuelve
    `(w, ruteo)` del cabezal, este devuelve `z = enc(x)`. Son dos porque la
    intrusion se califica por DISTANCIA LATENTE y no por `w`: lo que hace dano
    es la geometria, y si `w` lo ve o no es el hallazgo.

    Devuelve una clausura y no los tensores por la misma razon que el pesador:
    la calibracion del radio evalua la rejilla de `RADIOS_INTRUSO` radio por
    radio, asi que necesita poder PREGUNTARLE al codificador por posiciones que
    todavia no existen.

    Restaura el modo train/eval que encontro, y corre en `eval()`: con
    BatchNorm un forward en `train()` reestima `running_mean`/`running_var` sin
    gradiente, de modo que MEDIR movería el objeto medido.
    """
    def leer(X):
        modo = enc.training
        enc.eval()
        try:
            with torch.no_grad():
                return enc(torch.as_tensor(X)).numpy()
        finally:
            enc.train(modo)
    return leer


def distancias_al_centroide(Z, y, mascara):
    """La geometria que la licencia del experimento afirma, en UN solo lugar.

    Dos lecturas, porque son dos afirmaciones distintas:

    `intruso` / `limpio`  la distancia media al centroide de SU PROPIA clase.
                          Es la que dice que la estructura de segundo orden de
                          ese bloque quedo distorsionada, que es lo que un peso
                          podria quitar.
    `cercano_*`           la distancia al centroide MAS CERCANO de las tres. Es
                          la que dice que el punto es un INTRUSO y no un
                          miembro de otra clase --- `reubicado` aprobaba la
                          primera por construccion y reprobaba esta.

    **Los centroides salen de los puntos LIMPIOS de cada clase, nunca de todos.**
    Un centroide que incluyera a los intrusos se dejaria arrastrar por ellos y
    la distancia medida seria menor cuanto mas lejos estuvieran, que es la
    direccion contraria a la que la premisa afirma.

    Promedia por clase y despues entre clases, no sobre los puntos: con clases
    de 363/384/332 puntos un promedio plano le da a la clase mas poblada el voto
    de la premisa.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    mascara = np.asarray(mascara, dtype=bool)
    assert len(Z) == len(y) == len(mascara), (len(Z), len(y), len(mascara))
    clases = np.unique(y)
    centros = {}
    for c in clases:
        limpios_c = Z[(y == c) & ~mascara]
        assert len(limpios_c) > 0, (
            int(c), 'una clase sin puntos limpios: su centroide no existe y la '
            'premisa no se puede medir contra nada')
        centros[int(c)] = limpios_c.mean(0)
    todos = np.stack([centros[int(c)] for c in clases])
    cercano = np.linalg.norm(Z[:, None, :] - todos[None, :, :], axis=2).min(1)
    d_int, d_lim = [], []
    for c in clases:
        mc = centros[int(c)]
        sel_i = (y == c) & mascara
        sel_l = (y == c) & ~mascara
        if sel_i.any():
            d_int.append(float(np.linalg.norm(Z[sel_i] - mc, axis=1).mean()))
        d_lim.append(float(np.linalg.norm(Z[sel_l] - mc, axis=1).mean()))
    return dict(
        intruso=(float(np.mean(d_int)) if d_int else None),
        limpio=float(np.mean(d_lim)),
        por_clase_intruso=d_int, por_clase_limpio=d_lim,
        cercano_intruso=(float(cercano[mascara].mean()) if mascara.any()
                         else None),
        cercano_limpio=float(cercano[~mascara].mean()))


def intrusar(D, rho, semilla, latente=None):
    """REEMPLAZA una fraccion `rho` de cada clase del train del blanco por
    intrusos cuyo latente cae LEJOS de todos los centroides de clase.

    **Solo el train del blanco.** El termino se envenena donde entrena y la
    metrica se queda sobre datos que nadie movio, que es la unica razon por la
    que el acierto es comparable entre escenarios.

    **Reemplaza, no agrega ni reetiqueta.** La muestra que estaba en el indice
    `j` deja de estar y en su lugar queda un intruso con SU MISMA etiqueta, de
    modo que la composicion por clase del train es exactamente la que era ---
    `yt_tr` sale como el MISMO objeto --- y lo unico que cambio es donde caen
    `k_c` puntos de cada clase.

    **POR CLASE, con el redondeo declarado.** `k_c = round(rho * n_c)`. Un
    sorteo global del mismo total puede vaciar una clase y duplicar otra, que es
    otra intervencion con el mismo numero.

    **No se puede fijar el latente, asi que se CALIBRA y se MIDE.** El latente
    lo produce `enc_t`, que se mueve en cada paso; lo que se elige es la
    posicion de ENTRADA. El intruso va a

        x_j <- mu + R * r_rms * u_j,   ||u_j|| = 1

    con `mu` la media del train del blanco, `r_rms` su radio rms, y `R` el MENOR
    radio de `RADIOS_INTRUSO` cuya distancia latente medida alcanza
    `FACTOR_INTRUSO` veces la de los limpios. Si ningun radio de la rejilla lo
    alcanza, REHUSA: devolver el ultimo radio en silencio produciria una celda
    sin licencia que pasaria todos los demas tests de este archivo.

    **Las direcciones van en pares ANTIPODALES**, `+u` y `-u`, y eso no es un
    detalle: con direcciones sueltas la media global de la entrada se corre y la
    varianza se infla por un eje, lo que es un RESCALADO de las features
    disfrazado de intrusion. Medido en una inyeccion cruda por una direccion
    FIJA a 8 desvios, AdaBN subio +0.2139 en lunas y bajo -0.1657 en toroide y
    -0.1436 en rollo: el signo depende de si el rescalado le conviene a la
    familia, asi que un resultado positivo ahi seria un artefacto. Con pares
    antipodales la suma de los desplazamientos inyectados es cero cuando `k_c`
    es par, y la inflacion que queda es ISOTROPA --- un cambio de escala comun,
    que el `sigma='auto'` del kernel y el BatchNorm absorben --- en vez de una
    direccion privilegiada.

    **Los puntos limpios no se tocan.** Sus filas salen bit a bit iguales, de
    modo que su media y su varianza por feature son EXACTAMENTE las que eran, y
    su latente en el paso 0 tambien: el codificador congelado mapea punto por
    punto y en `eval()` no mira el lote.

    **Deterministico desde la semilla**: los indices y las direcciones salen de
    un `default_rng(semilla + SEMILLA_CONTAMINACION)`, el mismo flujo disjunto
    que usa `contaminar`.

    No muta lo que recibe: devuelve un diccionario nuevo cuyos demas splits son
    LOS MISMOS objetos, de modo que "el test no se toco" es estructural y no una
    comparacion que haya que acordarse de hacer.

    Devuelve `(D2, indices, premisa)`, con los indices ORDENADOS y `premisa` el
    diccionario de `distancias_al_centroide` mas el radio elegido --- o `None` a
    `rho = 0`, donde no hay premisa porque no hay intrusos.
    """
    assert 0.0 <= rho < 1.0, rho
    X = D['Xt_tr']
    y = D['yt_tr']
    D2 = dict(D)
    yn = y.numpy()
    clases = np.unique(yn)
    cuentas = {int(c): int((yn == c).sum()) for c in clases}
    if sum(int(round(rho * cuentas[int(c)])) for c in clases) == 0:
        # El cero es un no-op EXACTO, y por eso devuelve el mismo tensor: una
        # copia seria igual en bytes y distinta en identidad, y entonces el
        # escalon cero dejaria de poder afirmarse bit a bit. Es el control
        # contra el que se lee el dano, asi que tiene que ser exacto.
        return D2, np.zeros(0, dtype=np.int64), None
    assert latente is not None, (
        'la intrusion necesita el codificador congelado de la fase 1 para '
        'CALIBRAR el radio contra la distancia latente de los limpios; sin el '
        'no hay caida silenciosa a un radio fijo, que mide una intrusion '
        'distinta en cada familia')
    rng = np.random.default_rng(semilla + SEMILLA_CONTAMINACION)
    Xn = X.numpy().copy()
    mu = Xn.mean(0)
    r_rms = float(np.sqrt(((Xn - mu) ** 2).sum(1).mean()))
    assert r_rms > 0.0, 'el train del blanco es un solo punto'
    sel, dirs = [], []
    for c in clases:
        pos = np.nonzero(yn == c)[0]
        k = int(round(rho * cuentas[int(c)]))
        assert 0 < k < len(pos), (int(c), k, len(pos),
            'un nivel que reemplaza la clase entera deja a su centroide sin '
            'puntos limpios contra los que medir la premisa')
        sel.append(rng.permutation(pos)[:k])
        # Los pares antipodales: se sortean ceil(k/2) direcciones y cada una
        # entra dos veces, con los dos signos.
        base = rng.normal(size=((k + 1) // 2, Xn.shape[1]))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        dirs.append(np.concatenate([base, -base])[:k])
    sel = np.concatenate(sel)
    dirs = np.concatenate(dirs)
    assert len(sel) == len(dirs), (len(sel), len(dirs))
    mascara = np.zeros(len(Xn), dtype=bool)
    mascara[sel] = True
    # La referencia LIMPIA de la licencia, medida sobre los puntos que nadie
    # movio y con los centroides de ellos: es el denominador de la razon y no
    # puede depender de donde terminen los intrusos.
    base_limpia = distancias_al_centroide(latente(X), yn, mascara)
    elegido = None
    for R in RADIOS_INTRUSO:
        candidato = (mu + R * r_rms * dirs).astype(Xn.dtype)
        Xc = Xn.copy()
        Xc[sel] = candidato
        medido = distancias_al_centroide(latente(Xc), yn, mascara)
        # LAS DOS lecturas contra la misma referencia; ver `FACTOR_INTRUSO`.
        # Pedir solo la primera admite un punto lejos de su clase y pegado a
        # otra, que es lo que `reubicado` resulto ser.
        if min(medido['intruso'], medido['cercano_intruso']) >= (
                FACTOR_INTRUSO * base_limpia['limpio']):
            elegido = (float(R), candidato, medido)
            break
    assert elegido is not None, (
        float(RADIOS_INTRUSO[-1]), base_limpia['limpio'], FACTOR_INTRUSO,
        'ningun radio de la rejilla declarada alcanza la licencia, asi que no '
        'hay intrusion que medir: la celda se rehusa en vez de devolver el '
        'radio mas grande y pasar por buena')
    R, candidato, medido = elegido
    Xn[sel] = candidato
    D2['Xt_tr'] = torch.tensor(Xn, dtype=X.dtype)
    # Que de verdad se reemplazaron los que se dicen, y SOLO esos: un intruso
    # que cayera sobre su propio punto dejaria la celda plana, y una fila limpia
    # que se moviera volveria la premisa una afirmacion sobre otra cosa.
    movidas = ~np.isclose(Xn, X.numpy()).all(axis=1)
    assert int(movidas.sum()) == len(sel), (int(movidas.sum()), len(sel),
        'la intrusion no reemplazo los k puntos que declara')
    assert np.array_equal(movidas, mascara), (
        'la intrusion movio una fila que no es de las que declara')
    premisa = dict(radio=R, factor=float(FACTOR_INTRUSO), **medido)
    premisa['limpio_antes'] = float(base_limpia['limpio'])
    premisa['razon'] = float(medido['intruso'] / base_limpia['limpio'])
    return D2, np.sort(sel), premisa


def pesos_oraculo_intrusos(n, intrusos):
    """`w_j = 0` en los intrusos EXACTAMENTE, `1` en todo lo demas.

    **No es `pesos_oraculo` y no puede serlo.** Ese otro techo clava el peso en
    la correctitud de la PSEUDOETIQUETA sobre todo el train del blanco, asi que
    tambien pone en cero los puntos limpios mal ruteados --- y a intrusion cero
    no es un no-op, porque siempre hay limpios mal ruteados. Reusarlo aca
    contestaria otra pregunta: 'cuanto habria para ganar si el sopesado
    identificara el ruteo malo' y no 'cuanto habria para ganar si quitara los
    intrusos'.

    **LEE quienes son los intrusos, asi que NO es un metodo.** Es un TECHO, y lo
    puede ser con exactitud porque los inyectamos nosotros: no hay estimacion de
    por medio. `oraculo - apagado` acota lo que quitar los intrusos puede
    comprar y `declarado - apagado` dice cuanto de eso captura el estimador de
    entropia, de modo que cualquiera de los tres desenlaces es un resultado.

    Que sea inalcanzable desde la busqueda no se deja a la prosa:
    `PESOS_BUSCABLES` no lo contiene, `buscar_hp2` lo rechaza por nombre, y el
    brazo que lo usa lleva `PALABRA_TECHO` en el nombre Y en una clave del
    registro.
    """
    n = int(n)
    assert n > 0, n
    idx = np.asarray(list(intrusos), dtype=np.int64).reshape(-1)
    assert np.all((idx >= 0) & (idx < n)), (idx, n,
        'un indice de intruso fuera del train del blanco: el indicador seria '
        'de otro conjunto')
    w = np.ones(n, dtype=np.float32)
    w[idx] = 0.0
    return w


def covarianza(z, cresta=1e-4):
    # La cresta va RELATIVA a la traza, no como un numero absoluto. Una cresta
    # fija de 1e-4 no significa lo mismo sobre una matriz de escala 100 que sobre
    # una de escala 0.01, y los brazos de aca difieren en ordenes de magnitud:
    # medido cuando la fase 1 llevaba una KL, lunas sin BatchNorm daba el termino
    # no finito en 14 de 40 pasos registrados -- logdet divergiendo sobre una
    # covarianza casi singular. Ese termino ya no corre; la cresta relativa se
    # queda porque aca cond_s se lee como diagnostico y una matriz casi singular
    # da un numero sin sentido igual.
    d = z.shape[1]
    zc = z - z.mean(0)
    S = zc.T @ zc / (z.shape[0] - 1)
    return S + cresta * (torch.diagonal(S).sum() / d + 1e-12) * torch.eye(d)
#: El techo de la Ec. 21, `B_src = ln(1 + 1/eps)` con el `eps = 1e-8`
#: declarado: 18.4207. Se lee una sola vez al importar y no se recalcula, para
#: que los paneles y los tests puedan NOMBRAR la constante por la que se divide
#: el termino en vez de volver a derivarla cada uno por su lado.
B_FUENTE = source_bound()


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

    **Que cambia respecto de la cross-entropy cruda que corria antes.** La Ec.
    21 divide por `B_src`, el supremo exacto de la perdida de una unidad, de
    modo que el termino vive en [0, 1): cero para una prediccion correcta y
    certera, y nunca uno, porque softmax es estrictamente positivo y `g = 0` no
    se alcanza. La CE cruda arrancaba en `ln C = 1.0986` y no tenia techo, asi
    que el eje fijo de los paneles no podia ser el rango del termino --- era
    una convencion, y recortaba.

    **No mueve el minimizador.** `B_src` es una constante fijada por `eps`, asi
    que dividir por ella es reescalar los coeficientes libres; Adam la absorbe.
    Lo que si mueve es la ESCALA del gradiente de este termino FRENTE a los
    otros, y eso importa en la unica fila donde la fuente se suelta --- la de
    diagnostico de gradiente, que mide |g_renyi|/|g_ce| sobre el codificador de
    la fuente. Ese cociente sube por el factor `B_src` y no es un defecto: es
    la razon de los dos terminos una vez que los dos estan en su escala.

    `source_loss` toma PROBABILIDADES y etiquetas en one-hot --- una fila por
    unidad etiquetada, que aca es una muestra --- asi que el softmax y el
    one-hot se construyen aca y no se le pasan logits.
    """
    assert logits.dim() == 2, logits.shape
    assert logits.shape[1] == CLASES, (logits.shape, CLASES)
    G = F.softmax(logits, 1)
    Y = F.one_hot(y.reshape(-1), CLASES).to(G.dtype)
    return source_loss(G, Y)


def train_source_ce(encoder, head, x, y, xv, yv, *, epochs, batch_size,
                    generator, learning_rate=1e-2, weight_decay=0.0,
                    paciencia=None):
    # train_source de two_phase.py y nada mas: cross-entropy sobre las etiquetas
    # de la fuente. Esta fase tuvo tres terminos de latente a lo largo del
    # estudio -- la Hellinger global contra N(0,I), el par de entropias de Renyi
    # sobre las clases, y la KL -- y ninguno sobrevive. Medido sobre 18 celdas:
    # cambiar el termino abre 0.1322 de acierto en el blanco y cambiar la
    # normalizacion abre 0.2438, asi que lo que decide es la arquitectura y no
    # el termino. Devuelve la historia por epoca y la epoca que la parada
    # temprana eligio.
    paciencia = PACIENCIA_1 if paciencia is None else paciencia
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()),
                           lr=learning_rate, weight_decay=weight_decay)
    # Parada temprana sobre el split de VALID de la fuente --- valid, no test.
    # Esto no es fuga: y_s lo tenemos, es el dominio etiquetado.
    #
    # `xv`/`yv` eran el split de 'evaluacion', el MISMO que la tabla reportaba:
    # la epoca se elegia para maximizar exactamente el numero que despues se
    # publicaba. Con los tres splits, test ya no participa de ninguna decision.
    #
    # Se guardan los PESOS de la mejor epoca y se restauran al salir. Quedarse
    # con los ultimos y solo REPORTAR la mejor epoca seria mentir: el paso 2
    # transfiere lo que haya en el encoder, no lo que la tabla diga.
    hist, mejor, mejor_ep, mejor_w = [], -1.0, -1, None
    for epoca in range(epochs):
        encoder.train(); head.train()
        ce_ep, n_ep = 0.0, 0
        orden = torch.randperm(len(x), generator=generator)
        for ini in range(0, len(x), batch_size):
            idx = orden[ini:ini + batch_size]
            if len(idx) < 2: continue
            z = encoder(x[idx])
            # La Ec. 21 y no una CE cruda. Es la perdida DE VERDAD de la fase
            # 1, y lo que cambia es la escala: se divide por `B_FUENTE`, que
            # Adam absorbe, mientras la parada temprana mira el ACIERTO de
            # valid y no este numero. Medido, el modelo de fuente sale
            # practicamente igual; el delta se reporta donde se mide.
            ce = perdida_fuente(head(z), y[idx])
            ce_ep += float(ce.detach()); n_ep += 1
            opt.zero_grad(); ce.backward(); opt.step()
        encoder.eval(); head.eval()
        with torch.no_grad():
            _lg = head(encoder(xv))
            a_val = float((_lg.argmax(1) == yv).float().mean())
            # El CE de VALID, que antes no se registraba: la historia llevaba el
            # CE de TRAIN al lado del acierto de VALID, mezclados sin decirlo.
            # La misma Ec. 21 que entrena, para que las dos curvas del panel de
            # la fase 1 esten en la MISMA escala y en el mismo eje fijo. Una
            # mitad normalizada al lado de una cruda se dibuja igual y no es
            # comparable.
            ce_val = float(perdida_fuente(_lg, yv))
        encoder.train(); head.train()
        if a_val > mejor:
            mejor, mejor_ep = a_val, epoca
            mejor_w = ({k: v.detach().clone() for k, v in encoder.state_dict().items()},
                       {k: v.detach().clone() for k, v in head.state_dict().items()})
        if n_ep:
            _r = (epoca, ce_ep/n_ep, ce_val, a_val)
            assert len(_r) == CAMPOS_HS, (len(_r), CAMPOS_HS)
            hist.append(_r)
        # El BREAK de verdad. Antes corria las `epochs` enteras y solo recordaba
        # la mejor, asi que la parada temprana no ahorraba un segundo: medido,
        # rollo elige la epoca 14-37 de 200 y tiraba 163-186 epocas. Sobre
        # cientos de corridas de busqueda eso es la mayor parte del costo.
        if epoca - mejor_ep >= paciencia:
            break
    # Se vuelve a la mejor epoca. El assert existe porque un state_dict que no
    # se restaura deja TODO verde: el acierto reportado seria el de la mejor y
    # los pesos los de la ultima, y nada lo diria.
    assert mejor_w is not None, 'la fase 1 no dejo ninguna epoca'
    encoder.load_state_dict(mejor_w[0]); head.load_state_dict(mejor_w[1])
    encoder.eval(); head.eval()
    with torch.no_grad():
        _chk = float((head(encoder(xv)).argmax(1) == yv).float().mean())
    assert abs(_chk - mejor) < 1e-9, (_chk, mejor, 'los pesos restaurados no son los mejores')
    for q in encoder.parameters(): q.requires_grad_(False)
    for q in head.parameters():    q.requires_grad_(False)
    return hist, mejor_ep

def entre_dentro(H, y):
    # La descomposicion que este cambio existe para mover. Diagnostico: usa las
    # etiquetas verdaderas del blanco, que el entrenamiento nunca ve.
    mu = H.mean(0); entre = dentro = 0.0
    for c in np.unique(y):
        Hc = H[y == c]
        entre  += len(Hc)*float(((Hc.mean(0)-mu)**2).sum())/len(H)
        dentro += float(((Hc-Hc.mean(0))**2).sum())/len(H)
    return entre, dentro

def colapso(Ht, g_t):
    """Cuanto COLAPSO el blanco. Dos numeros, los dos SIN ETIQUETAS.

    El acierto no es la lente para esto. Con la cabeza siendo un solo `Linear`,
    el extractor no puede ser general y discriminante a la vez, asi que el
    acierto mide esa tension y no la calidad de la alineacion. Lo que si hace un
    modelo alimentado con puntos sin guia es colapsar --- a una clase
    equivocada o no --- y eso es un problema real.

    `cls_ef`   clases efectivas: `exp(H(g_promedio))`, en `[1, C]`. Lee el
               posterior PROMEDIO, no la confianza por punto: un modelo seguro
               en cada punto que usa dos clases no colapso, y confundir las dos
               cosas convertiria esto en una medida de confianza, que ya existe
               y es otra.
    `rank_ef`  dimension efectiva del embedding: `(sum l)^2 / sum l^2` sobre
               los autovalores de la covarianza. Una nube aplastada a una recta
               da 1. Es adimensional, asi que una nube que se ENCOGE no se lee
               como una que perdio rango --- sin eso, un cambio de unidades
               seria un colapso.

    Ninguno lee `y_t`, asi que no son diagnostico: son observables que un
    metodo podria usar. `tr_e`/`tr_d` siguen aparte, y aparte es el punto: el
    cociente confunde dos fenomenos opuestos --- clases que se fusionan y nubes
    que se desparraman.
    """
    with torch.no_grad():
        gm = (g_t if torch.is_tensor(g_t) else torch.as_tensor(g_t)).mean(0)
        gm = (gm / gm.sum()).clamp_min(1e-12)
        cls_ef = float(torch.exp(-(gm * gm.log()).sum()))
    H = np.asarray(Ht, dtype=np.float64)
    lam = np.linalg.eigvalsh(np.cov(H - H.mean(0), rowvar=False)
                             if H.shape[1] > 1 else np.atleast_2d(H.var()))
    lam = np.clip(lam, 0.0, None)
    rank_ef = float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-300)) \
        if lam.sum() > 0 else 1.0
    return dict(cls_ef=cls_ef, rank_ef=rank_ef)


def _k_cuartil(n):
    """El tamanio del cuartil, el MISMO para alto y bajo.

    La cardinalidad es parte de la medicion y no un detalle: el artefacto
    diagonal del termino escala como `1/n` --- medido, reducible 10.7x --- asi
    que dos subconjuntos de tamanio distinto no son comparables ni cuando el
    modelo no cambio. Dos es el piso: un kernel sobre un punto es la diagonal
    sola.
    """
    return max(2, n // 4)


def alineacion(Hs, Ht, ys, logits_t, yt, sigma, cfg):
    """Cuanto se ACERCO la nube del blanco a la de la fuente. Cuatro numeros.

    Esto existe porque faltaba. Las cuatro metricas del barrido --- acierto,
    confianza, concentracion, ruteo --- son del blanco solo o del clasificador,
    y ninguna contesta la pregunta que la Ec. 24 promete: alinear usando solo
    las muestras confiables. Un sopesado que cumple exactamente lo que promete
    y no mueve ninguna de esas cuatro es indistinguible, con esas cuatro, de uno
    que no hace nada.

    `creda`      el termino con `w = 1` sobre los embeddings finales. La VARA
                 ES COMUN a proposito: puntuar cada brazo con su propio
                 objetivo seria comparar dos funciones distintas y llamar
                 mejora a la diferencia. El sopesado del brazo entra en COMO
                 se entreno, nunca en como se mide.
    `creda_alto` el mismo numero sobre el cuartil de `w` mas alto, que es el
                 subconjunto que la formulacion dice que alinea.
    `creda_bajo` sobre el cuartil mas bajo: el que dice que descuenta.
    `cls`        la alineacion POR CLASE, adimensional: la distancia entre
                 centroides homologos sobre la dispersion intraclase. Lee
                 `yt`, asi que es diagnostico del mismo estatuto que `acc_te`
                 --- ni parada, ni busqueda, ni termino.

    Los dos cuartiles se definen con el `w` de la CORRIDA, asi que los brazos
    miran conjuntos distintos de puntos. Es intrinseco y no un defecto del
    instrumento: la pregunta es si cada brazo alineo a SUS confiables mejor que
    el otro a los suyos, y un conjunto fijado de afuera contestaria otra.
    """
    cfg = {k: v for k, v in cfg.items() if k != 'use_entropy_weighting'}
    vara = CREDALoss(sigma=sigma, use_entropy_weighting=False, **cfg)
    Hs_t = torch.as_tensor(np.ascontiguousarray(Hs, dtype=np.float32))
    Ht_t = torch.as_tensor(np.ascontiguousarray(Ht, dtype=np.float32))
    lg = logits_t if torch.is_tensor(logits_t) else torch.as_tensor(logits_t)
    ys_t = ys if torch.is_tensor(ys) else torch.as_tensor(ys)
    g = torch.softmax(lg, 1)
    yt = np.asarray(yt)
    with torch.no_grad():
        creda = float(vara(Hs_t, Ht_t, ys_t, g))
        w = pesos_declarados(lg)
        n = len(w); k = _k_cuartil(n)
        # Orden ESTABLE: con empates el subconjunto queda determinado por el
        # indice y no por el humor del algoritmo de sorteo, asi que el numero
        # es reproducible y el test puede fijar CUALES entran.
        ordn = np.argsort(w, kind='stable')
        sub = lambda idx: float(vara(Hs_t, Ht_t[idx], ys_t, g[idx]))
        c_alto, c_bajo = sub(ordn[n - k:]), sub(ordn[:k])
        # La alineacion por clase: solo las clases presentes en LAS DOS nubes.
        # Una clase que el blanco no tiene no esta desalineada, esta ausente, y
        # promediar su centroide vacio como si fuera una distancia es inventar.
        Hsn = np.asarray(Hs, dtype=np.float64); Htn = np.asarray(Ht, dtype=np.float64)
        ysn = np.asarray(ys_t)
        num = den = 0.0; m = 0
        for c in np.intersect1d(np.unique(ysn), np.unique(yt)):
            A, Bc = Hsn[ysn == c], Htn[yt == c]
            num += float(((A.mean(0) - Bc.mean(0)) ** 2).sum())
            den += float(((A - A.mean(0)) ** 2).sum(1).mean()
                         + ((Bc - Bc.mean(0)) ** 2).sum(1).mean()) / 2.0
            m += 1
        cls = (num / m) / max(den / max(m, 1), 1e-12) if m else float('nan')
    return dict(creda=creda, creda_alto=c_alto, creda_bajo=c_bajo, cls=cls)


#: El EJE de la seccion: con que se pondera el bloque del blanco. Tres valores,
#: y el tercero no es un metodo.
#:
#: `apagado`    `use_entropy_weighting=False`: todo peso en 1.
#: `declarado`  `w_j = 1 - H2(g_j)/ln C`, el de la formulacion (Ecs. 24 y 26).
#: `oraculo`    `w_j = 1` donde la pseudoetiqueta acierta y `0` donde se
#:              equivoca. LEE `y_t`, asi que es un TECHO y no un metodo.
#:
#: `oraculo - apagado` dice cuanto hay para ganar sopesando, y
#: `declarado - apagado` cuanto de eso captura el estimador de entropia. Con
#: los tres, cualquiera de los desenlaces es un resultado --- que es
#: exactamente lo que el diseno anterior, con dos brazos, no podia conseguir.
PESO_APAGADO = 'apagado'
PESO_DECLARADO = 'declarado'
PESO_ORACULO = 'oraculo'

#: El CUARTO brazo que es un METODO, y el unico que se agrego por un
#: diagnostico en vez de por una construccion: la misma señal del declarado
#: aplicada como {0,1} en vez de como un empujon. Ver `pesos_compuerta` para
#: las tres mediciones que lo pidieron. Es buscable porque no lee `y_t`.
PESO_COMPUERTA = 'compuerta'

#: El CONTROL, y no un metodo que alguien correria: los MISMOS valores de
#: `w` que el declarado, permutados entre los puntos del lote.
#:
#: Es lo que vuelve a la comparacion una prueba de MECANISMO y no de
#: desenlace. `declarado - apagado` dice si el sopesado cambia algo;
#: `declarado - barajado` dice si lo que cambio fue ELEGIR a quien pesar o
#: solo tener una distribucion de pesos desigual. Sin este brazo los dos
#: son indistinguibles, y todos los nulos medidos hasta hoy son
#: compatibles con 'el peso no selecciona nada'.
#:
#: No lee `y_t`, asi que no es un techo; pero tampoco esta en
#: `PESOS_BUSCABLES`, porque buscar hiperparametros para una permutacion
#: al azar no significa nada.
PESO_BARAJADO = 'barajado'

#: El CUARTO, y es un peso nuevo y no un reuso del anterior porque contesta otra
#: pregunta. `oraculo` clava el peso en la correctitud de la pseudoetiqueta
#: sobre todo el train del blanco, de modo que a intrusion cero YA pone ceros
#: --- los limpios mal ruteados --- y su brazo no es el piso ni ahi.
#: `oraculo-intrusos` pone cero en los intrusos EXACTAMENTE y uno en todo lo
#: demas, que es la cota de 'cuanto habria para ganar quitando lo que
#: inyectamos'. Ver `pesos_oraculo_intrusos`.
PESO_ORACULO_INTRUSOS = 'oraculo-intrusos'

#: El QUINTO peso, y el segundo que existe por un diagnostico: el oraculo
#: corrompido a una precision declarada, al mismo conteo. Ver
#: `pesos_oraculo_ruidoso`. Lleva `PALABRA_TECHO` porque lee `y_t`.
#: `oraculo-<que>`, la misma forma que `PESO_ORACULO_INTRUSOS`. La palabra
#: `PALABRA_TECHO` NO va aca y eso es la convencion del modulo, no un
#: descuido: `PESO_ORACULO` es 'oraculo' pelado, y lo que tiene que ser
#: imposible de leer como metodo es el BRAZO --- `BRAZO_TECHO` y la clave
#: `techo_no_es_metodo` del registro --- porque eso es lo que se imprime.
PESO_ORACULO_RUIDOSO = 'oraculo-ruidoso'

#: El SEXTO peso: el techo de la inyeccion en el embedding. Ver
#: `pesos_oraculo_emb`, y `intrusar_el_embedding` para la construccion.
PESO_ORACULO_EMB = 'oraculo-embedding'
PESOS = (PESO_APAGADO, PESO_DECLARADO, PESO_COMPUERTA, PESO_BARAJADO,
         PESO_ORACULO, PESO_ORACULO_INTRUSOS, PESO_ORACULO_RUIDOSO,
         PESO_ORACULO_EMB)

#: Los pesos que LEEN algo que un metodo no puede leer, y por eso son techos y
#: no metodos. Vive como tupla y no como una comparacion escrita en cada sitio
#: porque son tres los lugares que la preguntan --- el campo del registro, el
#: refuse de la busqueda y el rotulo del brazo --- y una condicion escrita tres
#: veces a mano es una que se queda atras en dos de ellas cuando aparece un
#: cuarto peso. Que aparecio.
PESOS_TECHO = (PESO_ORACULO, PESO_ORACULO_INTRUSOS, PESO_ORACULO_RUIDOSO,
               PESO_ORACULO_EMB)
assert set(PESOS_TECHO) < set(PESOS), (PESOS_TECHO, PESOS)

#: Los pesos que una busqueda de hiperparametros tiene PERMITIDO correr. El
#: oraculo no esta, y su ausencia la lee `buscar_hp2` para rechazarlo por
#: nombre: un techo alcanzable desde la busqueda deja de ser un techo y pasa a
#: ser un metodo que se eligio solo, con `y_t` adentro del criterio.
PESOS_BUSCABLES = (PESO_APAGADO, PESO_DECLARADO, PESO_COMPUERTA)
assert not (set(PESOS_BUSCABLES) & set(PESOS_TECHO)), (
    PESOS_BUSCABLES, PESOS_TECHO, 'un techo alcanzable desde la busqueda')

#: Minimo de puntos del lote que el oraculo puede dejar en pie antes de que el
#: termino deje de tener sentido. No es un fallback: si se cruza, la corrida
#: PARA. Un brazo que siguiera con un bloque de un punto reportaria un numero,
#: y ese numero no seria el del termino.
#:
#: Medido: con `pok` entre 0.30 y 0.46 y `LOTE_2=128`, el oraculo deja entre 38
#: y 59 puntos, asi que la guarda no llega a dispararse en este montaje. Esta
#: como assert y no como rama porque una rama que no se puede alcanzar se lee
#: como una guarda y no lo es.
MIN_ORACULO = 2


def corrida(fam, D, pos, semilla=None, cuadros=None, balance=True,
            mover_fuente=True, hp2=None, hp1=None, fase1=None, sopesado=True,
            peso=None, intrusos=None):
    # El plan, los tres pasos, en este orden y sin variantes:
    #
    #   1. source con cross-entropy sola, parando en su mejor epoca de
    #      validacion de FUENTE
    #   2. transferir pesos del encoder source al encoder target
    #   3. meter source + target y dejar mover encoder y predictor de los dos,
    #      con el regularizador de Renyi source-target prendido, normalizando
    #      las K sobre tr(K) incluida la mixta
    #
    # Lo que esto NO es, porque el cuaderno corria eso antes: la fase 1 no
    # lleva ningun termino de latente, no hay fase 2 con la fuente congelada, y
    # el CE no desaparece en el paso 3.
    # `peso` es el eje de verdad; `sopesado` queda como su alias booleano
    # porque es lo que los llamadores viejos pasan y lo que el registro ya
    # publica. Se derivan uno del otro en un solo sitio para que no puedan
    # discrepar: una corrida con `sopesado=True` y `peso='apagado'` correria
    # una cosa y se registraria como otra.
    if peso is None:
        peso = PESO_DECLARADO if sopesado else PESO_APAGADO
    assert peso in PESOS, (peso, PESOS)
    # `barajado` tambien pondera: lo que cambia es QUIEN recibe cada peso,
    # no si hay ponderacion. Si entrara como `sopesado=False`, el control
    # seria el brazo apagado con otro nombre y no controlaria nada.
    sopesado = peso in (PESO_DECLARADO, PESO_BARAJADO)
    # El indicador del techo de intrusos, resuelto UNA vez y aca. Falla cerrado:
    # un brazo que no supiera quienes son los intrusos ponderaria por unos y
    # seria el piso corriendo bajo el rotulo de un techo --- dos brazos
    # identicos con dos nombres distintos, y la tabla reportaria que quitar los
    # intrusos no compra nada sin que nada falle.
    if peso == PESO_ORACULO_INTRUSOS:
        assert intrusos is not None, (
            peso, 'el techo de intrusos necesita los indices que se '
            'inyectaron; sin ellos pondera por unos y es el piso')
        w_intrusos = pesos_oraculo_intrusos(len(D['Xt_tr']), intrusos)
        no_es_intruso = torch.as_tensor(w_intrusos > 0.5)
    else:
        w_intrusos, no_es_intruso = None, None
    sem = SEMILLA if semilla is None else semilla
    cuad = CUADROS if cuadros is None else cuadros
    # Refutada y no ignorada. Un `hp2` con la llave retirada viene de un
    # `barrido.json` anterior al sigma congelado, y esos HP de fase 2 son
    # INVALIDOS: se eligieron contra un ancho de banda movil y agrupado
    # fuente+blanco, que no es ni la magnitud ni la estabilidad de este. Dejar
    # pasar la llave sin leerla es como vuelve el alcance viejo sin que nadie
    # lo note.
    if 'SIGMA_FROM' in (hp2 or {}):
        raise ValueError(
            "SIGMA_FROM esta retirado: el ancho de banda ya no es un "
            "hiperparametro, lo calcula `sigma_de_la_fuente` sobre el train de "
            "la fuente tras la fase 1 y queda congelado. Un hp2 que trae esta "
            "llave viene de una busqueda hecha contra el sigma movil, asi que "
            f"sus valores no son validos aca; recibido {hp2['SIGMA_FROM']!r}.")
    # `TECHO` y `DELTA` se retiraron el 2026-10-11 por la misma razon de fondo
    # que `SIGMA_FROM`: dejaron de ser hiperparametros. No estan en
    # `research-concept-r29.md` --- cero menciones de rampa, ramp, schedule o
    # warmup --- y sobre un UNICO termino con gradiente un coeficiente no
    # balancea nada. Un `barrido.json` anterior los trae, y quien los manda
    # cree que decide algo.
    for _k in ('TECHO', 'DELTA'):
        if _k in (hp2 or {}):
            raise ValueError(
                f"{_k} esta retirado: el coeficiente y el warmup del termino "
                "se quitaron porque no estan en la formulacion, y sobre el "
                "unico termino con gradiente un escalar es una tasa de "
                "aprendizaje con otro nombre, redundante con LR_2. Un hp2 que "
                f"trae esta llave viene de una busqueda anterior; recibido "
                f"{hp2[_k]!r}.")
    h1 = {**HP_1, **(hp1 or {})}
    h2 = {**HP_2, **(hp2 or {})}
    gen = torch.Generator().manual_seed(sem)
    torch.manual_seed(sem); np.random.seed(sem)
    # `fase1` permite inyectar un (encoder, cabeza) YA entrenado. La busqueda de
    # fase 2 lo usa para no reentrenar la fuente en cada trial: asi todos los
    # trials comparan sobre EL MISMO modelo de fase 1 y la diferencia entre
    # ellos es apareada, sin la varianza de reentrenar la fuente adentro.
    if fase1 is not None:
        enc, cab, hist_s, ep_s = fase1
    else:
        enc, cab = MLP(fam['amb'], pos, dim=h1['DIM']), nn.Linear(h1['DIM'], CLASES)

    # ---- paso 1
    if fase1 is None:
        hist_s, ep_s = train_source_ce(
            enc, cab, D['Xs_tr'], D['ys_tr'], D['Xs_va'], D['ys_va'],
            epochs=EPOCAS_1, batch_size=h1['LOTE_1'], generator=gen,
            learning_rate=h1['LR_1'], weight_decay=h1['WD_1'])
    # La fuente de referencia, ANTES de que el paso 3 la mueva: es contra esto
    # que se lee si soltarla la degrado, y despues del paso 3 ya no se puede
    # medir porque los pesos son otros.
    with torch.no_grad():
        Hs_t = enc(D['Xs_tr']).clone()
        acc_s0 = float((cab(Hs_t).argmax(1) == D['ys_tr']).float().mean())
        # El ancho de banda, aca y una sola vez: `enc` ya esta entrenado y
        # congelado, y `Hs_t` ES el train completo de la fuente en el espacio
        # de representacion. Despues de la fase 2 ya no se puede calcular, por
        # la misma razon por la que `acc_s0` se mide aca.
        sigma_fijo = sigma_de_la_fuente(Hs_t)
        # La escala del BLANCO al paso 0, para la razon de deriva. `enc_t` es
        # todavia un clon de `enc`, asi que esto es la unidad en la que sigma
        # quedo calibrado.
        escala_t0 = sigma_de_la_fuente(enc(D['Xt_tr']))
    # train_source_ce congela al salir. El paso 3 los suelta, que es el cambio
    # central del plan; `mover_fuente=False` los deja quietos, que es la fila que
    # mide cuanto de lo que pasa se debe a esa libertad.
    for q in list(enc.parameters()) + list(cab.parameters()):
        q.requires_grad_(mover_fuente)

    # ---- paso 2
    enc_t = clone_for_target(enc)

    # ---- paso 3
    cr = CREDALoss(sigma=sigma_fijo,
                   normalize_by_bounds=NORMALIZAR_POR_COTAS,
                   bounds_form=FORMA_DE_COTAS,
                   weight_cross_block=h2['CRUZADO_SOPESADO'],
                   detach_target_entropy=h2['RHO_T_SIN_GRAD'],
                   unbiased_purity=h2['PUREZA_U'],
                   use_entropy_weighting=sopesado,
                   balance_mix_blocks=balance,
                   balance_mass=BALANCE_DE_MASA)
    # 'normalizando las k sobre tr de k, incluyendo la mixta'. Para un bloque
    # suelto eso es lo que H2 ya hace al dividir por su traza; para la mixta no,
    # y ahi vive todo el efecto: con lotes iguales el marginal queda identico a
    # 5.96e-08 (epsilon de float32) y el condicional se mueve 3.5e-03 a 1.3e-02,
    # porque las cuentas POR CLASE si difieren ([47, 38, 43] en un lote de 128).
    #
    # Y la masa que el balance iguala es la TRAZA de cada bloque (Ec. 32 de
    # r28), no su cardinalidad. En la fuente da lo mismo --- el kernel gaussiano
    # tiene 1 en la diagonal, asi que tr(K_s) = n_s --- pero el bloque del
    # blanco llega ya ponderado, con w_j^2 en la diagonal, de modo que
    # tr(K~_t) = suma de w_j^2. Dividir por n_t le daba al blanco la masa de su
    # CUENTA y no la que se gano: medido en un lote de 129 con pseudoetiquetas
    # de softmax, tr(K~_t)/n_t dio 0.470 / 0.774 / 0.482 en las tres clases,
    # mientras tr(K_s) daba exactamente n_s en las tres.
    assert cr.balance_mix_blocks is balance, (cr.balance_mix_blocks, balance)
    assert cr.bounds_form == FORMA_DE_COTAS, (cr.bounds_form, FORMA_DE_COTAS)
    assert cr.balance_mass == BALANCE_DE_MASA, (cr.balance_mass, BALANCE_DE_MASA)
    # El sopesado por confianza (Ecs. 24 y 26), que es el UNICO eje de la
    # seccion de contaminacion: los dos brazos de CREDA que esa seccion compara
    # difieren solo en esta bandera. Se afirma que llego donde tenia que llegar
    # porque `CREDALoss` la tiene en `True` por default, asi que un parametro
    # que no se pasara daria el brazo prendido en las dos filas --- dos brazos
    # identicos bajo dos rotulos distintos, y la tabla reportaria un empate
    # perfecto sin que nada fallara.
    assert cr.use_entropy_weighting is sopesado, (
        cr.use_entropy_weighting, sopesado)
    # `class_mixing` ya no se clava: lo fija `PASO_COND` en cada paso, mas
    # abajo. Se arranca en el valor que el paso 0 va a querer para que nada
    # lea un default que el bucle todavia no escribio.
    cr.class_mixing = 0.0 if h2['PASO_COND'] > 0 else 1.0
    params = list(enc_t.parameters())
    if mover_fuente: params += list(enc.parameters()) + list(cab.parameters())
    opt = torch.optim.Adam(params, lr=h2['LR_2'], weight_decay=h2['WD_2'])
    theta_s0 = torch.cat([q.detach().reshape(-1).clone() for q in enc.parameters()])
    def _bufs(m):
        b = [q.detach().reshape(-1).clone() for q in m.buffers()
             if q.dtype.is_floating_point]
        return torch.cat(b) if b else torch.zeros(1)
    buf_s0 = _bufs(enc)
    # El techo va en 1 por default, no en el 1e-4 publicado. Medido sobre el
    # encoder fuente: a 1e-4 la razon |g_renyi|/|g_ce| queda entre 4.6e-07 y
    # 4.2e-05, asi que la fuente no puede sentir al blanco. La busqueda lo puede
    # mover; `lambda_creda` queda clavado en 1.0 para no buscar dos veces lo
    # mismo, porque los dos se multiplican.

    mu = None
    with torch.no_grad():
        Hs = Hs_t.numpy()
        esc_s = float(Hs_t.std(0).norm()); disp_s = esc_s
        mu_s0 = Hs_t.mean(0)
        S_s = covarianza(Hs_t); cond_s = float(torch.linalg.cond(S_s))
    mu = Hs.mean(0); _, Sv, Vt = np.linalg.svd(Hs - mu, full_matrices=False)
    P = Vt[:2].T
    lam_ = (Sv**2) / max(len(Hs) - 1, 1)
    dim_ef = float(lam_.sum()**2 / (lam_**2).sum())
    # La proyeccion queda clavada a la fuente del paso 1 y no se recalcula: si
    # se moviera con la fuente, el GIF mostraria las dos nubes acercandose
    # cuando lo unico que cambio fue el sistema de coordenadas del dibujo.
    proy = lambda H: (H - mu) @ P
    with torch.no_grad(): pred_s = cab(Hs_t).argmax(1).numpy()

    # El lote FIJO de valid, sorteado una sola vez. Del MISMO tamanio que el de
    # train y no sobre el conjunto entero, y la razon cambio con r28 sin que la
    # restriccion cambie. Con las cotas de r25 el argumento era directo: los dos
    # extremos dependian de n_s y n_t, asi que un Renyi sobre 358 puntos no
    # estaba en la misma escala que uno sobre un lote de 128. Las cotas de r28
    # no dependen de las cuentas --- dependen de las dos entropias marginales
    # --- pero esas entropias si: H2 vive en [0, ln N], asi que 358 puntos
    # admiten una dispersion que 128 no pueden alcanzar, y U_c se mueve con eso.
    # El panel tiene el eje clavado en 0-1, asi que las dos curvas se verian
    # comparables sin serlo.
    _gv = torch.Generator().manual_seed(sem + 777)
    iv = torch.randperm(len(D['Xs_va']), generator=_gv)[:h2['LOTE_2']]
    jv = torch.randperm(len(D['Xt_va']), generator=_gv)[:h2['LOTE_2']]

    cu, br, hist_t, PS_t, diag, grad, w_cu = [], [], [], [], [], [], []
    # El indicador de RUTEO en cada punto de control: 1 donde la
    # pseudoetiqueta acierta, 0 donde se equivoca. Lo calcula
    # `pesos_oraculo`, el mismo criterio que el brazo del techo, y se
    # registra en LOS TRES BRAZOS --- no solo en el techo, que es donde
    # estaba.
    #
    # **Lee `y_t`, y por eso es INSTRUMENTO y nunca criterio**, del mismo
    # estatuto que `acc_te`: no entra en la parada, ni en la busqueda, ni
    # en el termino. Existe porque sin el no se puede contestar si `w`
    # ORDENA el mal ruteo --- un AUC de `w` contra esto --- y esa pregunta
    # separa 'la señal es la equivocada' de 'la señal es la correcta y la
    # escala no', que piden remedios distintos. Antes solo el brazo del
    # techo lo calculaba, asi que la pregunta se tenia que contestar
    # comparando DOS corridas con trayectorias distintas, lo cual no la
    # contesta.
    rut_cu = []
    # El peso que ESTE brazo multiplica de verdad, punto por punto y por punto
    # de control, al lado del declarado. Son dos registros y no uno porque
    # contestan dos preguntas: `w_cu` es siempre `1 - H2(g)/ln C` --- la
    # cantidad cuya premisa la seccion comprueba, y la que la figura dibuja, en
    # los tres brazos --- y `w_usado_cu` es lo que el brazo le paso al termino,
    # que en `apagado` son unos y en `oraculo` es el indicador de acierto.
    # Colapsarlos en uno dejaria al brazo del techo publicando una entropia que
    # no uso, o a la figura dibujando unos.
    w_usado_cu = []
    # LAS CUATRO CANTIDADES DE LA EC. 33 POR CLASE Y POR PUNTO DE CONTROL ---
    # `h_s`, `h_t`, `h_mix` y `J_c`. Una entrada por punto de control, cada una
    # un arreglo `(CAMPOS_EC33, clases vivas)`; ver `ec33_por_clase` para que
    # son y `ec33()` para leerlas por nombre.
    #
    # Es una MEDICION al lado de las que ya habia y no un cambio del termino ni
    # del criterio: nada en este modulo la lee para decidir. Existe porque el
    # criterio de parada de hoy --- el maximo de `cef` --- es una senal
    # INVENTADA para este banco, y las cuatro de abajo son terminos de la
    # formulacion que el metodo ya calcula en cada paso. Cual de ellas apunta
    # al pico es una medicion, y sin registrarlas no se puede hacer.
    #
    # Como `w_cu`, viaja en `info` como lista de ARREGLOS, lo que la deja
    # afuera del JSON del artefacto por la forma del valor y no por una lista
    # negra: el generador la excluye sola.
    ec33_cu = []
    # LA VARA CON LA QUE SE LEE LA EC. 33: una COPIA de `cr` y NUNCA `cr`.
    #
    # Esto no es prolijidad, es la unica forma de que la medicion no toque lo
    # que mide. La lectura necesita `class_mixing = 1.0` --- la Ec. 33 es el
    # termino POR CLASE, y por debajo de 1 `forward` suma ademas el bloque
    # marginal, que no es de ninguna clase --- y el bucle fija `class_mixing`
    # en cada paso desde `PASO_COND`. Hacerlo sobre la instancia del
    # entrenamiento y restaurarlo deja la corrida correcta pero su REGISTRO
    # falso: `tests/test_toy_entrega_marginal.py` espia `CREDALoss.forward`
    # por instancia y afirma que el termino del bucle es marginal PURO antes
    # de `PASO_COND` y condicional PURO despues. MEDIDO: con la instancia
    # compartida esa prueba se pone roja en dos de sus tests, porque la
    # lectura del punto de control aparece como una llamada del bucle con la
    # mezcla del otro extremo. La corrida publicaba los mismos numeros y la
    # propiedad observable dejaba de ser verdad.
    #
    # Y es una COPIA y no un `CREDALoss` nuevo porque un constructor nuevo
    # seria un segundo sitio donde se escriben las nueve banderas del termino,
    # y el que se quedara atras daria una serie de otra configuracion sin que
    # nada falle. `deepcopy` no puede divergir: no hay que mantenerla.
    cr_ec33 = copy.deepcopy(cr)
    cr_ec33.class_mixing = 1.0

    def _config_del_termino(q):
        """Las banderas escalares del termino, para comparar las dos varas."""
        return {k: v for k, v in q.__dict__.items()
                if isinstance(v, (bool, int, float, str))
                and k != 'class_mixing'}

    assert _config_del_termino(cr_ec33) == _config_del_termino(cr), (
        _config_del_termino(cr_ec33), _config_del_termino(cr))
    PASOS_GRAD = (20, PASOS_2//2, PASOS_2 - 20)
    # Estado de la parada temprana de fase 2. La senal es el MAXIMO DE `cef`
    # --- `clases_efectivas`, el numero efectivo de clases en el RUTEO del
    # blanco: `exp(H(histograma de argmax))`, en `[1, CLASES]`. Cero y_t: es el
    # histograma de las predicciones del propio modelo.
    #
    # QUE MIDE, literalmente: en cuantas clases distintas reparte el modelo al
    # blanco. `1` es el colapso --- todo a una clase --- y `CLASES` es un
    # reparto parejo.
    #
    # EL MECANISMO, que es lo que la hace criterio y no una regla con suerte:
    # el termino de adaptacion COLAPSA el ruteo del blanco si se lo deja
    # correr, y el mejor modelo esta justo ANTES de que el colapso arranque.
    # Asi que `cef` no es un proxy del acierto, es un detector del borde.
    #
    # Medido el 2026-10-16 sobre 90 celdas --- tres toysets, diez semillas,
    # tres posiciones --- es la UNICA senal que apunta al pico. Spearman con la
    # epoca del pico: `cef` `+0.612` (`p = 1.4e-10`), minimo del vigia `-0.345`
    # (`p = 0.062`), minimo de la loss `-0.011` (`p = 0.956`). Y es positiva en
    # 8 de 9 celdas (familia, normalizacion).
    #
    # ------------------------------------------------------------------
    # LO QUE HABIA ANTES, Y POR QUE SE CAYO. No se borra: era un argumento
    # medido, y lo que lo invalido fue OTRA medicion a otra escala.
    #
    # Hasta el 2026-10-16 la senal era el MINIMO DEL VIGIA ---
    # `acc(cab(enc_t(Xs_va)), ys_va)`, el encoder del blanco sobre la
    # validacion de la FUENTE, con etiquetas de la fuente. Lo que mide es
    # cuanto del comportamiento de la fuente CONSERVA el encoder del blanco, y
    # el criterio se apoyaba en una APUESTA: que el minimo de esa conservacion
    # coincide con el mejor momento del blanco.
    #
    # Lo que se habia medido a favor, y SIGUE SIENDO VERDAD de lo que midio:
    # sobre 9 celdas su argmin caia EXACTO en el pico en 4, una paciencia de 30
    # encontraba el minimo GLOBAL en 8 de 9 pese a haber hasta 11 minimos
    # locales, y le gano a las otras 27 reglas probadas.
    #
    # POR QUE ESO NO SOSTIENE EL CRITERIO DE HOY: aquello se midio POR PASO con
    # un techo de 200 pasos, ANTES de que el criterio por epoca existiera, y
    # sobre 9 celdas. El criterio de hoy lee al FINAL DE CADA EPOCA, y medido
    # por epoca sobre 90 celdas el vigia es la PEOR de las reglas probadas ---
    # peor que una constante:
    #
    #     regla                   GLOBAL  |  lunas   toroide   rollo
    #     ORACULO (techo)         0.7951  | 0.7071   0.8608   0.8175
    #     MAX cef                 0.7505  | 0.6227   0.8420   0.7868
    #     constante epoca 20      0.7048  | 0.5861   0.7601   0.7682
    #     min loss                0.6471  | 0.5379   0.8295   0.5739
    #     min vigia (lo de ayer)  0.6410  | 0.5219   0.7894   0.6117
    #     .predict                0.4086  | 0.4612   0.4192   0.3455
    #
    # Apareado sobre las 90 celdas, `MAX cef` contra el minimo del vigia:
    # `d = +0.1095`, `IC95 [+0.0796, +0.1394]`, `t = 7.18`, `p = 2.0e-10`,
    # Wilcoxon `p = 2.3e-10`, gana en 70 de 90.
    #
    # Y el agujero que ya estaba anotado aca es el que lo explica: perder la
    # fuente y adaptarse al blanco no son lo mismo. Un encoder que colapsa todo
    # a un punto TAMBIEN pierde la fuente, y el vigia bajo es compatible con
    # las dos cosas sin distinguirlas --- es decir, el vigia premiaba el
    # colapso que `cef` detecta. Medido ademas: bajo BatchNorm el vigia se da
    # vuelta (`+0.78` de correlacion con el acierto en rollo), porque ahi
    # mezcla 'los pesos se movieron' con 'los buffers se reestimaron'.
    #
    # El vigia SIGUE REGISTRANDOSE y sigue dibujandose, y no decide la parada.
    # Lo que si decide es la BUSQUEDA DE HP, que minimiza `mejor_vg`: eso no se
    # toco, ver el bloque del criterio mas abajo.
    #
    # Tampoco se borra esto, que sigue valiendo: el minimo de la loss TOTAL no
    # puede ser senal --- lleva lambda, que crece por la rampa, asi que su
    # minimo esta en el paso 0 --- y la loss cambia de SIGNO entre familias
    # (negativa con el acierto en lunas y toroide, POSITIVA en rollo), que es
    # por que ninguna regla que la minimice sirve a las tres.
    # ------------------------------------------------------------------
    #
    # El extremo importa, y para `cef` cae del otro lado que para el vigia: el
    # MAXIMO de `cef` NO es trivial en el paso 0 --- `enc_t` arranca siendo
    # `enc`, que rutea el blanco como lo haga la fuente --- mientras el ARGMAX
    # del vigia si caia trivialmente ahi.
    #
    # El mejor se rastrea solo DESDE la marca de epocas minimas, o
    # restaurariamos un modelo de antes de que la rampa prendiera.
    #
    # La ELECCION de esta senal se hizo comparando contra y_t, una vez, sobre
    # este banco. Su OPERACION no usa y_t en ningun paso. La paciencia y la
    # guarda estaban declaradas antes del hallazgo, y lo unico que cambio de
    # ellas es la DIRECCION que la paciencia cuenta: epocas sin un nuevo
    # MAXIMO.
    # Cuantas filas se reemplazaron en total. Es lo que distingue 'la
    # inyeccion corrio' de 'la bandera estaba puesta', y el techo del embedding
    # no puede existir sin ella.
    emb_total = 0
    # Cuantas filas se proyectaron al posterior uniforme. Distingue 'la
    # proyeccion corrio' de 'la bandera estaba puesta'.
    uni_total = 0
    mejor_vg, mejor_re, mejor_paso = float('inf'), float('inf'), -1
    # El criterio busca un MAXIMO, asi que su acumulador arranca en `-inf` y no
    # en `inf`. Arrancarlo en `inf` dejaria que ninguna epoca le gane nunca:
    # `mejor_paso` se quedaria en `-1`, no se guardarian pesos, y la corrida
    # publicaria el ULTIMO modelo como si fuera el elegido.
    mejor_cef = float('-inf')
    mejor_wt, paso_fin = None, PASOS_2
    # LA PASADA DE LA FASE 2. El blanco define la epoca y la fuente coincide;
    # `pasos_de_una_epoca` se niega si alguna vez no coinciden.
    n_s2, n_t2 = len(D['Xs_tr']), len(D['Xt_tr'])
    pasos_por_epoca = pasos_de_una_epoca(n_s2, n_t2, h2['LOTE_2'])
    # El bloque mas chico de la pasada es el PARCIAL del final, y el termino
    # necesita al menos dos puntos para que los kernels no degeneren. Con los
    # lotes de la rejilla el resto va de 24 a 56 puntos, asi que no muerde hoy;
    # es alcanzable (n=129 a lote 128 deja un resto de 1) y por eso esta.
    _restos = [len(b) for b in (pasada_en_lotes(n_t2, h2['LOTE_2'],
                                                torch.Generator().manual_seed(0)))]
    assert min(_restos) >= 2, (
        n_t2, h2['LOTE_2'], _restos,
        'el bloque parcial de la pasada deja menos de dos puntos')
    # Cuantos pasos del techo NO reciben lectura del criterio: el techo esta en
    # pasos y el criterio en epocas, asi que la ultima epoca puede quedar
    # incompleta. A lote 128 son 200 - 22*9 = 2 pasos, que se ENTRENAN y no se
    # leen. Queda en el registro para que nadie lo descubra midiendo.
    pasos_sin_leer = PASOS_2 - (PASOS_2 // pasos_por_epoca) * pasos_por_epoca
    lotes_s = lotes_t = None
    pasada_2, mejor_epoca = [], 0
    _vis_s, _vis_t, _sum_s, _sum_t = set(), set(), 0, 0
    for paso in range(PASOS_2 + 1):
        if paso in cuad:
            enc.eval(); cab.eval(); enc_t.eval()
            with torch.no_grad():
                Zt = enc_t(D['Xt_tr']); Zs = enc(D['Xs_tr'])
                br.append(float((mu_s0 - Zt.mean(0)).norm()) / esc_s)
                acc = float((cab(enc_t(D['Xt_va'])).argmax(1) == D['yt_va']).float().mean())
                _lg = cab(enc_t(D['Xt_tr']))
                rut = _lg.argmax(1)
                pok = float((rut == D['yt_tr']).float().mean())
                # La incertidumbre normalizada de la Ec. 24, promediada sobre el
                # blanco: U(g) = H2(g)/log C, con H2(g) = -log(sum g^2). SIN
                # ETIQUETAS -- es la unica senal de este bloque que un criterio
                # de parada puede mirar sin fugarse y_t, y no es maquinaria
                # nueva: es la que el metodo ya usa para ponderar el blanco.
                _u = incertidumbre_declarada(_lg)
                u_t = float(_u.mean())
                # El PESO POR PUNTO, `w_j = 1 - H2(g_j)/ln C`, que es lo que la
                # figura de contaminacion usa para el tamanio de cada marca.
                # Sale de `pesos_declarados`, que es el unico sitio del modulo
                # donde esa formula esta escrita.
                #
                # Y mide INCERTIDUMBRE, no error: un punto reubicado que cae
                # bien adentro de la clase equivocada tiene `w` ALTO ---
                # confiadamente equivocado --- y CRECE en la figura. Solo los
                # que caen entre clases se achican, que es la razon por la que
                # `MODO_FRONTERA` existe.
                w_cu.append(pesos_declarados(_lg))
                rut_cu.append(pesos_oraculo(_lg, D['yt_tr']))
                # Y lo que ESTE brazo multiplica. El del techo publica el
                # indicador de acierto, que es lo que le pasa al termino; el
                # apagado publica unos, porque un peso apagado ES un peso de
                # uno y registrar la entropia ahi diria que uso algo que no uso.
                if peso == PESO_ORACULO:
                    w_usado_cu.append(pesos_oraculo(_lg, D['yt_tr']))
                elif peso == PESO_ORACULO_INTRUSOS:
                    # El indicador de intrusos no depende del paso --- no se
                    # estima, se inyecto --- asi que es el MISMO vector en cada
                    # punto de control. Se publica una copia por punto igual
                    # que los otros: es lo que el brazo le paso al termino, y un
                    # registro con un solo vector obligaria al lector a saber
                    # que este peso es constante en el tiempo y los otros no.
                    w_usado_cu.append(w_intrusos.copy())
                elif peso == PESO_DECLARADO:
                    # Antes de `PASO_W` el brazo multiplico por UNOS, y el
                    # registro publica lo que uso y no lo que calculo: si
                    # publicara el declarado, la figura mostraria un sopesado
                    # que el termino nunca aplico.
                    w_usado_cu.append(
                        np.ones(len(w_cu[-1]), dtype=np.float32)
                        if paso < h2['PASO_W'] else w_cu[-1].copy())
                elif peso == PESO_BARAJADO:
                    # Los mismos valores, otra asignacion. El registro
                    # publica lo que el brazo USO, que aca es la
                    # permutacion y no el vector ordenado.
                    _rb = np.random.default_rng(sem * 100043 + paso)
                    w_usado_cu.append(
                        w_cu[-1][_rb.permutation(len(w_cu[-1]))].copy())
                elif peso == PESO_COMPUERTA:
                    w_usado_cu.append(pesos_compuerta(_lg, h2['Q_COMPUERTA']))
                elif peso == PESO_ORACULO_RUIDOSO:
                    w_usado_cu.append(pesos_oraculo_ruidoso(
                        _lg, D['yt_tr'], h2['PRECISION_TECHO'],
                        sem * 100003 + paso))
                else:
                    w_usado_cu.append(np.ones(len(w_cu[-1]), dtype=np.float32))
                # la fuente tambien se mueve ahora, asi que su acierto es una
                # serie y no un numero: sin esto, soltarla podria degradarla y
                # el cuaderno mostraria solo la mejora del blanco.
                a_s = float((cab(enc(D['Xs_va'])).argmax(1) == D['ys_va']).float().mean())
                # TEST en cada punto de control, PURO REPORTE. Antes se leia una
                # sola vez al final; ahora tambien a lo largo, para que el pico
                # y el paso elegido esten en el MISMO split --- si el pico fuera
                # de valid y el elegido de test, la brecha entre los dos
                # mezclaria 'la parada no acerto' con 'valid no es test' y no se
                # podria leer como ninguna de las dos.
                #
                # Nada lo lee para decidir: la parada mira el Renyi de valid, la
                # busqueda minimiza el Renyi de valid, y la epoca de fase 1 la
                # elige el acierto de la FUENTE. Que `y_t` se use para MEDIR el
                # resultado no es fuga; usarlo para ELEGIR si lo seria.
                acc_te_ck = float((cab(enc_t(D['Xt_te'])).argmax(1)
                                   == D['yt_te']).float().mean())
                # LA EC. 33 POR CLASE, sobre el material que este bloque YA
                # leyo: `Zs` es el train de la fuente en representacion, `Zt`
                # el del blanco y `_lg` sus logits. Nada vuelve a pasar por
                # los encoders, asi que la lectura no puede diferir del
                # modelo que el resto del punto de control describe.
                #
                # AQUI y NO en el cuerpo del paso. El termino se evalua en
                # cada paso y otra vez sobre el lote de valid, asi que un
                # registro por paso serian 54.861 lecturas en una corrida de
                # 200 epocas --- MEDIDO --- y una serie que no se puede
                # alinear con `cu`.
                #
                # Y bajo el `eval()` que este bloque ya fijo arriba, que es la
                # mitad que importa: un forward en `train()` reestima las
                # corridas de un BatchNorm SIN gradiente, asi que la medicion
                # moveria la epoca que el criterio elige. Este modulo ya lo
                # pago dos veces --- `acc_s1` y la senal del criterio --- y
                # las dos veces el sintoma fue otro numero y ningun error.
                # Con `cr_ec33`, la copia, y por eso se afirma en CADA punto
                # de control que las dos varas siguen pidiendo lo mismo: la
                # copia se saca antes del bucle, asi que una bandera que el
                # bucle empezara a mover manana dejaria la serie midiendo una
                # configuracion que el entrenamiento ya no usa.
                assert _config_del_termino(cr_ec33) == _config_del_termino(cr), (
                    paso, _config_del_termino(cr_ec33),
                    _config_del_termino(cr))
                ec33_cu.append(ec33_por_clase(cr_ec33, Zs, D['ys_tr'], Zt,
                                              F.softmax(_lg, 1)))
            _rec = (paso, proy(Zt.numpy()), acc, pok, rut.numpy(), a_s, u_t,
                    acc_te_ck)
            assert len(_rec) == CAMPOS_CU, (len(_rec), CAMPOS_CU)
            cu.append(_rec)
            PS_t.append(proy(Zs.numpy()))
            # d/sigma y K_st/K_s: distinguen 'el termino no ayudo' de 'el termino
            # no estaba midiendo'. Lote fijo para que la serie sea comparable.
            with torch.no_grad():
                i0 = torch.randperm(len(D['Xs_tr']),
                                    generator=torch.Generator().manual_seed(7))[:LOTE]
                j0 = torch.randperm(len(D['Xt_tr']),
                                    generator=torch.Generator().manual_seed(7))[:LOTE]
                a0, b0 = enc(D['Xs_tr'][i0]), enc_t(D['Xt_tr'][j0])
                dd = torch.cdist(a0, a0); sg = float(dd[dd > 0].median())
                kk = lambda U, W: float(torch.exp(-torch.cdist(U, W)**2/(2*sg**2)).mean())
                diag.append((paso, float((a0.mean(0) - b0.mean(0)).norm())/sg,
                             kk(a0, b0)/kk(a0, a0)))
            enc.train(); cab.train(); enc_t.train()
        if paso == PASOS_2: break
        # Si la fuente no se mueve, va en eval(): en train() un BatchNorm
        # reestima running_mean/running_var en el FORWARD, sin gradiente, y
        # entonces el encoder queda congelado en pesos y moviendose en
        # comportamiento. Medido: la CE sobre el source completo derivaba
        # 3.5e-02 con ||dtheta|| = 0 exacto.
        if mover_fuente: enc.train(); cab.train()
        else:            enc.eval();  cab.eval()
        enc_t.train()
        # EL LOTE SALE DE LA PASADA, no de un sorteo nuevo. `k_ep` es la
        # posicion dentro de la epoca y la permutacion se vuelve a sortear
        # SOLO al abrirla, que es lo que hace que cada punto salga exactamente
        # una vez por epoca. Antes estas dos lineas eran
        # `randperm(n)[:LOTE_2]`, de cero en cada paso --- muestreo con
        # reemplazo entre pasos, sin pasada y por lo tanto sin epoca.
        epoca, k_ep = divmod(paso, pasos_por_epoca)
        if k_ep == 0:
            lotes_s = pasada_en_lotes(n_s2, h2['LOTE_2'], gen)
            lotes_t = pasada_en_lotes(n_t2, h2['LOTE_2'], gen)
            _vis_s, _vis_t, _sum_s, _sum_t = set(), set(), 0, 0
        i, j = lotes_s[k_ep], lotes_t[k_ep]
        # Lo que la epoca DIBUJO de verdad, contado donde se consume y no donde
        # se reparte: `pasada_en_lotes` puede devolver una particion impecable
        # y el bucle consumir dos veces el mismo bloque, y ese es justo el
        # cruce que una prueba de la funcion sola no atraviesa.
        _vis_s |= {int(q) for q in i}; _sum_s += len(i)
        _vis_t |= {int(q) for q in j}; _sum_t += len(j)
        # con gradiente si la fuente se mueve; si no, el forward igual corre
        # porque el termino de Renyi necesita f_s para comparar contra f_t
        f_s = enc(D['Xs_tr'][i]) if mover_fuente else enc(D['Xs_tr'][i]).detach()
        f_t = enc_t(D['Xt_tr'][j])
        # La inyeccion va AQUI y el lugar es la decision entera: `enc_t` ya
        # corrio, asi que su BatchNorm vio solo puntos limpios, y sigma esta
        # congelado desde la fuente. De los tres canales por los que un atipico
        # hacia daño queda abierto UNO: el bloque de clase sopesado, que es el
        # que `w` multiplica.
        #
        # Y va ANTES de `g_t`, no despues: la cabeza tiene que VER el vector
        # atipico, o `w` se calcularia desde el punto limpio que ya no esta y el
        # experimento mediria otra cosa.
        #
        # Sorteo fresco por paso --- `(semilla, paso)` --- que es el diseño del
        # operador: un conjunto fijo se puede rutear por alrededor.
        emb_mal = None
        # La proyeccion va ANTES del empuje por rayo y antes de `g_t`, por la
        # misma razon: la cabeza tiene que VER el punto movido, o `w` se
        # calcularia desde el punto que ya no esta.
        if h2['RHO_UNI'] > 0.0:
            f_t, uni_mal = uniformar_el_embedding(
                f_t, cab, h2['RHO_UNI'], sem * 100057 + paso,
                desplazamiento=h2['DESPL_UNI'])
            uni_total += int(uni_mal.sum())
        if h2['RHO_EMB'] > 0.0:
            f_t, emb_mal = intrusar_el_embedding(
                f_t, h2['RHO_EMB'], h2['RADIO_EMB'], sem * 100019 + paso)
            emb_total += int(emb_mal.sum())
        # g_t se detacha: es el ruteo con el que el termino decide en que bloque
        # entra cada punto del blanco, no un camino para el gradiente.
        with torch.no_grad(): g_t = F.softmax(cab(f_t), 1)
        if peso == PESO_ORACULO_EMB:
            # El techo de la inyeccion, por SELECCION como los otros binarios.
            # Falla cerrado: un brazo que no supiera quienes son pondera por
            # unos y seria el PISO corriendo bajo el rotulo de un techo --- dos
            # brazos identicos con dos nombres distintos, y la tabla diria que
            # quitar los intrusos no compra nada sin que nada falle.
            assert emb_mal is not None, (
                peso, 'el techo del embedding necesita una inyeccion; con '
                'RHO_EMB=0 pondera por unos y es el piso')
            queda_e = ~emb_mal
            assert int(queda_e.sum()) >= MIN_ORACULO, (
                paso, int(queda_e.sum()), MIN_ORACULO,
                'el techo del embedding dejo el bloque por debajo del minimo')
            f_t = f_t[queda_e]
            g_t = g_t[queda_e]
        if peso == PESO_ORACULO:
            # El peso de ORACULO, `w_j in {0, 1}`, aplicado por SELECCION y no
            # por multiplicacion. Las dos cosas son la misma aca, y la
            # seleccion es la que no obliga a reescribir el termino:
            #
            #   `CREDALoss` calcula `target_weights` ADENTRO de `forward`, a
            #   partir de `g_t`, y no acepta un vector de pesos. Inyectar un
            #   indicador exigiria reimplementar `forward` en este modulo, que
            #   es exactamente la bifurcacion silenciosa de la formulacion que
            #   no se hace: el arnes llama a los modulos del metodo y nunca los
            #   reescribe.
            #
            #   Con `w` binario, `K~_t = K_t * outer(w,w)` y
            #   `K~_st = K_st * diag(w)` son el kernel del SUBCONJUNTO con
            #   filas y columnas de ceros alrededor, y un cero no aporta ni a
            #   `tr(M)` ni a `tr(M^2)`, asi que las tres entropias de Renyi y
            #   el balance por traza (`BALANCE_DE_MASA='trace'`) dan lo mismo
            #   sobre el subconjunto. Las cotas tambien: `FORMA_DE_COTAS` es
            #   `entropy-cosh`, que sale de las dos entropias marginales y no
            #   de las cardinalidades, de modo que dejar fuera los puntos de
            #   peso cero no mueve el mapa a [0,1].
            #
            # Y LEE `y_t`: es un techo, no un metodo. Ver `pesos_oraculo`.
            acierta = (g_t.argmax(1) == D['yt_tr'][j])
            # Para y no degrada: un bloque de un punto devolveria un numero que
            # no es el del termino, y un brazo que lo reportara seria peor que
            # un brazo que no corrio.
            assert int(acierta.sum()) >= MIN_ORACULO, (
                paso, int(acierta.sum()), MIN_ORACULO,
                'el oraculo dejo el bloque del blanco por debajo del minimo')
            f_t = f_t[acierta]
            g_t = g_t[acierta]
        elif peso == PESO_ORACULO_RUIDOSO:
            # El MISMO mecanismo de seleccion que el oraculo, con el mismo
            # argumento ya escrito ahi: con `w` binario el kernel ponderado ES
            # el del subconjunto, y un cero no aporta a `tr(M)` ni a `tr(M^2)`.
            #
            # El sorteo se siembra con `(semilla, paso)` y no con la semilla
            # sola: el conjunto mal ruteado CAMBIA en cada paso --- es la
            # pseudoetiqueta, que se mueve --- asi que un sorteo fijo elegiria
            # los sacrificados de un paso para todos los demas. Y no se siembra
            # al azar, porque entonces el delta de un peldaño llevaria adentro
            # el ruido del sorteo en vez de medir precision.
            #
            # LEE `y_t`: es un techo, no un metodo.
            _wr = pesos_oraculo_ruidoso(cab(f_t).detach(), D['yt_tr'][j],
                                        h2['PRECISION_TECHO'],
                                        sem * 100003 + paso)
            queda_r = torch.as_tensor(_wr > 0.5)
            assert int(queda_r.sum()) >= MIN_ORACULO, (
                paso, int(queda_r.sum()), MIN_ORACULO, h2['PRECISION_TECHO'],
                'el techo ruidoso dejo el bloque del blanco por debajo del '
                'minimo')
            f_t = f_t[queda_r]
            g_t = g_t[queda_r]
        elif peso == PESO_COMPUERTA:
            # La compuerta, por SELECCION y por las mismas razones que el
            # oraculo de arriba: con `w` binario `K~_t = K_t * outer(w,w)` y
            # `K~_st = K_st * diag(w)` son el kernel del SUBCONJUNTO, un cero no
            # aporta ni a `tr(M)` ni a `tr(M^2)`, y `FORMA_DE_COTAS` sale de las
            # dos entropias marginales y no de las cardinalidades. Asi que
            # seleccionar es multiplicar, sin reescribir `CREDALoss.forward`.
            #
            # **Y aqui NO se pondera encima**: `sopesado` es False para este
            # brazo, por el mismo `sopesado = (peso == PESO_DECLARADO)` que
            # apaga la ponderacion en los techos. Tiene que ser asi o el brazo
            # no contesta la pregunta: lo que se compara contra el oraculo es
            # una compuerta {0,1}, no una compuerta Y un peso continuo encima.
            # Con las dos, un resultado positivo no diria cual de las dos
            # operaciones lo compro.
            #
            # **No lee `y_t`**: el umbral es el cuantil `Q_COMPUERTA` del propio
            # peso declarado DENTRO de este lote. Es un metodo, y por eso esta
            # en `PESOS_BUSCABLES`.
            # Sin gradiente, por la misma razon que `g_t` se detacha: la
            # compuerta DECIDE que puntos entran al bloque, y una
            # decision de pertenencia no es un camino por el que el
            # termino deba empujar al encoder.
            with torch.no_grad():
                _w = pesos_compuerta(cab(f_t), h2['Q_COMPUERTA'])
            pasa = torch.as_tensor(_w > 0.5)
            # La misma guarda que los techos, y por la misma razon: un bloque
            # de un punto devuelve un numero que no es el del termino.
            assert int(pasa.sum()) >= MIN_ORACULO, (
                paso, int(pasa.sum()), MIN_ORACULO, h2['Q_COMPUERTA'],
                'la compuerta dejo el bloque del blanco por debajo del minimo')
            f_t = f_t[pasa]
            g_t = g_t[pasa]
        elif peso == PESO_ORACULO_INTRUSOS:
            # El MISMO mecanismo de seleccion que el oraculo de arriba, y por
            # las mismas razones: con `w` binario `K~_t = K_t * outer(w,w)` y
            # `K~_st = K_st * diag(w)` son el kernel del SUBCONJUNTO con filas y
            # columnas de ceros alrededor, un cero no aporta ni a `tr(M)` ni a
            # `tr(M^2)`, y `FORMA_DE_COTAS = 'entropy-cosh'` sale de las dos
            # entropias marginales y no de las cardinalidades. Asi que
            # seleccionar es multiplicar, sin reescribir `CREDALoss.forward` ---
            # que es la bifurcacion silenciosa de la formulacion que el arnes no
            # hace.
            #
            # Lo que este techo NO puede quitar, y se dice porque es el limite
            # de lo que mide: los intrusos ya pasaron por `enc_t`, asi que el
            # BatchNorm del blanco ya vio sus estadisticas de lote. El peso vive
            # en el termino y la contaminacion del BatchNorm vive antes; por eso
            # `FACTOR_INTRUSO` se alcanza con el MENOR radio que lo cumple.
            #
            # Y LEE quienes se inyectaron: es un techo, no un metodo.
            queda = no_es_intruso[j]
            assert int(queda.sum()) >= MIN_ORACULO, (
                paso, int(queda.sum()), MIN_ORACULO,
                'el techo de intrusos dejo el bloque del blanco por debajo '
                'del minimo')
            f_t = f_t[queda]
            g_t = g_t[queda]

        # La Ec. 21 tambien aca. En el regimen DECLARADO (`mover_fuente=False`)
        # este termino no tiene camino de gradiente a ningun parametro
        # entrenable --- `f_s` llega detachado y `cab` esta en eval() con sus
        # pesos congelados --- asi que lo unico que cambia es el numero que se
        # registra y se dibuja, ahora en la misma escala [0,1) que el de Renyi.
        # En la fila de diagnostico (`mover_fuente=True`) si tiene gradiente, y
        # ahi el cambio de escala se ve en |g_renyi|/|g_ce|: ver `perdida_fuente`.
        t_ce = perdida_fuente(cab(f_s), D['ys_tr'][i])
        # El control barajado: los MISMOS valores de `w`, reasignados al
        # azar entre los puntos de ESTE lote. Sorteo fresco por paso
        # --- `(semilla, paso)` --- porque una permutacion fija seria una
        # asignacion estable que el entrenamiento podria aprovechar, y lo
        # que el control tiene que destruir es justamente la
        # correspondencia entre punto y peso.
        # El retardo: antes de `PASO_W` el termino multiplica por unos, sea
        # cual sea el brazo. Se implementa pasando `target_weights` explicito
        # en vez de apagar `use_entropy_weighting`, porque esa bandera se fija
        # al construir el loss y el retardo es por PASO.
        # La ENTREGA, antes del sopesado porque decide QUE termino corre: el
        # marginal mientras las pseudoetiquetas no valgan nada, la condicional
        # despues. Los dos extremos, nunca una mezcla.
        cr.class_mixing = 0.0 if paso < h2['PASO_COND'] else 1.0
        _retardado = paso < h2['PASO_W']
        w_iny = None
        if _retardado and sopesado:
            w_iny = torch.ones(len(f_t))
        elif peso == PESO_BARAJADO:
            with torch.no_grad():
                _wd = torch.as_tensor(pesos_declarados(cab(f_t)))
            _rb = np.random.default_rng(sem * 100043 + paso)
            w_iny = _wd[torch.as_tensor(_rb.permutation(len(_wd)).copy())]
        t_re = cr(f_s, f_t, D['ys_tr'][i], g_t,
                          target_weights=w_iny)
        # La fuente aporta su CE y nada mas; el blanco entra solo por el
        # termino de Renyi. No hay termino de fuente: se retiraron los tres.
        term = t_ce + t_re
        if mover_fuente and paso in PASOS_GRAD:
            # Con que fuerza cada termino tira del encoder FUENTE. La pregunta no
            # se contesta con los pesos nominales: las magnitudes de gradiente
            # podrian compensarlos, asi que hay que medirlas.
            gn = {}
            for nom, tt in (('renyi', t_re), ('ce', t_ce)):
                gg = torch.autograd.grad(tt, list(enc.parameters()),
                                         retain_graph=True, allow_unused=True)
                gn[nom] = float(torch.cat([(x if x is not None else torch.zeros(1))
                                           .reshape(-1) for x in gg]).norm())
            grad.append((paso, 1.0, gn['renyi'], gn['ce']))
        opt.zero_grad(); term.backward(); opt.step()
        # Las senales de VALID, sobre el lote fijo, DESPUES del paso del
        # optimizador. Antes se median ANTES, y el snapshot de pesos se toma
        # despues: la senal describia un modelo distinto del que se guarda,
        # desfasado un paso. Medido al arreglarlo, eso valia +0.015 a +0.029 en
        # las tres celdas de lunas, que es la familia donde el vigia discrimina
        # menos --- el desfase importa cuando el criterio esta indeciso.
        #
        # El Renyi va SIN la rampa. El vigia es el encoder del BLANCO sobre la
        # fuente etiquetada: legal, usa y_s y nunca y_t.
        enc_t.eval()
        with torch.no_grad():
            fv_s = enc(D['Xs_va'][iv]); fv_t = enc_t(D['Xt_va'][jv])
            gv_t = F.softmax(cab(fv_t), 1)
            re_val = float(cr(fv_s, fv_t, D['ys_va'][iv], gv_t))
            vigia = float((cab(enc_t(D['Xs_va'])).argmax(1)
                           == D['ys_va']).float().mean())
        enc_t.train()
        _rt = (paso, float(t_ce.detach()), float(t_re.detach()),
               float(term.detach()), re_val, vigia)
        assert len(_rt) == CAMPOS_HT, (len(_rt), CAMPOS_HT)
        hist_t.append(_rt)
        # LA PARADA TEMPRANA DE FASE 2, AL FINAL DE CADA EPOCA. Declarada:
        # `PACIENCIA_2_EPOCAS` epocas sin mejorar el minimo del vigia, con
        # `MIN_EPOCAS_2` de guarda. Guarda los PESOS del mejor y los restaura
        # al salir --- quedarse con los ultimos y solo reportar la epoca seria
        # el mismo defecto que la fase 1 ya tenia.
        #
        # Se lee AL FINAL DE LA EPOCA y en ningun otro momento, que es la otra
        # mitad del arreglo. Antes se leia por PASO, o sea sobre un modelo que
        # habia visto un lote arbitrario en vez de uno que completo una pasada;
        # con el sorteo con reemplazo que habia, ese lote podia incluso repetir
        # puntos del paso anterior.
        if k_ep == pasos_por_epoca - 1:
            # La bitacora de la pasada, antes del criterio: es la evidencia de
            # que la epoca FUE una pasada, y se afirma aca para que una epoca
            # que no cubrio el train no llegue nunca a decidir nada.
            # LA SENAL DEL CRITERIO, al final de la epoca y en `eval()`: el
            # ruteo del blanco bajo la normalizacion que el modelo usa en
            # inferencia. En `train()` un BatchNorm normalizaria por el ultimo
            # lote, asi que el histograma seria de un modelo que nadie reporta.
            # `no_grad()` solo no alcanza --- apaga el gradiente, no BatchNorm
            # --- y es el mismo defecto que `acc_s1` ya tuvo.
            enc_t.eval()
            with torch.no_grad():
                cef = clases_efectivas(cab(enc_t(D['Xt_tr'])).argmax(1))
            enc_t.train()
            _fila_p2 = (epoca, pasos_por_epoca, len(_vis_s), len(_vis_t),
                        _sum_s, _sum_t, cef)
            assert len(_fila_p2) == CAMPOS_P2, (len(_fila_p2), CAMPOS_P2)
            pasada_2.append(_fila_p2)
            assert len(_vis_t) == n_t2 and _sum_t == n_t2, (
                epoca, len(_vis_t), _sum_t, n_t2,
                'la epoca no dibujo cada punto del train del blanco '
                'exactamente una vez')
            assert len(_vis_s) == n_s2 and _sum_s == n_s2, (
                epoca, len(_vis_s), _sum_s, n_s2,
                'la epoca no dibujo cada punto del train de la fuente '
                'exactamente una vez')
            if epoca + 1 >= MIN_EPOCAS_2:
                # EL VIGIA SIGUE SIENDO EL MINIMO, y no decide nada. Se sigue
                # rastreando porque `mejor_vg` es lo que la BUSQUEDA DE HP
                # minimiza --- `busqueda_fase2` devuelve `mean(mejor_vg)` ---
                # asi que redefinirlo como 'el vigia en la epoca elegida'
                # habria cambiado el objetivo de la busqueda sin tocar una
                # linea de la busqueda. El criterio de PARADA cambio; el
                # objetivo de la BUSQUEDA no se toco, porque nadie lo pidio.
                # `mejor_re` queda apareado con ESE minimo, como estaba.
                if vigia < mejor_vg:
                    mejor_vg, mejor_re = vigia, re_val
                # EL CRITERIO, Y SU DIRECCION. Se busca el MAXIMO de `cef` ---
                # MAXIMO, no minimo, y la paciencia cuenta epocas sin un NUEVO
                # MAXIMO. Era el minimo del vigia hasta el 2026-10-16.
                #
                # El signo es el peligro de este bloque: invertir la
                # comparacion no rompe nada que se vea. La corrida termina,
                # publica una epoca y un numero, y el unico sintoma es que la
                # tabla empeora. Lo que lo atrapa es el lock de contenido de
                # `tests/test_creda_criterio_cef.py`, que REPLICA la regla
                # sobre la columna `cef` de `pasada_2` y exige `mejor_paso` y
                # `mejor_cef` bit a bit.
                if cef > mejor_cef:
                    mejor_cef = cef
                    mejor_paso, mejor_epoca = paso + 1, epoca + 1
                    mejor_wt = {k: v.detach().clone()
                                for k, v in enc_t.state_dict().items()}
                elif (mejor_epoca >= 1
                      and (epoca + 1) - mejor_epoca >= PACIENCIA_2_EPOCAS):
                    paso_fin = paso + 1
                    break

    # eval() ANTES de medir, no solo no_grad(): no_grad apaga el gradiente y no
    # apaga BatchNorm. En train() un BatchNorm normaliza por el lote que recibe
    # -- estadisticas que el modelo nunca usa en inferencia -- y reestima sus
    # corridas en el forward. El ultimo checkpoint dejaba los tres en train(),
    # asi que acc_s1, que es la columna 'fuente final' de la tabla, se estaba
    # midiendo bajo otra normalizacion que la que el modelo usa.
    # Se vuelve al mejor paso que la parada eligio. Igual que en la fase 1: un
    # state_dict que no se restaura deja TODO verde, porque el paso reportado
    # seria el mejor y los pesos los del ultimo.
    if mejor_wt is not None:
        enc_t.load_state_dict(mejor_wt)
    enc.eval(); cab.eval(); enc_t.eval()
    with torch.no_grad():
        Ht_fin = enc_t(D['Xt_tr']).numpy()
        acc_s1 = float((cab(enc(D['Xs_va'])).argmax(1) == D['ys_va']).float().mean())
        disp_t = float(enc_t(D['Xt_tr']).std(0).norm())
        # TEST, la UNICA lectura de todo el modulo. No entra en ningun criterio
        # ni en ninguna busqueda: es el numero que la tabla publica, medido en el
        # paso que la parada eligio. Las curvas y los GIF usan valid.
        acc_te = float((cab(enc_t(D['Xt_te'])).argmax(1) == D['yt_te']).float().mean())
        acc_s_te = float((cab(enc(D['Xs_te'])).argmax(1) == D['ys_te']).float().mean())
        # El vigia en el punto donde paro, para la columna de reporte.
        vigia_fin = float((cab(enc_t(D['Xs_va'])).argmax(1)
                           == D['ys_va']).float().mean())
    # el control vale solo si la fuente de verdad quedo quieta, y el brazo libre
    # solo si de verdad se movio. Sin esto las dos filas pueden ser la misma.
    movio = float((torch.cat([q.detach().reshape(-1) for q in enc.parameters()])
                   - theta_s0).norm())
    # Los BUFFERS tambien: medir solo parameters() dejaba pasar un BatchNorm que
    # reestima sus corridas en el forward. Un encoder cuyos buffers se movieron
    # no esta quieto, aunque sus pesos no hayan cambiado un bit.
    movio_buf = float((_bufs(enc) - buf_s0).norm())
    assert (movio > 1e-6) == mover_fuente, (mover_fuente, movio)
    assert mover_fuente or movio_buf == 0.0, ('buffers de la fuente movidos', movio_buf)
    tr_e, tr_d = entre_dentro(Ht_fin, D['yt_tr'].numpy())
    # La ALINEACION, que es lo que el sopesado promete. Se mide con `w = 1`
    # para los DOS brazos --- ver `alineacion` --- porque puntuar cada brazo
    # con su propio objetivo seria comparar dos funciones distintas.
    with torch.no_grad():
        _lg_t = cab(enc_t(D['Xt_tr']))
    colap = colapso(Ht_fin, torch.softmax(_lg_t, 1))
    align = alineacion(
        Hs_t.numpy(), Ht_fin, D['ys_tr'], _lg_t, D['yt_tr'].numpy(),
        sigma=sigma_fijo,
        cfg=dict(normalize_by_bounds=NORMALIZAR_POR_COTAS,
                 bounds_form=FORMA_DE_COTAS,
                 weight_cross_block=h2['CRUZADO_SOPESADO'],
                 detach_target_entropy=h2['RHO_T_SIN_GRAD'],
                 unbiased_purity=h2['PUREZA_U'],
                 balance_mix_blocks=balance,
                 balance_mass=BALANCE_DE_MASA))
    # Lo que cuesta congelar la vara: cuanto se movio la escala del blanco
    # respecto del espacio en que sigma se calibro, y que fraccion del kernel
    # del blanco quedo en cero por eso. Un gradiente muerto no se ve en el
    # loss --- se ve aca.
    escala_t_fin = sigma_de_la_fuente(Ht_fin)
    sigma_deriva = escala_t_fin / escala_t0
    with torch.no_grad():
        _d2 = _HEURISTICO_DE_SIGMA._squared_euclidean_dist(
            torch.as_tensor(Ht_fin), torch.as_tensor(Ht_fin))
        _k = torch.exp(-_d2 / (2.0 * sigma_fijo ** 2))
        _fuera = ~torch.eye(_k.shape[0], dtype=bool)
        k_t_saturada = float((_k[_fuera] < 1e-6).float().mean())
    # `epoca_fin` son las epocas COMPLETAS que corrieron, y coincide con
    # `paso_fin // pasos_por_epoca` en los dos casos: si el criterio corto, en
    # el final de la epoca `mejor_epoca + PACIENCIA_2_EPOCAS`; si no corto,
    # porque el techo en pasos admite exactamente esas epocas enteras.
    epoca_fin = len(pasada_2)
    assert epoca_fin == paso_fin // pasos_por_epoca, (
        epoca_fin, paso_fin, pasos_por_epoca)
    assert mejor_paso < 0 or mejor_paso % pasos_por_epoca == 0, (
        mejor_paso, pasos_por_epoca,
        'el criterio eligio un paso que no es el final de una epoca')
    # LA CEF POR PUNTO DE CONTROL, DERIVADA de lo que `cu` ya guarda. No se
    # recalcula del modelo: `rut` ES el argmax sobre el train del blanco en ese
    # punto, asi que recomputarlo seria una segunda via para el mismo numero y
    # la que se quedara atras no daria error.
    #
    # La serie es del PUNTO DE CONTROL y el criterio lee el FIN DE EPOCA; las
    # dos rejillas no coinciden (`CUADROS` va cada 10 pasos, la epoca cae cada
    # `pasos_por_epoca`). Por eso la evidencia de la ELECCION es la columna
    # `cef` de `pasada_2` y esta serie es para dibujar.
    cef_cu = [clases_efectivas(punto(cu, j)['rut']) for j in range(len(cu))]
    info = dict(acc_s=acc_s0, acc_s_fin=acc_s1, ep_s=ep_s,
                acc_te=acc_te, acc_s_te=acc_s_te, paso_fin=paso_fin,
                mejor_paso=mejor_paso, mejor_re=mejor_re, mejor_vg=mejor_vg,
                # El numero que el criterio ELIGIO, y la serie que lo muestra.
                mejor_cef=mejor_cef, cef_cu=cef_cu,
                vigia_fin=vigia_fin,
                # LA PASADA Y EL CRITERIO, para que la propiedad se LEA del
                # registro en vez de creerle al codigo. `pasada_2` trae una
                # fila por epoca completa con cuantos puntos distintos dibujo
                # y cuantos en total, en los dos dominios: las dos mitades,
                # porque `distintos == n` sola admite que un punto salga dos
                # veces y otro se caiga.
                pasos_por_epoca=pasos_por_epoca,
                epoca_definida_por='blanco',
                bloques_s=-(-n_s2 // h2['LOTE_2']),
                bloques_t=-(-n_t2 // h2['LOTE_2']),
                min_epocas_2=MIN_EPOCAS_2,
                paciencia_epocas=PACIENCIA_2_EPOCAS,
                mejor_epoca=mejor_epoca, epoca_fin=epoca_fin,
                pasos_sin_leer=pasos_sin_leer,
                pasada_2=pasada_2,
                cond_s=cond_s, dim=dim_ef, disp_s=disp_s, disp_t=disp_t,
                tr_e=tr_e, tr_d=tr_d, hist_s=hist_s, hist_t=hist_t, PS_t=PS_t,
                align=align,
                colapso=colap,
                diag=diag, grad=grad, pos=pos, semilla=sem, balance=balance,
                mover_fuente=mover_fuente, movio_fuente=movio,
                movio_buffers=movio_buf,
                # La bandera del sopesado y los HP con los que esta corrida
                # CALCULO, no los que alguien quiso pasarle. Es la diferencia
                # entre medir y prometer: el barrido de contaminacion afirma
                # que los dos brazos comparten UNA busqueda de fase 2, y lo
                # unico que puede sostener eso es lo que cada corrida registra
                # de si misma.
                sopesado=sopesado, peso=peso, hp1=dict(h1), hp2=dict(h2),
                # El peso por punto en cada punto de control, sobre `Xt_tr`.
                # Fuera del JSON del artefacto principal a proposito: son
                # un float por punto del train del blanco y por punto de
                # control --- 1080 en lunas y 1079 en toroide y rollo, MEDIDO:
                # lunas sortea 360/360/360 y el 60% estratificado cae entero,
                # las otras dos sortean 363/384/332 y el piso por clase pierde
                # un punto --- y ese artefacto no tiene
                # ningun lector para ellos. El barrido de contaminacion
                # persiste los suyos en su propio .npz.
                # El ancho de banda que los kernels de VERDAD vieron, la
                # representacion de la que salio, y lo que costo congelarlo.
                # `Hs_fase1` deja que un lector recalcule sigma y verifique que
                # es el estadistico declarado, en vez de creerle al campo.
                sigma=sigma_fijo, Hs_fase1=Hs_t.numpy(), rut_cu=rut_cu,
                emb_inyectados=emb_total, uni_inyectados=uni_total,
                sigma_deriva=sigma_deriva, k_t_saturada=k_t_saturada,
                w_cu=w_cu, w_usado_cu=w_usado_cu,
                # Las cuatro cantidades de la Ec. 33 por clase y por punto de
                # control. Viaja como `hist_t`: por `info`, para el llamador.
                ec33_cu=ec33_cu)
    assert len(ec33_cu) == len(cu), (
        len(ec33_cu), len(cu), 'un punto de control sin su Ec. 33')
    assert all(np.asarray(A).shape[0] == CAMPOS_EC33 for A in ec33_cu), (
        [np.asarray(A).shape for A in ec33_cu], CAMPOS_EC33)
    assert len(w_cu) == len(cu), (len(w_cu), len(cu),
                                  'un punto de control sin sus pesos')
    assert len(w_usado_cu) == len(cu), (len(w_usado_cu), len(cu),
                                        'un punto de control sin el peso que '
                                        'el brazo uso')
    return proy(Hs), cu, pred_s, br, info


# (K_mix por bloques, la fuente se mueve) -> etiqueta de la fila
# El eje de FIGURAS dentro de un dataset: cada entrada es un cuadro de 3x4 con
# las tres normalizaciones como filas. Cuatro campos y no dos:
#
#   balance   la bandera de K_mix, hoy siempre True
#   mover     si la fase 2 suelta la fuente y la cabeza
#   etiqueta  lo que dice el titulo
#   clave     el nombre del archivo del GIF
#
# La clave es EXPLICITA y ya no se deriva de `mover`: en el cuaderno de una capa
# las tres figuras tienen la fuente suelta, y una clave derivada de `mover` les
# daria el mismo nombre a las tres --- dos GIF pisados en silencio.
FILAS = [
    (True, False, 'fuente y cabeza quietas', 'quieta'),
]
TABLA = [
    (True, False, 'fuente y cabeza quietas', 'quieta'),
]
# El primer campo de cada fila es `balance`, y `bounds_form='entropy-cosh'` solo
# esta demostrado sobre la mezcla BALANCEADA de la Ec. 32 --- `CREDALoss` lo
# refuta en su constructor. Esa refutacion llegaria dentro de `corrida`, o sea a
# los doce minutos de una corrida completa y despues de gastar las dos busquedas
# de Optuna. Aca se cobra al IMPORTAR el modulo, que es cuando todavia no cuesta
# nada: una fila con `balance=False` junto a estas banderas es una declaracion
# incoherente, no un brazo del barrido.
def verificar_coherencia(filas, tabla, normalizar, forma, balance_masa):
    """Que las banderas y las filas del barrido pidan lo mismo.

    Una funcion y no un bloque suelto para que se pueda probar con entradas
    incoherentes: un `assert` a nivel de modulo solo se puede comprobar
    leyendo el codigo, y una constante que nadie lee se ve igual que verde.
    """
    if not (normalizar and forma == 'entropy-cosh'):
        return
    incoherentes = [f for f in list(filas) + list(tabla) if not f[0]]
    if incoherentes:
        raise ValueError(
            f"{len(incoherentes)} fila(s) piden balance=False con "
            f"forma='entropy-cosh', que solo vale sobre la mezcla balanceada "
            f"de la Ec. 32: {incoherentes}")
    if balance_masa != 'trace':
        raise ValueError(
            f"forma='entropy-cosh' exige balance_masa='trace', "
            f"no {balance_masa!r}")


verificar_coherencia(FILAS, TABLA, NORMALIZAR_POR_COTAS, FORMA_DE_COTAS,
                     BALANCE_DE_MASA)
# La fuente suelta se retiro como brazo. Medido sobre 45 pares --- y APAREADOS,
# que es lo que el diseno permite porque las dos filas comparten semilla, datos
# y los pesos de la fase 1 bit a bit: media +0.0026, error estandar 0.0067,
# IC 95% [-0.0106, +0.0157]. Cualquier efecto esta acotado a +-1.6 puntos.
#
# Y la razon para quedarse con la QUIETA no es el empate: con la fuente quieta
# el termino de Renyi es el UNICO con camino de gradiente, asi que el numero del
# blanco es atribuible a el y a nada mas. Con la fuente suelta la CE tambien
# mueve el encoder compartido y el numero es una mezcla que no se puede repartir.
#
# Lo que si necesita la fuente suelta es el diagnostico de gradiente de abajo:
# |g Renyi|/|g CE| no existe si la fuente no se mueve, y es lo unico que
# justifica correr con techo 1.0 en vez del 1e-4 publicado. Vive aparte, como
# diagnostico, y no como fila de ninguna figura ni de ninguna tabla.
FILA_GRADIENTE = (True, True, 'fuente suelta · sólo diagnóstico', 'diag_grad')
# La etiqueta y la clave de la unica fila, derivadas y no escritas a mano. Las
# celdas de diagnostico pedian antes 'la fila con la fuente suelta', que ya no
# existe como brazo: lo que leen es la fila que hay.
assert len(TABLA) == len(FILAS) == 1, (len(TABLA), len(FILAS))
assert not TABLA[0][1], 'la fila de la tabla deberia tener la fuente quieta'
ETQ_UNICA = TABLA[0][2]
CLV_UNICA = TABLA[0][3]

# `SEMILLAS_TABLA` se retiro. Era LA lista cuando habia una sola, antes de que
# existiera la busqueda, y quedo como default de `barrer_tablas` para cuando
# nadie pasara semillas --- un default que nunca se ejecutaba, porque el
# cuaderno siempre pasa `SEMILLAS_REPORTE`.
#
# Y ya habia hecho dano: era la que el cuaderno de graficas leia para el rotulo,
# asi que la tabla decia '5 semillas' sobre numeros de 10. Dos nombres para el
# mismo concepto, uno vivo y uno muerto, y el rotulo tomo el muerto. Hoy
# `barrer_tablas` EXIGE las semillas: un default que puede mentir es peor que
# una llamada que falta un argumento.
CTRL = list(range(0, PASOS_2 + 1, 20))

def adabn(fam, D, semilla, hp1=None):
    # El paso 1 igual que todos, y despues SOLO forwards. No hay optimizador:
    # si lo hubiera, esto no seria AdaBN sino otro brazo mas.
    gen = torch.Generator().manual_seed(semilla)
    torch.manual_seed(semilla); np.random.seed(semilla)
    h1 = {**HP_1, **(hp1 or {})}
    enc = MLP(fam['amb'], 'batchnorm', dim=h1['DIM'])
    cab = nn.Linear(h1['DIM'], CLASES)
    # AdaBN hereda la MISMA fase 1 sintonizada que CREDA: la fuente y la
    # arquitectura las comparten, asi que sintonizarlas una vez sirve a los dos.
    # Lo que AdaBN no recibe son las perillas de fase 2, que no existen para el.
    train_source_ce(enc, cab, D['Xs_tr'], D['ys_tr'], D['Xs_va'], D['ys_va'],
                    epochs=EPOCAS_1, batch_size=h1['LOTE_1'], generator=gen,
                    learning_rate=h1['LR_1'], weight_decay=h1['WD_1'])
    enc.eval(); cab.eval()
    with torch.no_grad():
        a0 = float((cab(enc(D['Xt_va'])).argmax(1) == D['yt_va']).float().mean())
        s0 = float((cab(enc(D['Xs_va'])).argmax(1) == D['ys_va']).float().mean())
    enc_t = clone_for_target(enc)
    enc_t.train()                       # train() es lo que reestima las corridas
    bufs0 = torch.cat([b.detach().reshape(-1).clone() for b in enc_t.buffers()
                       if b.dtype.is_floating_point])
    with torch.no_grad():
        for _ in range(PASOS_2):
            jj = torch.randperm(len(D['Xt_tr']), generator=gen)[:LOTE]
            enc_t(D['Xt_tr'][jj])
    enc_t.eval()
    bufs1 = torch.cat([b.detach().reshape(-1).clone() for b in enc_t.buffers()
                       if b.dtype.is_floating_point])
    # Que de verdad haya adaptado algo: si los buffers no se movieron, esto no
    # es AdaBN, es el clon pelado -- y daria el mismo numero sin avisar.
    assert float((bufs1 - bufs0).norm()) > 1e-6, 'los buffers no se movieron'
    with torch.no_grad():
        a1 = float((cab(enc_t(D['Xt_va'])).argmax(1) == D['yt_va']).float().mean())
        # TEST, para la tabla. AdaBN no tiene parada: sus buffers se reestiman
        # el numero de pasos declarado y ahi queda.
        a1_te = float((cab(enc_t(D['Xt_te'])).argmax(1) == D['yt_te']).float().mean())
        s0_te = float((cab(enc(D['Xs_te'])).argmax(1) == D['ys_te']).float().mean())
    return dict(acc_s=s0, acc_s_fin=s0, acc_s_te=s0_te,
                blanco_ini=a0, blanco_fin=a1, acc_te=a1_te)


# ---------------------------------------------------------------------------
# Las banderas de la corrida, en un solo lugar y escritas junto a los
# resultados. Un .npz sin esto es un monton de numeros sin decir de que red
# salieron: el cuaderno de graficas los lee y los imprime, asi que la figura
# siempre dice bajo que configuracion se midio.
# ---------------------------------------------------------------------------
def banderas():
    return dict(
        UNA_CAPA=UNA_CAPA, DIM=DIM, CLASES=CLASES, LOTE=LOTE, SEMILLA=SEMILLA,
        EPOCAS_1=EPOCAS_1, PASOS_2=PASOS_2, LR_1=LR_1, LR_2=LR_2,
        # Para que el rotulo de una figura no pueda afirmar un alcance que ya
        # no existe. El valor es una frase y no una llave porque no hay nada
        # que elegir: es como se calcula.
        SIGMA_DE='train completo de la fuente, congelado tras la fase 1',
        NORMALIZAR_POR_COTAS=NORMALIZAR_POR_COTAS,
        FORMA_DE_COTAS=FORMA_DE_COTAS, BALANCE_DE_MASA=BALANCE_DE_MASA,
        CUADROS=len(CUADROS),
        POS_FILAS=list(POS_FILAS),
        SEMILLAS_BUSQUEDA=list(SEMILLAS_BUSQUEDA),
        SEMILLAS_REPORTE=list(SEMILLAS_REPORTE),
        TRIALS_1=TRIALS_1, TRIALS_2=TRIALS_2,
        TRASLADO_REL=TRASLADO_REL, ANGULO_SIGMA_GRADOS=ANGULO_SIGMA,
        FRAC_TRAIN=FRAC_TRAIN, FRAC_VALID=FRAC_VALID,
        PACIENCIA_1=PACIENCIA_1, PACIENCIA_2=PACIENCIA_2,
        MIN_PASOS_2=MIN_PASOS_2,
        # EL CRITERIO DE LA FASE 2 Y SU CONVERSION, publicados para que se
        # puedan LEER del artefacto y no haya que abrir el codigo. Los dos
        # pasos de arriba quedan al lado y no se borran: son la procedencia de
        # `MIN_EPOCAS_2` y lo que la escala de `MIL_CREDA_Benchmark` lee.
        CRITERIO_2_EN='epocas',
        # LA SENAL Y SU DIRECCION, que `CRITERIO_2_EN` no decia: esa bandera
        # dice la UNIDAD del criterio y no QUE cantidad lee ni hacia donde.
        # Sin estas dos, una corrida vieja --- minimo del vigia --- y una nueva
        # publicaban banderas IDENTICAS aunque eligieran por cantidades
        # opuestas, asi que el artefacto no permitia distinguirlas.
        CRITERIO_2_SENAL='cef',
        CRITERIO_2_DIRECCION='maximo',
        CRITERIO_2_SENAL_ES='exp(H(histograma de argmax sobre el train del '
                            'blanco)), en [1, CLASES]. No usa y_t',
        CRITERIO_2_CAMBIO='2026-10-16: era el MINIMO del vigia. Medido sobre '
                          '90 celdas el vigia era la peor de las reglas '
                          'probadas (0.6410 global) y cef la mejor sin '
                          'etiquetas (0.7505), con el oraculo en 0.7951',
        MIN_EPOCAS_2=MIN_EPOCAS_2,
        PACIENCIA_2_EPOCAS=PACIENCIA_2_EPOCAS,
        PACIENCIA_2_EPOCAS_ES='declarada por el operador, no derivada del '
                              'lote: el lote se busca',
        MIN_EPOCAS_2_ES=f'derivada: ceil(MIN_PASOS_2 / '
                        f'BLOQUES_AL_LOTE_DECLARADO) = '
                        f'ceil({MIN_PASOS_2} / {BLOQUES_AL_LOTE_DECLARADO})',
        BLOQUES_AL_LOTE_DECLARADO=BLOQUES_AL_LOTE_DECLARADO,
        LOTES_2_BUSCADOS=list(LOTES_2_BUSCADOS),
        RUIDO_REL=RUIDO_REL, N_POR_DOMINIO=N_POR_DOMINIO,
        TERMINO_DE_FUENTE='ninguno: cross-entropy sola',
        CAMPOS_CU=CAMPOS_CU, CAMPOS_HS=CAMPOS_HS, CAMPOS_HT=CAMPOS_HT,
    )


# ---------------------------------------------------------------------------
# El barrido de los GIF: una corrida por (normalizacion, regimen) y por familia,
# con la semilla fija y los cuadros cada 10 pasos. Es lo que antes hacia la
# primera mitad de `mirar()`, y se corto justo donde empezaba a dibujar.
# ---------------------------------------------------------------------------
def barrer_animaciones(familias=None, al_paso=print, HP=None):
    familias = FAMILIAS if familias is None else familias
    POR, DAT = {}, {}
    for fam in familias:
        D = preparar(fam)
        DAT[fam['nombre']] = D
        h1, h2 = (HP or {}).get(fam['nombre'], (None, None))
        for q in POS_FILAS:
            for bal, mov, etq, clv in FILAS:
                al_paso(f"    {fam['nombre']} · {NOMBRE_FILA[q]} · {etq}")
                Ps, cu, pr, br, nf = corrida(fam, D, q, balance=bal,
                                             mover_fuente=mov,
                                             hp1=h1, hp2=h2)
                assert nf['balance'] is bal and nf['mover_fuente'] is mov
                POR[(fam['nombre'], q, clv)] = (Ps, cu, nf, pr, br)
    esperado = len(familias)*len(POS_FILAS)*len(FILAS)
    assert len(POR) == esperado, (len(POR), esperado)
    return POR, DAT


# ---------------------------------------------------------------------------
# El barrido de las tablas: cinco semillas de DATOS por celda, con los puntos de
# control cada 20 pasos, mas AdaBN como piso de comparacion.
# ---------------------------------------------------------------------------
def barrer_tablas(familias=None, *, semillas, al_paso=print, HP=None):
    familias = FAMILIAS if familias is None else familias
    assert semillas, 'barrer_tablas necesita las semillas, sin default'
    SEM = list(semillas)
    CELDAS, ADA = {}, {}
    t0 = time.time()
    for fam in familias:
        h1, h2 = (HP or {}).get(fam['nombre'], (None, None))
        D5 = {s: datos_semilla(fam, s) for s in SEM}
        for pos in POS_FILAS:
            for bal, mov, etq, clv in TABLA:
                sem = []
                for s in SEM:
                    _, cu, _, _, nf = corrida(fam, D5[s], pos, semilla=s,
                                              balance=bal, mover_fuente=mov,
                                              cuadros=CTRL, hp1=h1, hp2=h2)
                    assert nf['balance'] is bal and nf['mover_fuente'] is mov
                    # Solo lo que la tabla lee. Guardar el `cu` entero aca
                    # duplicaria los arreglos grandes que ya viajan en el .npz
                    # del barrido de animaciones, con otra semilla y otra
                    # rejilla de pasos: dos fuentes para el mismo numero.
                    # `.predict`: el modelo de la FUENTE aplicado al blanco,
                    # sin ninguna adaptacion. Es el punto de control del paso 0
                    # --- `clone_for_target` es un deepcopy puro y el punto se
                    # mide en eval(), asi que ahi el encoder del blanco es el de
                    # la fuente y nada mas.
                    #
                    # Sale de la misma corrida y no de un barrido aparte, y eso
                    # es mejor y no peor: comparte la fase 1 BIT A BIT con el
                    # brazo que va a comparar, asi que la diferencia es
                    # apareada. Un barrido aparte entrenaria otra fase 1 y la
                    # diferencia llevaria adentro esa variacion.
                    assert int(cu[0][0]) == 0, (
                        cu[0][0], 'el primer punto de control no es el paso 0, '
                                  'asi que no mide el modelo sin adaptar')
                    sem.append(dict(pasos=[int(c[0]) for c in cu],
                                    # la serie es de VALID; test es un numero
                                    # solo, medido donde la parada eligio
                                    acc=[float(c[2]) for c in cu],
                                    acc_te=float(nf['acc_te']),
                                    # la serie de TEST a lo largo, para que el
                                    # pico salga del mismo split que el elegido
                                    acc_te_ser=[float(c[7]) for c in cu],
                                    acc_s_te=float(nf['acc_s_te']),
                                    paso_fin=int(nf['paso_fin']),
                                    mejor_paso=int(nf['mejor_paso']),
                                    vigia_fin=float(nf['vigia_fin']),
                                    mejor_vg=float(nf['mejor_vg']),
                                    mejor_re=float(nf['mejor_re']),
                                    # LA EVIDENCIA DEL CRITERIO, para que
                                    # viaje con el artefacto: el `cef` que
                                    # eligio, la serie por punto de control, y
                                    # la epoca elegida. Sin la serie, el
                                    # cuaderno de graficas no puede dibujar la
                                    # senal que decide y habria que volver a
                                    # correr el barrido para verla.
                                    mejor_cef=float(nf['mejor_cef']),
                                    cef=[float(v) for v in nf['cef_cu']],
                                    mejor_epoca=int(nf['mejor_epoca']),
                                    # La columna por epoca, que es la rejilla
                                    # en la que el criterio REALMENTE leyo.
                                    # POR NOMBRE: el desempaque posicional de
                                    # estas bitacoras ya se rompio cinco veces
                                    # al agregar campos.
                                    cef_epoca=[
                                        float(dict(zip(CAMPOS_P2_NOMBRES,
                                                       f))['cef'])
                                        for f in nf['pasada_2']],
                                    # El modelo de la fase 1 sobre el blanco,
                                    # SIN adaptar: es el paso 0, antes de que
                                    # el optimizador toque nada, porque `enc_t`
                                    # arranca siendo un clon de `enc`.
                                    #
                                    # Sale de TEST, no de valid, y eso fue un
                                    # defecto: AdaBN reporta `a1_te` (test) y
                                    # CREDA `acc_te` (test), asi que un
                                    # `.predict` sobre valid comparaba tres
                                    # brazos de un split contra uno de otro.
                                    # Medido sobre el artefacto anterior, el
                                    # corrimiento llegaba a 0.0556 EN UNA
                                    # SEMILLA de lunas --- y los test son
                                    # apareados por semilla, asi que entraba
                                    # entero en la comparacion.
                                    #
                                    # Por nombre y no por indice: el desempaque
                                    # posicional de `cu` ya se rompio cinco
                                    # veces al agregar campos.
                                    acc_predict=float(punto(cu, 0)['acc_te']),
                                    # El de VALID se conserva SOLO para el
                                    # cross-check de mas abajo contra
                                    # `blanco_ini` de AdaBN, que tambien es
                                    # valid. No se reporta en ninguna tabla.
                                    acc_predict_va=float(punto(cu, 0)['acc_va']),
                                    # la serie del acierto de la FUENTE, que la
                                    # rejilla de trayectorias dibuja punteada
                                    acc_s_ser=[float(c[5]) for c in cu],
                                    # d/sigma y K_st/K_s, promediados sobre las
                                    # semillas por la rejilla del kernel
                                    diag=[[float(x) for x in d]
                                          for d in nf['diag']],
                                    ep_s=int(nf['ep_s']),
                                    acc_s=float(nf['acc_s']),
                                    acc_s_fin=float(nf['acc_s_fin'])))
                CELDAS[(fam['nombre'], pos, etq)] = sem
        ADA[fam['nombre']] = [{k: float(v)
                               for k, v in adabn(fam, D5[s], s, hp1=h1).items()}
                              for s in SEM]
        # AdaBN mide su propio `blanco_ini` --- el mismo modelo sin adaptar ---
        # con su propia fase 1. Si las dos fases 1 no son la misma, uno de los
        # dos barridos no esta haciendo lo que dice, y el unico sintoma seria un
        # par de numeros que no coinciden sin que nadie los compare.
        _pred_bn = [r['acc_predict_va']
                    for r in CELDAS[(fam['nombre'], 'batchnorm', ETQ_UNICA)]]
        _ini_bn = [r['blanco_ini'] for r in ADA[fam['nombre']]]
        assert all(abs(a - b) < 1e-9 for a, b in zip(_pred_bn, _ini_bn)), (
            fam['nombre'], _pred_bn, _ini_bn,
            'la fase 1 de corrida() y la de adabn() divergieron')
        al_paso(f"  {fam['nombre']:8s} "
                f"{len(POS_FILAS)*len(TABLA)*len(SEM):3d} corridas "
                f"+ {len(SEM)} AdaBN · {time.time()-t0:6.0f}s acumulados")
    assert len(CELDAS) == len(familias)*len(POS_FILAS)*len(TABLA), len(CELDAS)
    assert len(ADA) == len(familias)
    return CELDAS, ADA


# ---------------------------------------------------------------------------
# El diagnostico de gradiente. Una corrida por (familia, normalizacion) con la
# fuente SUELTA, de la que se guarda unicamente el registro `grad`.
#
# Existe porque |g Renyi| / |g CE| sobre el encoder de la FUENTE no se puede
# medir si la fuente no se mueve --- no hay gradiente que medir --- y es lo
# unico que justifica correr con techo 1.0 en vez del 1e-4 publicado. No es un
# brazo: su acierto no se reporta, no sale en ninguna tabla y no tiene figura.
# ---------------------------------------------------------------------------
def barrer_gradiente(familias=None, al_paso=print, HP=None):
    familias = FAMILIAS if familias is None else familias
    bal, mov, etq, clv = FILA_GRADIENTE
    assert mov, 'la fila de diagnostico tiene que soltar la fuente'
    G = {}
    for fam in familias:
        D = preparar(fam)
        h1, h2 = (HP or {}).get(fam['nombre'], (None, None))
        for pos in POS_FILAS:
            # `info` es el ULTIMO campo de lo que corrida() devuelve
            # (proy, cu, pred_s, br, info), no el tercero.
            *_, nf = corrida(fam, D, pos, balance=bal, mover_fuente=mov,
                             hp1=h1, hp2=h2)
            # Si esto queda vacio el diagnostico no midio nada, y la seccion
            # que justifica el techo saldria en blanco sin decir por que.
            assert nf['grad'], (fam['nombre'], pos, 'no se registro gradiente')
            G[(fam['nombre'], pos)] = [[float(x) for x in g] for g in nf['grad']]
        al_paso(f"  {fam['nombre']:8s} {len(POS_FILAS)} corridas de diagnóstico")
    assert len(G) == len(familias)*len(POS_FILAS), len(G)
    return G


# ---------------------------------------------------------------------------
# El artefacto intermedio. El cuaderno de generacion escribe; el de graficas
# lee. Partir el cuaderno en dos no sirve de nada si el de graficas tiene que
# volver a correr las dos fases para tener los datos.
#
#   banderas.json           bajo que configuracion se midio
#   barrido.json            lo que las tablas leen, y solo eso
#   <familia>.npz           las trayectorias que los GIF animan
#
# Las etiquetas de las celdas llevan '|' como separador porque ni los nombres de
# familia ni las normalizaciones ni las etiquetas de regimen lo contienen, y se
# comprueba al escribir en vez de confiarlo.
# ---------------------------------------------------------------------------
SEP = '|'


def _clave(*partes):
    for p in partes:
        assert SEP not in str(p), (p, f'una parte de la clave contiene {SEP!r}')
    return SEP.join(str(p) for p in partes)


def guardar(destino, POR, DAT, CELDAS, ADA, GRAD, HP1=None, HP2=None,
            R2=None):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    (destino/'banderas.json').write_text(
        json.dumps(banderas(), indent=2, ensure_ascii=False) + '\n')
    (destino/'barrido.json').write_text(json.dumps(dict(
        celdas={_clave(f, p, e): v for (f, p, e), v in CELDAS.items()},
        adabn=ADA,
        gradiente={_clave(f, p): v for (f, p), v in GRAD.items()},
        # Los HP que cada busqueda eligio, por toyset. Van al artefacto para que
        # el cuaderno de graficas los imprima: una tabla sin decir bajo que
        # hiperparametros se midio no se puede leer.
        hp1=HP1 or {}, hp2=HP2 or {},
        # Los trials de la busqueda de fase 2, con los DOS escalares sin
        # etiquetas y el test de cada uno. Sin esto no se puede contestar cual
        # de los dos habria ordenado mejor las configuraciones --- y antes solo
        # se guardaba el ganador.
        trials2={n: v.get('por_trial', []) for n, v in (R2 or {}).items()}),
        indent=2, ensure_ascii=False) + '\n')
    escritos = [destino/'banderas.json', destino/'barrido.json']
    for nombre, D in DAT.items():
        # Las etiquetas del split de TRAIN, que es el que los paneles
        # colorean: `cu_proy` es la proyeccion de Xt_tr.
        A = {'ys_tr': D['ys_tr'].numpy(), 'yt_tr': D['yt_tr'].numpy()}
        for (f, q, clv), (Ps, cu, nf, pr, br) in POR.items():
            if f != nombre: continue
            k = lambda campo: _clave(q, clv, campo)
            assert all(len(c) == CAMPOS_CU for c in cu), (len(cu[0]), CAMPOS_CU)
            A[k('Ps')]      = Ps.astype(np.float32)
            A[k('pred_s')]  = np.asarray(pr, dtype=np.int16)
            A[k('br')]      = np.asarray(br, dtype=np.float32)
            A[k('cu_paso')] = np.asarray([c[0] for c in cu], dtype=np.int32)
            A[k('cu_proy')] = np.stack([c[1] for c in cu]).astype(np.float32)
            A[k('cu_acc')]  = np.asarray([c[2] for c in cu], dtype=np.float32)
            A[k('cu_pok')]  = np.asarray([c[3] for c in cu], dtype=np.float32)
            A[k('cu_rut')]  = np.stack([c[4] for c in cu]).astype(np.int16)
            A[k('cu_as')]   = np.asarray([c[5] for c in cu], dtype=np.float32)
            A[k('cu_ut')]   = np.asarray([c[6] for c in cu], dtype=np.float32)
            A[k('cu_te')]   = np.asarray([c[7] for c in cu], dtype=np.float32)
            A[k('hist_s')]  = np.asarray(nf['hist_s'], dtype=np.float32)
            A[k('hist_t')]  = np.asarray(nf['hist_t'], dtype=np.float32)
            A[k('PS_t')]    = np.stack(nf['PS_t']).astype(np.float32)
            A[k('diag')]    = np.asarray(nf['diag'], dtype=np.float32)
            A[k('grad')]    = np.asarray(nf['grad'], dtype=np.float32)
            # Los escalares del `info` viajan como un arreglo de 0 dimensiones
            # con el JSON adentro: un .npz no guarda diccionarios, y partirlo en
            # un segundo archivo por familia deja dos cosas que se pueden perder
            # por separado.
            # bool ANTES que int, porque bool ES subclase de int: sin esta
            # rama `mover_fuente=True` volvia como 1.0, y el assert de `animar`
            # --- que compara con `is` --- fallaba al leer el artefacto. La
            # escritura salia verde: el defecto solo existe de vuelta.
            def _plano(v):
                if isinstance(v, bool): return v
                if isinstance(v, (int, float)): return float(v)
                return v
            # Lo que NO entra al `info` se decide por la FORMA del valor, no
            # por una lista de nombres.
            #
            # Habia una lista negra, y fallo exactamente como falla una lista
            # negra: alguien agrego `w_usado_cu` --- los pesos efectivamente
            # usados por punto y por cuadro --- al lado de `w_cu`, que si
            # estaba nombrado, y `json.dumps` reviento con
            # `Object of type ndarray is not JSON serializable`. Costo 15
            # minutos de barrido COMPLETO, porque esta es la ultima linea del
            # generador: las dos busquedas, los tres barridos y el de gradiente
            # ya se habian pagado cuando la escritura murio.
            #
            # Un campo nuevo que sea un arreglo ahora se excluye solo. Los
            # nombres siguen listados porque sus razones no son la forma:
            # `hist_*`, `PS_t`, `diag` y `grad` viajan como arreglos propios
            # unas lineas mas arriba, y los `w_*` por punto no tienen lector en
            # este artefacto --- el barrido de contaminacion persiste los suyos
            # aparte, donde si se leen. El formato del .npz no cambia, asi que
            # `cargar` sigue leyendo los artefactos que ya estan en disco.
            _POR_NOMBRE = ('hist_s', 'hist_t', 'PS_t', 'diag', 'grad')

            def _es_arreglo(v):
                if isinstance(v, np.ndarray):
                    return True
                return (isinstance(v, (list, tuple)) and len(v) > 0
                        and isinstance(v[0], np.ndarray))

            _info = {c: _plano(v) for c, v in nf.items()
                     if c not in _POR_NOMBRE and not _es_arreglo(v)}
            # Y si algo todavia no es serializable, que lo diga NOMBRANDOLO,
            # en vez de dejar que `json.dumps` reporte solo el tipo: el error
            # original no decia cual de los treinta y tres campos era.
            for _c, _v in _info.items():
                try:
                    json.dumps(_v)
                except TypeError as _e:
                    raise TypeError(
                        f"el campo {_c!r} de {nombre}/{k('info')} no es "
                        f"serializable ({type(_v).__name__}): agregalo a "
                        f"_POR_NOMBRE con su razon, o hacelo un arreglo "
                        f"propio como hist_s") from _e
            A[k('info')] = np.array(json.dumps(_info, ensure_ascii=False))
        ruta = destino/f'{nombre}.npz'
        np.savez_compressed(ruta, **A)
        escritos.append(ruta)
    # Leer de vuelta lo que se acaba de escribir. np.savez no falla con un
    # arreglo que no se puede serializar: lo guarda como objeto y revienta al
    # cargarlo, en el otro cuaderno y sin ninguna pista de cual es.
    for nombre in DAT:
        with np.load(destino/f'{nombre}.npz', allow_pickle=False) as z:
            assert f'{_clave(POS_FILAS[0], FILAS[0][3], "cu_proy")}' in z.files
    return escritos


def cargar(origen):
    origen = Path(origen)
    banderas_ = json.loads((origen/'banderas.json').read_text())
    b = json.loads((origen/'barrido.json').read_text())
    CELDAS = {tuple(k.split(SEP)): v for k, v in b['celdas'].items()}
    GRAD = {tuple(k.split(SEP)): v for k, v in b.get('gradiente', {}).items()}
    POR, DAT = {}, {}
    for fam in FAMILIAS:
        ruta = origen/f"{fam['nombre']}.npz"
        if not ruta.exists(): continue
        with np.load(ruta, allow_pickle=False) as z:
            DAT[fam['nombre']] = {'ys_tr': z['ys_tr'], 'yt_tr': z['yt_tr']}
            for q in banderas_['POS_FILAS']:
                for *_, clv in FILAS:
                    k = lambda campo: _clave(q, clv, campo)
                    if k('Ps') not in z.files: continue
                    cu = list(zip(z[k('cu_paso')].tolist(), z[k('cu_proy')],
                                  z[k('cu_acc')].tolist(), z[k('cu_pok')].tolist(),
                                  z[k('cu_rut')], z[k('cu_as')].tolist(),
                                  z[k('cu_ut')].tolist(), z[k('cu_te')].tolist()))
                    assert all(len(c) == CAMPOS_CU for c in cu), CAMPOS_CU
                    nf = json.loads(str(z[k('info')]))
                    nf['hist_s'] = [tuple(r) for r in z[k('hist_s')].tolist()]
                    nf['hist_t'] = [tuple(r) for r in z[k('hist_t')].tolist()]
                    nf['PS_t'] = list(z[k('PS_t')])
                    nf['diag'] = [tuple(r) for r in z[k('diag')].tolist()]
                    nf['grad'] = [tuple(r) for r in z[k('grad')].tolist()]
                    POR[(fam['nombre'], q, clv)] = (
                        z[k('Ps')], cu, nf, z[k('pred_s')], z[k('br')].tolist())
    return (banderas_, POR, DAT, CELDAS, b['adabn'], GRAD,
            b.get('hp1', {}), b.get('hp2', {}), b.get('trials2', {}))


# ---------------------------------------------------------------------------
# Las dos busquedas. Ninguna mira y_t: la de fase 1 maximiza el acierto de VALID
# de la fuente, y la de fase 2 minimiza el termino de Renyi de VALID en el paso
# que la parada eligio. El oraculo no aparece --- ni para validar el criterio,
# que seria seleccion sobre y_t un nivel mas arriba.
#
# Las semillas de busqueda son DISJUNTAS de las que reportan: si se buscara en
# {0,1,2} y se reportara en {0..9}, los mejores HP quedarian ajustados a tres
# semillas que despues se publican.
# ---------------------------------------------------------------------------
SEMILLAS_BUSQUEDA = [100, 101, 102]
SEMILLAS_REPORTE = list(range(10))
TRIALS_1, TRIALS_2 = 25, 25


def _optuna():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def buscar_hp1(fam, trials=None, semillas=None, al_paso=print):
    """Los HP de la fase 1, UNA config por toyset.

    Compartida por las tres normalizaciones a proposito: si cada brazo tuviera
    sus propios HP, la comparacion entre normalizaciones quedaria confundida con
    los HP y el eje del experimento se arruina. Asi que el objetivo agrega las
    tres.

    El objetivo es el ACIERTO de valid, por decision del operador. Y como 37 de
    45 corridas empatan dentro de 1e-2 --- con 358 puntos de valid, 1e-2 son
    cuatro muestras --- los empates los rompe el CE de valid, que si discrimina
    donde el acierto esta contra el techo. La arquitectura NO se busca: es el eje
    del experimento, y elegirla por la fuente cuesta 0.13-0.17 de acierto en el
    blanco (medido).
    """
    opt = _optuna()
    trials = TRIALS_1 if trials is None else trials
    semillas = SEMILLAS_BUSQUEDA if semillas is None else semillas
    D = {s: datos_semilla(fam, s) for s in semillas}

    def objetivo(trial):
        h1 = dict(LR_1=trial.suggest_float('LR_1', 1e-4, 1e-1, log=True),
                  WD_1=trial.suggest_float('WD_1', 1e-7, 1e-2, log=True),
                  DIM=trial.suggest_categorical('DIM', [8, 16, 32, 64]),
                  LOTE_1=trial.suggest_categorical('LOTE_1', [32, 64, 128, 256]))
        accs, ces = [], []
        for pos in POS_FILAS:
            for s in semillas:
                gen = torch.Generator().manual_seed(s)
                torch.manual_seed(s); np.random.seed(s)
                enc = MLP(fam['amb'], pos, dim=h1['DIM'])
                cab = nn.Linear(h1['DIM'], CLASES)
                hist, ep = train_source_ce(
                    enc, cab, D[s]['Xs_tr'], D[s]['ys_tr'],
                    D[s]['Xs_va'], D[s]['ys_va'], epochs=EPOCAS_1,
                    batch_size=h1['LOTE_1'], generator=gen,
                    learning_rate=h1['LR_1'], weight_decay=h1['WD_1'])
                r = [x for x in hist if x[0] == ep]
                assert len(r) == 1, (ep, 'la epoca elegida no esta en la historia')
                accs.append(r[0][3]); ces.append(r[0][2])
        # El CE viaja como atributo, no sumado al objetivo: sumarlos inventaria
        # una ponderacion que nadie puede justificar.
        trial.set_user_attr('ce_valid', float(np.mean(ces)))
        return float(np.mean(accs))

    est = opt.create_study(direction='maximize',
                           sampler=opt.samplers.TPESampler(seed=0))
    est.optimize(objetivo, n_trials=trials)
    # El desempate, y el aviso de empate. Si el top-k cae dentro del ruido del
    # objetivo, coronar uno es elegir por ruido: se dice y se desempata por CE.
    hechos = [t for t in est.trials if t.value is not None]
    tope = max(t.value for t in hechos)
    empatan = [t for t in hechos if t.value >= tope - 1e-2]
    ganador = min(empatan, key=lambda t: t.user_attrs['ce_valid'])
    al_paso(f"    {fam['nombre']:8s} fase 1: {len(hechos)} trials, "
            f"acc {tope:.4f}, {len(empatan)} dentro de 1e-2 "
            f"-> desempata el CE ({ganador.user_attrs['ce_valid']:.4f})")
    return dict(ganador.params), dict(
        trials=len(hechos), mejor_acc=float(tope), empatados=len(empatan),
        ce_valid=float(ganador.user_attrs['ce_valid']),
        acc_por_trial=[float(t.value) for t in hechos])


def buscar_hp2(fam, hp1, trials=None, semillas=None, al_paso=print,
               rho=0.0, pos_filas=None, sopesado=True, modo=MODO_REUBICADO,
               peso=None, perturbar_con=None):
    """Los HP de la fase 2, con la fase 1 CONGELADA y entrenada una sola vez.

    `rho` y `modo` perturban el TRAIN del blanco de cada semilla de busqueda
    ANTES de buscar, de modo que cada escenario busca en SUS condiciones y no
    en las limpias. Reusar los HP limpios a contaminacion alta leeria el
    resultado en un punto sintonizado para condiciones que ya no valen.

    **El oraculo no se puede buscar, y eso se rechaza por nombre.** `peso` solo
    admite lo que `PESOS_BUSCABLES` declara. Un techo alcanzable desde la
    busqueda deja de ser un techo: seria un brazo que Optuna eligio con `y_t`
    adentro del criterio, y ninguna tabla podria volver a leerlo como una cota.

    `pos_filas` angosta la busqueda a las arquitecturas que el escenario va a
    USAR. La corrida principal busca sobre las tres a proposito --- compartir
    HP es lo que mantiene limpio el eje de normalizacion --- pero la seccion de
    contaminacion compara cuatro brazos de los cuales uno es AdaBN, que solo
    existe donde hay buffers que reestimar, asi que su celda es `batchnorm` y
    nada mas. Buscar sobre las tres dejaria que dos arquitecturas que la
    seccion no reporta decidieran su HP.

    `sopesado` entra para que la busqueda corra el MISMO termino que los brazos
    que la van a usar. Queda en `True`, que es el default de `CREDALoss`, y se
    pasa explicito: la decision es que los dos brazos de sopesado comparten UNA
    busqueda, asi que ninguno de los dos busca con su propia bandera.

    Reusar un unico (encoder, cabeza) por (normalizacion, semilla) hace dos
    cosas: la busqueda corre al doble de velocidad, y --- mas importante ---
    todos los trials comparan sobre el MISMO modelo de fase 1, asi que la
    diferencia entre ellos es apareada y no lleva adentro la varianza de
    reentrenar la fuente.

    El objetivo es el MINIMO DEL VIGIA --- `acc(cab(enc_t(Xs_va)), ys_va)`, el
    encoder del blanco sobre la validacion de la FUENTE --- a minimizar. Cero
    etiquetas del blanco, asi que el mismo criterio corre sobre datos que no las
    tienen.

    El Renyi de valid, que era el objetivo antes, se sigue registrando por trial
    para poder comparar. Y el acierto en test tambien, pero NUNCA como valor
    devuelto: eso haria que Optuna optimizara contra y_t.
    """
    opt = _optuna()
    trials = TRIALS_2 if trials is None else trials
    semillas = SEMILLAS_BUSQUEDA if semillas is None else semillas
    posiciones = list(POS_FILAS if pos_filas is None else pos_filas)
    assert posiciones, 'la busqueda de fase 2 necesita al menos una arquitectura'
    if peso is None:
        peso = PESO_DECLARADO if sopesado else PESO_APAGADO
    assert peso in PESOS_BUSCABLES, (
        peso, PESOS_BUSCABLES, 'la busqueda no puede correr este peso: un '
        'techo que la busqueda pudiera elegir deja de ser un techo')
    # La fase 1 va PRIMERO y sobre los datos limpios, y eso no es un atajo: la
    # fase 1 solo mira `Xs_*`/`ys_*`, las dos perturbaciones solo tocan
    # `Xt_tr`, y `contaminar` devuelve los demas splits como LOS MISMOS
    # objetos --- asi que el cabezal es el mismo con y sin perturbacion. El
    # orden importa porque `MODO_FRONTERA` NECESITA ese cabezal para saber
    # donde esta la frontera, y antes de este cambio la perturbacion corria
    # antes de que existiera.
    limpios = {s: datos_semilla(fam, s) for s in semillas}
    cache = {}
    for pos in posiciones:
        for s in semillas:
            cache[(pos, s)] = fase1_congelada(fam, limpios[s], pos, s, hp1)
    # Un dato por (normalizacion, semilla) y no por semilla: con
    # `MODO_FRONTERA` la perturbacion depende del cabezal, y el cabezal depende
    # de la arquitectura. Con `MODO_REUBICADO` las entradas de un mismo `s`
    # salen identicas --- el mecanismo no lee el cabezal --- y la clave mas
    # fina no cambia nada.
    D = {}
    for pos in posiciones:
        for s in semillas:
            enc, cab, _, _ = cache[(pos, s)]
            if perturbar_con is None:
                D[(pos, s)] = contaminar(limpios[s], rho, s, modo=modo,
                                         pesar=pesador_del_cabezal(enc, cab))[0]
            else:
                # La perturbacion que el LLAMADOR declara, con la fase 1 de
                # esta celda. Existe porque el barrido de intrusos no es uno de
                # los dos modos de `contaminar` --- su conteo es por clase, ver
                # `MODO_INTRUSOS` --- y la alternativa era que la busqueda
                # corriera sobre datos LIMPIOS: eso leeria el resultado de cada
                # escenario en un punto sintonizado para condiciones que ya no
                # valen, que es exactamente lo que `rho` existe para evitar.
                # El default `None` deja el camino de los dos modos intacto.
                D[(pos, s)] = perturbar_con(limpios[s], s, enc, cab)

    def objetivo(trial):
        h2 = dict(
            LR_2=trial.suggest_float('LR_2', 1e-5, 1e-2, log=True),
            WD_2=trial.suggest_float('WD_2', 1e-7, 1e-2, log=True),
            LOTE_2=trial.suggest_categorical('LOTE_2', list(LOTES_2_BUSCADOS)),
            # `SIGMA_FROM` ya no se busca, y lo que lo saco no fue r28 sino una
            # medicion: las tres familias ELIGIERON `batch-pooled`, el alcance
            # cuyo propio docstring en `models.SIGMA_SCOPES` registra el
            # defecto --- la mediana agrupada lee las features del blanco, que
            # son exactamente lo que el optimizador mueve, asi que la unidad de
            # medida se mueve junto con lo que mide. Desplazas una clase del
            # blanco y su puntaje BAJA: 0.505 a desplazamiento 10, 0.437 a 30,
            # 0.397 a 100. El signo invertido. Con un sigma compartido el mismo
            # barrido sube monotono y satura en ln 2 = 0.69315.
            #
            # `models.SIGMA_SCOPES` conserva los cuatro valores para reproducir
            # numeros publicados; lo que cambia es que el ancho de banda dejo de
            # ser una dimension de esta busqueda y pasa a ser `sigma_fijo`,
            # calibrado en la fuente y congelado.
            )
        # NI `TECHO` NI `DELTA` se buscan: los dos se retiraron el 2026-10-11.
        # No estan en `research-concept-r29.md`, y sobre un UNICO termino con
        # gradiente un coeficiente no balancea nada --- medido inerte en cuatro
        # ordenes porque Adam lo absorbe, y redundante con `LR_2`.
        res, vig, tes, porarq = [], [], [], {}
        for pos in posiciones:
            v_p, t_p, r_p = [], [], []
            for s in semillas:
                e0, c0, hi, ep = cache[(pos, s)]
                f1 = (copy.deepcopy(e0), copy.deepcopy(c0), hi, ep)
                *_, nf = corrida(fam, D[(pos, s)], pos, semilla=s,
                                 mover_fuente=False, cuadros=CTRL, hp2=h2,
                                 hp1=hp1, fase1=f1, peso=peso)
                res.append(nf['mejor_re']); vig.append(nf['mejor_vg'])
                tes.append(nf['acc_te'])
                v_p.append(nf['mejor_vg']); t_p.append(nf['acc_te'])
                r_p.append(nf['mejor_re'])
            # El desglose POR ARQUITECTURA. El objetivo promedia las tres, y sus
            # vigias no estan en la misma escala --- medido en la corrida final,
            # el rango entre arquitecturas va de 0.0497 en lunas a 0.1485 en
            # toroide. Promediar escalas distintas deja que la arquitectura mas
            # variable decida el HP y las otras dos viajen de pasajeras.
            #
            # Sin este desglose no se puede medir si un HP por arquitectura, o
            # un vigia normalizado por arquitectura antes de promediar, habrian
            # elegido mejor. Se guarda; no cambia el objetivo.
            porarq[pos] = dict(vg=float(np.mean(v_p)), te=float(np.mean(t_p)),
                               re=float(np.mean(r_p)))
        # Los tres se GUARDAN; solo uno se DEVUELVE. El acierto en test va al
        # registro para poder medir DESPUES cual de los dos escalares sin
        # etiquetas habria ordenado mejor los trials --- esa medicion usa y_t y
        # su costo se declara, igual que la eleccion de la senal de parada.
        #
        # Lo que NUNCA puede pasar es que `acc_te` sea el valor devuelto: ahi
        # Optuna optimizaria contra y_t y todo lo de abajo seria inadmisible.
        # El OBJETIVO es el minimo del vigia, a minimizar. Era el Renyi de
        # valid, y medido sobre 75 trials el vigia ordena mucho mejor:
        # Pearson -0.984 contra -0.129 en toroide, -0.895 contra -0.712 en
        # rollo. Y lo que decide: el HP que cada objetivo habria elegido da
        # 0.6584 de media con el vigia contra 0.5749 con el Renyi, con el
        # oraculo en 0.6856. En toroide el vigia elige EXACTO la mejor de las 25
        # (0.8399) y el Renyi una que da 0.6069.
        #
        # Es la MISMA cantidad que la senal de parada, y eso es coherente y no
        # casualidad: las dos preguntas son 'cuanto se adapto', una en el tiempo
        # y otra entre configuraciones.
        _obj = float(np.mean(vig))
        trial.set_user_attr('mejor_re', float(np.mean(res)))
        trial.set_user_attr('mejor_vg', _obj)
        trial.set_user_attr('acc_te', float(np.mean(tes)))
        trial.set_user_attr('por_arq', porarq)
        return _obj

    # SIN pruner. Habia un MedianPruner puesto, y no podaba nada: el objetivo
    # se calcula al FINAL del trial, asi que nunca hay un valor intermedio que
    # reportar con `trial.report`, y sin eso `should_prune` no se consulta. Un
    # pruner que no puede podar se lee como un ahorro de tiempo que no existe.
    est = opt.create_study(direction='minimize',
                           sampler=opt.samplers.TPESampler(seed=0))
    est.optimize(objetivo, n_trials=trials)
    hechos = [t for t in est.trials if t.value is not None]
    mejor = min(hechos, key=lambda t: t.value)
    al_paso(f"    {fam['nombre']:8s} fase 2: {len(hechos)} trials, "
            f"vigía {mejor.value:.4f} (el objetivo), "
            f"Rényi de valid {mejor.user_attrs['mejor_re']:.4f}")
    # La tabla completa de trials: los dos escalares sin etiquetas y el test de
    # cada uno. Es lo que contesta 'cual de los dos habria elegido mejor HP', y
    # antes no se guardaba --- solo el ganador.
    tabla = [dict(mejor_re=float(t.user_attrs['mejor_re']),
                  mejor_vg=float(t.user_attrs['mejor_vg']),
                  acc_te=float(t.user_attrs['acc_te']),
                  por_arq=t.user_attrs['por_arq'],
                  params={k: (float(v) if isinstance(v, float) else v)
                          for k, v in t.params.items()})
             for t in hechos]
    return dict(mejor.params), dict(
        trials=len(hechos), objetivo='min(vigia)',
        mejor_vigia=float(mejor.value),
        mejor_renyi=float(mejor.user_attrs['mejor_re']), por_trial=tabla)


# ===========================================================================
# La seccion de sopesado bajo contaminacion
# ---------------------------------------------------------------------------
# Lo que la seccion pregunta: el sopesado por confianza (Ecs. 24 y 26) gana
# algo cuando el dano al blanco es DESIGUAL. Bajo el balance por traza (Ec. 32)
# la escala COMUN de `w` se cancela --- medido, y ahora tambien probado en
# `tests/test_toy_contamination.py` --- asi que una perturbacion que degrade
# todo por igual mide cero por construccion. Lo que sobrevive a la cancelacion
# es el descuento RELATIVO, y para que se vea hace falta que unos puntos esten
# danados y otros no.
#
# Los CUATRO brazos, en escalera de intervencion, y el ultimo par es el unico
# eje del experimento: `sin sopesado` contra `con sopesado` difieren SOLO en
# `use_entropy_weighting`.
#
# La celda es `batchnorm` y eso es DERIVADO, no elegido: uno de los cuatro
# brazos es AdaBN y AdaBN solo existe donde hay buffers que reestimar.
# ===========================================================================

#: Los dos brazos de CREDA que la seccion compara, declarados aca --- en la
#: mitad que CALCULA --- para que la mitad que dibuja los importe en vez de
#: escribirlos por segunda vez. Dos listas a mano en estos dos archivos ya
#: divergieron una vez.
BRAZO_SIN_SOP = 'CREDA sin sopesado'
BRAZO_CON_SOP = 'CREDA con sopesado'
#: El brazo de la COMPUERTA. Se rotula 'compuerta' y no 'sopesado duro'
#: porque no es una variante de intensidad del declarado: es otra
#: operacion --- quitar filas en vez de reescalarlas --- y un rotulo que
#: sugiriera un grado haria leer un resultado positivo como 'el sopesado
#: funciona mas fuerte' cuando lo que diria es 'el sopesado funciona de
#: otra forma'. No lleva `PALABRA_TECHO`: no lee `y_t` y es un metodo.
BRAZO_COMPUERTA = 'CREDA con compuerta'
#: El brazo del CONTROL barajado. Sin `PALABRA_TECHO`: no lee `y_t`.
BRAZO_BARAJADO = 'CREDA con w barajado'

#: La palabra que marca una columna como TECHO y no como metodo. Vive como
#: constante porque se lee en tres sitios --- el nombre del brazo, una clave del
#: registro y el test que comprueba las dos --- y una palabra escrita tres veces
#: a mano es una que se puede quedar atras en dos de ellas.
PALABRA_TECHO = 'techo'


#: El brazo del ORACULO. Lee `y_t`, asi que NO es un metodo: es un techo, del
#: mismo tipo que la columna del pico que este proyecto ya reporta.
#:
#: Su nombre lleva `PALABRA_TECHO` adentro a proposito, y la fila que escribe
#: lleva la misma palabra en una CLAVE. Un booleano suelto lo leeria quien se
#: acordara de buscarlo; la palabra en el nombre y en la clave la lee cualquiera
#: que imprima el registro o mire el encabezado de la tabla.
BRAZO_TECHO = f'CREDA {PALABRA_TECHO} de oráculo'
assert PALABRA_TECHO in BRAZO_TECHO.lower(), BRAZO_TECHO

#: Los cinco, EN ORDEN de intervencion: el piso sin tocar nada, despues solo
#: los buffers, despues el regularizador con peso uno, despues el peso de la
#: formulacion, y al final el techo que lee las etiquetas.
BRAZOS_CONTAMINACION = ('.predict', 'AdaBN', BRAZO_SIN_SOP, BRAZO_CON_SOP,
                        BRAZO_TECHO)

#: Los tres brazos de CREDA de un escenario, y el peso de cada uno. UNA lista,
#: porque el barrido la recorre para correr y el registro la recorre para
#: rotular: dos listas a mano en este archivo ya divergieron una vez.
BRAZOS_POR_PESO = ((BRAZO_SIN_SOP, PESO_APAGADO),
                   (BRAZO_CON_SOP, PESO_DECLARADO),
                   (BRAZO_TECHO, PESO_ORACULO))
assert all(b in BRAZOS_CONTAMINACION for b, _ in BRAZOS_POR_PESO), (
    BRAZOS_POR_PESO, BRAZOS_CONTAMINACION)

#: El techo de la seccion de INTRUSOS, con `PALABRA_TECHO` adentro por la misma
#: razon que el otro: la palabra en el nombre y en una clave del registro la lee
#: cualquiera que imprima la tabla, y un booleano suelto no.
BRAZO_TECHO_INTRUSOS = f'CREDA {PALABRA_TECHO} de intrusos'

#: El brazo de la escalera de precision. Lleva `PALABRA_TECHO` porque es
#: uno, y quien imprima la tabla tiene que poner la precision al lado: la
#: escalera entera son peldaños del MISMO brazo, y 'techo ruidoso'
#: repetido seis veces sin el numero no se puede leer.
BRAZO_TECHO_RUIDOSO = f'CREDA {PALABRA_TECHO} ruidoso'
#: El brazo del techo de la inyeccion en el embedding.
BRAZO_TECHO_EMB = f'CREDA {PALABRA_TECHO} de embedding'
assert PALABRA_TECHO in BRAZO_TECHO_INTRUSOS.lower(), BRAZO_TECHO_INTRUSOS
assert BRAZO_TECHO_INTRUSOS != BRAZO_TECHO, BRAZO_TECHO_INTRUSOS

#: Los CINCO brazos del barrido de intrusos, en orden de intervencion.
#:
#: `.predict` y `AdaBN` no son adornos heredados del otro barrido y se declaran
#: porque estan MEDIDOS bajo esta intervencion. `.predict` no toca el blanco en
#: ningun paso, asi que es el unico brazo inmune por construccion. AdaBN es el
#: MAS expuesto y no el menos: reestima `running_mean`/`running_var` desde
#: `D['Xt_tr']` --- exactamente donde van los intrusos --- y una media y una
#: varianza planas son lo menos robusto que hay a un punto lejano. Medido en una
#: inyeccion cruda por una direccion fija, AdaBN se movio -0.1657 y -0.1436 en
#: toroide y rollo y +0.2139 en lunas. Que sea la referencia que recibe el dano
#: y no tiene mecanismo para resistirlo, ni en principio, es lo que vuelve
#: significativa la resistencia de un brazo ponderado en vez de supuesta.
#:
#: Y el signo POSITIVO de esa medicion es la razon por la que las direcciones de
#: `intrusar` van en pares antipodales: un AdaBN que MEJORA delata un rescalado
#: de features disfrazado de intrusion. El delta de AdaBN se reporta por familia
#: al lado del de CREDA justamente para que ese artefacto no pase inadvertido.
BRAZOS_INTRUSOS = ('.predict', 'AdaBN', BRAZO_SIN_SOP, BRAZO_CON_SOP,
                   BRAZO_TECHO_RUIDOSO, BRAZO_TECHO_INTRUSOS)

#: Los tres brazos de CREDA del barrido de intrusos y el peso de cada uno. La
#: misma forma que `BRAZOS_POR_PESO` y por la misma razon: una lista que el
#: barrido recorre para correr y el registro recorre para rotular.
BRAZOS_POR_PESO_INTRUSOS = ((BRAZO_SIN_SOP, PESO_APAGADO),
                            (BRAZO_CON_SOP, PESO_DECLARADO),
                            (BRAZO_TECHO_RUIDOSO, PESO_ORACULO_RUIDOSO),
                            (BRAZO_TECHO_INTRUSOS, PESO_ORACULO_INTRUSOS))
assert all(b in BRAZOS_INTRUSOS for b, _ in BRAZOS_POR_PESO_INTRUSOS), (
    BRAZOS_POR_PESO_INTRUSOS, BRAZOS_INTRUSOS)

# Y la cobertura: entre los dos padrones, CADA peso declarado tiene un brazo que
# lo corre. Era una igualdad contra `BRAZOS_POR_PESO` sola, y dejo de poder
# serlo cuando aparecio el cuarto peso: la union es lo que afirma lo mismo sin
# volverse falsa al agregar un barrido. Un peso sin brazo es un modo que nadie
# ejercita, que es como un peso se queda roto en silencio.
#: Los pesos que NINGUN barrido declarado corre, cada uno con su razon al
#: lado --- la misma forma que las demas exenciones de este arbol, y por la
#: misma razon: un nombre pelado en una exencion y un olvido se leen igual.
#:
#: Los tres son INSTRUMENTOS de diagnostico y no brazos de una seccion. Los
#: corren los scripts que diagnostican, llamando a `corrida` con `peso=`
#: directamente, y meterlos en un barrido declarado lo ensancha: mas computo y
#: columnas que la seccion no reporta. Eso ya paso una vez y lo encontro la
#: suite.
PESOS_SIN_BRAZO = {
    PESO_COMPUERTA: (
        'la señal declarada aplicada como {0,1}. Existe para contestar si el '
        'peso elige o solo tiene pesos, y esa pregunta se contesta con un '
        'diagnostico y no con una columna de la tabla'),
    PESO_BARAJADO: (
        'el CONTROL de mecanismo: los mismos valores de w permutados. Una '
        'tabla que lo reportara como brazo sugeriria que es un metodo que '
        'alguien correria, y no lo es'),
    PESO_ORACULO_EMB: (
        'el techo de la inyeccion en el EMBEDDING, que es otra construccion '
        'que los dos barridos declarados: exige RHO_EMB > 0 y falla cerrado '
        'sin ella, asi que no puede vivir en el padron de intrusos'),
}
assert not (set(PESOS_SIN_BRAZO) & ({p for _, p in BRAZOS_POR_PESO}
                                    | {p for _, p in BRAZOS_POR_PESO_INTRUSOS})), (
    'un peso no puede estar exento Y tener brazo', PESOS_SIN_BRAZO)
assert all(r.strip() for r in PESOS_SIN_BRAZO.values()), PESOS_SIN_BRAZO

# Y la cobertura: entre los dos padrones y las exenciones declaradas, CADA peso
# esta nombrado. Un peso sin brazo Y sin exencion es un modo que nadie
# ejercita, que es como un peso se queda roto en silencio.
assert ({p for _, p in BRAZOS_POR_PESO}
        | {p for _, p in BRAZOS_POR_PESO_INTRUSOS}
        | set(PESOS_SIN_BRAZO)) == set(PESOS), (
    BRAZOS_POR_PESO, BRAZOS_POR_PESO_INTRUSOS, PESOS_SIN_BRAZO, PESOS)

#: La arquitectura de la celda, derivada del brazo que la exige.
POS_CONTAMINACION = 'batchnorm'


def _fila_brazo(nf, cu, peso):
    """Lo que la tabla lee de UNA corrida de CREDA, y nada mas.

    `techo_no_es_metodo` lleva `PALABRA_TECHO` en el NOMBRE de la clave y no
    solo en su valor: la columna del oraculo tiene que ser imposible de leer
    como un metodo, y un `oraculo=True` entre quince campos no lo consigue.
    """
    assert peso in PESOS, (peso, PESOS)
    return dict(acc_te=float(nf['acc_te']), split='test', peso=peso,
                sopesado=bool(peso == PESO_DECLARADO),
                # `in PESOS_TECHO` y no `== PESO_ORACULO`: hay DOS techos desde
                # que existe el de intrusos, y una comparacion contra uno solo
                # habria dejado al otro escribiendo una fila que se lee como un
                # metodo. Es el campo entero de la marca; ver `PESOS_TECHO`.
                techo_no_es_metodo=bool(peso in PESOS_TECHO),
                hp2=dict(nf['hp2']),
                acc_s_te=float(nf['acc_s_te']),
                paso_fin=int(nf['paso_fin']), mejor_paso=int(nf['mejor_paso']),
                vigia_fin=float(nf['vigia_fin']),
                # `mejor_cef` NO entra en esta fila, y es deliberado. La fila
                # de contaminacion es lo que la TABLA lee, y la tabla publica
                # el vigia; agregar la senal del criterio aca obligaria a cada
                # `nf` sintetico de los tests a crecer una llave sin que
                # ninguna tabla la lea. La evidencia del criterio viaja en
                # `barrido.json` --- `mejor_cef`, `cef`, `cef_epoca` --- y en
                # `pasada_2`, que son los registros que SI se leen para eso.
                #
                # Y no se usa `nf.get('mejor_cef')`: un default silencioso
                # dejaria a un registro sin el campo publicando un numero
                # inventado en vez de fallar.
                # El AGRUPAMIENTO, que es la segunda cosa que la seccion 2
                # compara. SUELTAS y nunca el cociente: `tr_e` cayendo son
                # clases que se FUSIONAN y `tr_d` creciendo son nubes que se
                # DESPARRAMAN, dos fenomenos opuestos. Medido en el barrido del
                # sopesado, `tr_e` cae 4.6% (p<0.001) mientras `tr_d` no se
                # mueve (-0.7%, p=0.052); el cociente habria reportado un solo
                # numero y la lectura habria sido otra. El cociente NO se
                # guarda a proposito: tenerlo al lado invita a leerlo.
                #
                # Leen `y_t`, asi que son DIAGNOSTICO, del mismo estatuto que
                # `acc_te`: ni parada, ni busqueda, ni termino.
                tr_e=float(nf['tr_e']), tr_d=float(nf['tr_d']),
                # El pico sobre TEST a lo largo de los puntos de control. Igual
                # que en la tabla principal: informativo, no eligio nada.
                pico_te=float(max(punto(cu, k)['acc_te']
                                  for k in range(len(cu)))))


def _resumen_w(w_cu, idx, yt, k=-1):
    """La dispersion de `w`, que es la cantidad de la que depende la prediccion.

    Se resume y no se persiste entero por semilla: lo que la seccion afirma es
    que las confianzas se SEPARAN entre los puntos contaminados y los limpios,
    y entre clases. Guardar un float por punto del train del blanco, por punto
    de control y por semilla pesaria gigabytes para contestar eso.

    **No es el mismo conteo en las tres familias y por eso no se escribe uno.**
    Medido: 1080 puntos en lunas y 1079 en toroide y rollo. Lunas sortea
    360/360/360 y el 60% estratificado cae entero; las otras dos sortean
    363/384/332 y el piso por clase pierde un punto. Un numero clavado en la
    prosa es correcto en una familia de tres y falso en dos.

    `contaminado` y `limpio` se promedian sobre los puntos de cada grupo en el
    punto de control `k`, que por default es el ULTIMO --- el que la figura
    dibuja. `k=0` da la lectura del PASO 0, que es la premisa de la
    perturbacion: ahi `enc_t` todavia es `enc` y lo que se mide es el cabezal
    congelado de la fuente, que es el unico momento en que la premisa es sobre
    la perturbacion y no sobre lo que la adaptacion haya hecho despues.

    `por_clase` es la dispersion ENTRE clases de la media por clase --- la
    cantidad que estaba declarada como no medida.
    """
    w = np.asarray(w_cu[k], dtype=float)
    mask = np.zeros(len(w), dtype=bool)
    mask[np.asarray(idx, dtype=int)] = True
    medias = [float(w[yt == c].mean()) for c in np.unique(yt)]
    return dict(
        contaminado=(float(w[mask].mean()) if mask.any() else None),
        limpio=float(w[~mask].mean()),
        por_clase=medias,
        dispersion_entre_clases=float(np.std(medias)),
        media=float(w.mean()), minimo=float(w.min()), maximo=float(w.max()))


def barrer_contaminacion(familias=None, *, semillas, rhos=None, HP1=None,
                         trials1=None, trials2=None, al_paso=print,
                         semillas_busqueda=None, modo=MODO_REUBICADO):
    """Un escenario por `(familia, rho)`, CINCO brazos cada uno.

    **Tres de los cinco son CREDA y difieren solo en el peso del bloque del
    blanco**: `apagado`, `declarado` y `oraculo`. Los otros dos son el piso sin
    tocar nada y AdaBN. El tercero de CREDA lee `y_t` y es un TECHO, no un
    metodo; ver `pesos_oraculo` y `BRAZO_TECHO`.

    Que esten los tres es lo que hace informativo al experimento sea cual sea
    el desenlace: `oraculo - apagado` dice cuanto hay para ganar sopesando y
    `declarado - apagado` cuanto de eso captura el estimador de entropia. Con
    dos brazos, un resultado nulo no distinguia 'el sopesado no sirve' de 'no
    habia nada que ganar aca'.

    **UNA busqueda de fase 2 por escenario, corrida una sola vez y compartida
    por los TRES brazos de peso.** No por brazo y no reusada del nivel limpio:

    - buscar por brazo daria 'el mejor de w prendido' contra 'el mejor de w
      apagado', que es otra afirmacion, y volveria a meter el confundido entre
      la busqueda y el termino;
    - reusar los HP limpios leeria el resultado en un punto sintonizado para
      condiciones que ya no valen.

    Con los HP fijos lo unico que varia es el peso, asi que el peso es la
    causa. El limite de alcance se escribe en una linea: la afirmacion es 'con
    ESTOS HP', no 'con cualquier HP'. Y la busqueda corre con el peso
    DECLARADO: el oraculo no es buscable y `buscar_hp2` lo rechaza por nombre.

    **UNA fase 1 por (familia, semilla), construida aca y pasada a los tres
    brazos.** Antes cada `corrida` entrenaba la suya y la igualdad entre brazos
    se apoyaba en que la semilla las hiciera coincidir. Ahora es el MISMO
    objeto, y hace falta que lo sea: `MODO_FRONTERA` elige donde poner cada
    punto leyendo ese cabezal, asi que el cabezal tiene que existir antes de la
    perturbacion y ser el que despues corre.

    **Los HP de la fase 1 se buscan UNA vez por familia**, como hoy: la fase 1
    no mira el blanco y la perturbacion es del lado del blanco, de modo que la
    fuente de un escenario es la misma que la del nivel limpio.

    **La perturbacion entra solo en el TRAIN del blanco**; el test se queda
    intacto, que es lo que hace comparable el acierto entre escenarios.

    Devuelve un `CONTAM` con una celda por `(familia, rho)`. `trials1` y
    `trials2` existen para el ENSAYO: dejan correr el camino entero a escala
    chica sin tocar los valores declarados del modulo.
    """
    familias = FAMILIAS if familias is None else familias
    assert semillas, 'barrer_contaminacion necesita las semillas, sin default'
    assert modo in MODOS_CONTAMINACION, (modo, MODOS_CONTAMINACION)
    SEM = list(semillas)
    LADDER = list(RHOS if rhos is None else rhos)
    assert LADDER, 'la escalera no puede estar vacia'
    # Cada escalon tiene que ser uno de los DECLARADOS.
    #
    # El cero vuelve a correr desde el 2026-10-11: la seccion 2 del cuaderno
    # compara con y sin sopesado POR NIVEL, y sin la fila limpia no hay contra
    # que afirmar que el ruido hizo dano. La fila limpia de la seccion 1 no
    # sirve de reemplazo --- la contaminacion busca fase 2 por escenario, asi
    # que empalmarla mezclaria dos regimenes de HP en una tabla.
    #
    # Lo que el cero probaba ADEMAS --- que el arnes no mueve nada por su
    # cuenta --- sigue probandose donde corresponde, en
    # `tests/test_toy_frontera.py::test_a_zero_fraction_is_a_no_op_for_both_modes`,
    # bit a bit y por identidad de objeto, porque eso una celda de computo no
    # lo demuestra.
    assert all(float(r) in set(float(x) for x in RHOS) for r in LADDER), (
        LADDER, RHOS, 'un escalon que no esta declarado en RHOS')
    assert len(set(float(r) for r in LADDER)) == len(LADDER), (
        LADDER, 'un escalon repetido')
    busq = SEMILLAS_BUSQUEDA if semillas_busqueda is None else list(semillas_busqueda)
    # Disjuntas de las de reporte, por la misma razon que en la corrida
    # principal: buscar y reportar en las mismas semillas ajusta los HP a lo
    # que despues se publica.
    assert not (set(busq) & set(SEM)), (busq, SEM,
        'las semillas de busqueda y las de reporte se solapan')

    hp1_por_fam = dict(HP1 or {})
    # `perturbaciones` es una LISTA y no el nombre suelto que habia antes: un
    # artefacto sostiene las DOS perturbaciones, asi que el campo que las
    # declara tiene que poder nombrar mas de una. Un barrido produce una, de
    # modo que sale con un solo elemento, y `fusionar_contaminacion` es lo que
    # junta dos barridos en el artefacto que el lado que dibuja lee.
    #
    # Y la perturbacion va DENTRO de la clave de la celda, no solo en el padron
    # de arriba: a un mismo rho corren las dos y predicen lo CONTRARIO sobre el
    # par de medias, asi que una clave `(familia, rho)` no distingue los dos
    # escenarios y el segundo barrido pisaria al primero sin que nada fallara.
    CONTAM = dict(rhos=[float(r) for r in LADDER], semillas=list(SEM),
                  pos=POS_CONTAMINACION, brazos=list(BRAZOS_CONTAMINACION),
                  perturbaciones=[modo], hp1={}, hp2={}, celdas={})
    t0 = time.time()
    for fam in familias:
        nombre = fam['nombre']
        if nombre not in hp1_por_fam:
            al_paso(f'  {nombre}: busqueda de fase 1 (una sola, compartida)')
            hp1_por_fam[nombre], _ = buscar_hp1(fam, trials=trials1,
                                                semillas=busq, al_paso=al_paso)
        hp1 = hp1_por_fam[nombre]
        CONTAM['hp1'][nombre] = dict(hp1)
        # Los datos LIMPIOS por semilla, sorteados una vez: cada escenario
        # perturba a partir de ellos, de modo que dos escenarios comparten el
        # sorteo bit a bit y lo unico que los separa es la perturbacion.
        limpios = {s: datos_semilla(fam, s) for s in SEM}
        # Y la fase 1 por semilla, UNA, construida aca. Dos razones, y la
        # segunda es la que obliga: los tres brazos tienen que compartirla bit
        # a bit, y `MODO_FRONTERA` necesita leerla ANTES de perturbar.
        fase1_por_sem = {
            s: fase1_congelada(fam, limpios[s], POS_CONTAMINACION, s, hp1)
            for s in SEM}
        for rho in LADDER:
            al_paso(f'  {nombre} {chr(183)} {modo} {chr(183)} rho={rho:.2f} '
                    f'{chr(183)} busqueda de fase 2 (UNA, para los tres brazos)')
            hp2, _r2 = buscar_hp2(fam, hp1, trials=trials2, semillas=busq,
                                  al_paso=al_paso, rho=rho, modo=modo,
                                  pos_filas=[POS_CONTAMINACION],
                                  peso=PESO_DECLARADO)
            # Lo EFECTIVO, no lo elegido: `corrida` fusiona lo que recibe sobre
            # `HP_2`, asi que `TECHO` --- que no se busca, va clavado en 1.0 y
            # esta medido inerte en cuatro ordenes --- forma parte de lo que la
            # corrida realmente usa. El registro dice lo que CORRIO; lo que
            # Optuna eligio se guarda al lado y por separado.
            hp2_efectivo = {**HP_2, **hp2}
            brazos = {b: [] for b in BRAZOS_CONTAMINACION}
            indices, figuras = {}, {}
            resumen = {b: [] for b, _ in BRAZOS_POR_PESO}
            usado = {b: [] for b, _ in BRAZOS_POR_PESO}
            premisa = {}
            for s in SEM:
                enc0, cab0, hist0, ep0 = fase1_por_sem[s]
                pesar = pesador_del_cabezal(enc0, cab0)
                D, idx = contaminar(limpios[s], rho, s, modo=modo, pesar=pesar)
                indices[s] = [int(j) for j in np.asarray(idx)]
                por_brazo = {}
                for etiqueta, pes in BRAZOS_POR_PESO:
                    # Un deepcopy de la fase 1 POR BRAZO: los tres parten del
                    # mismo cabezal y ninguno puede ver lo que otro le hizo a
                    # sus buffers. Compartir el objeto haria que el orden de
                    # los brazos cambiara los numeros.
                    f1 = (copy.deepcopy(enc0), copy.deepcopy(cab0), hist0, ep0)
                    Ps, cu, pred_s, br, nf = corrida(
                        fam, D, POS_CONTAMINACION, semilla=s,
                        mover_fuente=False, cuadros=CTRL, hp1=hp1, hp2=hp2,
                        fase1=f1, peso=pes)
                    assert nf['peso'] == pes, (nf['peso'], pes)
                    assert nf['hp2'] == hp2_efectivo, (nf['hp2'], hp2_efectivo)
                    brazos[etiqueta].append(_fila_brazo(nf, cu, pes))
                    resumen[etiqueta].append(
                        _resumen_w(nf['w_cu'], idx, D['yt_tr'].numpy()))
                    # El peso que el brazo USO, resumido por sus valores
                    # distintos: es lo que hace legible desde el registro que
                    # el techo pondera por un indicador y el apagado por unos.
                    w0 = np.asarray(nf['w_usado_cu'][0], dtype=float)
                    usado[etiqueta].append(dict(
                        semilla=int(s), paso=0,
                        valores=sorted(set(round(float(v), 6)
                                           for v in np.unique(w0))),
                        media=float(w0.mean()), n=int(len(w0))))
                    por_brazo[etiqueta] = (Ps, cu, pred_s, nf)
                # LA PREMISA de la perturbacion, en el PASO 0 y sobre el peso
                # DECLARADO, que es la cantidad de la que la seccion depende.
                # Va en el artefacto y no solo en un test porque un test
                # sostiene la premisa mientras alguien lo corra; el registro la
                # deja re-medible desde el disco.
                nf_dec = por_brazo[BRAZO_CON_SOP][3]
                p0 = _resumen_w(nf_dec['w_cu'], idx, D['yt_tr'].numpy(), k=0)
                # Y que ese paso 0 sea el cabezal que la perturbacion leyo, no
                # otro: cruza el empalme entre quien eligio donde poner los
                # puntos y quien los pondera despues. Sin esto las dos mitades
                # se verifican por separado y la union nunca.
                w_leido, _ = pesar(D['Xt_tr'])
                assert np.allclose(nf_dec['w_cu'][0], w_leido, atol=1e-5), (
                    float(np.abs(np.asarray(nf_dec['w_cu'][0]) - w_leido).max()),
                    'el paso 0 de la corrida no es el cabezal que la '
                    'perturbacion leyo')
                premisa[s] = dict(
                    perturbado=p0['contaminado'], limpio=p0['limpio'],
                    delta=(None if p0['contaminado'] is None
                           else p0['contaminado'] - p0['limpio']))
                # El piso, del punto de control 0 de la MISMA corrida y sobre
                # TEST --- no sobre valid. `clone_for_target` es un deepcopy
                # puro y el punto 0 se mide en eval(), asi que ahi el encoder
                # del blanco ES el de la fuente. Que salga de la misma corrida
                # es lo que hace apareada la diferencia: comparte la fase 1 bit
                # a bit con los brazos contra los que se compara.
                cus = {b: por_brazo[b][1] for b, _ in BRAZOS_POR_PESO}
                assert int(punto(cus[BRAZO_SIN_SOP], 0)['paso']) == 0, (
                    'el primer punto de control no es el paso 0, asi que no '
                    'mide el modelo sin adaptar')
                piso = float(punto(cus[BRAZO_SIN_SOP], 0)['acc_te'])
                # La fase 1 no depende del peso, asi que el piso de los TRES
                # brazos tiene que ser el MISMO numero. Si no lo es, algo de la
                # fase 1 esta leyendo el peso y la comparacion dejo de ser
                # apareada.
                # Igualdad EXACTA y sin tolerancia: tres aciertos medidos sobre
                # el mismo split con los mismos pesos son el mismo float.
                for b in cus:
                    assert piso == float(punto(cus[b], 0)['acc_te']), (
                        b, piso, float(punto(cus[b], 0)['acc_te']),
                        'los brazos no comparten la fase 1')
                brazos['.predict'].append(dict(acc_te=piso, split='test'))
                ada = adabn(fam, D, s, hp1=hp1)
                brazos['AdaBN'].append(dict(acc_te=float(ada['acc_te']),
                                            split='test'))
                # UNA figura POR BRAZO de peso, no una sola. El gif de la
                # seccion 2 lleva cuatro columnas declaradas por el operador
                # --- con sopesado, sin sopesado, loss de fase 2 con, loss de
                # fase 2 sin --- y las dos de "sin" son del brazo APAGADO.
                # Guardando solo el sopesado, la mitad del gif no tenia de
                # donde salir.
                #
                # Se guarda la PRIMERA semilla, igual que antes: es una
                # trayectoria para dibujar, no una medicion, y las diez no
                # caben en el artefacto.
                if not figuras:
                    for _b, _ in BRAZOS_POR_PESO:
                        figuras[_b] = _figura_contaminacion(
                            s, idx, D, *por_brazo[_b])
            # Largos IGUALES, comprobado aca y no en la tabla: un brazo corto
            # da un Friedman sobre bloques que no existen, y el defecto de
            # largos desiguales ya mordio a este cuaderno tres veces.
            largos = {b: len(v) for b, v in brazos.items()}
            assert set(largos.values()) == {len(SEM)}, (largos, len(SEM))
            assert set(len(v) for v in usado.values()) == {len(SEM)}, usado
            assert set(premisa) == set(SEM), (list(premisa), SEM)
            CONTAM['celdas'][(nombre, modo, float(rho))] = dict(
                hp2=dict(hp2_efectivo), hp2_elegido=dict(hp2),
                busquedas=1, brazos=brazos, indices=indices,
                w_resumen=resumen, w_usado=usado, premisa_w=premisa,
                figuras=figuras,
                # `figura` singular se conserva como ALIAS del brazo con
                # sopesado: el lado que dibuja la seccion vieja lo lee por ese
                # nombre, y romperlo seria un cambio que nadie pidio.
                figura=figuras[BRAZO_CON_SOP])
            al_paso(f'    {nombre} {chr(183)} rho={rho:.2f} listo {chr(183)} '
                    f'{time.time()-t0:6.0f}s acumulados')
        CONTAM['hp2'][nombre] = {
            f'{r:.2f}': dict(CONTAM['celdas'][(nombre, modo, float(r))]['hp2'])
            for r in LADDER}
    esperado = len(list(familias)) * len(LADDER)
    assert len(CONTAM['celdas']) == esperado, (len(CONTAM['celdas']), esperado)
    return CONTAM


def _figura_contaminacion(semilla, idx, D, Ps, cu, pred_s, nf):
    """El cuadro que la figura de UN escenario dibuja, en el paso que eligio.

    Se guarda de UNA semilla --- la primera de las de reporte --- porque la
    figura dibuja puntos y no promedios: una figura de diez semillas
    superpuestas no muestra el tamanio de ninguna marca.

    El punto de control que se elige es el ULTIMO en o antes de `mejor_paso`,
    que es donde la parada dejo el encoder y es el modelo que la tabla reporta.
    Dibujar el ultimo cuadro CORRIDO mostraria un modelo que nadie reporta.
    """
    mejor = int(nf['mejor_paso'])
    pasos = [int(punto(cu, k)['paso']) for k in range(len(cu))]
    alcanzables = [k for k, p in enumerate(pasos) if p <= max(mejor, 0)]
    k = alcanzables[-1] if alcanzables else 0
    return dict(
        semilla=int(semilla), k=int(k), paso=int(pasos[k]), pasos=pasos,
        mejor_paso=mejor, ep_s=int(nf['ep_s']),
        Ps=np.asarray(nf['PS_t'][k], dtype=np.float32),
        Pt=np.stack([punto(cu, j)['proy'] for j in range(len(cu))]
                    ).astype(np.float32),
        rut=np.stack([punto(cu, j)['rut'] for j in range(len(cu))]
                     ).astype(np.int16),
        acc_va=[float(punto(cu, j)['acc_va']) for j in range(len(cu))],
        acc_s=[float(punto(cu, j)['acc_s']) for j in range(len(cu))],
        pok=[float(punto(cu, j)['pok']) for j in range(len(cu))],
        u_t=[float(punto(cu, j)['u_t']) for j in range(len(cu))],
        # La senal del criterio por punto, DERIVADA del `rut` de arriba y no
        # recalculada: es el mismo array que esta figura ya dibuja.
        cef=[float(v) for v in nf['cef_cu']],
        acc_te=[float(punto(cu, j)['acc_te']) for j in range(len(cu))],
        w=[np.asarray(w, dtype=np.float32) for w in nf['w_cu']],
        PS_t=[np.asarray(p, dtype=np.float32) for p in nf['PS_t']],
        pred_s=np.asarray(pred_s, dtype=np.int16),
        idx=np.asarray(idx, dtype=np.int64),
        ys_tr=D['ys_tr'].numpy(), yt_tr=D['yt_tr'].numpy(),
        hist_s=[tuple(float(x) for x in r) for r in nf['hist_s']],
        hist_t=[tuple(float(x) for x in r) for r in nf['hist_t']])


#: El nombre del JSON y el prefijo de los .npz del artefacto de contaminacion.
#: Separado del artefacto principal a proposito: la seccion se puede recorrer,
#: re-dibujar y re-tabular sin volver a correr el barrido de las tablas, que es
#: la misma razon por la que el cuaderno esta partido en dos.
ARCHIVO_CONTAMINACION = 'contaminacion.json'


def fusionar_contaminacion(*CONTAMS):
    """Los barridos de las DOS perturbaciones, en UN artefacto.

    Un barrido corre una perturbacion, asi que cada `CONTAM` que entra aca
    declara una sola. El artefacto que sale declara las de todos y sus celdas
    siguen llevando la suya DENTRO de la clave, de modo que el lado que dibuja
    lee un solo archivo y elige el escenario por el par `(perturbacion, rho)`.

    Antes esto se resolvia con un sufijo por perturbacion y un archivo cada
    una, y el lado que dibuja tenia que acordarse de cual pedir --- un default
    ahi dibuja un escenario bajo el pie del otro, que es el defecto que
    `escenario_del_gif` existe para no cometer.

    **Lo que no es de la celda tiene que COINCIDIR, y no se promedia ni se
    prefiere.** Las semillas, la arquitectura y el padron de brazos son del
    experimento y no de la perturbacion; dos barridos que no coinciden en ellos
    no son dos mitades de un artefacto y juntarlos produciria una tabla que
    compara brazos medidos bajo condiciones distintas sin decirlo.
    """
    assert CONTAMS, 'fusionar_contaminacion no recibio ningun barrido'
    primero = CONTAMS[0]
    for campo in ('semillas', 'pos', 'brazos'):
        valores = [tuple(C[campo]) if isinstance(C[campo], list) else C[campo]
                   for C in CONTAMS]
        assert len(set(valores)) == 1, (campo, valores,
            'los barridos no coinciden en algo que no es de la perturbacion, '
            'asi que no son dos mitades del mismo artefacto')
    nombres, celdas, rhos = [], {}, []
    for C in CONTAMS:
        declaradas = list(C['perturbaciones'])
        assert declaradas, (C.get('perturbaciones'),
                            'un barrido no declara ninguna perturbacion')
        for p in declaradas:
            assert p not in nombres, (p, nombres,
                'dos barridos declaran la misma perturbacion; fusionarlos '
                'dejaria una de las dos mitades pisada en silencio')
            nombres.append(p)
        for clave, celda in C['celdas'].items():
            assert len(clave) == 3, (clave,
                'una celda sin la perturbacion en la clave no se puede '
                'fusionar: su escenario no se distingue del de la otra')
            assert clave[1] in declaradas, (clave, declaradas,
                'la celda nombra una perturbacion que su barrido no declara')
            assert clave not in celdas, (clave, 'celda repetida')
            celdas[clave] = celda
        for r in C['rhos']:
            if float(r) not in rhos: rhos.append(float(r))
    return dict(rhos=sorted(rhos), semillas=list(primero['semillas']),
                pos=primero['pos'], brazos=list(primero['brazos']),
                perturbaciones=list(nombres),
                hp1={p: dict(C['hp1']) for C, p in zip(CONTAMS, nombres)
                     } if len(CONTAMS) > 1 else dict(primero['hp1']),
                hp2={p: dict(C['hp2']) for C, p in zip(CONTAMS, nombres)
                     } if len(CONTAMS) > 1 else dict(primero['hp2']),
                celdas=celdas)


# Un nombre de familia que coincidiera con un nombre de perturbacion haria que
# la clave `familia|perturbacion|rho` de una celda dejara de ser legible: la
# parte del medio y la primera dirian lo mismo y no habria como saber cual es
# cual. Se comprueba al importar y no al escribir: descubrirlo despues del
# barrido cuesta el barrido.
assert not ({f['nombre'] for f in FAMILIAS} & set(MODOS_CONTAMINACION)), (
    'una familia se llama igual que una perturbacion')


def _clave_rho(rho):
    return f'{float(rho):.2f}'


def guardar_entradas(destino, familias=None, *, semillas=None):
    """Persiste las nubes de ENTRADA por `(familia, semilla)`. Devuelve la ruta.

    **Por que existe como artefacto y no como una llamada del cuaderno.** El
    primer panel de la seccion del sopesado es el dato de ENTRADA con los
    puntos atipicos marcados, y no se podia dibujar por una razon estructural y
    no por un olvido: `toy_plots` no importa `torch` --- invariante declarado,
    con test --- asi que no puede llamar a `datos_semilla`, y las entradas 3-D
    no estaban en ningun artefacto. Lo unico que estaba era el latente
    PROYECTADO, en dos dimensiones.

    **Quienes son los atipicos ya era derivable**: el npz del barrido lleva
    `yt_tr` y, por punto de control, `rut` --- la clase predicha --- de modo que
    el conjunto mal ruteado es `rut[k] != yt_tr`. Faltaba solo la nube.

    **Y vive en su propio archivo y no dentro del npz de cada barrido** porque
    es propiedad de `(familia, semilla)` y de nada mas: la misma semilla da las
    mismas nubes para cada seccion, cada rho y cada brazo. Copiarlas en cada
    barrido serian los mismos bytes bajo varios nombres, que es como dos copias
    de una cosa se separan sin que nada falle.
    """
    fams = FAMILIAS if familias is None else familias
    sems = SEMILLAS_REPORTE if semillas is None else semillas
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    A = {}
    for fam in fams:
        for s in sems:
            # `datos_semilla` y no una segunda via de generacion que se le
            # parezca: si se separan, la figura muestra puntos que la corrida
            # nunca vio. Un test lo fija contra esta misma llamada.
            D = datos_semilla(fam, s)
            for campo in ('Xs_tr', 'Xt_tr', 'ys_tr', 'yt_tr'):
                A[f"{fam['nombre']}|{s}|{campo}"] = D[campo].numpy()
    ruta = destino/'entradas.npz'
    np.savez_compressed(ruta, **A)
    return ruta


def guardar_contaminacion(destino, CONTAM):
    """El JSON con todo lo que la tabla lee, y un .npz por familia con la figura.

    **UN artefacto, con las DOS perturbaciones adentro.** Antes era uno por
    perturbacion, distinguidos por un sufijo en el nombre del archivo, y el
    lado que dibuja tenia que pedir el que queria; ahora la perturbacion va en
    la CLAVE de cada celda --- `familia|perturbacion|rho` --- y el padron de
    arriba las declara todas en `perturbaciones`. `perturbaciones_del_artefacto`
    cruza las dos mitades y rechaza que no coincidan: comprobar solo el padron
    verifica la mitad que nadie rompe.

    El campo es una LISTA incluso con una sola perturbacion corrida. Un escalar
    no puede crecer a dos sin que todo lo que lo lee cambie de forma, y la
    forma de lo escrito no deberia depender de cuantas corridas hubo.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    perturbaciones = [str(p) for p in CONTAM['perturbaciones']]
    assert perturbaciones, 'el artefacto no declara ninguna perturbacion'
    for p in perturbaciones:
        assert p in MODOS_CONTAMINACION, (p, MODOS_CONTAMINACION,
            'una perturbacion que el modulo no declara')
    # El padron contra las claves, ANTES de escribir: un artefacto cuyas dos
    # mitades no coinciden se descubre al dibujarlo, y ahi ya costo el barrido.
    en_claves = {str(k[1]) for k in CONTAM['celdas']}
    assert en_claves == set(perturbaciones), (
        sorted(en_claves), sorted(perturbaciones),
        'el padron de perturbaciones y lo que las celdas nombran no coinciden')
    plano = dict(rhos=CONTAM['rhos'], semillas=CONTAM['semillas'],
                 pos=CONTAM['pos'], brazos=CONTAM['brazos'],
                 perturbaciones=perturbaciones,
                 hp1=CONTAM['hp1'], hp2=CONTAM['hp2'], celdas={})
    for (f, perturbacion, rho), celda in CONTAM['celdas'].items():
        plano['celdas'][_clave(f, perturbacion, _clave_rho(rho))] = dict(
            hp2=celda['hp2'], busquedas=celda['busquedas'],
            brazos=celda['brazos'],
            indices={str(s): v for s, v in celda['indices'].items()},
            w_resumen=celda['w_resumen'],
            w_usado=celda['w_usado'],
            # La premisa de la perturbacion, con la semilla por clave de texto
            # como los indices: JSON no tiene claves enteras.
            premisa_w={str(s): v for s, v in celda['premisa_w'].items()})
    ruta = destino/ARCHIVO_CONTAMINACION
    ruta.write_text(json.dumps(plano, indent=2, ensure_ascii=False) + '\n')
    escritos = [ruta]
    familias = sorted({f for f, _, _ in CONTAM['celdas']})
    for nombre in familias:
        A = {}
        for (f, perturbacion, rho), celda in CONTAM['celdas'].items():
            if f != nombre: continue
            # UNA figura POR BRAZO, con el brazo en la clave. Antes se
            # guardaba una sola: el barrido ya las producia las dos, el
            # artefacto se llevaba una, y el gif del sopesado reventaba
            # al dibujar la columna del apagado. Una mitad verificada
            # --- el CONTAM en memoria --- no es la prueba del todo.
            for _brazo, _fig in celda['figuras'].items():
                fig = _fig
                k = lambda campo, _b=_brazo: _clave(
                    perturbacion, _clave_rho(rho), _b, campo)
                A[k('semilla')] = np.asarray(fig['semilla'], dtype=np.int64)
                A[k('k')] = np.asarray(fig['k'], dtype=np.int64)
                A[k('pasos')] = np.asarray(fig['pasos'], dtype=np.int32)
                A[k('mejor_paso')] = np.asarray(fig['mejor_paso'], dtype=np.int32)
                A[k('ep_s')] = np.asarray(fig['ep_s'], dtype=np.int32)
                A[k('Pt')] = fig['Pt']
                A[k('rut')] = fig['rut']
                A[k('PS_t')] = np.stack(fig['PS_t'])
                A[k('w')] = np.stack(fig['w'])
                A[k('idx')] = fig['idx']
                A[k('pred_s')] = fig['pred_s']
                A[k('ys_tr')] = fig['ys_tr']
                A[k('yt_tr')] = fig['yt_tr']
                A[k('acc_va')] = np.asarray(fig['acc_va'], dtype=np.float32)
                A[k('acc_s')] = np.asarray(fig['acc_s'], dtype=np.float32)
                A[k('pok')] = np.asarray(fig['pok'], dtype=np.float32)
                A[k('u_t')] = np.asarray(fig['u_t'], dtype=np.float32)
                A[k('acc_te')] = np.asarray(fig['acc_te'], dtype=np.float32)
                A[k('hist_s')] = np.asarray(fig['hist_s'], dtype=np.float32)
                A[k('hist_t')] = np.asarray(fig['hist_t'], dtype=np.float32)
        ruta = destino/f'contaminacion_{nombre}.npz'
        np.savez_compressed(ruta, **A)
        escritos.append(ruta)
    # Leer de vuelta lo que se acaba de escribir: `np.savez` guarda como objeto
    # un arreglo que no puede serializar y revienta al CARGARLO, en el otro
    # cuaderno y sin ninguna pista de cual es.
    #
    # Se comprueba una clave POR CELDA y no solo la del primer rho: con dos
    # perturbaciones en el mismo archivo, mirar un rho suelto pasa por el lado
    # de la mitad entera que podria faltar.
    for nombre in familias:
        with np.load(destino/f'contaminacion_{nombre}.npz',
                     allow_pickle=False) as z:
            for (f, perturbacion, rho), celda in CONTAM['celdas'].items():
                if f != nombre: continue
                # Una clave por CELDA y por BRAZO: con dos perturbaciones y
                # dos brazos en el mismo archivo, mirar una sola pasa por el
                # lado de las tres mitades que podrian faltar.
                for _brazo in celda['figuras']:
                    esperada = _clave(perturbacion, _clave_rho(rho), _brazo, 'w')
                    assert esperada in z.files, (
                        esperada, sorted(z.files)[:8])
    return escritos


def cargar_contaminacion(origen):
    """Lo que `guardar_contaminacion` dejo, con las claves vueltas a su forma.

    UN artefacto, con todas las perturbaciones que se corrieron adentro. No hay
    selector: elegir cual se lee era elegir el escenario sin decir cual, y la
    eleccion es de `escenario_del_gif`, que la hace por el par y se niega
    cuando hay dos.

    **Lee las DOS formas de clave y nunca inventa la que falta.** La de tres
    partes `familia|perturbacion|rho` es la que escribe el barrido de hoy; la de
    dos, `familia|rho`, es la del artefacto que ya esta en disco, que se corrio
    antes de que la perturbacion se declarara. Esa vuelve como una clave de dos
    y su perturbacion queda en `None` --- no en el nombre de una de las dos ---
    porque leer una declaracion ausente como un valor concreto es lo que dibuja
    un escenario bajo el pie del contrario.
    """
    origen = Path(origen)
    plano = json.loads((origen/ARCHIVO_CONTAMINACION).read_text())
    CONTAM = dict(rhos=plano['rhos'], semillas=plano['semillas'],
                  pos=plano['pos'], brazos=plano['brazos'],
                  hp1=plano['hp1'], hp2=plano['hp2'], celdas={})
    # El padron solo si el artefacto lo trae: el que esta en disco no lo trae y
    # ponerselo vacio o inventado haria que `perturbaciones_del_artefacto`
    # cruzara el padron contra si mismo.
    if 'perturbaciones' in plano:
        CONTAM['perturbaciones'] = [str(p) for p in plano['perturbaciones']]
    for clave, celda in plano['celdas'].items():
        partes = clave.split(SEP)
        if len(partes) == 3:
            f, perturbacion, rho = partes
            nueva = (f, perturbacion, float(rho))
        elif len(partes) == 2:
            f, rho = partes
            nueva = (f, float(rho))
        else:
            raise ValueError(
                f'la clave de celda {clave!r} no tiene 2 ni 3 partes, asi que '
                'no se puede leer de ella ni la familia ni la perturbacion ni '
                'el rho')
        CONTAM['celdas'][nueva] = dict(
            hp2=celda['hp2'], busquedas=celda['busquedas'],
            brazos=celda['brazos'],
            indices={int(s): v for s, v in celda['indices'].items()},
            w_resumen=celda['w_resumen'],
            w_usado=celda.get('w_usado', {}),
            premisa_w={int(s): v
                       for s, v in celda.get('premisa_w', {}).items()},
            figura=None)
    for nombre in sorted({k[0] for k in CONTAM['celdas']}):
        ruta = origen/f'contaminacion_{nombre}.npz'
        if not ruta.exists(): continue
        with np.load(ruta, allow_pickle=False) as z:
            for clave in list(CONTAM['celdas']):
                if clave[0] != nombre: continue
                rho = clave[-1]
                # La clave del arreglo lleva la perturbacion cuando la celda la
                # lleva, y no cuando no: las dos formas de artefacto se leen
                # con el mismo codigo y ninguna se lee con la clave de la otra.
                figs = {}
                for _brazo, _ in BRAZOS_POR_PESO:
                    kb = (lambda campo, _b=_brazo, c=clave, r=rho:
                          _clave(_clave_rho(r), _b, campo) if len(c) == 2
                          else _clave(c[1], _clave_rho(r), _b, campo))
                    if kb('w') not in z.files: continue
                    k = kb
                    figs[_brazo] = dict(
                        semilla=int(z[k('semilla')]), k=int(z[k('k')]),
                        pasos=z[k('pasos')].tolist(),
                        paso=int(z[k('pasos')][int(z[k('k')])]),
                        mejor_paso=int(z[k('mejor_paso')]),
                        ep_s=int(z[k('ep_s')]),
                        Ps=z[k('PS_t')][int(z[k('k')])],
                        Pt=z[k('Pt')], rut=z[k('rut')],
                        PS_t=list(z[k('PS_t')]), w=list(z[k('w')]),
                        idx=z[k('idx')], pred_s=z[k('pred_s')],
                        ys_tr=z[k('ys_tr')], yt_tr=z[k('yt_tr')],
                        acc_va=z[k('acc_va')].tolist(),
                        acc_s=z[k('acc_s')].tolist(),
                        pok=z[k('pok')].tolist(), u_t=z[k('u_t')].tolist(),
                        acc_te=z[k('acc_te')].tolist(),
                        hist_s=[tuple(r) for r in z[k('hist_s')].tolist()],
                        hist_t=[tuple(r) for r in z[k('hist_t')].tolist()])
                if not figs: continue
                CONTAM['celdas'][clave]['figuras'] = figs
                # Alias del brazo sopesado, que es como la seccion
                # vieja lo lee por nombre.
                CONTAM['celdas'][clave]['figura'] = figs.get(
                    BRAZO_CON_SOP, next(iter(figs.values())))
    return CONTAM


# ===========================================================================
# La seccion de INTRUSOS: dano por distancia latente, y quien lo puede quitar
# ---------------------------------------------------------------------------
# Lo que esta seccion pregunta y las dos anteriores no podian: el sopesado por
# confianza (Ecs. 24 y 26) gana algo cuando el dano al bloque del blanco es
# GEOMETRICO. Las dos construcciones anteriores volvieron nulas y los dos nulos
# estan explicados por medicion --- ver `MODO_INTRUSOS` --- de modo que esta no
# es una variante mas: es el unico regimen declarado en el que hay algo para
# que un peso quite.
#
# Los DOS numeros por los que existe, y ninguno se infiere del otro:
#
#   el dano       `apagado` con intrusos contra `apagado` sin ninguno, la misma
#                 maquinaria, las mismas semillas, los mismos HP. Sin el, toda
#                 afirmacion de esta seccion es sobre un dano que nadie midio.
#   la recuperacion  `oraculo-intrusos - apagado` acota lo que quitar los
#                 intrusos puede comprar, y `declarado - apagado` dice cuanto de
#                 eso captura el estimador de entropia.
#
# Dano cerca de cero significa que la intervencion fallo y la seccion no dice
# nada; eso tambien es un desenlace y se reporta. Dano con recuperacion del
# oraculo y sin recuperacion del declarado es el caso informativo. Dano sin
# ninguna recuperacion es un tercer hallazgo.
#
# La celda es `batchnorm`, DERIVADO y no elegido: uno de los brazos es AdaBN y
# AdaBN solo existe donde hay buffers que reestimar.
# ===========================================================================

#: El nombre del artefacto. Es un JSONL y no un JSON por una razon medida: un
#: barrido anterior perdio ~198 escenarios completos porque solo escribia al
#: final y una celda levanto en la ultima. Aca cada celda se escribe cuando
#: termina, de modo que una interrupcion cuesta la celda en curso y no el
#: barrido.
ARCHIVO_INTRUSOS = 'intrusos.jsonl'


def _escribir_jsonl(destino, registro):
    """Un registro por linea, abierto y cerrado en cada llamada.

    Abrir y cerrar cuesta microsegundos contra los segundos de una celda, y es
    lo que hace que el archivo este completo en disco cuando la celda termina en
    vez de cuando el proceso sale: un handle abierto con buffer pierde
    exactamente lo que este formato existe para no perder.
    """
    if destino is None:
        return
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(registro, sort_keys=True) + chr(10))


def barrer_intrusos(familias=None, *, semillas, rhos=None, HP1=None,
                    trials1=None, trials2=None, al_paso=print,
                    semillas_busqueda=None, destino=None):
    """Un escenario por `(familia, rho)`, CINCO brazos cada uno.

    **Tres de los cinco son CREDA y difieren solo en el peso del bloque del
    blanco**: `apagado`, `declarado` y `oraculo-intrusos`. Los otros dos son el
    piso que no toca el blanco y AdaBN, que es el MAS expuesto de todos; ver
    `BRAZOS_INTRUSOS`. El tercero de CREDA lee quienes se inyectaron y es un
    TECHO, no un metodo; ver `pesos_oraculo_intrusos`.

    **UNA busqueda de fase 2 por escenario, compartida por los TRES brazos de
    peso**, y corrida sobre los datos INTRUIDOS de ese escenario. No por brazo
    --- eso daria 'el mejor de w prendido' contra 'el mejor de w apagado', que
    es otra afirmacion --- y no reusada del nivel cero, que leeria el resultado
    en un punto sintonizado para condiciones que ya no valen. Con los HP fijos
    lo unico que varia es el peso, asi que el peso es la causa, y el limite de
    alcance se escribe en una linea: la afirmacion es 'con ESTOS HP'.

    **La escalera incluye el cero y el cero CORRE.** Es el control contra el que
    se lee el dano, y nadie lo tiene medido: `RHOS[0] == 0.0` sigue declarado y
    un test lo afirma, de modo que el control se expresa sin tocar la escalera.

    **UNA fase 1 por (familia, semilla)**, construida aca y pasada a los tres
    brazos por `deepcopy`: tienen que compartirla bit a bit, y `intrusar`
    necesita leer su codificador ANTES de intruir para calibrar el radio.

    **Serializa INCREMENTALMENTE** en `destino`, una linea por celda terminada.

    Devuelve un `INTRUS` con una celda por `(familia, modo, rho)`. `trials1` y
    `trials2` existen para el ENSAYO: dejan correr el camino entero a escala
    chica sin tocar los valores declarados del modulo.
    """
    familias = FAMILIAS if familias is None else familias
    assert semillas, 'barrer_intrusos necesita las semillas, sin default'
    SEM = list(semillas)
    LADDER = list(RHOS if rhos is None else rhos)
    assert LADDER, 'la escalera no puede estar vacia'
    assert all(float(r) in set(float(x) for x in RHOS) for r in LADDER), (
        LADDER, RHOS, 'un escalon que no esta declarado en RHOS')
    assert len(set(float(r) for r in LADDER)) == len(LADDER), (
        LADDER, 'un escalon repetido')
    # El cero tiene que estar, y primero: es el control del dano y leerlo
    # despues de los demas lo volveria un escalon mas de la curva.
    assert float(LADDER[0]) == 0.0, (
        LADDER, 'el control de dano es el escalon cero y va primero')
    busq = SEMILLAS_BUSQUEDA if semillas_busqueda is None else list(semillas_busqueda)
    assert not (set(busq) & set(SEM)), (busq, SEM,
        'las semillas de busqueda y las de reporte se solapan')

    INTRUS = dict(rhos=[float(r) for r in LADDER], semillas=list(SEM),
                  pos=POS_CONTAMINACION, brazos=list(BRAZOS_INTRUSOS),
                  perturbaciones=[MODO_INTRUSOS], factor=float(FACTOR_INTRUSO),
                  radios=[float(r) for r in RADIOS_INTRUSO],
                  hp1={}, hp2={}, celdas={})
    _escribir_jsonl(destino, dict(
        tipo='cabecera', perturbacion=MODO_INTRUSOS,
        familias=[f['nombre'] for f in familias],
        rhos=INTRUS['rhos'], semillas=INTRUS['semillas'],
        semillas_busqueda=list(busq), pos=POS_CONTAMINACION,
        brazos=list(BRAZOS_INTRUSOS), factor=float(FACTOR_INTRUSO),
        radios=INTRUS['radios'], arms_por_peso=[list(x) for x in
                                                BRAZOS_POR_PESO_INTRUSOS]))
    hp1_por_fam = dict(HP1 or {})
    t0 = time.time()
    for fam in familias:
        nombre = fam['nombre']
        if nombre not in hp1_por_fam:
            al_paso(f'  {nombre}: busqueda de fase 1 (una sola, compartida)')
            hp1_por_fam[nombre], _ = buscar_hp1(fam, trials=trials1,
                                                semillas=busq, al_paso=al_paso)
        hp1 = hp1_por_fam[nombre]
        INTRUS['hp1'][nombre] = dict(hp1)
        limpios = {s: datos_semilla(fam, s) for s in SEM}
        fase1_por_sem = {
            s: fase1_congelada(fam, limpios[s], POS_CONTAMINACION, s, hp1)
            for s in SEM}
        for rho in LADDER:
            al_paso(f'  {nombre} {chr(183)} {MODO_INTRUSOS} '
                    f'{chr(183)} rho={rho:.2f} {chr(183)} busqueda de fase 2 '
                    f'(UNA, para los tres brazos)')

            def _perturbar(D_limpio, s, enc, _cab, _rho=rho):
                return intrusar(D_limpio, _rho, s,
                                latente=latente_del_codificador(enc))[0]

            hp2, _r2 = buscar_hp2(fam, hp1, trials=trials2, semillas=busq,
                                  al_paso=al_paso, rho=rho,
                                  pos_filas=[POS_CONTAMINACION],
                                  peso=PESO_DECLARADO,
                                  perturbar_con=_perturbar)
            hp2_efectivo = {**HP_2, **hp2}
            brazos = {b: [] for b in BRAZOS_INTRUSOS}
            indices, premisa, limpios_intactos, w_paso0 = {}, {}, {}, {}
            usado = {b: [] for b, _ in BRAZOS_POR_PESO_INTRUSOS}
            resumen = {b: [] for b, _ in BRAZOS_POR_PESO_INTRUSOS}
            for s in SEM:
                enc0, cab0, hist0, ep0 = fase1_por_sem[s]
                leer = latente_del_codificador(enc0)
                D, idx, prem = intrusar(limpios[s], rho, s, latente=leer)
                indices[str(s)] = [int(j) for j in np.asarray(idx)]
                yt = D['yt_tr'].numpy()
                mascara = np.zeros(len(yt), dtype=bool)
                mascara[np.asarray(idx, dtype=int)] = True
                # LA LICENCIA, re-medida aca sobre el codificador que la corrida
                # va a usar y no leida de lo que `intrusar` devolvio. `intrusar`
                # ELIGIO su radio contra esta cantidad, asi que creerle su
                # propio numero seria la premisa afirmandose contra si misma.
                if prem is not None:
                    vista = distancias_al_centroide(leer(D['Xt_tr']), yt,
                                                    mascara)
                    assert abs(vista['intruso'] - prem['intruso']) < 1e-4, (
                        vista['intruso'], prem['intruso'],
                        'la premisa que la celda registra no es la que el '
                        'codificador de la corrida mide')
                    assert vista['intruso'] >= FACTOR_INTRUSO * vista['limpio'], (
                        vista, FACTOR_INTRUSO,
                        'la celda no tiene licencia: los intrusos no caen '
                        'materialmente mas lejos que los limpios')
                    premisa[str(s)] = dict(prem)
                # Y lo que la correccion del coordinador obliga a mirar: que la
                # REPRESENTACION de los puntos limpios no se movio. En el paso 0
                # el codificador es el congelado de la fase 1 y corre en
                # `eval()`, asi que mapea punto por punto: el latente de los
                # limpios tiene que ser IDENTICO con y sin intrusos. Si no lo
                # fuera, lo que se mide despues seria un rescalado de features y
                # no una intrusion --- el artefacto que una inyeccion por
                # direccion fija produjo, con AdaBN SUBIENDO +0.2139 en lunas.
                z_antes = leer(limpios[s]['Xt_tr'])[~mascara]
                z_ahora = leer(D['Xt_tr'])[~mascara]
                deriva = float(np.abs(z_antes - z_ahora).max())
                assert deriva == 0.0, (deriva,
                    'el latente de los puntos limpios se movio en el paso 0, '
                    'asi que la intervencion rescala las features en vez de '
                    'intruir')
                # La estadistica por feature de la ENTRADA de los limpios, que
                # es exactamente la que era porque sus filas no se tocan. Se
                # mide en vez de afirmarse: es la unica forma de que el
                # reemplazo siga siendo un reemplazo.
                x_antes = limpios[s]['Xt_tr'].numpy()[~mascara]
                x_ahora = D['Xt_tr'].numpy()[~mascara]
                limpios_intactos[str(s)] = dict(
                    latente_max=deriva,
                    media_max=float(np.abs(x_antes.mean(0)
                                           - x_ahora.mean(0)).max()),
                    varianza_max=float(np.abs(x_antes.var(0)
                                              - x_ahora.var(0)).max()),
                    n_limpios=int((~mascara).sum()))
                por_brazo = {}
                for etiqueta, pes in BRAZOS_POR_PESO_INTRUSOS:
                    f1 = (copy.deepcopy(enc0), copy.deepcopy(cab0), hist0, ep0)
                    Ps, cu, pred_s, br, nf = corrida(
                        fam, D, POS_CONTAMINACION, semilla=s,
                        mover_fuente=False, cuadros=CTRL, hp1=hp1, hp2=hp2,
                        fase1=f1, peso=pes, intrusos=idx)
                    assert nf['peso'] == pes, (nf['peso'], pes)
                    assert nf['hp2'] == hp2_efectivo, (nf['hp2'], hp2_efectivo)
                    brazos[etiqueta].append(_fila_brazo(nf, cu, pes))
                    resumen[etiqueta].append(_resumen_w(nf['w_cu'], idx, yt))
                    w0 = np.asarray(nf['w_usado_cu'][0], dtype=float)
                    usado[etiqueta].append(dict(
                        semilla=int(s), paso=0,
                        valores=sorted(set(round(float(v), 6)
                                           for v in np.unique(w0))),
                        media=float(w0.mean()), n=int(len(w0))))
                    por_brazo[etiqueta] = (cu, nf)
                # EL `w` DE LOS INTRUSOS EN EL PASO 0, que es el numero que dice
                # POR ADELANTADO si el brazo declarado puede ganarle al
                # apagado. Esta cantidad no se infiere del dano: `w` alto en los
                # intrusos significa que el estimador de entropia no los ve, y
                # entonces `declarado - apagado` puede salir cero o negativo con
                # el techo recuperando todo --- que es un hallazgo y no un
                # fracaso. Va en el artefacto y no solo en un reporte, porque un
                # numero que nadie puede re-medir desde el disco es un numero
                # que nadie puede re-medir.
                #
                # Del paso 0 y no del ultimo: ahi `enc_t` todavia es `enc` y lo
                # que se lee es el cabezal congelado de la fuente, el unico
                # momento en que la lectura es sobre la INTRUSION y no sobre lo
                # que la adaptacion haya hecho despues.
                nf_ap = por_brazo[BRAZO_SIN_SOP][1]
                p0 = _resumen_w(nf_ap['w_cu'], idx, yt, k=0)
                # Y que ese paso 0 sea el cabezal que la celda calibro, no otro:
                # cruza el empalme entre quien eligio donde poner los puntos y
                # quien los ponderaria despues. Sin esto las dos mitades se
                # verifican por separado y la union nunca.
                w_leido, _ = pesador_del_cabezal(enc0, cab0)(D['Xt_tr'])
                assert np.allclose(nf_ap['w_cu'][0], w_leido, atol=1e-5), (
                    float(np.abs(np.asarray(nf_ap['w_cu'][0]) - w_leido).max()),
                    'el paso 0 de la corrida no es el cabezal que la celda leyo')
                w_paso0[str(s)] = dict(
                    intruso=p0['contaminado'], limpio=p0['limpio'],
                    delta=(None if p0['contaminado'] is None
                           else p0['contaminado'] - p0['limpio']))
                cus = {b: por_brazo[b][0] for b, _ in BRAZOS_POR_PESO_INTRUSOS}
                assert int(punto(cus[BRAZO_SIN_SOP], 0)['paso']) == 0, (
                    'el primer punto de control no es el paso 0, asi que no '
                    'mide el modelo sin adaptar')
                piso = float(punto(cus[BRAZO_SIN_SOP], 0)['acc_te'])
                for b in cus:
                    assert piso == float(punto(cus[b], 0)['acc_te']), (
                        b, piso, float(punto(cus[b], 0)['acc_te']),
                        'los brazos no comparten la fase 1')
                brazos['.predict'].append(dict(acc_te=piso, split='test'))
                ada = adabn(fam, D, s, hp1=hp1)
                brazos['AdaBN'].append(dict(acc_te=float(ada['acc_te']),
                                            split='test'))
            largos = {b: len(v) for b, v in brazos.items()}
            assert set(largos.values()) == {len(SEM)}, (largos, len(SEM))
            assert set(len(v) for v in usado.values()) == {len(SEM)}, usado
            assert set(indices) == {str(s) for s in SEM}, list(indices)
            celda = dict(
                familia=nombre, perturbacion=MODO_INTRUSOS, rho=float(rho),
                hp2=dict(hp2_efectivo), hp2_elegido=dict(hp2), busquedas=1,
                brazos=brazos, indices=indices, premisa=premisa,
                limpios_intactos=limpios_intactos, w_usado=usado,
                w_resumen=resumen, w_paso0=w_paso0,
                segundos=float(time.time() - t0))
            INTRUS['celdas'][(nombre, MODO_INTRUSOS, float(rho))] = celda
            _escribir_jsonl(destino, dict(tipo='celda', **celda))
            al_paso(f'    {nombre} {chr(183)} rho={rho:.2f} listo {chr(183)} '
                    f'{time.time()-t0:6.0f}s acumulados')
        INTRUS['hp2'][nombre] = {
            f'{r:.2f}': dict(INTRUS['celdas'][(nombre, MODO_INTRUSOS,
                                               float(r))]['hp2'])
            for r in LADDER}
    esperado = len(list(familias)) * len(LADDER)
    assert len(INTRUS['celdas']) == esperado, (len(INTRUS['celdas']), esperado)
    return INTRUS
