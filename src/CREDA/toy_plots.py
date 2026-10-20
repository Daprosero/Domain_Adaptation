"""Las graficas del toyset: los paneles, los GIF y las tablas en markdown.

Nada de aca entrena. Lee el artefacto que `toy_experiment.guardar()` dejo en
`CREDA/Results/tesis/toyset/` y dibuja. Esa es toda la razon de que el cuaderno
este partido en dos: retocar un color, un limite o una etiqueta no deberia
costar el barrido entero --- y costaba cuarenta minutos.

La rejilla (FILAS, TABLA, POS_FILAS), los nombres y los umbrales se importan de
`toy_experiment` y no se redeclaran. Dos listas escritas aparte ya divergieron
una vez: las figuras mostraban un brazo que la tabla no tenia, y el brazo que la
tabla reportaba como mejor no tenia figura.

**This module implements no equation, and the provenance below says so.** It
reads an artifact and draws it; nothing here computes a term, and the two
paragraphs above are the two claims it can be held to instead: the grid it
draws is the experiment's own objects rather than a second copy of them, and
nothing in this half of the split trains -- which is why it imports no `torch`
and reads numpy arrays off the artifact. `tests/test_creda_provenance.py` holds
both; neither had a test before this declaration, and both are new.

It does import `scipy.stats`, for the two declared statistical tables at the
bottom of the file. That is not a crack in the rule above: `scipy` reads the
arrays this module already read off the artifact and runs a closed-form test
over them. The rule is about `torch`, which is what would mean something here
had begun computing the term again, and that is the rule the test holds.
"""
import textwrap
from pathlib import Path

import numpy as np
import scipy.stats as _st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D            # el proyector 3D  # noqa: F401
import imageio.v2 as imageio
from PIL import Image as ImagenPIL
from IPython.display import Markdown

from CREDA.toy_experiment import (
    ASPECTO_1D, ASPECTO_MAX, BRAZO_CON_SOP, BRAZO_SIN_SOP, BRAZO_TECHO,
    BRAZOS_CONTAMINACION, CAMPOS_CU, CAMPOS_HS, CAMPOS_HT, CLASES, COLOR,
    COL_DURA, VISTA_DURA, caminata_al_centro, puntos_de_frontera,
    ESCALERA_CONTAMINACION,
    CUADROS,
    ETQ_UNICA, EPOCAS_2, FAMILIAS, FILAS, MARCA, NOMBRE_FILA, PALABRA_TECHO,
    PESO_APAGADO, PESO_DECLARADO, POS_CONTAMINACION, POS_FILAS, RHOS, TABLA,
    cargar, cargar_contaminacion, punto,
)

# `cargar` y `cargar_contaminacion` se RE-EXPORTAN a proposito y no son un
# import muerto: el cuaderno de graficas pide sus dos lectores por un solo
# nombre --- este modulo --- y no por dos. Que el de contaminacion viniera de
# `toy_experiment` y el otro de aca obligaria a un lector a saber cual vive
# donde para contestar la misma pregunta. No se declara un `__all__`: eso
# afirmaria que estos dos son la superficie publica del modulo, y la superficie
# publica son las tablas y las figuras.

__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        "the toy's drawing half, not a term; see the module docstring"),
    "invariants": [
        "the_drawing_module_reads_the_grid_it_draws_from_the_experiment",
        "nothing_in_the_drawing_module_imports_torch",
        # Las dos tablas estadisticas declaradas. No son ecuaciones de la
        # propuesta --- son como se LEE lo medido --- pero cada una tiene una
        # propiedad que se puede romper sin que ningun numero cambie, y las dos
        # ya se rompieron: el signo salio invertido una vez, y una tabla de
        # este repositorio ya reporto 30 pares donde hay 10.
        "the_sign_of_a_toy_table_delta_is_the_first_named_arm_minus_the_second",
        "the_averaged_row_of_a_toy_table_is_a_within_seed_mean",
        # La tabla de cuatro brazos de la seccion de contaminacion, y el eje
        # fijo que la Ec. 21 hizo posible. Las dos se pueden romper sin que
        # ningun numero cambie: el signo ya salio invertido una vez, y un eje
        # log sobre un termino que llega a cero no dibuja su mejor caso.
        "the_four_arm_sign_is_the_first_named_arm_minus_the_second",
        # Era `..._within_its_own_six_pairs`, y el nombre dejo de ser cierto.
        # El seis era todos los pares de cuatro brazos; la familia de hoy es un
        # conjunto DECLARADO de cuatro, con el techo en uno de ellos, y los
        # otros seis se reportan descriptivos. Un invariante cuyo nombre afirma
        # un numero que ya no se corrige es peor que ninguno: su test sigue
        # verde sobre la fixtura vieja y nadie lee el nombre otra vez.
        "the_four_arm_holm_corrects_within_its_own_declared_pairs",
        # El techo fuera del omnibus, que es lo que la herencia del padron de
        # contaminacion habia roto en silencio.
        "the_oracle_ceiling_is_outside_the_friedman_and_outside_the_ranks",
        "the_four_arm_arity_does_not_follow_the_contamination_roster",
        "the_loss_panels_are_fixed_to_the_terms_own_zero_to_one_range",
        # El GIF de un escenario de contaminacion. Los cuadros son los puntos
        # de control de la fase 2 y el sujeto es el train del blanco, asi que
        # cada propiedad de aca se puede romper sin que ningun numero cambie:
        # un conteo de cuadros tomado de la primera fila revienta en la mas
        # corta (ya paso tres veces), un rotulo con la parada de otra fila
        # afirma un paso donde el criterio no disparo, un conjunto de
        # contaminados recalculado dibuja el tamanio correcto en el lugar
        # equivocado, y un par de medias sobre la particion equivocada
        # contesta otra pregunta con la misma forma.
        "the_contamination_gif_truncates_every_row_to_the_shortest",
        "each_contamination_row_is_labelled_with_its_own_stop_step",
        "the_contaminated_points_drawn_are_exactly_the_recorded_indices",
        "the_marker_size_is_monotone_in_w_between_the_declared_endpoints",
        "the_mean_weight_pair_splits_on_the_recorded_indices",
        "the_contamination_figure_names_the_arm_the_artifact_records",
        "the_animated_scenario_is_the_highest_rung_of_the_recorded_ladder",
        # La identidad de un escenario es el PAR (perturbacion, rho): a un
        # mismo rho corren dos perturbaciones que predicen lo CONTRARIO sobre
        # el par de medias, y un brazo que lee `y_t` es un TECHO y no un
        # metodo. Las dos se rompen sin que ningun numero cambie: una dibuja un
        # escenario bajo el pie del otro, la otra publica un oraculo como
        # competidor.
        "two_perturbations_at_one_rho_refuse_to_pick_one_by_default",
        "a_ceiling_arm_is_marked_as_one_or_the_table_says_it_is_not",
        "the_contamination_gif_is_written_with_more_than_one_frame",
        "the_weight_panel_reports_the_ordering_it_measured",
        "the_gif_caption_states_the_measured_ending_and_the_split",
        "the_contamination_accuracy_table_names_what_the_record_cannot_fill",
        # La ESCALERA de la seccion 2: una figura de entradas, un GIF y cuatro
        # tablas. Cada propiedad de aca se puede romper sin que ningun numero
        # cambie, y esa es toda la razon de que esten declaradas: una rejilla
        # transpuesta dibuja las mismas nueve nubes, una quinta columna es otra
        # figura con el mismo nombre, una tabla sin el escalon cero se lee
        # igual que una que lo tiene, un cociente de las dos trazas reporta un
        # solo numero sobre dos fenomenos opuestos, una media sin su mediana ya
        # publico una cola como una tendencia, y Holm a lo largo de la escalera
        # corrige sobre una familia que nadie declaro.
        "the_input_ladder_is_one_figure_of_levels_by_toysets",
        "the_extracted_input_panel_leaves_the_two_column_figure_pixel_identical",
        "the_ladder_frame_keeps_the_four_columns_of_the_trajectory_frame",
        "the_ladder_gif_is_written_with_more_than_one_frame",
        # Renombrado el 2026-10-19 al salir el cero de la escalera de esta
        # seccion: lo que el test mide ahora es el MECANISMO --- un escalon
        # declarado que el barrido salteo se imprime como ausente --- y el cero
        # ya no es uno de los declarados aqui. El padron lo encontro solo.
        "a_declared_rung_the_sweep_skipped_is_printed_as_absent",
        "the_clustering_table_reports_only_the_within_class_trace",
        "no_section_two_table_prints_the_ratio_of_the_two_traces",
        "a_section_two_table_with_a_test_reports_the_median_and_the_win_count",
        "the_weighting_family_is_corrected_within_one_level_and_never_across_the_ladder",
        "the_weighted_and_unweighted_arms_are_read_from_the_recorded_weight",
        "a_ceiling_arm_stays_marked_in_the_ladder_accuracy_table",
        "the_ladder_tables_count_their_seeds_from_the_data",
    ],
}

RB = chr(9671)          # el rombo del centroide, que cuadro_config dibuja

def panel_embedding(a, Ps, cu, k, lim, D, pred_s, por_pred, cuadrada=True):
    # D llega del artefacto, asi que sus etiquetas son arreglos de numpy y no
    # tensores: este modulo no importa torch y no debe. Antes esto llamaba
    # .numpy() sobre ellas, que es lo que hacia falta cuando el mismo cuaderno
    # entrenaba y dibujaba.
    # Ps es la fuente DE ESTE CUADRO, no la del paso 1: en el plan la fuente se
    # mueve, y dibujarla quieta mostraria al blanco acercandose a un lugar que
    # la fuente ya dejo.
    assert len(cu[k]) == CAMPOS_CU, (len(cu[k]), CAMPOS_CU)
    # OCHO nombres, no siete. Al entrar la serie de test el registro crecio, y
    # aunque se agrego AL FINAL para no correr ningun indice, un unpack posicional
    # de siete falla igual --- el assert de arriba comprueba la aridad y pasa, y
    # despues esta linea revienta. Cuarta vez que esta aridad declarada atrapa el
    # mismo defecto.
    # POR NOMBRE, no por posicion. Este unpack se rompio CINCO veces --- cada
    # campo nuevo del registro, aunque se agregara al final, hacia fallar un
    # unpack de N nombres. `punto()` lee por nombre y agregar un campo ya no
    # rompe nada.
    _c = punto(cu, k)
    paso, Pt, acc = _c['paso'], _c['proy'], _c['acc_va']
    pok, pred_t, acc_s, u_t = _c['pok'], _c['rut'], _c['acc_s'], _c['u_t']
    # El COLOR del punto sale de la verdad o del predict. Los rombos son siempre
    # los centroides de la clase VERDADERA, asi que un punto de un color sobre el
    # rombo de otro es un error de lectura.
    col_s = pred_s if por_pred else D['ys_tr']
    col_t = pred_t if por_pred else D['yt_tr']

    # La razon de participacion, calculada ANTES de dibujar: 1 = toda la
    # varianza en una direccion, 2 = ocupa el plano. Medido sobre el cuaderno de
    # una capa, de 1.005 a 1.99 --- varias nubes estan literalmente sobre una
    # recta. Avisarlo no alcanzaba: un scatter 2-D de algo que es 1-D no tiene
    # nada que mostrar, por mucho que el titulo lo aclare. Cuando eso pasa, el
    # panel cambia de DIBUJO y no de rotulo.
    _Z = np.concatenate([Ps, Pt]); _Zc = _Z - _Z.mean(0)
    _lam, _vec = np.linalg.eigh(_Zc.T @ _Zc / max(len(_Zc)-1, 1))
    _lam = _lam.clip(0)
    _part = float(_lam.sum()**2 / max((_lam**2).sum(), 1e-30))

    if _part < ASPECTO_1D:
        # Vista 1-D: todo proyectado sobre la direccion principal. La fuente
        # arriba, el blanco abajo, una fila por clase. Asi se ve lo unico que hay
        # para ver --- si las clases se separan A LO LARGO de la recta, y si el
        # blanco cae donde cae la fuente --- en vez de una raya sin estructura.
        _u = _vec[:, int(np.argmax(_lam))]
        _ps, _pt = (Ps - _Z.mean(0)) @ _u, (Pt - _Z.mean(0)) @ _u
        _vy = D['ys_tr']; _vt = D['yt_tr']
        _rng = np.random.default_rng(0)
        for c in range(3):
            ms, mt = col_s == c, col_t == c
            if ms.any():
                a.scatter(_ps[ms], c + 0.18 + _rng.uniform(-.07, .07, ms.sum()),
                          c=COLOR[c], marker=MARCA['fuente'], s=14,
                          linewidths=.9, alpha=.55)
            if mt.any():
                a.scatter(_pt[mt], c - 0.18 + _rng.uniform(-.07, .07, mt.sum()),
                          c=COLOR[c], marker=MARCA['blanco'], s=12, alpha=.45,
                          edgecolors='none')
            a.plot([_ps[_vy == c].mean()]*2, [c+.02, c+.34], color=COLOR[c], lw=2.4)
            a.plot([_pt[_vt == c].mean()]*2, [c-.34, c-.02], color=COLOR[c], lw=2.4,
                   ls=':')
        a.set_yticks(range(3)); a.set_yticklabels([f'clase {c}' for c in range(3)],
                                                   fontsize=8)
        a.set_ylim(-.6, 2.6); a.set_xlim(lim[0])
        a.set_xlabel('proyección sobre la dirección principal', fontsize=8.5)
        a.grid(alpha=.25, axis='x'); a.tick_params(labelsize=8)
        a.set_title(f'blanco {acc:.3f} {chr(183)} fuente {acc_s:.3f} '
                    f'{chr(183)} ruteo {100*pok:.0f}%' + chr(10) +
                    f'VISTA 1-D {chr(183)} la nube está sobre una recta '
                    f'(part. {_part:.2f})', fontsize=10, linespacing=1.3)
        return

    for c in range(3):
        ms, mt = col_s == c, col_t == c
        mvs, mvt = D['ys_tr'] == c, D['yt_tr'] == c
        if ms.any(): a.scatter(Ps[ms,0], Ps[ms,1], c=COLOR[c], marker=MARCA['fuente'],
                               s=15, linewidths=.9, alpha=.55)
        if mt.any(): a.scatter(Pt[mt,0], Pt[mt,1], c=COLOR[c], marker=MARCA['blanco'],
                               s=13, alpha=.45, edgecolors='none')
        a.scatter(*Ps[mvs].mean(0), marker='D', s=150, facecolors='none',
                  edgecolors=COLOR[c], linewidths=2.2, zorder=5)
        a.scatter(*Pt[mvt].mean(0), marker='D', s=105, c=COLOR[c],
                  edgecolors='k', linewidths=1.0, zorder=6)
    _plano = ''
    a.set_title(f'blanco {acc:.3f} {chr(183)} fuente {acc_s:.3f} '
                f'{chr(183)} ruteo {100*pok:.0f}%' + _plano, fontsize=11)
    a.set_xlim(lim[0]); a.set_ylim(lim[1]); a.grid(alpha=.25)
    # aspect='equal' existe para que la caja CUADRADA valga, y se va con ella:
    # con una caja de 191 a 1 obliga a dibujar el panel como una tira de 191 a 1
    # y desperdicia mas espacio del que la caja cuadrada desperdiciaba. Cuando
    # los ejes no estan a la misma escala el panel lo dice en su rotulo.
    if cuadrada:
        a.set_aspect('equal')
    a.tick_params(labelsize=8)

TINTA = {'renyi': '#2ca02c', 'ce': '#1f77b4', 'total': '#333333'}

def limites_fila(PS, CU, INFO_PS, llaves):
    # Caja CENTRADA y CUADRADA, no el min/max por eje.
    #
    # Centrada: el blanco arranca lejos y termina cerca, asi que el extremo de la
    # caja lo fija UN cuadro y en todos los demas la nube queda pegada a un borde.
    #
    # Cuadrada: una alineacion se juzga por distancias, y con los dos ejes a
    # distinta escala las distancias que se ven no son las que hay -- dos nubes
    # que el dibujo muestra juntas pueden estar lejos en la direccion comprimida.
    # Las dos nubes se mueven, asi que la caja tiene que contener las dos
    # trayectorias enteras. Antes la fuente era un solo arreglo fijo.
    emb = np.concatenate([PS[q] for q in llaves]
                         + [c[1] for q in llaves for c in CU[q]]
                         + [p for q in llaves for p in INFO_PS[q]])
    #
    # Y cuadrada SOLO mientras el dato sea casi isotropo. Medido sobre las
    # familias: lunas sin normalizacion da una relacion de aspecto de 44.5 a 1 y
    # lunas con BatchNorm de 14 a 1 -- el embedding es genuinamente una linea en
    # el plano del PCA. Con la caja cuadrada eso deja el 97.8% del panel vacio y
    # la nube se ve como una raya: no se aprecia nada, que es el motivo de este
    # cambio. toroide da 2 a 1 y ahi la caja cuadrada esta bien.
    #
    # Pasado el umbral se usa un limite POR EJE y el panel lo DICE en su titulo,
    # porque ahi las distancias que se ven ya no son las que hay y un lector que
    # no lo sepa leeria dos nubes como juntas estando lejos en el eje comprimido.
    centro = (emb.max(0) + emb.min(0)) / 2
    rango = emb.max(0) - emb.min(0)
    razon = float(max(rango) / max(float(min(rango)), 1e-9))
    if razon <= ASPECTO_MAX:
        medio = float(np.max(rango)) / 2 * 1.06
        return [(float(centro[d] - medio), float(centro[d] + medio))
                for d in range(2)], razon
    return [(float(centro[d] - rango[d]/2*1.06), float(centro[d] + rango[d]/2*1.06))
            for d in range(2)], razon

def panel_fase1(a, hist_s, titulo, ep_s):
    # La fase 1, hasta la epoca que ELIGIO. Dibuja los DOS cross-entropy --- el
    # de train y el de VALID --- porque es valid el que decide, y antes el panel
    # mostraba el de train al lado del acierto de valid sin decirlo.
    #
    # Y hasta `ep_s`, no hasta el techo de epocas: la fase 1 restaura los pesos
    # de esa epoca y es ESE encoder el que el paso 2 transfiere. Las epocas de
    # paciencia que vienen despues existen para corroborar que era la mejor, no
    # para mostrarse.
    assert hist_s, 'la fase 1 siempre deja historia'
    assert all(len(r) == CAMPOS_HS for r in hist_s), (
        len(hist_s[0]), CAMPOS_HS, 'la historia de la fase 1 cambio de aridad')
    rec = [r for r in hist_s if r[0] <= ep_s]
    assert rec, (ep_s, hist_s[0][0], 'la epoca elegida no esta en la historia')
    assert rec[-1][0] == ep_s, (rec[-1][0], ep_s,
        'la ultima epoca dibujada no es la que la fase 1 eligio')
    # Las POSICIONES, nombradas: al entrar el CE de valid los campos se
    # corrieron y el acierto paso de r[2] a r[3]. El assert de aridad de arriba
    # NO lo habria visto --- la aridad estaba bien y la lectura mal.
    e  = [r[0] for r in rec]
    ce = [r[1] for r in rec]        # CE de TRAIN
    cv = [r[2] for r in rec]        # CE de VALID  <- el que decide
    av = [r[3] for r in rec]        # acierto de VALID
    a.plot(e, ce, color=TINTA['ce'], lw=1.4, alpha=.7,
           label='CE de la FUENTE ' + chr(183) + ' train')
    a.plot(e, cv, color=TINTA['total'], lw=2.0,
           label='CE de la FUENTE ' + chr(183) + ' valid')
    # Lineal y 0 a 1 FIJO, igual que el panel de la fase 2 --- pero NO por el
    # mismo motivo, desde r31: ahi el [0,1] sale del teorema de r30 sobre el
    # span, y aca sale de la construccion. Con la Ec. 21 el termino supervisado
    # vive en [0,1] por construccion --- `B_src = ln(1 + 1/eps)` es el supremo exacto de la
    # perdida de una unidad, y dividir por el manda una prediccion correcta y
    # certera a 0 exacto y el otro extremo a 1 --- asi que el eje fijo NO es
    # una convencion: es el rango del termino, y los dos paneles de la fila se
    # leen de un vistazo en la misma escala.
    #
    # Antes habia un eje LOG, y tenia que haberlo: la cross-entropy cruda
    # arranca en `ln 3 = 1.0986`, por encima de cualquier tope de 1, y no tiene
    # techo arriba, asi que un eje fijo la habria recortado en silencio. Por
    # eso el termino tuvo que cambiar ANTES que el eje y no al revés. Y un eje
    # log tampoco podia dibujar el mejor caso del termino nuevo, que es 0.
    a.set_ylim(0.0, 1.0)
    # El rango REALIZADO, al lado del titulo. Con `B_src = 18.42` el termino
    # ocupa medido el 4% del eje, asi que la curva sale casi plana contra el
    # piso --- y sin este numero eso se lee como un grafico roto en vez de como
    # lo que es: la fuente entrenada hasta una perdida normalizada diminuta. El
    # eje fijo es una decision declarada y se respeta; lo que se agrega es el
    # dato que la vuelve legible.
    _lo, _hi = float(min(min(ce), min(cv))), float(max(max(ce), max(cv)))
    a.set_title(f"{titulo}\n(CE realizada: [{_lo:.4f}, {_hi:.4f}] "
                f"= {100*(_hi-_lo):.1f}% del eje)", fontsize=9)
    a.set_xlim(0, max(e)); a.grid(alpha=.25); a.tick_params(labelsize=8)
    a.set_xlabel('época de la fase 1', fontsize=9)
    a.legend(loc='upper right', fontsize=8, framealpha=.85)
    # El acierto de VALID va en eje propio porque es un acierto y no una
    # perdida. Es el que la parada temprana mira.
    b = a.twinx()
    b.plot(e, av, color='#8c564b', lw=1.2, alpha=.75, ls='--')
    b.set_ylim(0, 1.02); b.tick_params(labelsize=7, colors='#8c564b')
    b.set_ylabel('acierto valid fuente', fontsize=8, color='#8c564b')
    k_mej = int(np.argmax(av))
    assert e[k_mej] == ep_s, (e[k_mej], ep_s,
        'el maximo de la curva dibujada no es la epoca que la fase 1 eligio')
    b.plot([e[k_mej]], [av[k_mej]], 'o', ms=6, color='#8c564b')
    b.annotate(f'para en {e[k_mej]}' + chr(10) + f'val {av[k_mej]:.3f}',
               (e[k_mej], av[k_mej]), textcoords='offset points',
               xytext=(-52, -16), fontsize=7.5, color='#8c564b')

def pasos_por_epoca_del_cuadro(pasos) -> int:
    """El pase de ESTA corrida, recuperado de su propia rejilla de cuadros.

    **Por que se recupera en vez de leerse.** `pasos_por_epoca` sale de
    `LOTE_2`, que la busqueda elige por celda, y el artefacto lo guarda en el
    registro del barrido pero NO en cada figura de contaminacion --- que es
    justo donde los otros cuatro paneles de fase 2 lo necesitan. Escribir una
    constante seria peor que recuperarlo: medido sobre el barrido de hoy el
    pase es 34 en lunas y toroide y 9 en rollo, asi que un literal rotularia
    bien una familia y mal otra sin que nada falle.

    **De donde sale.** `CUADROS` declara la rejilla EN EPOCAS y `corrida` la
    convierte a pasos multiplicando por el pase de la celda, asi que dividir
    la deshace exactamente. No es una estimacion: las tres comprobaciones de
    abajo --- la rejilla tiene el largo declarado, el cociente es unico, y el
    producto reconstruye cada paso --- dejan pasar un solo valor y revientan
    ante cualquier otro. Medido sobre el artefacto de hoy: 90 celdas del
    barrido y 120 brazos de contaminacion, cero cuadros fuera de una frontera
    de epoca.
    """
    pasos = [int(x) for x in pasos]
    # **Un PREFIJO de la rejilla declarada, no necesariamente toda.** La figura
    # de un escenario corta en el punto de control que el criterio eligio, asi
    # que su rejilla es la declarada hasta ahi; exigir el largo completo
    # rechazaria una figura legitima. Lo que NO se afloja es la aritmetica: el
    # cuadro `i` tiene que ser exactamente `CUADROS[i] * pasos_por_epoca`, que
    # es lo que distingue una rejilla en epocas de uno cualquiera en pasos.
    assert 0 < len(pasos) <= len(CUADROS), (
        len(pasos), len(CUADROS),
        'la rejilla del cuadro es mas larga que la rejilla declarada en epocas')
    cuad = CUADROS[:len(pasos)]
    cocientes = {p // e for p, e in zip(pasos, cuad) if e}
    assert len(cocientes) == 1, (
        sorted(cocientes), pasos, cuad,
        'la rejilla no sale de un solo pasos_por_epoca')
    por_epoca = int(next(iter(cocientes)))
    assert all(p == e * por_epoca for p, e in zip(pasos, cuad)), (
        pasos, cuad, por_epoca,
        'algun cuadro no cae en una frontera de epoca')
    return por_epoca


def epoca_elegida_del_brazo(r) -> int:
    """La EPOCA que el criterio eligio en un brazo de contaminacion.

    **Por que se deriva y no se lee.** El barrido guarda `mejor_epoca` celda
    por celda; el registro de contaminacion no --- medido sobre el artefacto de
    hoy, los 120 brazos con traza tienen `mejor_paso` y `paso_fin` y ninguno
    tiene `mejor_epoca`. Asi que o se deriva aca, o la tabla sigue imprimiendo
    el paso de adentro de la epoca, que es el numero que el criterio nunca
    miro.

    **El pase sale de la corrida, no de una constante.** Desde el 2026-10-19 la
    fase 2 no tiene parada temprana: corre su techo entero, asi que `paso_fin`
    ES `EPOCAS_2` pasadas y el cociente es el pase de esta celda. Las dos
    comprobaciones estan para el dia que eso deje de valer: si `paso_fin` no es
    multiplo del techo, la corrida no llego, y derivar un pase de ella seria
    inventarlo. Medido: 0 de 120 brazos fallan cualquiera de las dos.

    Devuelve `-1` tal cual cuando el criterio no eligio --- es el valor que
    `barrer_contaminacion` escribe y la tabla de arriba ya lo imprime como una
    raya con su aviso, no como la epoca 0.
    """
    mejor = int(r['mejor_paso'])
    if mejor < 0:
        return mejor
    fin = int(r['paso_fin'])
    assert fin % EPOCAS_2 == 0, (
        fin, EPOCAS_2,
        'el techo corrido no es un multiplo de las epocas declaradas, asi que '
        'no hay pase que derivar de el')
    por_epoca = fin // EPOCAS_2
    assert por_epoca > 0 and mejor % por_epoca == 0, (
        mejor, por_epoca, 'el paso elegido no cae en una frontera de epoca')
    return mejor // por_epoca


def panel_fase2(a, h, k, titulo, mejor_paso, por_epoca):
    """El panel de la fase 2, dibujado en EPOCAS.

    `h` es la historia de la fase 2 y cada fila suya es UN paso del optimizador
    --- eso es lo que el arreglo guarda ---, asi que el corte por `k` y por
    `mejor_paso` se hace en pasos, que es el indice propio del arreglo.
    Convertir antes de cortar compararia una epoca contra un paso y el filtro
    se quedaria con las filas equivocadas.

    Lo que llega al LECTOR es otra cosa: la abscisa, su limite y el rotulo van
    en epocas, que es la unidad en la que este paquete decide y reporta. El
    divisor entra por parametro y no se deriva aca: `pasos_por_epoca` sale de
    `LOTE_2`, que la busqueda elige por celda --- medido sobre el barrido de
    hoy da 34 en lunas y toroide y 9 en rollo ---, asi que una constante
    escrita aca rotularia bien una familia y mal otra.

    En pasos los tres paneles corrian 0-3400, 0-3400 y 0-900 y no se podian
    mirar uno al lado del otro; en epocas los tres corren 0-100. Esa es la
    razon del cambio, y no la palabra.
    """
    # La fase 2 hasta el cuadro k, y nunca mas alla de `mejor_paso`: ahi es donde
    # la parada temprana dejo el encoder del blanco, y es ese el que se reporta.
    # Los pasos de paciencia que siguen corroboraron que era el mejor; no se
    # dibujan, igual que en la fase 1.
    assert h, 'el paso 3 siempre tiene sus dos terminos'
    # La aridad subio a 7 al entrar el Renyi de valid y el vigia, PERO las cinco
    # primeras posiciones no se movieron, asi que estas lecturas siguen
    # significando lo mismo. El assert queda: la proxima vez pueden moverse.
    assert all(len(r) == CAMPOS_HT for r in h), (
        len(h[0]), CAMPOS_HT, 'la historia de la fase 2 cambio de aridad')
    tope = min(k, mejor_paso)
    hh = [r for r in h if r[0] <= tope] or h[:1]
    assert por_epoca > 0, (por_epoca, 'pasos por epoca tiene que ser positivo')
    s = [r[0] / por_epoca for r in hh]
    ce  = [r[1] for r in hh]        # CE de la fuente, de train
    re_ = [r[2] for r in hh]        # CREDA de train, CRUDO (ya sin factor)
    tot = [r[3] for r in hh]        # el total que recibio backward()
    rv  = [r[4] for r in hh]        # CREDA de VALID <- el que para
    recon = [x + y for x, y in zip(ce, re_)]
    assert all(abs(x - y) < 1e-4*max(1.0, abs(y)) for x, y in zip(recon, tot)), (
        'el total registrado no coincide con los terminos crudos')
    # La CE se calcula pero no tiene camino de gradiente a ningun parametro
    # entrenable: no se dibuja. Con la Ec. 21 ahora vive en la MISMA escala
    # [0,1] que el Renyi, asi que la escala dejo de ser el motivo --- el motivo
    # que queda es el unico que habia: en el regimen declarado no tira de nada,
    # y una curva en el panel de la perdida que no es parte de la perdida se
    # lee como si lo fuera. Dibujarla seria una decision aparte.
    # Y al lado del Renyi de train va el de VALID sin
    # ponderar, que es el que PARA --- dibujarlo no es decoracion: si el
    # criterio lo lee, el lector tiene que poder verlo.
    a.plot(s, re_, color=TINTA['renyi'], lw=2.0,
           label='CREDA ' + chr(183) + ' train')
    a.plot(s, rv, color=TINTA['total'], lw=1.6, ls='--',
           label='CREDA ' + chr(183) + ' valid  (el que para)')
    # Lineal y 0 a 1 FIJO: el eje compartido permite comparar los paneles de un
    # vistazo. El de valid va sobre el MISMO lote de 128 que el de train, no
    # sobre el conjunto entero, justamente para que compartan escala. Con
    # `bounds_form='entropy-cosh'` (r31, Ec. 36) el mapa es genuinamente a
    # [0,1], con la alineacion perfecta en 0 exacto, asi que el eje fijo ya no
    # es una convencion: es el rango del termino. Con las cotas viejas, que
    # dependian de la cardinalidad, todo caia entre 0.457 y 0.509.
    #
    # 2026-10-17: el eje no se movio y la RAZON si. Este comentario derivaba el
    # [0,1] de los dos extremos --- `L_c = -ln 2` y `U_c = 2 ln cosh(...)` ---
    # que es como r28 lo construyo: una afin entre las dos puntas. r31 divide
    # por el literal `ln 2` y el intervalo sale del teorema de r30,
    # `0 <= U_c - J_c <= ln 2`, sin leer el span. Queda MAS fijo que antes, no
    # menos: bajo r29 el tope `ln 2 / (U_c + ln 2)` era inalcanzable en toda
    # clase cuyas entropias marginales difirieran, asi que el 1 del eje no era
    # un valor que el termino pudiera tomar. Ahora lo es.
    a.set_ylim(0.0, 1.0)
    a.set_title(titulo, fontsize=10)
    a.set_xlim(0, max(mejor_paso / por_epoca, 1))
    a.grid(alpha=.25); a.tick_params(labelsize=8)
    a.set_xlabel('época de la fase 2', fontsize=9)
    a.legend(loc='lower right', fontsize=8, framealpha=.9)
    # El vigia: el encoder del BLANCO sobre la fuente etiquetada. Mide cuanto
    # del comportamiento de la FUENTE conserva --- no 'cuanto se movio', que
    # seria una inferencia: un colapso tambien pierde la fuente y no adapta
    # nada. Es legal (usa y_s).
    #
    # YA NO ELIGE EL PASO. Hasta el 2026-10-16 esta nota decia que «decide
    # adentro de la corrida, eligiendo el paso», y desde esa fecha el criterio
    # de la fase 2 es el MAXIMO de `cef` --- el numero efectivo de clases en el
    # ruteo del blanco. El vigia se sigue dibujando porque sigue siendo la
    # lectura de cuanta fuente queda, y porque es lo que la BUSQUEDA DE HP
    # minimiza; lo que dejo de hacer es elegir la epoca.
    #
    # Se dibuja y no decide, que es una combinacion que conviene decir en vez
    # de dejar que el lector la infiera de la curva.
    b = a.twinx()
    # `r[5]` y no `r[6]`: la aridad bajo a 6 el 2026-10-11 al irse el warmup.
    b.plot(s, [r[5] for r in hh], color='#8c564b', lw=1.1, alpha=.7, ls=':')
    b.set_ylim(0, 1.02); b.tick_params(labelsize=7, colors='#8c564b')
    b.set_ylabel('vigía: fuente conservada por enc_t', fontsize=8,
                 color='#8c564b')


#: Los rotulos de las DOS columnas de embedding de un cuadro, escritos UNA vez.
#:
#: Los leen `cuadro_config`, la figura de contaminacion y la escalera. Eran
#: tres literales iguales escritos aparte, y el acuerdo del operador es que
#: estas cuatro columnas no cambian: un literal repetido es como la tercera se
#: queda atras sin que nada falle, y entonces dos GIF de la misma seccion
#: rotulan la misma columna de dos formas.
COL_EMB_REAL = 'embedding ' + chr(183) + ' color = clase REAL'
COL_EMB_PRED = 'embedding ' + chr(183) + ' color = PREDICCIÓN'


def cuadro_config(fam, D, mov, POR_NORM, pred_s, k, lim, asp):
    # Un cuadro por familia. Las filas son las tres normalizaciones, no las
    # variantes de K_mix: asi la figura compara lo que el paso 4 del plan
    # compara. El regimen de la fuente era el segundo eje y se retiro: hoy la
    # fuente esta quieta siempre, asi que hay una figura por familia y no dos.
    #
    # Antes el eje era al reves -- un GIF por normalizacion, las filas eran
    # K_mix crudo / normalizado / fuente quieta -- y la fase 1 se dibujaba una
    # sola vez porque era bit-identica en las tres. Aca NO lo es: cada fila es
    # otra arquitectura y entrena su propia fuente. Las cuatro columnas estan
    # vivas en las tres filas, y la nota de "la fase 1 es la misma de arriba"
    # seria falsa.
    # squeeze=False para que `ax` sea SIEMPRE 2-D. Con una sola fila
    # matplotlib devuelve un arreglo de una dimension y `ax[r,0]` revienta: pasa
    # solo cuando se corre con una normalizacion, que es justo la forma en que
    # se prueba el modulo.
    fig, ax = plt.subplots(len(POS_FILAS), 4, squeeze=False,
                           figsize=(23.5, 5.8*len(POS_FILAS)))
    for r, q in enumerate(POS_FILAS):
        Ps_f, cu, info = POR_NORM[q]
        Ps_k = info['PS_t'][k]            # la fuente de ESTE cuadro
        panel_embedding(ax[r,0], Ps_k, cu, k, lim[q], D, pred_s[q], False, asp[q] <= ASPECTO_MAX)
        ax[r,0].set_title(COL_EMB_REAL + chr(10)
                          + ax[r,0].get_title(), fontsize=10, linespacing=1.4)
        panel_embedding(ax[r,1], Ps_k, cu, k, lim[q], D, pred_s[q], True, asp[q] <= ASPECTO_MAX)
        ax[r,1].set_title(COL_EMB_PRED + chr(10)
                          + ax[r,1].get_title(), fontsize=10, linespacing=1.4)
        panel_fase1(ax[r,2], info['hist_s'],
                    'fase 1 ' + chr(183) + ' hasta la época que eligió',
                    int(info['ep_s']))
        panel_fase2(ax[r,3], info['hist_t'], cu[k][0],
                    'fase 2 ' + chr(183) + ' hasta la época que eligió',
                    int(info['mejor_paso']), int(info['pasos_por_epoca']))
        # Cuando la caja dejo de ser cuadrada hay que DECIRLO: ahi las
        # distancias que se ven no son las que hay.
        # El rotulo de la fila dice que EPOCA eligio ESA fila. Sin esto, una
        # animacion que llega al cuadro de la mas corta deja creer que las tres
        # pararon ahi.
        etq_asp = (NOMBRE_FILA[q] if asp[q] <= ASPECTO_MAX else
                   NOMBRE_FILA[q] + chr(10) + f'(ejes {asp[q]:.0f}:1, no cuadrados)')
        etq_asp += chr(10) + f"eligió la época {int(info['mejor_epoca'])}"
        ax[r,0].set_ylabel(etq_asp, fontsize=11.5, labelpad=10, linespacing=1.4)
    assert not mov, 'la fuente suelta se retiro como brazo'
    reg = 'fuente y cabeza QUIETAS'
    # EL CUADRO, EN EPOCAS. `cu[k][0]` es el PASO del punto de control, no la
    # epoca: la rejilla se declara en epocas y `corrida` la convierte a paso
    # contra su propio `pasos_por_epoca`, asi que para volver hay que dividir
    # por el mismo numero. Escribir `cu[k][0]` bajo la palabra «época» seria
    # rotular un paso como una pasada --- a lote 128 el cuadro 5 cae en el paso
    # 45, y «época 45 de 100» es una figura que miente sobre su propio eje.
    _ppe = int(POR_NORM[POS_FILAS[0]][2]['pasos_por_epoca'])
    _epoca_k = int(POR_NORM[POS_FILAS[0]][1][k][0]) // _ppe
    fig.suptitle(f"{fam['nombre']} {chr(183)} {reg} {chr(183)} "
                 f"época {_epoca_k} de {EPOCAS_2} {chr(183)} "
                 f"azar {1/CLASES:.3f}", fontsize=13, y=0.985)
    pie = (f"`x` fuente, `o` blanco {chr(183)} {RB} hueco = centroide fuente, "
           f"{RB} relleno = centroide blanco, ambos por clase REAL {chr(183)} "
           + ("la fuente TAMBIÉN se mueve en las tres filas"
              if mov else
              "la fuente está quieta: el único término con gradiente es el de Rényi"))
    fig.text(0.5, 0.012, pie, ha='center', fontsize=9.5)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.895, bottom=0.065,
                        hspace=0.26, wspace=0.21)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img

#: El paso de un cuadro y la pausa del ULTIMO, en MILISEGUNDOS. En imageio
#: 2.37 `duration` va en milisegundos y no en segundos: con 0.30 redondeaba a 0
#: y los GIF salian SIN tiempo asignado.
MS_CUADRO, MS_PAUSA = 140, 2500


def cuadros_comunes(largos):
    """Hasta donde puede llegar una animacion cuyas series tienen largos distintos.

    El MINIMO, nunca el primero. Con la parada temprana cada fila corta en un
    paso distinto --- medido, las tres normalizaciones de `lunas` dieron
    [21, 13, 11] cuadros, y la escalera de contaminacion dio [5, 4, 5, 7] ---
    asi que tomar el conteo de la PRIMERA e indexar las demas con el revienta
    con `IndexError` dentro de la mas corta. Paso de verdad tres veces en este
    cuaderno, y `nbconvert` salio 0 igual.

    Alargar hasta la mas larga seria la otra salida y es peor: obligaria a
    congelar las filas cortas, y entonces un mismo cuadro mostraria filas en
    pasos distintos bajo un titulo que dice un solo paso.

    Y nunca se rellena: un largo 0 es una celda que no dejo ningun cuadro, que
    es un defecto del artefacto y no un GIF de cero cuadros.
    """
    largos = [int(v) for v in largos]
    assert largos, 'no hay ninguna serie que animar'
    assert all(v > 0 for v in largos), (largos, 'alguna serie no dejo ningun cuadro')
    return min(largos)


def _escribir_gif(ruta, cuadros, al_paso=print):
    """Codifica los cuadros y LEE DE VUELTA lo que escribio.

    La pausa final va como duracion del ultimo cuadro y no repitiendolo: el
    codificador une cuadros identicos consecutivos y los duplicados
    desaparecen del archivo sin que nada falle.
    """
    ruta = Path(ruta)
    assert cuadros, 'no hay cuadros para escribir'
    ms = [MS_CUADRO]*(len(cuadros) - 1) + [MS_PAUSA]
    imageio.mimsave(ruta, cuadros, duration=ms, loop=0)
    with ImagenPIL.open(ruta) as _g:
        assert _g.n_frames == len(cuadros), (_g.n_frames, len(cuadros))
        _g.seek(_g.n_frames - 1)
        assert _g.info.get('duration') == MS_PAUSA, _g.info.get('duration')
    al_paso(f"      {ruta.name}  {ruta.stat().st_size/1e6:.1f} MB")
    return ruta


def animar(fam, POR_TODO, DAT, destino, al_paso=print):
    # NO corre nada: lee el artefacto que el cuaderno de generacion dejo. Antes
    # esto entrenaba las seis corridas y despues dibujaba, asi que retocar un
    # color costaba el barrido entero. Ahora el generador escribe una vez y esto
    # se puede correr todas las veces que haga falta.
    #
    # Tres entradas: las tres normalizaciones, con la fuente quieta. Eran seis
    # cuando el regimen era un eje.
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    D = DAT[fam['nombre']]
    POR = {}
    for q in POS_FILAS:
        for bal, mov, etq, clv in FILAS:
            k = (fam['nombre'], q, clv)
            assert k in POR_TODO, (k, 'falta en el artefacto: corré el generador')
            Ps, cu, nf, pr, br = POR_TODO[k]
            assert nf['balance'] is bal and nf['mover_fuente'] is mov, (
                k, nf['balance'], nf['mover_fuente'],
                'el artefacto no corresponde a esta rejilla')
            POR[(q, clv)] = (Ps, cu, nf, pr, br)
    # Los limites son POR ARQUITECTURA y no comunes a la figura: dos
    # arquitecturas tienen su propia base de PCA y su propia escala, asi que una
    # caja comun las aplastaria. Antes se compartian entre los dos regimenes de
    # la misma arquitectura, que era lo que habia que poder comparar de un
    # vistazo; con un solo regimen el conjunto tiene una clave sola y el calculo
    # es el mismo.
    lim, asp = {}, {}
    for q in POS_FILAS:
        cl = [(q, clv) for *_, clv in FILAS]
        PS  = {c: POR[c][0] for c in cl}
        CU  = {c: POR[c][1] for c in cl}
        IPS = {c: POR[c][2]['PS_t'] for c in cl}
        lim[q], asp[q] = limites_fila(PS, CU, IPS, cl)
    # Que ninguna figura pise a otra, comprobado ANTES de escribir ninguna: un
    # nombre repetido no da error, da un archivo de menos y nadie lo mira.
    _nombres = [f"paso_{fam['nombre']}_{f[3]}.gif" for f in FILAS]
    assert len(set(_nombres)) == len(_nombres), f"nombres repetidos: {_nombres}"
    gifs = {}
    for bal, mov, etq, clv in FILAS:
        pnorm = {q: (POR[(q, clv)][0], POR[(q, clv)][1], POR[(q, clv)][2])
                 for q in POS_FILAS}
        preds = {q: POR[(q, clv)][3] for q in POS_FILAS}
        # El MINIMO de las tres, no el de la primera. Con la parada temprana
        # cada normalizacion corta en un paso distinto --- medido, lunas da
        # [21, 13, 11] cuadros --- asi que tomar el conteo de la primera e
        # indexar las tres con el revienta con IndexError en la mas corta. Paso
        # de verdad, y nbconvert salio 0 igual.
        #
        # La animacion llega hasta donde la PRIMERA de las tres paro. Alargarla
        # hasta la mas larga obligaria a congelar las otras filas, y entonces un
        # mismo cuadro mostraria tres filas en pasos distintos bajo un titulo
        # que dice un solo paso.
        n = cuadros_comunes(len(POR[(q, clv)][1]) for q in POS_FILAS)
        cuadros = [cuadro_config(fam, D, mov, pnorm, preds, k, lim, asp)
                   for k in range(n)]
        # El nombre sale de `clv`, la clave explicita de FILAS, y NO de una
        # derivada de `mover`. Con la derivada, varias figuras que sueltan la
        # fuente se llamaban igual y se pisaban: quedaba una sola, la ultima.
        # Paso de verdad, despues de 40 minutos de corrida.
        gif = destino/f"paso_{fam['nombre']}_{clv}.gif"
        # El codificador y su lectura de vuelta son los MISMOS que los del GIF
        # de contaminacion: dos escritores del mismo formato divergen, y el que
        # nadie revisa es el que deja un archivo sin tiempo asignado.
        gifs[clv] = _escribir_gif(gif, cuadros, al_paso)
    return gifs

_m = lambda v: f"{np.mean(v):.4f} " + chr(177) + f" {np.std(v):.4f}"

def tabla_dataset(nombre, CELDAS, ADA, semillas):
    # Las filas en ESCALERA DE INTERVENCION, de menos a mas, que es el orden en
    # que hay que leerla: `.predict` (el piso, sin tocar nada), AdaBN (solo las
    # corridas de BatchNorm, sin optimizador), CREDA (el regularizador moviendo
    # el encoder del blanco).
    #
    # Y todo sobre TEST. Las columnas de antes reportaban el split que la parada
    # temprana usaba para elegir, asi que el numero estaba optimistamente
    # sesgado: la epoca se elegia para maximizar exactamente lo que se
    # publicaba.
    #
    # Test se lee en cada punto de control --- no una sola vez --- porque el
    # pico y el paso elegido tienen que estar en el MISMO split para que la
    # brecha entre ellos se pueda leer. Lo que importa sigue en pie: test no
    # participa de NINGUNA decision. La parada mira el Renyi de valid, la
    # busqueda minimiza el Renyi de valid, y la epoca de fase 1 la elige el
    # acierto de la FUENTE. Usar y_t para MEDIR el resultado no es fuga; usarlo
    # para ELEGIR si lo seria.
    # `época que eligió` y no «paso que eligió»: lo que el criterio elige ES
    # una epoca --- el maximo de `cef` se evalua una vez por pasada --- y la
    # columna reportaba el paso de adentro, un numero que el criterio nunca
    # miro. El registro guarda los dos, asi que esto cambia QUE CAMPO se lee y
    # no recalcula nada: medido sobre el barrido de hoy, los 90 `mejor_paso`
    # caen exactamente en su propia `mejor_epoca`.
    enc = ('método', 'normalización', 'época fuente', 'época que eligió',
           'fuente · test', 'blanco · test', 'vigía: fuente conservada',
           'blanco pico · test (oráculo)')
    _med = lambda ac, k: f"{int(np.median([r[k] for r in ac]))}"
    fil = []
    # ---- 1. sin adaptar, una fila por arquitectura
    for pos in POS_FILAS:
        ac = CELDAS[(nombre, pos, ETQ_UNICA)]
        fil.append(('`.predict` (sin adaptar)', NOMBRE_FILA[pos],
                    _med(ac, 'ep_s'), '—',
                    _m([r['acc_s_te'] for r in ac]),
                    _m([r['acc_predict'] for r in ac]), '—', '—'))
    # ---- 2. AdaBN, que solo existe con BatchNorm porque es el unico con
    #         buffers que reestimar. Medido: ninguna y layernorm tienen cero.
    a = ADA[nombre]
    fil.append(('**AdaBN** (sólo buffers, sin optimizador)', 'BatchNorm',
                '—', '—', _m([r['acc_s_te'] for r in a]),
                _m([r['acc_te'] for r in a]), '—', '—'))
    # ---- 3. CREDA, en la epoca que su criterio eligio
    for pos in POS_FILAS:
        for bal, mov, etq, clv in TABLA:
            ac = CELDAS[(nombre, pos, etq)]
            # El pico sobre TEST, el MISMO split que la columna de la epoca
            # elegida. Antes era sobre valid, y entonces la brecha entre las dos
            # mezclaba 'el criterio no acerto la mejor epoca' con 'valid no es
            # test': no se podia leer como ninguna de las dos.
            #
            # Es informativo y nada mas: no eligio ninguna epoca y no entro en
            # ninguna busqueda. El pico es sobre los PUNTOS DE CONTROL, asi que
            # es una cota inferior del verdadero.
            pico = _m([max(r['acc_te_ser']) for r in ac])
            fil.append(('**CREDA** · HP sintonizados', NOMBRE_FILA[pos],
                        _med(ac, 'ep_s'), _med(ac, 'mejor_epoca'),
                        _m([r['acc_s_te'] for r in ac]),
                        _m([r['acc_te'] for r in ac]),
                        _m([r['vigia_fin'] for r in ac]), pico))
    # Que cada fila tenga tantas celdas como el encabezado. Sacar una columna y
    # olvidar una fila da una tabla que markdown renderiza igual, corrida un
    # lugar, y los numeros quedan bajo el rotulo de al lado.
    assert all(len(f) == len(enc) for f in fil), (len(fil[0]), len(enc))
    assert len(fil) == 2*len(POS_FILAS) + 1, (len(fil), len(POS_FILAS))
    md_ = ['| ' + ' | '.join(enc) + ' |', '|' + '|'.join(['---']*len(enc)) + '|']
    md_ += ['| ' + ' | '.join(r) + ' |' for r in fil]
    return Markdown(f"**{nombre} {chr(183)} {semillas} semillas "
                    f"{chr(183)} media {chr(177)} desvío {chr(183)} "
                    f"azar {1/CLASES:.4f} {chr(183)} test, sólo reporte**"
                    + chr(10)*2 + chr(10).join(md_))

#: El rotulo del balance de `K_mix`, derivado de la bandera y no escrito a mano.
#: Un valor nuevo de `BALANCE_DE_MASA` revienta aca en vez de dejar un titulo
#: que afirma una normalizacion que la corrida no uso --- que es exactamente lo
#: que paso con "K_mix / n_s, n_t" cuando el balance paso a ser por traza.
ETIQUETAS_BALANCE = {
    'trace': 'balanceado por traza (Ec. 32)',
    'cardinality': 'balanceado por cardinalidad',
}


def etiqueta_balance(banderas):
    """Como nombrar el balance de `K_mix` de ESTA corrida."""
    clave = banderas['BALANCE_DE_MASA']
    if clave not in ETIQUETAS_BALANCE:
        raise KeyError(
            f"no hay rotulo para BALANCE_DE_MASA={clave!r}; "
            f"conocidos: {sorted(ETIQUETAS_BALANCE)}")
    return ETIQUETAS_BALANCE[clave]


# ===========================================================================
# Las dos tablas estadisticas declaradas
# ---------------------------------------------------------------------------
# El metodo esta FIJADO, antes de implementar, en
# `odd/tasks/toyset-tests-estadisticos-y-ruido.md`, seccion "DECLARED METHOD,
# fixed 2026-10-10 before implementation", y la referencia es Demsar (2006),
# *Statistical Comparisons of Classifiers over Multiple Data Sets*:
#
#   dos brazos    -> t pareado, con Shapiro-Wilk al lado (la normalidad es lo
#                    que LICENCIA el t, no algo que se asume) y Wilcoxon
#                    tambien al lado, para que un lector vea si los dos
#                    coinciden.
#   tres brazos   -> Friedman con 3 tratamientos x 10 bloques de semilla, y
#                    post-hoc de Wilcoxon por pares corregido con Holm DENTRO
#                    de la familia de 3 de ese toyset.
#
# Se fijo una sola vez a proposito: sobre estos mismos datos se corrieron
# cuatro metodos distintos en respuesta a objeciones sucesivas, y el efecto
# acumulado de eso es method-shopping aunque cada paso contestara algo justo.
# No se sustituye por otro test aca.
#
# NINGUNA de las dos imprime un veredicto. Reportan la diferencia, el
# estadistico, la p, el IC95, la p ajustada y el conteo de ganadas: el numero
# es el reporte y el umbral lo aplica quien lee.
#
# El emparejamiento es REAL y no una suposicion: los dos brazos de cada par
# comparten semilla, datos y pesos de fase 1 bit a bit, y `toy_experiment`
# construye `CELDAS` y `ADA` recorriendo `SEMILLAS_REPORTE` en el mismo orden
# --- con un assert propio de que `acc_predict` de la celda de BatchNorm es el
# `blanco_ini` de AdaBN semilla por semilla. Por eso el indice posicional ES la
# semilla, y por eso un par se puede restar.

#: Los tres brazos de la celda de BatchNorm, en ESCALERA DE INTERVENCION: el
#: piso sin tocar nada, despues solo los buffers, despues el regularizador.
#: Declarados una sola vez porque el que compara y el que imprime los leen los
#: dos, y dos listas escritas aparte en este archivo ya divergieron una vez.
BRAZO_PISO = '.predict'
BRAZO_ADABN = 'AdaBN'
BRAZO_CREDA = 'CREDA'

#: Los pares del post-hoc, en ese mismo orden, y cada uno con el brazo MAS
#: intervenido primero. El orden del par es fijo y no sigue al ganador: si
#: siguiera al ganador, un signo siempre positivo no diria nada.
PARES_TRES_BRAZOS = (
    (BRAZO_ADABN, BRAZO_PISO),
    (BRAZO_CREDA, BRAZO_PISO),
    (BRAZO_CREDA, BRAZO_ADABN),
)

#: La normalizacion que AdaBN necesita: es el unico lugar donde hay buffers que
#: reestimar. Medido: `ninguna` y `layernorm` no tienen ninguno.
POS_ADABN = 'batchnorm'

#: El rotulo de la fila que promedia. No es una normalizacion mas, asi que su
#: `clave` es None y nada la confunde con una.
ROTULO_PROMEDIO = 'PROMEDIO'

#: El eje que esa fila colapsa, dicho donde se colapsa. Una agregacion que no
#: nombra su eje produce algo con la forma correcta y otro significado.
EJE_PROMEDIO = 'la normalizaci\u00f3n, DENTRO de cada semilla'

#: Por debajo de este rango, las diferencias pareadas no tienen dispersion que
#: estimar y lo que se mide es ruido de punto flotante. El numero no es una
#: tolerancia elegida a ojo: lo que se resta son aciertos sobre el split de
#: test, que tiene ~360 puntos, asi que la granularidad de una diferencia real
#: es ~1/360 = 3e-3 y cualquier rango por debajo de 1e-12 es cancelacion
#: catastrofica, no datos. Sin el, un conjunto de diferencias identicas hasta
#: 1e-17 da `t = 7.5e15`, que se imprime como si fuera un estadistico.
RANGO_MINIMO = 1e-12


def _sin_dispersion(dif):
    return bool(np.ptp(dif) <= RANGO_MINIMO)

_NADA = chr(8212)                       # la raya para una celda sin numero
_DELTA = chr(916)                       # el Delta de la columna
_CHI = chr(967)                         # la chi de Friedman


def _holm(crudas):
    """Holm paso a paso sobre UNA familia: `m*p` al menor, `m-1` al siguiente.

    Monotono por construccion --- el ajustado de cada p no puede quedar por
    debajo del de una p menor --- y topado en 1. `m` es el largo de lo que se
    le pasa, de modo que la familia es una decision de quien llama y no algo
    que esta funcion adivina: las dos tablas corrigen dentro de los 3 pares de
    UN toyset, nunca sobre los 9 de los tres juntos.
    """
    # Una fila SIN p no es parte de la familia, y por eso `m` la excluye.
    #
    # `_par` devuelve `wilcoxon=None` cuando las diferencias pareadas son todas
    # exactamente cero --- dos brazos que salieron bit a bit iguales, que en la
    # corrida real paso en 3 de 12 celdas de contaminacion (`rollo|0.30`,
    # `toroide|0.00`, `toroide|0.10`): el acierto vive en una rejilla de 1/362,
    # asi que una diferencia de perdida del 0.9% puede no mover ni un punto de
    # test. Eso no es un test que fallo, es un test que no se pudo hacer.
    #
    # Contarla en `m` corregiria por una comparacion que no existe y aflojaria
    # a las demas; inventarle `p=1.0` afirmaria que se midio y no se rechazo.
    # Se la saca del denominador y vuelve como `None`, que la tabla imprime
    # como raya. Esto reventaba con `TypeError: float() ... not 'NoneType'`
    # despues de 29 minutos de generador, asi que el caso ahora tiene test.
    vivas = [(i, float(p)) for i, p in enumerate(crudas) if p is not None]
    salida = [None] * len(crudas)
    m = len(vivas)
    if m == 0:
        return salida
    orden = sorted(range(m), key=lambda k: vivas[k][1])
    corriendo = 0.0
    for i, k in enumerate(orden):
        j, p = vivas[k]
        corriendo = max(corriendo, (m - i) * p)
        salida[j] = min(corriendo, 1.0)
    return salida


def _semillas(largos, banderas, que):
    """El conteo de semillas sale del DATO, con la bandera como contraste.

    `largos` son los largos de lo que hay en el artefacto. La bandera dice lo
    que se declaro para reporte. Si no coinciden, no hay tabla que rotular: una
    tabla de este repositorio ya dijo "5 semillas" sobre numeros de 10, porque
    leia la bandera. Que no coincidan se RECHAZA en vez de preferir una de las
    dos --- preferir el dato callaria que la corrida quedo corta, y preferir la
    bandera es el defecto que ya paso.

    Es un `ValueError` y no un `assert` porque un assert desaparece bajo
    `python -O` y entonces el rotulo vuelve a poder mentir.
    """
    largos = list(largos)
    if not largos:
        raise ValueError(f'{que}: no hay celdas de las que leer las semillas')
    if len(set(largos)) != 1:
        raise ValueError(
            f'{que}: las celdas no tienen todas las mismas semillas ({largos}); '
            'promediar dentro de la semilla sobre celdas de largos distintos '
            'no esta definido')
    n = largos[0]
    declaradas = len(banderas['SEMILLAS_REPORTE'])
    if n != declaradas:
        raise ValueError(
            f'{que}: el artefacto tiene {n} semillas y SEMILLAS_REPORTE '
            f'declara {declaradas}; una de las dos quedo vieja y rotular la '
            'tabla con cualquiera de ellas afirma algo que no se midio')
    return n


def _celda(CELDAS, nombre, pos):
    clave = (nombre, pos, ETQ_UNICA)
    if clave not in CELDAS:
        raise ValueError(
            f'no hay celda {clave} en el artefacto; POS_FILAS la declara y el '
            'barrido no la dejo')
    return CELDAS[clave]


def _par(primero, segundo, a, b, n, *, campo):
    """Un par ya restado, con TODO lo que la tabla reporta de el.

    `delta = a - b`, y `a` es el brazo nombrado PRIMERO. Esa es la propiedad
    que la tabla tiene que mantener: `delta > 0` quiere decir que gana el
    primero. Salio invertida una vez --- el brazo restado iba en la primera
    columna, asi que `+0.0819` al lado de `.predict` se leia como que gana
    `.predict` cuando quiere decir lo contrario --- y una tabla cuyo signo se
    lee al reves es peor que no tener tabla, porque se lee con confianza.
    """
    dif = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta = float(dif.mean())
    # En que direccion se gana ESTA cantidad. Sin `campo` queda la polaridad
    # historica (`sube`), que es la del acierto y la de casi todo lo demas.
    # Sin default: `campo` es obligatorio. Un default que coincide con la
    # mitad de las cantidades es exactamente como `tr_d` quedo contando al
    # reves sin que nada fallara.
    if campo not in MEJOR_SI:
        raise ValueError(
            f'{campo!r} no declara en que direccion se gana; agregalo a '
            f'MEJOR_SI. Sin eso el conteo de victorias usa el signo del delta, '
            f'que para una cantidad que gana BAJANDO cuenta las derrotas.')
    _dir = MEJOR_SI[campo]
    # El IC95 del t pareado sobre la diferencia. Es el intervalo que este
    # proyecto ya reporto (lunas CREDA-AdaBN: -0.0508, [-0.1609, +0.0592]), y
    # se reporta tambien al lado de un Wilcoxon: el test decide la p, el
    # intervalo dice de que tamano es la diferencia, y las dos cosas hacen
    # falta porque una p sola no tiene magnitud.
    if _sin_dispersion(dif):
        # Sin dispersion no hay error estandar que estimar. Se dice que no se
        # pudo en vez de imprimir un `nan` o un `inf`, que se leen como numeros.
        ic = (delta, delta)
    else:
        err = float(dif.std(ddof=1) / np.sqrt(len(dif)))
        tc = float(_st.t.ppf(0.975, len(dif) - 1))
        ic = (delta - tc * err, delta + tc * err)
    return {
        'rotulo': f'{primero} vs {segundo}',
        'primero': primero,
        'segundo': segundo,
        'delta': delta,
        # La MEDIANA de la misma diferencia pareada, al lado de su media y
        # calculada aca --- en el nucleo --- y no en cada tabla.
        #
        # El motivo esta medido: una metrica de este proyecto leyo media
        # -0.0254, que se ve como una mejora, mientras la mediana era +0.0119 y
        # solo 28 de 90 pares mejoraban. Era una COLA y no una tendencia, y la
        # media sola no podia decirlo. Un campo que cada tabla calculara por su
        # cuenta es uno que alguna tabla no calcula.
        'mediana': float(np.median(dif)),
        'ic': ic,
        'gana': int((dif > 0).sum() if _dir == 'sube' else (dif < 0).sum()),
        'mejor_si': _dir,
        'n': n,
        'wilcoxon': (None if np.all(dif == 0)
                     else float(_st.wilcoxon(dif).pvalue)),
    }


def _t_pareado(f, a, b):
    """El t pareado sobre `a - b`, su Shapiro y sus grados de libertad.

    Extraido de `_fila_dos_brazos` para que las tablas de la escalera corran el
    MISMO t y el MISMO Shapiro en vez de una segunda copia. Lo que cada tabla
    aporta es de donde salen sus dos brazos; el test es este.

    Sin dispersion no hay error estandar que estimar y Shapiro no esta
    definido: se reportan como no medidos en vez de como un infinito y un
    `nan`, que se leen como numeros.
    """
    dif = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if _sin_dispersion(dif):
        f['t'], f['p'], f['shapiro'] = None, None, None
    else:
        r = _st.ttest_rel(np.asarray(a, dtype=float),
                          np.asarray(b, dtype=float))
        f['t'] = float(r.statistic)
        f['p'] = float(r.pvalue)
        f['shapiro'] = float(_st.shapiro(dif).pvalue)
    f['gl'] = len(dif) - 1
    return f


def comparar_sin_contra_con(nombre, CELDAS, banderas):
    """Los tres brazos sin CREDA contra los tres con CREDA, en UN toyset.

    Dos brazos, asi que t pareado --- con Shapiro-Wilk sobre las diferencias
    pareadas al lado, porque la normalidad es lo que licencia el t en vez de
    algo que se asume --- y Wilcoxon reportado tambien, para que se vea si los
    dos coinciden. Nada de Friedman: Friedman es para tres o mas.

    `sin CREDA` es `acc_predict` y `con CREDA` es `acc_te`, el valor FINAL por
    semilla de reporte, en el paso que el criterio eligio. Nunca un punto de
    trayectoria, nunca el pico de oraculo, nunca una semilla de busqueda.

    **La fila PROMEDIO promedia las tres normalizaciones DENTRO de cada
    semilla, lo que da 10 pares independientes, y NUNCA agrupa los 30 pares de
    celda-semilla.** Esa es toda la razon de que se calcule asi: la semilla 0
    aparece en las tres normalizaciones, asi que los 30 pares no son 30
    observaciones independientes y agruparlos es pseudo-replicacion. El
    promedio puntual sale igual de las dos formas --- es el denominador el que
    cambia --- de modo que agrupar no mueve la columna del delta y multiplica
    el estadistico, que es la forma mas silenciosa que tiene un numero de
    quedar mal: la tabla sigue teniendo la forma correcta.

    `delta = con CREDA - sin CREDA`, de modo que `delta > 0` quiere decir que
    gana CREDA.
    """
    pos_filas = list(banderas['POS_FILAS'])
    celdas = {pos: _celda(CELDAS, nombre, pos) for pos in pos_filas}
    n = _semillas((len(c) for c in celdas.values()), banderas,
                  f'{nombre}: celdas de tabla')

    filas = []
    sin_por_pos, con_por_pos = [], []
    for pos in pos_filas:
        ac = celdas[pos]
        sin = np.array([r['acc_predict'] for r in ac], dtype=float)
        con = np.array([r['acc_te'] for r in ac], dtype=float)
        sin_por_pos.append(sin)
        con_por_pos.append(con)
        filas.append(_fila_dos_brazos(NOMBRE_FILA[pos], pos, sin, con, n))

    # El promedio DENTRO de la semilla: un vector de 10, no una pila de 30.
    filas.append(_fila_dos_brazos(
        ROTULO_PROMEDIO, None,
        np.mean(sin_por_pos, axis=0), np.mean(con_por_pos, axis=0), n))

    return {
        'familia': nombre,
        'n': n,
        'primero': 'con CREDA',
        'segundo': 'sin CREDA',
        'eje_colapsado': EJE_PROMEDIO,
        'filas': filas,
    }


def _fila_dos_brazos(rotulo, clave, sin, con, n):
    # El t, el Shapiro y los gl los pone `_t_pareado`, que es el mismo que
    # corren las tablas de la escalera. Lo propio de esta fila son sus dos
    # columnas de nivel --- `sin` y `con` --- que son de esta tabla y de
    # ninguna otra.
    # `campo` EXPLICITO aunque el default coincida: que el conteo salga bien
    # por suerte y no por declaracion es como `tr_d` quedo invertido.
    f = _par('con CREDA', 'sin CREDA', con, sin, n, campo=CAMPO_ACIERTO)
    _t_pareado(f, con, sin)
    f['rotulo'] = rotulo
    f['clave'] = clave
    f['sin'] = float(np.mean(sin))
    f['con'] = float(np.mean(con))
    return f


def comparar_n_brazos(brazos, pares, n, *, familia, normalizacion=None,
                      escenario=None, extra=None):
    """Friedman sobre `len(brazos)` tratamientos y `n` bloques, con Holm en `pares`.

    **El NUCLEO, uno solo, del que salen las dos tablas de tres y de cuatro
    brazos.** No hay un segundo Friedman ni un segundo Holm en este archivo:
    dos implementaciones del mismo test divergen, y la que divergiera seria la
    que nadie volvio a leer. Lo que cada tabla aporta es DE DONDE salen sus
    brazos y como se rotula; el test es este.

    `brazos` es un diccionario ORDENADO etiqueta -> una observacion por bloque
    de semilla. El orden importa dos veces: es el orden en que Friedman recibe
    los tratamientos y el orden en que se imprimen los rangos.

    `pares` son los pares del post-hoc, cada uno `(primero, segundo)` con el
    brazo MAS intervenido primero. El orden del par es fijo y NO sigue al
    ganador: si siguiera al ganador, un signo siempre positivo no diria nada.

    La familia de correccion es una decision de quien llama --- los 3 pares de
    un toyset, los 6 de un escenario --- y nunca algo que esta funcion adivine.
    `m` de Holm es el numero de pares que PRODUJERON una p: un par cuyos dos
    brazos salieron bit a bit iguales no se pudo testear, asi que sale del
    denominador en vez de aflojar la correccion de los demas. `familia_de`
    reporta ese numero real, no `len(pares)`, y cuando difieren la tabla lo
    dice.

    Los rangos son rangos MEDIOS sobre los bloques y el rango ALTO es el brazo
    mejor: `len(brazos)` es el mejor de `len(brazos)`.

    `extra` son brazos que un par puede nombrar y que NO son tratamientos: el
    techo de oraculo, hoy. Entran en el post-hoc y nunca en el Friedman ni en
    los rangos, que es toda la razon de que esten separados del diccionario de
    arriba en vez de marcados adentro de el --- una marca se puede ignorar al
    armar la pila, y una clave que no esta en el diccionario no.

    `delta = (primero nombrado) - (segundo nombrado)`, de modo que `delta > 0`
    quiere decir que gana el PRIMERO.
    """
    etiquetas = list(brazos)
    if len(etiquetas) < 3:
        raise ValueError(
            f'{familia}: Friedman necesita 3 o mas tratamientos relacionados y '
            f'recibio {len(etiquetas)} ({etiquetas}); para dos brazos la que '
            'corresponde es la tabla del t pareado')
    datos = {b: np.asarray(brazos[b], dtype=float) for b in etiquetas}
    largos = sorted({len(v) for v in datos.values()})
    if largos != [n]:
        raise ValueError(
            f'{familia}: los brazos no tienen todos las {n} semillas '
            f'({ {b: len(v) for b, v in datos.items()} }); un brazo corto da '
            'un Friedman sobre bloques que no existen')
    # Los que un par puede nombrar sin ser tratamiento. Se guardan APARTE de
    # `datos` hasta despues del Friedman, de modo que no haya ninguna via por la
    # que entren a la pila de rangos.
    fuera = {b: np.asarray(v, dtype=float)
             for b, v in (extra or {}).items() if b not in datos}
    for b, v in fuera.items():
        if len(v) != n:
            raise ValueError(
                f'{familia}: el brazo {b!r} tiene {len(v)} semillas y los '
                f'tratamientos tienen {n}; un brazo corto da un par sobre '
                'bloques que no existen')
    for a, b in pares:
        for brazo in (a, b):
            if brazo not in datos and brazo not in fuera:
                raise ValueError(
                    f'{familia}: el par ({a!r}, {b!r}) nombra un brazo que no '
                    f'esta entre {etiquetas} ni entre los no tratamientos '
                    f'{sorted(fuera)}')
    if len({frozenset(p) for p in pares}) != len(pares):
        raise ValueError(f'{familia}: un par se repite en {pares}')

    chi, p_fried = _st.friedmanchisquare(*(datos[b] for b in etiquetas))
    # Rangos dentro de cada bloque de semilla, promediados sobre los bloques.
    # `axis=0` rankea los brazos de una misma semilla entre si; rankear el otro
    # eje ordenaria las semillas, que no es la pregunta.
    pila = np.vstack([datos[b] for b in etiquetas])
    rangos = _st.rankdata(pila, axis=0).mean(axis=1)

    # Despues de la pila: a partir de aca los no tratamientos se pueden leer
    # para restar un par, y los rangos ya estan calculados sin ellos.
    lectura = {**datos, **fuera}
    filas = [_par(a, b, lectura[a], lectura[b], n, campo=CAMPO_ACIERTO)
             for a, b in pares]
    for f in filas:
        f['p'] = f.pop('wilcoxon')
    for f, aj in zip(filas, _holm([f['p'] for f in filas])):
        f['holm'] = aj

    return {
        'familia': familia,
        'normalizacion': normalizacion,
        'escenario': escenario,
        'n': n,
        'bloques': n,
        'brazos': {b: datos[b].tolist() for b in etiquetas},
        # Lo que un par pudo nombrar y el Friedman no vio. Va en el resultado
        # para que se pueda COMPROBAR que quedo fuera, en vez de deducirlo de
        # que no aparece: una ausencia se afirma mirando donde viviria.
        'no_tratamientos': sorted(fuera),
        'chi2': float(chi),
        'gl_friedman': len(etiquetas) - 1,
        'p_friedman': float(p_fried),
        'rangos': {b: float(r) for b, r in zip(etiquetas, rangos)},
        'mejor_rango': len(etiquetas),
        # El denominador REAL de Holm, no `len(pares)`: los pares sin p
        # quedaron fuera, y rotular la tabla con el conteo nominal afirmaria
        # una correccion que no se aplico.
        'familia_de': sum(1 for f in filas if f['p'] is not None),
        'pares_sin_p': [f['rotulo'] for f in filas if f['p'] is None],
        'filas': filas,
    }


def comparar_tres_brazos(nombre, CELDAS, ADA, banderas):
    """Los tres brazos de BatchNorm de UN toyset: piso, AdaBN y CREDA.

    Tres brazos, asi que **Friedman** con 3 tratamientos y `n` bloques de
    semilla, y despues **Wilcoxon por pares con Holm DENTRO de la familia de 3
    de este toyset** --- nunca sobre los 9 pares de los tres toysets juntos,
    que corregiria de mas y es una familia que nadie declaro.

    Esta funcion LEE el artefacto y delega el test a `comparar_n_brazos`. Lo
    que es suyo es de donde salen los tres brazos; el Friedman, el Wilcoxon y
    el Holm son los del nucleo compartido.

    **Los tres brazos salen del MISMO split, test.** No siempre fue asi:
    `acc_predict` se grababa desde `acc_va` mientras AdaBN y CREDA salian de
    `acc_te`, de modo que la tabla comparaba tres brazos de un split contra uno
    de otro. Medido sobre el artefacto anterior, el corrimiento llegaba a
    0.0556 en una semilla de lunas, y como el test es apareado por semilla
    entraba entero. Corregido en `barrer_tablas`, que ahora graba
    `punto(cu, 0)['acc_te']`; el valor de valid sobrevive como
    `acc_predict_va` solo para su propio cross-check contra `blanco_ini` de
    AdaBN, que tambien es valid.
    """
    if POS_ADABN not in banderas['POS_FILAS']:
        raise ValueError(
            f'{nombre}: esta tabla son los tres brazos de {POS_ADABN!r} y '
            f'POS_FILAS no lo tiene ({banderas["POS_FILAS"]}); AdaBN solo '
            'existe donde hay buffers que reestimar')
    ac = _celda(CELDAS, nombre, POS_ADABN)
    if nombre not in ADA:
        raise ValueError(f'{nombre}: el artefacto no tiene brazo AdaBN')
    n = _semillas([len(ac), len(ADA[nombre])], banderas,
                  f'{nombre}: celda de {POS_ADABN} y AdaBN')

    brazos = {
        BRAZO_PISO: np.array([r['acc_predict'] for r in ac], dtype=float),
        BRAZO_ADABN: np.array([r['acc_te'] for r in ADA[nombre]], dtype=float),
        BRAZO_CREDA: np.array([r['acc_te'] for r in ac], dtype=float),
    }

    return comparar_n_brazos(brazos, PARES_TRES_BRAZOS, n, familia=nombre,
                             normalizacion=POS_ADABN)


def _num(v, fmt):
    return _NADA if v is None else format(v, fmt)


def _md(enc, fil):
    # Que cada fila tenga tantas celdas como el encabezado. Sacar una columna y
    # olvidar una fila da una tabla que markdown renderiza igual, corrida un
    # lugar, y los numeros quedan bajo el rotulo de al lado.
    assert all(len(f) == len(enc) for f in fil), (len(fil[0]), len(enc))
    return chr(10).join(
        ['| ' + ' | '.join(enc) + ' |',
         '|' + '|'.join(['---'] * len(enc)) + '|']
        + ['| ' + ' | '.join(f) + ' |' for f in fil])


def tabla_sin_contra_con(nombre, CELDAS, banderas):
    """La tabla A de UN toyset, en markdown. Los numeros los hace comparar()."""
    res = comparar_sin_contra_con(nombre, CELDAS, banderas)
    # `con` va PRIMERO: con `sin` delante, un lector recorre la fila de
    # izquierda a derecha y lee el delta al reves de como esta definido.
    enc = ('normalización', 'con CREDA', 'sin CREDA',
           f'{_DELTA} = con {chr(8722)} sin', f'IC95 de {_DELTA}',
           f't({res["filas"][0]["gl"]})', 'p', 'Shapiro p', 'Wilcoxon p',
           'gana')
    fil = [(f['rotulo'],
            f'{f["con"]:.4f}', f'{f["sin"]:.4f}',
            f'{f["delta"]:+.4f}',
            f'[{f["ic"][0]:+.4f}, {f["ic"][1]:+.4f}]',
            _num(f['t'], '.2f'), _num(f['p'], '.5f'),
            _num(f['shapiro'], '.3f'), _num(f['wilcoxon'], '.4f'),
            f'{f["gana"]}/{f["n"]}')
           for f in res['filas']]
    pie = (f'`{_DELTA} = con CREDA {chr(8722)} sin CREDA`, as\u00ed que '
           f'**{_DELTA}>0 quiere decir que gana `con CREDA`** '
           f'{chr(183)} `{ROTULO_PROMEDIO}` promedia {EJE_PROMEDIO}, o sea '
           f'{res["n"]} pares, y no agrupa los '
           f'{res["n"] * (len(res["filas"]) - 1)} de celda-semilla: la semilla '
           f'0 est\u00e1 en las tres normalizaciones '
           f'{chr(183)} t pareado, Shapiro-Wilk sobre las diferencias '
           f'pareadas, Wilcoxon al lado')
    return Markdown(
        f'**{nombre} {chr(183)} {res["n"]} semillas {chr(183)} test, sólo '
        f'reporte {chr(183)} {etiqueta_balance(banderas)}**'
        + chr(10) * 2 + _md(enc, fil) + chr(10) * 2 + pie)


#: El orden de lectura de la escalera del sopesado, de menos intervencion a
#: mas. Derivado de `BRAZOS_CONTAMINACION` y no escrito a mano: el padron de
#: brazos es lo que el barrido corre, y una tabla con su propio orden se
#: desincroniza del barrido sin que nada falle.
#:
#: **Sin el techo de oraculo.** Lo decidio el operador el 2026-10-18 para ESTA
#: tabla, mientras el brazo seguia corriendo y seguia en el registro; el
#: 2026-10-19 lo retiro de la seccion entera, asi que `BRAZOS_CONTAMINACION` ya
#: no lo trae y `barrer_contaminacion` ya no lo mide. Ver la declaracion de
#: `BRAZOS_CONTAMINACION` en `toy_experiment`, donde esta la medicion que lo
#: retiro.
#:
#: Se deriva por RESTA del padron y no se escribe la lista a mano: un brazo
#: nuevo entra solo, y --- esto es lo que esa decision escribio aqui de
#: antemano --- si el techo dejara de existir esta resta no se quedaria
#: nombrando algo que no esta. Dejo de existir, y la resta no se movio.
ESCALERA_SOPESADO = tuple(b for b in BRAZOS_CONTAMINACION
                          if PALABRA_TECHO not in b)
#: La resta ya no saca nada, y eso es LO QUE SE ASERTA. Decia
#: `== len(BRAZOS_CONTAMINACION) - 1` mientras el padron traia exactamente un
#: techo; con el techo retirado ese `-1` afirmaria que el padron todavia trae
#: uno. La asercion se invierte en vez de borrarse: un archivo sin ella no
#: podria distinguir «el techo se retiro» de «nadie comprobo nunca que la resta
#: correspondiera al padron».
assert ESCALERA_SOPESADO == tuple(BRAZOS_CONTAMINACION), (
    ESCALERA_SOPESADO, BRAZOS_CONTAMINACION,
    'el padron volvio a traer un techo: esta tabla lo dejaria fuera en '
    'silencio, que es el defecto que la resta y esta asercion existen para '
    'hacer ruidoso')


def _celda_sopesado(CONTAM, nombre, modo, rho):
    """La celda `(familia, modo, rho)` del registro de contaminacion.

    Refuta en vez de devolver vacio: una celda ausente con una tabla que
    renderiza igual es la forma en que un escenario que nunca corrio se publica
    como si hubiera corrido.
    """
    clave = (nombre, modo, float(rho))
    celdas = CONTAM['celdas']
    if clave not in celdas:
        raise KeyError(
            f'el registro de contaminacion no tiene la celda {clave!r}; '
            f'tiene {sorted(celdas)!r}. Una tabla no puede reportar un '
            f'escenario que el barrido no corrio.')
    return celdas[clave]


def tabla_sopesado_acc(nombre, CONTAM, modo, rho):
    """Los CINCO brazos del sopesado sobre una familia, en escalera.

    **La misma forma que `tabla_dataset` de la seccion anterior, con una sola
    arquitectura.** `POS_CONTAMINACION` es `batchnorm`, y no porque sea la
    mejor: la escalera incluye AdaBN, que necesita los buffers de BatchNorm
    para existir, de modo que la arquitectura la EXIGE un brazo y no la elige
    una comparacion. Se dice en el titulo para que no se lea como una eleccion.

    Todo sobre TEST y en el paso que el criterio eligio --- `acc_te` de
    `_fila_brazo` sale de `nf`, que es el modelo que la regla de puntos de
    control restauro, nunca el ultimo cuadro de la trayectoria ni el pico.

    **El techo lleva su marca en la fila.** Lee `y_t`, asi que no es un metodo
    que nadie pueda correr, y una fila sin marcar al lado de cuatro metodos se
    lee como un quinto metodo.

    **Y si el criterio no llego a elegir, lo dice.** Con menos de
    `MIN_EPOCAS_2` epocas el criterio nunca es legal, no se restauran pesos y
    `mejor_paso` queda en `-1`: la corrida termina y publica el ULTIMO modelo.
    Un ensayo corto que no avisara de eso reportaria la trayectoria final bajo
    el rotulo del modelo elegido, que es exactamente lo que esta columna
    existe para distinguir.
    """
    celda = _celda_sopesado(CONTAM, nombre, modo, rho)
    brazos = celda['brazos']
    n = len(brazos[ESCALERA_SOPESADO[0]])

    enc = ('método', 'época que eligió', 'acc fuente (test)',
           'acc blanco (test)', 'vigía final')
    fil, sin_elegir = [], []
    for etq in ESCALERA_SOPESADO:
        filas = brazos[etq]
        assert len(filas) == n, (etq, len(filas), n)
        assert not any(r.get('techo_no_es_metodo') for r in filas), (
            etq, 'un techo llego a la escalera: lee `y_t` y esta tabla no lo '
            'imprime, asi que una fila suya aqui se leeria como un metodo')
        rotulo = f'**{etq}**'
        # `.predict` y AdaBN no corren fase 2, asi que no tienen epoca ni
        # vigia: un guion y no un cero, que se leeria como que eligieron la
        # epoca 0.
        tiene_fase2 = 'mejor_paso' in filas[0]
        if tiene_fase2:
            epocas = [epoca_elegida_del_brazo(r) for r in filas]
            if any(q < 0 for q in epocas):
                sin_elegir.append(etq)
            paso = ('—' if any(q < 0 for q in epocas)
                    else f'{int(np.median(epocas))}')
            vigia = _m([r['vigia_fin'] for r in filas])
        else:
            paso, vigia = '—', '—'
        fil.append((rotulo, paso,
                    _m([r['acc_s_te'] for r in filas]) if 'acc_s_te' in filas[0] else '—',
                    _m([r['acc_te'] for r in filas]), vigia))

    aviso = ''
    if sin_elegir:
        aviso = (chr(10) * 2 + f'> **El criterio no eligió en {sorted(sin_elegir)}**: '
                 f'`mejor_paso = -1`, o sea que la fase 2 no alcanzó '
                 f'`MIN_EPOCAS_2` épocas, no se restauraron pesos y lo que la '
                 f'columna del blanco reporta es el ÚLTIMO modelo y no el '
                 f'elegido. Es un ensayo corto, no un resultado.')
    return Markdown(
        f'**{nombre} {chr(183)} {n} semillas {chr(183)} '
        f'{POS_CONTAMINACION} (la exige AdaBN, no la elige una comparación) '
        f'{chr(183)} {modo} {chr(183)} ' + chr(961) + f' = {float(rho):.2f} '
        f'{chr(183)} media {chr(177)} desvío {chr(183)} azar {1/CLASES:.4f}**'
        + chr(10) * 2 + _md(enc, fil) + aviso)


def tabla_sopesado_test(nombre, CONTAM, modo, rho):
    """El eje del sopesado, testeado: con sopesado contra sin sopesado.

    **La misma forma que `tabla_sin_contra_con`, y el mismo t pareado.** Dos
    brazos, asi que t pareado con Shapiro-Wilk sobre las diferencias pareadas
    al lado y Wilcoxon reportado; nada de Friedman, que es para tres o mas. No
    hay un segundo t en este archivo: `_fila_dos_brazos` es el que corren las
    tablas de la escalera y es el que corre aqui.

    `delta = con sopesado - sin sopesado`, de modo que `delta > 0` quiere decir
    que gana el sopesado. Los dos brazos difieren SOLO en la bandera del peso:
    misma fase 1, mismos HP de fase 2, misma semilla, misma arquitectura.

    **El techo va en una fila SIN test**, y eso es una declaracion y no una
    omision: lee `y_t`, asi que testear si le gana a un metodo pregunta por una
    hipotesis sobre la que nadie va a actuar. Lleva delta e IC95 --- cuanto hay
    para ganar sopesando --- y ninguna p.
    """
    celda = _celda_sopesado(CONTAM, nombre, modo, rho)
    brazos = celda['brazos']
    sin = np.array([r['acc_te'] for r in brazos[BRAZO_SIN_SOP]], dtype=float)
    con = np.array([r['acc_te'] for r in brazos[BRAZO_CON_SOP]], dtype=float)
    n = len(sin)
    assert len(con) == n, (n, len(con))

    probado = _fila_dos_brazos(f'con sopesado {chr(8722)} sin sopesado',
                               None, sin, con, n)
    enc = ('par', 'con', 'sin', f'{_DELTA}', f'IC95 de {_DELTA}',
           f't({probado["gl"]})', 'p', 'Shapiro p', 'Wilcoxon p', 'gana')
    fil = [(probado['rotulo'], f'{probado["con"]:.4f}', f'{probado["sin"]:.4f}',
            f'{probado["delta"]:+.4f}',
            f'[{probado["ic"][0]:+.4f}, {probado["ic"][1]:+.4f}]',
            f'{probado["t"]:+.3f}', f'{probado["p"]:.4f}',
            f'{probado["shapiro"]:.4f}', f'{probado["wilcoxon"]:.4f}',
            f'{probado["gana"]}/{probado["n"]}')]
    # El techo NO lleva fila, por la misma decision que lo saco de la escalera.
    # Sigue corriendo y sigue en el registro; lo que no hace es imprimirse.
    pie = (f'`{_DELTA} = con sopesado {chr(8722)} sin sopesado`, así que '
           f'**{_DELTA}>0 quiere decir que gana el sopesado** {chr(183)} '
           f'los dos brazos difieren SOLO en la bandera del peso {chr(183)} '
           f't pareado, Shapiro-Wilk sobre las diferencias pareadas, Wilcoxon '
           f'al lado')
    return Markdown(
        f'**{nombre} {chr(183)} {n} semillas {chr(183)} '
        f'{POS_CONTAMINACION} {chr(183)} {modo} {chr(183)} '
        + chr(961) + f' = {float(rho):.2f} {chr(183)} test, sólo reporte**'
        + chr(10) * 2 + _md(enc, fil) + chr(10) * 2 + pie)


def tabla_n_brazos(res, titulo, nota_familia):
    """El markdown de una tabla de N brazos. Los numeros los hace comparar().

    Uno solo, igual que el test: dos renderizadores del mismo cuadro divergen y
    el signo se lee en el que nadie reviso. La primera columna se llama
    `gana si {D}>0` DENTRO de la propia tabla y no solo en un docstring --- el
    signo se lee donde se imprime, y donde se leyo al reves fue ahi.

    `nota_familia` dice contra que NO se corrige, que es la mitad de la
    afirmacion que un lector no puede deducir de la tabla.

    `res['descriptivos']` y `res['filas_techo']`, si estan, se imprimen por
    ESTE mismo renderizador y no por uno al lado. La regla de este archivo es
    que haya un solo cuadro: dos renderizadores del mismo cuadro divergen y el
    signo se lee en el que nadie reviso. La tabla de tres brazos no los manda y
    sale exactamente igual que antes.
    """
    enc = (f'gana si {_DELTA}>0', 'contra',
           f'{_DELTA} = 1\u00aa {chr(8722)} 2\u00aa', f'IC95 de {_DELTA}',
           'p (Wilcoxon)', f'p Holm (familia de {res["familia_de"]})', 'gana')

    def _linea(f, sufijo=''):
        return (f['primero'] + sufijo, f['segundo'],
                f'{f["delta"]:+.4f}',
                f'[{f["ic"][0]:+.4f}, {f["ic"][1]:+.4f}]',
                _num(f['p'], '.4f'), _num(f['holm'], '.4f'),
                f'{f["gana"]}/{f["n"]}')

    # La marca de techo va en la celda que se IMPRIME y no solo en el dict: la
    # fila de un techo al lado de las de los metodos se lee como un metodo mas,
    # y es la fila impresa la que alguien lee.
    fil = [_linea(f, f' {MARCA_TECHO_PAR}' if f.get('techo') else '')
           for f in res['filas']]
    # Los pares DECLARADOS que se reportan sin test, entre la familia corregida
    # y lo no declarado. Ese lugar es la mitad de lo que la fila dice: arriba
    # esta lo que Holm corrigio, abajo lo que nadie declaro, y esto es lo
    # declarado-y-no-testeado, que no es ninguno de los dos.
    for f in res.get('sin_p', ()):
        fil.append(_linea(f, (f' {MARCA_TECHO_PAR}' if f.get('techo') else '')
                          + ' *(declarado, sin test)*'))
    # Los pares no declarados, DESPUES de los declarados y con las dos columnas
    # de p vacias. El orden importa: la familia corregida va primero y lo
    # descriptivo despues, de modo que no se lea como parte de ella.
    for f in res.get('descriptivos', ()):
        fil.append(_linea(f, (f' {MARCA_TECHO_PAR}' if f.get('techo') else '')
                          + ' *(no declarado)*'))
    # Y la fila propia del techo: su nivel, no una diferencia. Las columnas de
    # par quedan vacias porque un techo no compite contra nadie en esta fila.
    for t in res.get('filas_techo', ()):
        desv = '' if t['desvio'] is None else f' {chr(177)} {t["desvio"]:.4f}'
        fil.append((f'{t["brazo"]} {MARCA_TECHO_PAR}', _NADA,
                    f'{t["media"]:.4f}{desv}', _NADA, _NADA, _NADA,
                    f'{chr(8212)}/{t["n"]}'))
    rangos = ' | '.join(f'`{b}` {r:.2f}' for b, r in res['rangos'].items())
    pie = (f'`{_DELTA} = (columna \u00abgana si {_DELTA}>0\u00bb) {chr(8722)} '
           f'(columna \u00abcontra\u00bb)`, as\u00ed que **{_DELTA}>0 quiere decir que gana '
           f'el PRIMERO de los dos nombrados** {chr(183)} el orden del par es '
           f'fijo y no sigue al ganador {chr(183)} {nota_familia}')
    return Markdown(
        f'**{titulo}**'
        + chr(10) * 2
        + f'Friedman {_CHI}\u00b2({res["gl_friedman"]}) = {res["chi2"]:.2f} '
        f'{chr(183)} p = {res["p_friedman"]:.5f} {chr(183)} '
        f'{res["bloques"]} bloques de semilla {chr(183)} rangos medios '
        f'({res["mejor_rango"]} = mejor): {rangos}'
        + chr(10) * 2 + _md(enc, fil) + chr(10) * 2 + pie)


def tabla_tres_brazos(nombre, CELDAS, ADA, banderas):
    """La tabla B de UN toyset, en markdown. Los numeros los hace comparar()."""
    res = comparar_tres_brazos(nombre, CELDAS, ADA, banderas)
    return tabla_n_brazos(
        res,
        f'{nombre.upper()} {chr(183)} {NOMBRE_FILA[res["normalizacion"]]} '
        f'{chr(183)} {res["n"]} semillas {chr(183)} test, s\u00f3lo reporte',
        f'Holm dentro de los {res["familia_de"]} pares de este toyset, no '
        'sobre los 9')


# ===========================================================================
# La seccion de sopesado bajo contaminacion: la tabla y la figura
# ---------------------------------------------------------------------------
# Cuatro METODOS por escenario, asi que el MISMO Friedman del nucleo compartido
# --- 4 tratamientos x n bloques de semilla --- con post-hoc de Wilcoxon sobre
# un conjunto DECLARADO de pares y Holm dentro de esa familia. Nunca a lo largo
# de la escalera: los pares de los cuatro niveles son una familia que nadie
# declaro, y corregir sobre ella corrige de mas. Y nunca sobre todos los pares
# que el padron admita, que es el mismo error adentro de un escenario: la
# familia se declara antes de correr y se cuenta de la declaracion, nunca de un
# literal al lado de un padron que puede crecer.
#
# El techo de oraculo NO es uno de los tratamientos NI de la familia de Holm.
# Lee `y_t`, asi que rankearlo contra los metodos pregunta cual metodo gana
# contra algo que ningun metodo puede ser, y testearlo pregunta por una
# hipotesis sobre la que nadie va a actuar; tiene fila propia y marcada, y
# aparece en el par DECLARADO SIN TEST que lo nombra, con delta, IC95 y
# victorias. Ver `BRAZOS_CUATRO`, donde esta escrito por que la herencia del
# padron de contaminacion era el defecto, y `PARES_CUATRO_BRAZOS`, donde estan
# las tres razones por las que la familia son los tres pares entre metodos.
#
# La celda es `batchnorm` y es DERIVADO: uno de los cuatro metodos es AdaBN y
# AdaBN solo existe donde hay buffers que reestimar.
# ===========================================================================

#: Los CUATRO METODOS que el Friedman de esta tabla compara, declarados aca y
#: **severados a proposito del padron de contaminacion**.
#:
#: Esto fue `BRAZOS_CUATRO = BRAZOS_CONTAMINACION`, y la herencia era el
#: defecto. El padron de contaminacion crecio de cuatro brazos a cinco --- se
#: le sumo el techo de oraculo, que lee `y_t` --- y esta tabla crecio con el sin
#: que nada lo dijera: Friedman con 4 grados de libertad sobre 5 tratamientos,
#: rangos medios que incluian el techo, y el techo sentado en los rangos sin
#: aparecer en ninguna fila de pares. Hoy no publica eso solo porque el
#: artefacto en disco todavia tiene cuatro brazos y la tabla se NIEGA; en
#: cuanto corra el barrido de cinco deja de negarse y empieza a publicar.
#:
#: Un techo no es un metodo: lee las etiquetas del blanco. Rankearlo contra los
#: metodos pregunta cual metodo gana contra algo que ningun metodo puede ser.
#: Este proyecto ya trata asi a la columna del pico --- se reporta y no se
#: testea --- y el techo se trata igual: fila propia y marcada, fuera del
#: omnibus.
BRAZOS_CUATRO = (BRAZO_PISO, BRAZO_ADABN, BRAZO_SIN_SOP, BRAZO_CON_SOP)

#: El TECHO de esta seccion, importado de la mitad que lo corre. Tiene fila
#: propia en la tabla y no entra en los tratamientos del Friedman.
BRAZO_TECHO_CUATRO = BRAZO_TECHO

# Que los cuatro sean METODOS, comprobado y no supuesto: la palabra que marca un
# techo se lee del modulo que la escribe, de modo que un brazo nuevo que la
# lleve no puede entrar a los tratamientos por descuido.
assert all(PALABRA_TECHO not in b.lower() for b in BRAZOS_CUATRO), (
    BRAZOS_CUATRO, 'un brazo de los tratamientos lleva la palabra de techo')
assert PALABRA_TECHO in BRAZO_TECHO_CUATRO.lower(), BRAZO_TECHO_CUATRO
# Y que el barrido los corra. No se DERIVAN de `BRAZOS_CONTAMINACION` --- eso es
# justamente la cadena que se corto --- pero un metodo que el barrido no mide es
# una tabla de nada, asi que se cruza el empalme en una direccion sola: esta
# lista se comprueba CONTRA el padron y nunca se toma de el.
assert all(b in BRAZOS_CONTAMINACION for b in BRAZOS_CUATRO), (
    BRAZOS_CUATRO, BRAZOS_CONTAMINACION,
    'la tabla declara un metodo que el barrido de contaminacion no corre')
assert len(set(BRAZOS_CUATRO)) == len(BRAZOS_CUATRO), BRAZOS_CUATRO

#: Los pares del post-hoc, DECLARADOS antes de la corrida y no todos los que
#: existen. Cada uno con el brazo mas intervenido primero; el orden es fijo y no
#: sigue al ganador.
#:
#: **Declarar el conjunto antes de correr es lo que mantiene honesta la
#: correccion.** Holm reparte sobre la familia que se declaro; pescar un par
#: despues de ver los numeros y meterlo en la misma familia es elegir el
#: denominador con el resultado a la vista. Los otros seis pares existen y se
#: reportan, pero DESCRIPTIVOS: delta y cuenta de victorias, sin p.
#:
#: **El techo NO esta entre ellos, y los tres METODOS son toda la familia.**
#: Decidido por el operador, y escrito aca porque es aca donde el proximo lector
#: lo vuelve a agregar: un par contra el techo se ve como una comparacion mas.
#: Tres razones, y ninguna es de estilo:
#:
#: 1. **Un techo no es una hipotesis sobre la que alguien vaya a actuar.** Que un
#:    oraculo que lee `y_t` gane ya se sabe; lo que su fila contesta es CUANTO
#:    hay para ganar, que es un tamanio de efecto y no un test.
#: 2. **Meterlo infla `m` y afloja la correccion de las comparaciones que
#:    deciden.** Medido sobre los cuatro declarados: la p cruda del par del techo
#:    era 0.0020 y llegaba a 0.0078 de Holm con m=4, mientras los pares entre
#:    metodos quedaban en 0.0059 con m=3. El techo le estaba pidiendo prestada
#:    correccion a lo que importa.
#: 3. **Es como este proyecto ya trata la columna del pico de oraculo**: se
#:    reporta y no se testea.
#:
#: El par del techo no se borra: sigue en la tabla, DECLARADO, en
#: `PARES_CUATRO_SIN_P`, con delta, IC95 y victorias y sin p ni Holm.
PARES_CUATRO_BRAZOS = (
    # El unico eje del experimento: los dos brazos de CREDA difieren SOLO en
    # `use_entropy_weighting`. Va primero porque es la pregunta de la seccion.
    (BRAZO_CON_SOP, BRAZO_SIN_SOP),
    (BRAZO_CON_SOP, BRAZO_ADABN),
    (BRAZO_SIN_SOP, BRAZO_ADABN),
)

#: Los pares DECLARADOS que se reportan con tamanio de efecto y SIN test: delta,
#: IC95 y victorias, con `p` y `holm` en `None`.
#:
#: **Declarados, y por eso no caen en el cubo de lo no declarado.** Los dos
#: terminan con `p` nula, asi que la tentacion es dejar que este par caiga en
#: `descriptivos` y ahorrarse una tupla. No se puede: esa fila se imprime
#: rotulada «no declarado», y para este par eso afirma lo contrario de la
#: decision que lo puso ahi. Un par que el operador declaro y resolvio no testear
#: no es un par que nadie declaro.
#:
#: `techo {chr(8722)} sin sopesado` dice cuanto hay para ganar sopesando, que es
#: la unica pregunta para la que un techo sirve.
PARES_CUATRO_SIN_P = (
    (BRAZO_TECHO_CUATRO, BRAZO_SIN_SOP),
)

#: Los pares declarados que comparan contra un TECHO, DERIVADOS de la palabra
#: que marca uno y no de una segunda lista escrita a mano. Una lista paralela
#: aca se queda atras en cuanto el conjunto declarado cambia.
#:
#: Se deriva sobre los DOS conjuntos declarados y no sobre el testeado solo: el
#: par del techo se movio al de sin test, y una derivacion que mirara un conjunto
#: solo habria contestado la tupla vacia sin que nada lo dijera --- y lo que lee
#: esto es la marca de la fila.
PARES_CUATRO_CON_TECHO = tuple(
    p for p in PARES_CUATRO_BRAZOS + PARES_CUATRO_SIN_P
    if any(PALABRA_TECHO in b.lower() for b in p))

#: La marca que distingue una fila de techo de una fila de metodo, en la tabla.
MARCA_TECHO_PAR = '(techo)'


def _validar_pares_declarados(padron, techos, pares, *, sin_p=()):
    """Los pares declarados contra el padron, o una negativa. Corre al IMPORTAR.

    **Esto es el guard que reemplaza un conteo literal.** La tabla decia «Holm
    dentro de los 6 pares» con un 6 escrito al lado de un padron que podia
    crecer, y crecio: el padron paso a cinco brazos y el 6 se quedo. Un numero
    de pares escrito como literal al lado de un padron que puede crecer es el
    mismo defecto en un lugar nuevo, asi que aca no hay ningun conteo: se
    comprueba que el conjunto declarado y el padron se CORRESPONDAN, y el
    numero se cuenta de la tupla donde haga falta.

    Lo que se comprueba, y lo que a proposito NO:

    - **un par que nombra un brazo inexistente** --- ni metodo ni techo --- es
      un par que no se puede medir, y es una negativa;
    - **ningun par se repite**, porque un par repetido infla la familia de Holm
      sin agregar ninguna comparacion;
    - **ningun techo entra sin marcar**: un par que nombra un techo tiene que
      ser uno de los declarados como tal, de modo que la fila salga marcada;
    - **ningun par TESTEADO nombra un techo.** Esta es la que sostiene la
      decision de `m = 3`, y esta aca y no en un comentario porque un comentario
      no se ejecuta: el proximo lector vuelve a agregar el par del techo a la
      familia --- se ve como una comparacion mas --- y lo unico que lo detiene es
      que el import se niegue. Los dos conjuntos juntos tienen que ser
      DISJUNTOS, ademas: un par en los dos se testearia y se reportaria como no
      testeado a la vez, y la tabla afirmaria las dos cosas;
    - **un metodo sin par declarado NO es un error.** El piso `.predict` es
      exactamente ese caso: entra al Friedman como tratamiento y sus tres pares
      salen DESCRIPTIVOS por decision. Lo que lo deja cubierto es que el
      conjunto descriptivo se DERIVA del padron de la celda, de modo que un
      metodo nuevo aparece ahi por si solo y ningun conteo queda viejo. Exigirlo
      en un par declarado seria obligar a la familia de Holm a crecer con el
      padron, que es la cadena que esta tabla acaba de cortar.

    El eje de la seccion si se exige: los dos brazos de sopesado difieren SOLO
    en la bandera, y una declaracion que no los enfrentara seria una tabla que
    no contesta su propia pregunta.
    """
    conocidos = set(padron) | set(techos)
    sin_p = tuple(sin_p)
    # Todos los declarados: los testeados y los que se reportan sin test. Las
    # comprobaciones de existencia, de repeticion y de techo sin marcar valen
    # para los dos --- las dos clases se imprimen, y una fila mal armada se lee
    # igual de mal venga de donde venga.
    todos = tuple(pares) + sin_p
    for a, b in todos:
        for brazo in (a, b):
            if brazo not in conocidos:
                raise ValueError(
                    f'el par ({a!r}, {b!r}) nombra {brazo!r}, que no esta ni '
                    f'entre los metodos {sorted(padron)} ni entre los techos '
                    f'{sorted(techos)}; un par declarado sobre un brazo que no '
                    'existe no se puede medir')
    if len({frozenset(p) for p in todos}) != len(todos):
        raise ValueError(
            f'un par se repite, o esta declarado dos veces --- una entre los '
            f'testeados {[list(p) for p in pares]} y otra entre los que se '
            f'reportan sin test {[list(p) for p in sin_p]}. Un par en los dos '
            'se testearia y se reportaria como no testeado a la vez')
    # Y la que sostiene `m = 3`: un par TESTEADO no nombra un techo. Un techo
    # lee `y_t`, asi que no es una hipotesis sobre la que alguien actue, y
    # meterlo en la familia infla `m` y afloja la correccion de los pares entre
    # metodos, que son los que deciden. Su fila sigue en la tabla, declarada,
    # entre los que se reportan sin test.
    for a, b in pares:
        for brazo in (a, b):
            if PALABRA_TECHO in brazo.lower():
                raise ValueError(
                    f'el par TESTEADO ({a!r}, {b!r}) nombra {brazo!r}, que es '
                    f'un techo. La familia de Holm son los pares entre METODOS '
                    f'y nada mas: un techo infla `m` y afloja la correccion de '
                    f'los pares que deciden (medido: 0.0020 crudo llegaba a '
                    f'0.0078 con m=4 mientras los metodos quedaban en 0.0059 '
                    f'con m=3). El par del techo se declara en '
                    f'`PARES_CUATRO_SIN_P`, con delta, IC95 y victorias y sin p')
    for a, b in todos:
        for brazo in (a, b):
            if PALABRA_TECHO in brazo.lower() and brazo not in set(techos):
                raise ValueError(
                    f'el par ({a!r}, {b!r}) nombra {brazo!r}, que lleva la '
                    f'palabra de techo y no esta entre los techos declarados '
                    f'{sorted(techos)}; un techo sin declarar sale como una '
                    'fila sin marca y se lee como un metodo que compite')
    if frozenset((BRAZO_CON_SOP, BRAZO_SIN_SOP)) not in {
            frozenset(p) for p in pares}:
        raise ValueError(
            f'los pares declarados {[list(p) for p in pares]} no enfrentan '
            f'{BRAZO_CON_SOP!r} con {BRAZO_SIN_SOP!r}, que es el unico eje del '
            'experimento: los dos difieren SOLO en la bandera de sopesado. Una '
            'tabla que no declara ese par no contesta la pregunta de la seccion')
    return pares


_validar_pares_declarados(BRAZOS_CUATRO, (BRAZO_TECHO_CUATRO,),
                          PARES_CUATRO_BRAZOS, sin_p=PARES_CUATRO_SIN_P)

#: El mapa de tamanio de marca de la figura, DECLARADO: un lector no puede
#: distinguir `w=0.6` de `w=0.95` si el mapa no esta escrito, y una figura cuyo
#: tamanio no se puede leer no dice nada que el panel de al lado no diga.
#:
#:     tamanio = TAM_MIN_W + (TAM_MAX_W - TAM_MIN_W) * w
#:
#: Lineal en `w` y no en `w^2`: `w` ya es la cantidad que la Ec. 26 aplica a la
#: matriz, asi que elevarla pondria el ojo en otra escala que la perdida.
TAM_MIN_W, TAM_MAX_W = 6.0, 170.0

#: Los valores de `w` que la leyenda muestra con su tamanio, para que el mapa
#: se pueda LEER y no solo creer.
REFERENCIAS_W = (0.0, 0.25, 0.50, 0.75, 1.0)

#: El color de los dos grupos de la segunda columna. Los limpios de fondo, en
#: gris claro, y los contaminados en negro: el contraste es el que separa los
#: dos grupos, y el tamanio es el que dice el peso.
#: El split que estos paneles dibujan, escrito UNA vez y leido por el titulo
#: del panel, por el rotulo de la fila y por el pie. Las tablas de esta misma
#: seccion reportan TEST, asi que un panel sin rotulo al lado de una tabla de
#: test se lee como test --- y `w` no existe sobre test: no hay ningun termino
#: evaluado ahi. La contaminacion se inyecta en el train del blanco y la
#: metrica se lee sobre test, a proposito: el veneno esta donde el termino
#: entrena y el numero se mide sobre datos que nadie movio.
SPLIT_DIBUJADO = 'train del blanco'

COL_LIMPIO = '#cccccc'
COL_SUCIO = '#000000'


def tamano_por_w(w):
    """El tamanio de marca de un peso, por el mapa declarado arriba."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
    return TAM_MIN_W + (TAM_MAX_W - TAM_MIN_W) * w


def _celda_contaminacion(CONTAM, nombre, rho, perturbacion=None):
    """La celda de un escenario, por la clave de DOS o de TRES partes.

    `(familia, perturbacion, rho)` es la que el barrido escribe hoy y
    `(familia, rho)` la del artefacto que ya esta en disco, que se corrio antes
    de que la perturbacion se declarara. Las dos se resuelven y ninguna se lee
    con la clave de la otra.

    `perturbacion=None` NO quiere decir `reubicado`: quiere decir «la que la
    celda tenga, si hay exactamente una». Con dos es una negativa que las
    nombra, por la misma razon que en `escenario_del_gif` --- las dos predicen
    lo CONTRARIO sobre el par de medias, asi que elegir por defecto es elegir la
    conclusion --- y nunca una caida silenciosa a `None`.
    """
    if perturbacion is not None:
        clave = (nombre, str(perturbacion), float(rho))
        if clave not in CONTAM['celdas']:
            raise ValueError(
                f'no hay escenario {clave} en el artefacto de contaminacion; '
                f'las claves de {nombre!r} son '
                f'{sorted(str(k) for k in CONTAM["celdas"] if k[0] == nombre)}')
        return CONTAM['celdas'][clave]
    candidatas = [k for k in CONTAM['celdas']
                  if k[0] == nombre and float(k[-1]) == float(rho)]
    if not candidatas:
        disponibles = sorted(float(k[-1]) for k in CONTAM['celdas']
                             if k[0] == nombre)
        raise ValueError(
            f'no hay escenario {(nombre, float(rho))} en el artefacto de '
            f'contaminacion; la escalera que corrio es {disponibles}')
    if len(candidatas) > 1:
        nombres = sorted(str(k[1]) for k in candidatas if len(k) == 3)
        raise ValueError(
            f'{nombre} rho={rho}: el artefacto tiene {len(candidatas)} '
            f'escenarios a ese rho ({nombres}); nombra la perturbacion con '
            '`perturbacion=`. No se elige por defecto: las declaradas predicen '
            'lo contrario sobre el par de medias, asi que un defecto elegiria '
            'la conclusion')
    return CONTAM['celdas'][candidatas[0]]


def comparar_cuatro_brazos(nombre, rho, CONTAM, banderas, perturbacion=None):
    """Los cuatro METODOS de UN escenario, con el techo aparte y marcado.

    Lee el artefacto de contaminacion y delega el test al nucleo compartido:
    Friedman con 4 tratamientos y `n` bloques, Wilcoxon sobre los pares
    DECLARADOS y Holm dentro de esa familia.

    **Los tratamientos son los cuatro metodos y el techo no es uno de ellos.**
    `BRAZOS_CUATRO` esta severado de `BRAZOS_CONTAMINACION` justamente aca: el
    padron del barrido crecio a cinco y, heredandolo, esta tabla habria
    rankeado el techo contra los metodos. Un techo lee `y_t`, asi que
    preguntarle a un rango cual metodo le gana es preguntar cual gana contra
    algo que ningun metodo puede ser. Su fila va aparte y marcada, como la
    columna del pico que este proyecto ya reporta sin testear.

    **El techo aparece en un par DECLARADO y SIN TEST**, marcado, con delta,
    IC95 y victorias y sin p ni Holm: `techo - sin sopesado` dice cuanto hay
    para ganar sopesando, que es la unica pregunta para la que un techo sirve, y
    esa es una magnitud y no una hipotesis. La familia de Holm son los pares
    entre METODOS --- ver `PARES_CUATRO_BRAZOS`, donde estan las tres razones.

    **Lo no declarado sale DESCRIPTIVO**: delta y cuenta de victorias, sin p.
    Declarar la familia antes de correr es lo que mantiene honesta la
    correccion; pescar un par despues de ver los numeros es elegir el
    denominador con el resultado a la vista.

    **Los brazos se leen sobre TEST.** Incluido el piso: la tabla de tres brazos
    lee `.predict` sobre VALID --- un defecto previo de `barrer_tablas`,
    reportado y dejado en pie porque los numeros publicados se midieron asi ---
    y esta tabla no lo hereda. `barrer_contaminacion` registra el piso del punto
    de control 0 sobre test, y cada fila lleva su `split`.

    `delta = (primero nombrado) - (segundo nombrado)`, de modo que `delta > 0`
    quiere decir que gana el PRIMERO.
    """
    celda = _celda_contaminacion(CONTAM, nombre, rho, perturbacion)
    brazos_art = celda['brazos']
    faltan = [b for b in BRAZOS_CUATRO if b not in brazos_art]
    if faltan:
        raise ValueError(
            f'{nombre} rho={rho}: el artefacto no tiene los brazos {faltan}')
    # Los techos que la celda trae, LEIDOS de la marca del registro y no de una
    # lista escrita aca. Un techo presente y sin marcar no se adopta en
    # silencio: se nombra abajo, porque un techo sin marca se lee como un metodo
    # que gana.
    presentes_techo = [b for b in (BRAZO_TECHO_CUATRO,) if b in brazos_art]
    n = _semillas([len(brazos_art[b]) for b in BRAZOS_CUATRO], banderas,
                  f'{nombre} rho={rho}: los cuatro métodos')
    # El split de cada fila, comprobado y no supuesto: la tabla afirma que todos
    # se miden sobre el mismo split, y lo unico que puede sostenerlo es lo que
    # cada fila registro de si misma.
    for b in list(BRAZOS_CUATRO) + presentes_techo:
        for fila in brazos_art[b]:
            if fila.get('split') != 'test':
                raise ValueError(
                    f'{nombre} rho={rho}: el brazo {b!r} registra el split '
                    f'{fila.get("split")!r} y la tabla compara sobre test; '
                    'cuatro brazos en dos splits no son cuatro brazos')
    lectura = {b: [float(r['acc_te']) for r in brazos_art[b]]
               for b in list(BRAZOS_CUATRO) + presentes_techo}
    # Los tratamientos del omnibus: los METODOS, en orden de intervencion. El
    # techo queda fuera del diccionario que Friedman recibe, asi que no puede
    # entrar a los rangos por ninguna via.
    brazos = {b: lectura[b] for b in BRAZOS_CUATRO}
    # Los pares declarados que esta celda puede medir, y los que no. Un par
    # declarado cuyo brazo la celda no tiene NO se descarta en silencio: sale
    # del denominador de Holm y se nombra, porque un denominador que se achica
    # calladamente afloja la correccion y la tabla seguiria diciendo «familia
    # de» con el numero que se midio sin decir que faltaba uno.
    medibles = [p for p in PARES_CUATRO_BRAZOS
                if all(b in lectura for b in p)]
    ausentes = [f'{a} vs {b}' for a, b in PARES_CUATRO_BRAZOS
                if not all(x in lectura for x in (a, b))]
    if not medibles:
        raise ValueError(
            f'{nombre} rho={rho}: la celda no puede medir ninguno de los '
            f'{len(PARES_CUATRO_BRAZOS)} pares declarados '
            f'({[list(p) for p in PARES_CUATRO_BRAZOS]}); una tabla sin ningun '
            'par declarado no tiene post-hoc que corregir')
    # Largos iguales tambien para el techo: entra en un par declarado, asi que
    # un techo corto daria un Wilcoxon sobre pares que no existen.
    for b in presentes_techo:
        if len(lectura[b]) != n:
            raise ValueError(
                f'{nombre} rho={rho}: el brazo {b!r} tiene '
                f'{len(lectura[b])} semillas y los métodos tienen {n}; un '
                'brazo corto da un par sobre bloques que no existen')
    res = comparar_n_brazos(brazos, medibles, n, familia=nombre,
                            normalizacion=CONTAM['pos'],
                            escenario=float(rho),
                            extra=lectura)
    # La fila propia del techo, y la marca en cada par que lo nombra.
    for f in res['filas']:
        f['techo'] = any(PALABRA_TECHO in b.lower()
                         for b in (f['primero'], f['segundo']))
    res['pares_declarados'] = [f'{a} vs {b}' for a, b in PARES_CUATRO_BRAZOS]
    res['pares_ausentes'] = ausentes
    res['techos'] = list(presentes_techo)
    res['filas_techo'] = [
        dict(brazo=b, media=float(np.mean(lectura[b])),
             desvio=float(np.std(lectura[b], ddof=1))
             if len(lectura[b]) > 1 else None,
             n=len(lectura[b]))
        for b in presentes_techo]
    # Los pares DECLARADOS que se reportan sin test: delta, IC95 y victorias,
    # con `p` y `holm` en None. El par del techo es el unico hoy. No caen en
    # `descriptivos` a proposito: esa fila se imprime rotulada «no declarado», y
    # para un par que el operador declaro y resolvio no testear eso afirma lo
    # contrario de la decision que lo puso ahi.
    res['pares_sin_test'] = [f'{a} vs {b}' for a, b in PARES_CUATRO_SIN_P]
    res['sin_p'] = []
    for a, b in PARES_CUATRO_SIN_P:
        if not all(x in lectura for x in (a, b)):
            # La celda no tiene uno de los dos brazos. Se nombra arriba, en
            # `pares_sin_test`, y no se inventa una fila: un par que no se pudo
            # medir y uno que se midio sin testear no son lo mismo.
            continue
        d = _par(a, b, lectura[a], lectura[b], n, campo=CAMPO_ACIERTO)
        # La p se BORRA y no se deja pasar. Igual que en los descriptivos: una p
        # cruda impresa al lado de unas corregidas se lee como una corregida
        # mas, y aca la decision es justamente que no haya ninguna.
        d['p'], d['holm'], d['wilcoxon'] = None, None, None
        d['techo'] = any(PALABRA_TECHO in x.lower() for x in (a, b))
        d['declarado'] = True
        res['sin_p'].append(d)
    # Los pares que NO se declararon, descriptivos: delta y victorias, sin p.
    # Los DOS conjuntos declarados salen de aca: uno ya se imprimio con su p y
    # el otro sin ella, y repetir cualquiera de los dos seria la misma medicion
    # en dos lugares, que es lo que despues divergen.
    declarados = {frozenset(p)
                  for p in PARES_CUATRO_BRAZOS + PARES_CUATRO_SIN_P}
    orden = list(BRAZOS_CUATRO) + presentes_techo
    res['descriptivos'] = []
    for i, a in enumerate(orden):
        for b in orden[:i]:
            if frozenset((a, b)) in declarados: continue
            d = _par(a, b, lectura[a], lectura[b], n, campo=CAMPO_ACIERTO)
            # La p se BORRA y no se deja pasar: el par no se declaro, de modo
            # que no entra en la familia de Holm, y una p cruda impresa al lado
            # de unas corregidas se lee como una corregida mas.
            d['p'], d['holm'] = None, None
            d['wilcoxon'] = None
            d['techo'] = any(PALABRA_TECHO in x.lower() for x in (a, b))
            res['descriptivos'].append(d)
    res['hp2'] = dict(celda['hp2'])
    res['busquedas'] = celda['busquedas']
    res['contaminados'] = {s: len(v) for s, v in celda['indices'].items()}
    return res


def tabla_cuatro_brazos(nombre, rho, CONTAM, banderas, perturbacion=None):
    """La tabla de UN escenario, en markdown. Los numeros los hace comparar()."""
    res = comparar_cuatro_brazos(nombre, rho, CONTAM, banderas, perturbacion)
    sucios = sorted(set(res['contaminados'].values()))
    cuenta = sucios[0] if len(sucios) == 1 else sucios
    # Lo que la familia de Holm NO cubre, DICHO. Las tres piezas van juntas
    # porque sueltas cada una se lee como la familia entera: cuantos pares se
    # declararon, cuantos se pudieron medir, y cuantos se reportan sin p.
    declarados = len(res['pares_declarados'])
    nota_declarada = (
        f'los {declarados} pares del post-hoc se DECLARARON antes de la '
        f'corrida y Holm reparte dentro de esa familia, y los {declarados} son '
        f'entre MÉTODOS; los otros '
        f'{len(res["descriptivos"])} se reportan DESCRIPTIVOS, con {_DELTA} y '
        f'victorias y sin p, porque no se declararon --- pescar un par después '
        f'de ver los números es elegir el denominador con el resultado a la '
        f'vista')
    # Lo declarado-y-no-testeado, DICHO con su razon. Sin esta frase la fila del
    # techo se lee como una comparacion a la que le falta la p, y lo que pasa es
    # lo contrario: no se testea a proposito.
    if res['sin_p']:
        nota_declarada += (
            f' {chr(183)} {len(res["sin_p"])} declarado(s) **SIN test** '
            f'({", ".join(f"`{f["rotulo"]}`" for f in res["sin_p"])}): un techo '
            f'lee `y_t`, así que no es una hipótesis sobre la que alguien actúe '
            f'--- lo que su fila contesta es CUÁNTO hay para ganar, que es un '
            f'tamaño de efecto y no un test. Va con {_DELTA}, IC95 y victorias '
            f'y **sin p ni Holm**: meterlo en la familia inflaría `m` y '
            f'aflojaría la corrección de los pares que deciden')
    faltan_sin_test = [p for p in res['pares_sin_test']
                       if p not in {f['rotulo'] for f in res['sin_p']}]
    if faltan_sin_test:
        nota_declarada += (
            f' {chr(183)} **{len(faltan_sin_test)} declarado(s) sin test que '
            f'esta celda no puede medir**: '
            + ', '.join(f'`{p}`' for p in faltan_sin_test))
    if res['pares_ausentes']:
        nota_declarada += (
            f' {chr(183)} **{len(res["pares_ausentes"])} par(es) declarados '
            f'que esta celda no puede medir** y que por eso NO están en la '
            f'familia de {res["familia_de"]}: '
            + ', '.join(f'`{p}`' for p in res['pares_ausentes']))
    if res['techos']:
        nota_declarada += (
            f' {chr(183)} `{"`, `".join(res["techos"])}` '
            f'{MARCA_TECHO_PAR} lee `y_t`: **no es un tratamiento del '
            f'Friedman** y no está en los rangos; tiene fila propia y aparece '
            f'en los pares declarados SIN test que lo nombran, marcado')
    else:
        nota_declarada += (
            f' {chr(183)} **el registro no trae ningún techo**: la fila de '
            f'`{BRAZO_TECHO_CUATRO}` y los pares declarados que lo nombran '
            f'quedan fuera de esta tabla')
    return tabla_n_brazos(
        res,
        f'{nombre.upper()} {chr(183)} {NOMBRE_FILA[res["normalizacion"]]} '
        f'{chr(183)} {chr(961)} = {float(rho):.2f} ({cuenta} puntos del train '
        f'del blanco reubicados) {chr(183)} {res["n"]} semillas {chr(183)} '
        f'test, sólo reporte {chr(183)} {etiqueta_balance(banderas)}',
        f'Holm dentro de los {res["familia_de"]} pares DECLARADOS de ESTE '
        f'escenario, no sobre los '
        f'{res["familia_de"] * len(CONTAM["rhos"])} de la escalera '
        f'{chr(183)} {nota_declarada} '
        f'{chr(183)} UNA búsqueda de fase 2 por escenario, compartida por '
        f'los dos brazos de sopesado, así que la única diferencia entre '
        f'ellos es la bandera {chr(183)} el alcance es «con ESTOS HP», no '
        '«con cualquier HP»')


# --------------------------------------------------------------- la figura


#: El pie de la figura, DECLARADO antes de la corrida. Dice lo que va a pasar y
#: por que no es un defecto, de modo que la figura se lea como el limite real
#: del sopesado y no como un dibujo roto.
PIE_CONTAMINACION = (
    f'`w_j = 1 {chr(8722)} H{chr(8322)}(g_j)/ln C` mide **INCERTIDUMBRE, no '
    f'error**. Un punto contaminado que cae bien adentro de la clase '
    f'equivocada queda **confiadamente equivocado**: su `w` es ALTO y su marca '
    f'CRECE. Solo los que caen ENTRE clases se achican. Esto está predicho '
    f'por escrito antes de la corrida y es el límite real del sopesado, no '
    f'un panel roto. {chr(183)} Tamaño de marca: '
    f'`tamaño = {TAM_MIN_W:.0f} + {TAM_MAX_W - TAM_MIN_W:.0f}'
    f'{chr(183)}w`, lineal en `w`.'
)


def panel_contaminacion(a, Pt, w, idx, lim, cuadrada=True):
    """Los limpios de fondo en gris, los contaminados en negro y por su `w`.

    El tamanio dice el peso y el color dice el grupo, que son dos preguntas
    distintas y por eso dos canales distintos. Los limpios van TODOS y de un
    tamanio fijo chico: son el fondo contra el que se lee donde cayeron los
    otros, no una segunda serie de pesos.
    """
    Pt = np.asarray(Pt, dtype=float)
    w = np.asarray(w, dtype=float)
    assert len(w) == len(Pt), (len(w), len(Pt),
                              'un peso por punto del train del blanco')
    sucio = np.zeros(len(Pt), dtype=bool)
    idx = np.asarray(idx, dtype=int)
    if len(idx):
        sucio[idx] = True
    a.scatter(Pt[~sucio, 0], Pt[~sucio, 1], s=7, c=COL_LIMPIO,
              edgecolors='none', zorder=1, label='limpios (fondo)')
    if sucio.any():
        a.scatter(Pt[sucio, 0], Pt[sucio, 1], s=tamano_por_w(w[sucio]),
                  facecolors='none', edgecolors=COL_SUCIO, linewidths=1.1,
                  zorder=3)
    # La leyenda LLEVA el mapa: con solo "tamanio proporcional a w" nadie
    # distingue 0.60 de 0.95, que es exactamente lo que la figura tiene que
    # poder mostrar.
    #
    # Las manijas son `Line2D` y NO `plt.scatter([], [])`, y esto no es estilo.
    # `plt.scatter` dibuja sobre los ejes ACTUALES --- `plt.gca()` --- que en
    # una rejilla son los ULTIMOS que `subplots` creo, no los que esta funcion
    # recibio. Medido en el ensayo: las cinco manijas de cada fila aterrizaban
    # en el panel de abajo a la derecha y la leyenda de `panel_fase2` las
    # adoptaba, de modo que el panel de la perdida mostraba una escala de `w`
    # que no dibuja. `Line2D` no se adjunta a ningunos ejes.
    manijas = [Line2D([], [], linestyle='none', marker='o',
                      markersize=float(np.sqrt(tamano_por_w(v))),
                      markerfacecolor='none', markeredgecolor=COL_SUCIO,
                      markeredgewidth=1.1, label=f'w = {v:.2f}')
               for v in REFERENCIAS_W]
    a.legend(handles=manijas, loc='upper right', fontsize=7, framealpha=.9,
             labelspacing=1.5, borderpad=0.9, handletextpad=1.4,
             title='contaminados', title_fontsize=7.5)
    a.set_xlim(lim[0]); a.set_ylim(lim[1]); a.grid(alpha=.25)
    if cuadrada:
        a.set_aspect('equal')
    a.tick_params(labelsize=8)
    n_sucio = int(sucio.sum())
    medio = float(w[sucio].mean()) if n_sucio else float('nan')
    a.set_title(
        f'{SPLIT_DIBUJADO} {chr(183)} contaminados {n_sucio} de {len(Pt)} '
        f'{chr(183)} tamaño = w {chr(183)} w medio {medio:.3f}' if n_sucio else
        f'{SPLIT_DIBUJADO} {chr(183)} sin contaminación {chr(183)} '
        f'{len(Pt)} puntos limpios',
        fontsize=10)


def caja_contaminacion(figs):
    """La caja COMUN a todo lo que se dibuje de este toyset, y si es cuadrada.

    Comun y no por fila ni por cuadro: las filas comparten arquitectura y base
    de PCA, asi que una caja por fila --- o peor, una por CUADRO de la
    animacion --- haria que el blanco pareciera moverse cuando lo unico que
    cambio fue el sistema de coordenadas del dibujo.

    La calculan la figura estatica y la animacion, por el mismo lector: dos
    cajas calculadas aparte divergen y entonces el PNG y el GIF del mismo
    escenario no son el mismo dibujo en dos formatos.
    """
    assert figs, 'no hay ningun cuadro del que sacar la caja'
    emb = np.concatenate(
        [np.asarray(p, dtype=float) for f in figs for p in f['PS_t']]
        + [np.asarray(p, dtype=float) for f in figs for p in f['Pt']])
    centro = (emb.max(0) + emb.min(0)) / 2
    rango = emb.max(0) - emb.min(0)
    razon = float(max(rango) / max(float(min(rango)), 1e-9))
    if razon <= ASPECTO_MAX:
        medio = float(np.max(rango)) / 2 * 1.06
        lim = [(float(centro[d] - medio), float(centro[d] + medio))
               for d in range(2)]
    else:
        lim = [(float(centro[d] - rango[d]/2*1.06),
                float(centro[d] + rango[d]/2*1.06)) for d in range(2)]
    return lim, razon <= ASPECTO_MAX


#: Los rotulos de las dos columnas, declarados UNA vez y usados por la figura
#: de la seccion 1 y por la de la escalera. Dos listas escritas aparte
#: divergen, y la que divergiera seria la que nadie volvio a mirar.
ROT_FUENTE = 'FUENTE' + chr(183) + ' color = clase real'
ROT_BLANCO = 'BLANCO' + chr(183) + ' círculo hueco = mal ruteado'
#: El mismo rotulo para la escalera, que marca otra cosa. Las dos figuras
#: dibujan el mismo circulo hueco sobre conjuntos DISTINTOS --- la seccion 1
#: marca los MAL RUTEADOS, la escalera los PERTURBADOS --- y tenerlo clavado
#: hacia que la figura afirmara una cosa en el panel y otra en el pie.
ROT_BLANCO_PERTURBADO = ('BLANCO' + chr(183)
                         + ' círculo hueco = PERTURBADO')


def _panel_entradas(a, X, y, rotulo, mal=None):
    """UN panel de nube de entrada en 3-D: las clases por color, los atipicos
    en circulo hueco.

    Extraido del cuerpo de `figura_entradas` para que la ESCALERA lo dibuje en
    vez de llevar una segunda copia. Dos copias del mismo panel divergen, y la
    que divergiera seria la que nadie volvio a mirar: este archivo ya tuvo dos
    listas de brazos escritas aparte y las figuras mostraban un brazo que la
    tabla no tenia.

    `mal=None` es la columna SIN atipicos --- la fuente, donde no hay ruteo que
    comparar --- y no «no se marcaron»: con una mascara, el panel ademas cuenta
    los marcados en su propio titulo, que es el numero que un lector necesita
    para saber que esta mirando. `tests/test_toy_seccion2.py` compara los
    pixeles de `figura_entradas` contra una replica de su cuerpo original, que
    es lo unico que puede sostener que la extraccion no cambio el dibujo.
    """
    for c in range(CLASES):
        m = y == c
        a.scatter(X[m, 0], X[m, 1], X[m, 2], s=7, alpha=0.45,
                  label=f'clase {c}')
    if mal is not None and mal.any():
        a.scatter(X[mal, 0], X[mal, 1], X[mal, 2], s=70,
                  facecolors='none', edgecolors='k', linewidths=0.9)
    # El SUSTANTIVO sale del rotulo, no clavado: la seccion 1 marca los mal
    # ruteados y la escalera los perturbados, y el conteo tiene que nombrar lo
    # que de verdad conto. Clavado, la figura decia "mal ruteados" encima de
    # 324 perturbados mientras su propio pie decia "PERTURBADO".
    _qué = 'perturbados' if 'PERTURBADO' in rotulo else 'mal ruteados'
    a.set_title(rotulo + chr(10) + f'{int(mal.sum())} de {len(mal)} '
                f'{_qué} ({100*mal.mean():.0f}%)'
                if mal is not None else rotulo, fontsize=10, linespacing=1.4)
    a.set_xticklabels([]); a.set_yticklabels([]); a.set_zticklabels([])


def figura_entradas(nombre, CONTAM, ENTR, banderas, destino=None,
                    perturbacion=None, rho=0.0):
    """El PRIMER panel de la seccion: el dato de ENTRADA, atipicos marcados.

    Dos columnas, fuente y blanco, porque estos toysets SON el corrimiento: una
    nube del blanco sola muestra una forma y no una transferencia.

    **Quienes son los atipicos.** Sobre dato limpio no hay perturbacion
    inyectada, asi que los atipicos son los MAL RUTEADOS, y el npz del barrido
    los nombra sin ambiguedad: `rut[k]` es la clase predicha en el punto de
    control donde la parada eligio y `yt_tr` la real, de modo que el conjunto es
    `rut[k] != yt_tr`. No se estima nada.

    Esta figura no se pudo dibujar durante cuatro intentos, y la razon era
    estructural: con una perturbacion INYECTADA el artefacto registraba CUALES
    puntos se movieron y nunca DONDE cayeron, y los aterrizajes de `frontera`
    dependen de un cabezal de fase 1 entrenado en proceso y persistido en
    ningun lado. Sobre dato limpio ese obstaculo no existe.

    Las nubes salen de `toy_experiment.guardar_entradas` y no de
    `datos_semilla`: este modulo no importa `torch` --- invariante declarado,
    con test --- asi que no puede generarlas.
    """
    f = _celda_contaminacion(CONTAM, nombre, rho, perturbacion)['figura']
    k = int(f['k'])
    sem = int(np.asarray(f['semilla']))
    rut = np.asarray(f['rut'][k]).ravel()
    yt = np.asarray(f['yt_tr']).ravel()
    # La semilla tiene que tener nube persistida. Falla cerrado: dibujar la de
    # otra semilla marcaria puntos que esta corrida no vio y la figura se veria
    # perfecta.
    faltan = [c for c in ('Xs_tr', 'Xt_tr', 'ys_tr')
              if f'{nombre}|{sem}|{c}' not in ENTR]
    assert not faltan, (
        nombre, sem, faltan, 'esa semilla no tiene nube de entrada '
        'persistida; corré `guardar_entradas` con ella')
    Xs = np.asarray(ENTR[f'{nombre}|{sem}|Xs_tr'], dtype=float)
    Xt = np.asarray(ENTR[f'{nombre}|{sem}|Xt_tr'], dtype=float)
    ys = np.asarray(ENTR[f'{nombre}|{sem}|ys_tr']).ravel()
    # Y los largos tienen que coincidir. Sin esto, una nube de 1080 contra un
    # ruteo de 1079 se emparejan por posicion, marcan los puntos equivocados y
    # no falla nada --- medido: lunas sortea 1080 y las otras dos 1079.
    assert len(Xt) == len(rut) == len(yt), (
        nombre, len(Xt), len(rut), len(yt),
        'la nube y el ruteo no tienen el mismo largo; emparejarlos por '
        'posicion marcaria los puntos equivocados sin que nada falle')
    mal = rut != yt
    destino = Path('.') if destino is None else Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.5, 6.2))
    for col, (X, y, rot) in enumerate((
            (Xs, ys, ROT_FUENTE), (Xt, yt, ROT_BLANCO))):
        a = fig.add_subplot(1, 2, col + 1, projection='3d')
        # La mascara va SOLO en la columna del blanco: el ruteo es del blanco y
        # la fuente no tiene con que compararse. Es el mismo panel que la
        # escalera dibuja, y por eso vive afuera.
        _panel_entradas(a, X, y, rot, mal if col == 1 else None)
    fig.suptitle(f'{nombre}: el dato de entrada, semilla {sem}, '
                 f'en la época que el criterio eligió (época '
                 f'{int(f["pasos"][k]) // pasos_por_epoca_del_cuadro(f["pasos"])})',
                 fontsize=12)
    fig.tight_layout()
    ruta = destino/f'entradas_{nombre}.png'
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    return ruta


def figura_contaminacion(nombre, CONTAM, banderas, destino=None,
                         perturbacion=None):
    """Una figura por toyset: una FILA por escenario de la escalera, cuatro columnas.

    **`perturbacion` se nombra y no se elige por defecto.** La identidad de un
    escenario es el par `(perturbacion, rho)` y no el rho solo: a un mismo
    `rho = 0.30` corren las dos y predicen lo CONTRARIO sobre el par de medias,
    asi que seleccionar por rho dibujaria una de las dos sin decir cual --- y
    una figura con el pie del escenario contrario afirma lo opuesto de lo que
    midio. Con una sola perturbacion declarada `None` la resuelve; con dos,
    `_celda_contaminacion` se niega y nombra las dos. La tabla y la animacion
    toman el mismo parametro: las tres son una regla y no tres costumbres.

    Las columnas, en el orden en que hay que leerlas:

      1. el embedding como ya existe, coloreado por la clase REAL
      2. los limpios en gris de fondo y los contaminados en negro, con el
         tamanio de cada marca proporcional a su `w` en ESE punto de control
      3. la fase 1 sobre el eje fijo 0-1 de la Ec. 21
      4. la fase 2 sobre el mismo eje fijo

    El punto de control que se dibuja es el que la parada eligio, no el ultimo
    corrido: es el modelo que la tabla reporta.

    La caja es COMUN a todas las filas, por escenario del mismo toyset: las
    cuatro filas comparten arquitectura y base de PCA, asi que una caja por
    fila haria que el blanco pareciera moverse cuando lo unico que cambio fue
    el sistema de coordenadas del dibujo.
    """
    rhos = list(CONTAM['rhos'])
    celdas = [_celda_contaminacion(CONTAM, nombre, r, perturbacion)
              for r in rhos]
    # La que se DIBUJO, no la que se pidio: con una sola declarada `None` la
    # resuelve, y el titulo tiene que nombrar lo que hay en el dibujo. El pie no
    # puede quedarse en `None` mientras las filas son de una perturbacion
    # concreta --- eso es exactamente la figura rotulada con el escenario
    # contrario, una vuelta mas abajo.
    declaradas = perturbaciones_del_artefacto(CONTAM)
    dibujada = perturbacion if perturbacion is not None else (
        declaradas[0] if len(declaradas) == 1 else None)
    faltan = [r for r, c in zip(rhos, celdas) if c.get('figura') is None]
    if faltan:
        raise ValueError(
            f'{nombre}: los escenarios {faltan} no tienen cuadro de figura en '
            'el artefacto; corré el generador')
    figs = [c['figura'] for c in celdas]
    lim, cuadrada = caja_contaminacion(figs)

    fig, ax = plt.subplots(len(rhos), 4, squeeze=False,
                           figsize=(23.5, 5.8*len(rhos)))
    for r, (rho, f) in enumerate(zip(rhos, figs)):
        k = int(f['k'])
        # El `cu` que `panel_embedding` espera, rearmado del cuadro guardado y
        # con los campos EN EL ORDEN declarado por `CAMPOS_CU_NOMBRES`. Se
        # comprueba la aridad: este unpack ya se rompio cinco veces.
        cu = [(f['pasos'][j], np.asarray(f['Pt'][j], dtype=float),
               f['acc_va'][j], f['pok'][j], np.asarray(f['rut'][j]),
               f['acc_s'][j], f['u_t'][j], f['acc_te'][j])
              for j in range(len(f['pasos']))]
        assert all(len(c) == CAMPOS_CU for c in cu), (len(cu[0]), CAMPOS_CU)
        D = {'ys_tr': np.asarray(f['ys_tr']), 'yt_tr': np.asarray(f['yt_tr'])}
        panel_embedding(ax[r, 0], np.asarray(f['PS_t'][k], dtype=float), cu, k,
                        lim, D, np.asarray(f['pred_s']), False, cuadrada)
        ax[r, 0].set_title('embedding ' + chr(183) + ' color = clase REAL'
                           + chr(10) + ax[r, 0].get_title(),
                           fontsize=10, linespacing=1.4)
        panel_contaminacion(ax[r, 1], f['Pt'][k], f['w'][k], f['idx'], lim,
                            cuadrada)
        panel_fase1(ax[r, 2], [tuple(x) for x in f['hist_s']],
                    'fase 1 ' + chr(183) + ' Ec. 21, eje fijo 0-1',
                    int(f['ep_s']))
        panel_fase2(ax[r, 3], [tuple(x) for x in f['hist_t']],
                    f['pasos'][k], 'fase 2 ' + chr(183) + ' eje fijo 0-1',
                    int(f['mejor_paso']),
                    pasos_por_epoca_del_cuadro(f['pasos']))
        etq = (f'{chr(961)} = {rho:.2f}' + chr(10)
               + f'{len(np.asarray(f["idx"]))} reubicados' + chr(10)
               + f"eligió la época "
                 f"{int(f['mejor_paso']) // pasos_por_epoca_del_cuadro(f['pasos'])}")
        ax[r, 0].set_ylabel(etq, fontsize=11.5, labelpad=10, linespacing=1.4)
    sem = sorted({int(f['semilla']) for f in figs})
    fig.suptitle(
        f"{nombre} {chr(183)} sopesado bajo contaminación {chr(183)} "
        f"perturbación {rotulo_perturbacion(dibujada)} {chr(183)} "
        f"{NOMBRE_FILA[CONTAM['pos']]} {chr(183)} brazo "
        f"«{BRAZO_CON_SOP}» {chr(183)} semilla "
        f"{sem[0] if len(sem) == 1 else sem} {chr(183)} "
        f"{etiqueta_balance(banderas)}", fontsize=13, y=0.99)
    fig.text(0.5, 0.008, PIE_CONTAMINACION, ha='center', fontsize=9.5,
             wrap=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.075,
                        hspace=0.3, wspace=0.21)
    if destino is None:
        return fig
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    # El nombre lleva la perturbacion: dibujar las dos sobre un solo archivo
    # deja una pisada en silencio, que es la misma perdida que la clave de dos
    # partes causaba adentro del artefacto. SIEMPRE, incluso sin perturbacion
    # declarada --- la misma ranura que el GIF ya usa, y por la misma razon: un
    # nombre cuya forma depende de si hay una declarada hace que el archivo de
    # un artefacto viejo y el de una perturbacion nueva se puedan pisar.
    ranura = RANURA_SIN_PERTURBACION if dibujada is None else dibujada
    ruta = destino/f'contaminacion_{nombre}_{ranura}.png'
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    return ruta


# ===========================================================================
# El GIF de UN escenario: los cuadros son los puntos de control de la fase 2
# ---------------------------------------------------------------------------
# El SUJETO es el train del blanco y como se mueve su `w` a lo largo de los
# pasos de la fase 2. Eso es todo lo que anima, y es la unica razon para
# animarlo: `w` solo existe ahi --- no hay ningun termino evaluado sobre test,
# asi que un `w` sobre test seria un numero sin definicion detras.
#
# UN escenario y no la escalera: la escalera era el eje de la figura estatica,
# que es donde se lee de un vistazo. Con el tiempo como eje la escalera sobra,
# y el escenario que se anima es el MAS ALTO --- el que mas contaminados tiene
# y por lo tanto el unico donde el par de medias se puede ver.
# ===========================================================================


#: Los campos que `tabla_dataset` lee de una celda del barrido PRINCIPAL. Se
#: declara para poder MEDIR cuales de ellos el registro de contaminacion no
#: tiene, en vez de afirmarlo en una nota que envejece sin que nada falle.
#: `tests/test_toy_contamination.py` lee el cuerpo de `tabla_dataset` y
#: comprueba que cada nombre de aca sigue siendo uno que esa tabla lee.
CAMPOS_TABLA_DATASET = ('ep_s', 'acc_predict', 'acc_s_te', 'acc_te',
                        'vigia_fin', 'acc_te_ser', 'mejor_epoca')

#: Por que falta cada uno en el registro de contaminacion. El nombre del campo
#: es un contrato de datos y se imprime como esta; la razon es para el lector.
RAZON_CAMPO_AUSENTE = {
    'ep_s': 'la epoca de fase 1 se guarda del cuadro de la figura, de UNA '
            'semilla, no por semilla',
    'acc_predict': '`.predict` es un BRAZO de esta tabla y no una columna de '
                   'las otras filas',
    'acc_te_ser': 'la serie de test por punto de control no se persiste por '
                  'semilla; solo su maximo, en la columna del pico',
    'mejor_epoca': 'el registro de contaminacion guarda `mejor_paso` y '
                   '`paso_fin` y no la epoca; la tabla la deriva con '
                   '`epoca_elegida_del_brazo`',
}

#: El pie del GIF: el de la figura estatica MAS lo que solo la animacion puede
#: decir. Extiende y no reemplaza --- dos pies del mismo escenario divergen, y
#: el que nadie revisa es el que sigue afirmando lo que la corrida desmintio.
#:
#: Lo que esta aca es lo que vale para CUALQUIER perturbacion: el mecanismo y
#: el split. La prediccion de hacia donde se mueve el par NO esta aca, porque
#: depende de la perturbacion y las dos declaradas predicen lo CONTRARIO ---
#: un pie que afirmara una de las dos estaria equivocado en la otra mitad de
#: la seccion. Esa va en `PREDICCION_PERTURBACION`, y el orden realmente
#: observado lo pone `pie_del_gif` MEDIDO.
PIE_CONTAMINACION_GIF = (
    PIE_CONTAMINACION
    + f' {chr(183)} **Lo que se anima es el {SPLIT_DIBUJADO}**: `w` solo '
      f'existe ahí porque no hay ningún término evaluado sobre test. La '
      f'perturbación se inyecta en el train y el acierto se mide sobre test, '
      f'así que el acierto es comparable a lo largo de la escalera '
      f'{chr(183)} El par de medias **se mide en cada cuadro** y el panel '
      f'dice cuál de las dos quedó arriba: el pie no afirma un resultado que '
      f'la corrida podría desmentir.'
)

#: Lo que cada perturbacion declarada PREDICE sobre el par de medias, por
#: escrito antes de la corrida. Las dos predicen lo CONTRARIO y por eso existen
#: las dos: una es el regimen donde el sopesado se rompe demostrablemente y la
#: otra el regimen donde puede funcionar por sus meritos. Un nombre que no esta
#: aca es una negativa y nunca uno de estos dos por defecto --- dibujar un
#: escenario bajo el pie del otro es exactamente el defecto que un valor por
#: defecto produce, y en silencio.
PREDICCION_PERTURBACION = {
    'reubicado': (
        f'**Perturbación `reubicado` (adversaria).** Los puntos se reubican '
        f'DENTRO de la región de otra clase, así que caen decididos y '
        f'equivocados: `w` es ALTO y su marca CRECE. **Lo PREDICHO antes de '
        f'la corrida** es que la media de los perturbados queda ARRIBA de la '
        f'de los limpios. Es el límite real del sopesado y la razón de que '
        f'este escenario se conserve. Lo que la corrida hizo lo dice el par '
        f'MEDIDO de abajo, cuadro por cuadro, y no tiene por qué coincidir: '
        f'una predicción es una predicción.'),
    'frontera': (
        f'**Perturbación `frontera` (favorable).** Los puntos se colocan '
        f'ENTRE clases, donde `g` se aplana de verdad, así que `w` BAJA y '
        f'descontarlos es lo correcto. Lo predicho es que la media de los '
        f'perturbados termina **DEBAJO** de la de los limpios: es el régimen '
        f'donde el sopesado puede funcionar por sus méritos.'),
}

#: El rotulo de una perturbacion que el artefacto NO declara, en el pie y en el
#: nombre del archivo. Se escribe en los dos lados a proposito: el GIF de un
#: artefacto que no dice que perturbo no puede llamarse igual que el de uno que
#: si lo dice, o el segundo pisaria al primero sin que nada falle.
SIN_PERTURBACION_DECLARADA = 'perturbación no declarada'
RANURA_SIN_PERTURBACION = 'perturbacion-no-declarada'


def rotulo_perturbacion(perturbacion):
    """Como nombrar la perturbacion de un escenario, o su ausencia."""
    if perturbacion is None:
        return SIN_PERTURBACION_DECLARADA
    if perturbacion not in PREDICCION_PERTURBACION:
        raise ValueError(
            f'el artefacto declara la perturbación {perturbacion!r} y no hay '
            f'predicción escrita para ella; declaradas: '
            f'{sorted(PREDICCION_PERTURBACION)}. No se dibuja bajo el pie de '
            'otra: un escenario rotulado con la predicción del contrario dice '
            'lo opuesto de lo que midió.')
    return perturbacion


def pie_del_gif(perturbacion, sucio, limpio):
    """El pie entero: el mecanismo, la prediccion de ESTA perturbacion, y lo medido.

    La prediccion solo entra si el artefacto declara cual perturbacion corrio.
    Sin eso no se adjunta ninguna --- las dos declaradas predicen lo contrario,
    asi que elegir una seria elegir la conclusion. Lo MEDIDO entra siempre,
    porque eso no depende de ninguna declaracion.
    """
    partes = [PIE_CONTAMINACION_GIF]
    if perturbacion is None:
        partes.append(
            f'{chr(183)} El artefacto **no declara qué perturbación** corrió, '
            f'así que no se adjunta ninguna predicción: las dos declaradas '
            f'({", ".join(sorted(PREDICCION_PERTURBACION))}) predicen lo '
            f'contrario, y elegir una sería elegir la conclusión.')
    else:
        partes.append(chr(183) + ' '
                      + PREDICCION_PERTURBACION[rotulo_perturbacion(perturbacion)])
    if sucio is None:
        partes.append(f'{chr(183)} **Medido** en este cuadro: sin puntos '
                      f'perturbados, no hay par de medias.')
    else:
        orden = ('ARRIBA de' if sucio > limpio else
                 'DEBAJO de' if sucio < limpio else 'IGUAL que')
        partes.append(
            f'{chr(183)} **Medido** en el último cuadro: perturbados '
            f'{sucio:.4f}, **{orden}** limpios {limpio:.4f} '
            f'({sucio - limpio:+.4f}).')
    # Partido a mano y no con `wrap=True`: matplotlib re-envuelve la cadena
    # ENTERA e ignora los saltos, asi que el pie salia en una sola linea de mil
    # caracteres pisando los rotulos del eje. Cada parte arranca en su propio
    # renglon --- son oraciones distintas --- y despues se parte por ancho.
    return chr(10).join(
        chr(10).join(textwrap.wrap(parte, ANCHO_PIE)) for parte in partes)


#: El ancho del pie en caracteres. Medido sobre la figura de seis columnas:
#: 3400 px de ancho a 8.5 pt dan sitio de sobra para esto y deja el pie en
#: media docena de renglones en vez de en uno solo.
ANCHO_PIE = 260


#: Como se marca un brazo TECHO: uno que lee `y_t` y por lo tanto NO es un
#: metodo que compita. Se imprime pegado al nombre del brazo en la tabla, para
#: que no se pueda leer como un competidor mas de la escalera.
MARCA_TECHO = f'{chr(183)} **TECHO** (lee `y_t`, no es un método)'


def brazos_techo(CONTAM):
    """Los brazos que el REGISTRO declara techo, por la convencion que lo ESCRIBE.

    La convencion no es un campo con un nombre que este archivo elija: es
    «un campo cuyo NOMBRE lleva `PALABRA_TECHO` adentro, con valor verdadero»,
    y `PALABRA_TECHO` se importa de la mitad que escribe el registro en vez de
    copiarse aca. `_fila_brazo` escribe hoy `techo_no_es_metodo`; leer un
    `'techo'` pelado habria pasado por al lado y la tabla habria impreso el
    oraculo como un metodo mas, sin que nada fallara.

    Esa es la razon de leer la palabra y no el nombre exacto del campo: el
    nombre puede crecer --- ya es `techo_no_es_metodo` y no `techo` --- y lo
    que la otra mitad promete es la palabra, no el campo.

    NO se adivina por el nombre del BRAZO: un oraculo llamado de otra forma
    quedaria sin marcar. Un registro que no marca ninguno devuelve el conjunto
    vacio, y la tabla lo DICE en vez de callarlo: un techo sin marcar se lee
    como un metodo que gana.
    """
    techos = set()
    for celda in CONTAM['celdas'].values():
        for b, filas in celda['brazos'].items():
            if filas and all(_dice_techo(f) for f in filas):
                techos.add(b)
    return techos


#: Como un nombre de campo puede llevar `PALABRA_TECHO` sin que el campo hable
#: de un techo: como PALABRA entera, separada por `_`, y no como subcadena.
#:
#: **Un subcadena pelado da falsos positivos medibles, no hipoteticos.** `techo`
#: es tambien el nombre de un HIPERPARAMETRO de esta seccion --- el techo de la
#: rampa de CREDA, `HP_2['TECHO'] = 1.0` --- de modo que `techo_rampa`,
#: `techo_lambda` o un `TECHO` suelto llevan la palabra adentro, tienen valor
#: verdadero y no dicen nada sobre un techo de oraculo. Medido: con la lectura
#: por subcadena, `_dice_techo({'techo_rampa': 1e-4})` devolvia `True`. Hoy esos
#: HP viven anidados bajo `hp2` y no se ven desde el nivel de arriba, asi que el
#: falso positivo esta a un aplanamiento de distancia --- y un aplanamiento
#: marcaria TODOS los brazos como techo y la tabla no reportaria ningun metodo.
def _partes_del_campo(clave):
    """El nombre de un campo partido en palabras, en minuscula."""
    return [p for p in str(clave).lower().replace('-', '_').split('_') if p]


def _dice_techo(fila):
    """Si algun campo de la fila nombra un TECHO como palabra y es verdadero.

    `PALABRA_TECHO` tiene que aparecer como una palabra entera del nombre del
    campo --- `techo_no_es_metodo` si, `techo_rampa` tambien lo haria por
    construccion y por eso ademas se exige que el valor sea BOOLEANO.

    **Las dos condiciones juntas, y cada una tapa lo que la otra deja pasar.**
    La palabra entera descarta un `sobretecho`; el booleano descarta un
    `techo_rampa = 1e-4`, que es un hiperparametro real de esta seccion y el
    falso positivo que la lectura anterior daba. Un umbral no es una marca: lo
    que la mitad que escribe promete es una bandera, y `_fila_brazo` escribe
    `techo_no_es_metodo=bool(...)`.
    """
    for k, v in fila.items():
        if PALABRA_TECHO not in _partes_del_campo(k): continue
        if not isinstance(v, bool): continue
        if v: return True
    return False


def _perturbacion_de_clave(clave):
    """La perturbacion que una clave de celda nombra, o `None` si no nombra ninguna.

    `(familia, rho)` es la forma vieja y no declara perturbacion; `(familia,
    perturbacion, rho)` la declara. La ausencia se devuelve como `None` y NUNCA
    como el nombre de una de las dos: leer una declaracion ausente como un
    valor concreto es lo que dibuja un escenario bajo el pie del otro.
    """
    if len(clave) == 2:
        return None
    if len(clave) == 3:
        return str(clave[1])
    raise ValueError(
        f'la clave de celda {clave!r} no tiene 2 ni 3 partes, así que no se '
        'puede leer de ella ni la familia ni la perturbación ni el rho')


def perturbaciones_del_artefacto(CONTAM):
    """Las perturbaciones que el artefacto declara, LEIDAS de sus claves.

    Y cruzadas contra el padron que declare arriba, si declara uno: un nombre
    en las claves que el padron no tiene, o al revés, es una negativa. Las dos
    mitades del artefacto tienen que decir lo mismo, y comprobar solo una
    verifica la mitad que nadie rompe.
    """
    nombres = {_perturbacion_de_clave(k) for k in CONTAM['celdas']}
    padron = CONTAM.get('perturbaciones')
    if padron is not None:
        padron = {str(v) for v in padron}
        sobran = {n for n in nombres if n is not None} - padron
        faltan = padron - {n for n in nombres if n is not None}
        if sobran or faltan:
            raise ValueError(
                f'el padrón de perturbaciones del artefacto ({sorted(padron)}) '
                f'y lo que sus celdas nombran no coinciden: sobran '
                f'{sorted(sobran)}, faltan {sorted(faltan)}')
    return sorted(nombres, key=lambda v: (v is None, v))


def escenario_del_gif(CONTAM, perturbacion=None):
    """El escenario que se anima, como el PAR `(perturbacion, rho)`.

    La identidad de un escenario es el par y no el rho solo: a un mismo
    `rho = 0.30` corren dos perturbaciones que predicen lo CONTRARIO sobre el
    par de medias, asi que seleccionar por rho elegiria una de las dos sin
    decir cual.

    `perturbacion=None` NO es un valor por defecto: significa «la que el
    artefacto declare, si declara exactamente una». Con dos declaradas es una
    negativa que las nombra, porque elegir una seria elegir la conclusion. Un
    nombre que el artefacto no declara tambien es una negativa.

    El rho es el MAS ALTO de esa perturbacion y se lee de la escalera que
    corrio: un nivel clavado anima un escenario que el barrido no corrio en
    cuanto la escalera se mueve, y la escalera ya se movio una vez.
    """
    disponibles = perturbaciones_del_artefacto(CONTAM)
    assert disponibles, 'el artefacto no tiene ninguna celda'
    if perturbacion is None:
        if len(disponibles) != 1:
            raise ValueError(
                f'el artefacto declara {len(disponibles)} perturbaciones '
                f'({disponibles}); nombrá una con `perturbacion=`. No se elige '
                'por defecto: las declaradas predicen lo contrario sobre el '
                'par de medias, así que un defecto elegiría la conclusión.')
        perturbacion = disponibles[0]
    elif perturbacion not in disponibles:
        raise ValueError(
            f'el artefacto no tiene la perturbación {perturbacion!r}; '
            f'declara {disponibles}')
    rhos = [float(k[-1]) for k in CONTAM['celdas']
            if _perturbacion_de_clave(k) == perturbacion]
    assert rhos, (perturbacion, 'ninguna celda de esa perturbación')
    return perturbacion, max(rhos)


def _celda_escenario(CONTAM, nombre, perturbacion, rho):
    """La celda de UN escenario, buscada por el par y no por el rho solo."""
    clave = ((nombre, float(rho)) if perturbacion is None
             else (nombre, perturbacion, float(rho)))
    if clave not in CONTAM['celdas']:
        raise ValueError(
            f'no hay escenario {clave} en el artefacto de contaminación; '
            f'las claves que hay son '
            f'{sorted(str(k) for k in CONTAM["celdas"])[:8]}')
    return CONTAM['celdas'][clave]


def brazo_de_la_figura(CONTAM):
    """El brazo cuyo `w` guarda el artefacto, DERIVADO de lo que cada fila dice.

    El npz tiene UNA trayectoria y ninguna clave de brazo, asi que el nombre
    sale de la mitad del registro que si lo dice: exactamente un brazo declara
    `sopesado: True` en sus filas, y `w` solo tiene sentido para ese. Afirmarlo
    con una constante de este archivo seria escribir por segunda vez algo que
    el artefacto ya sabe, y las dos copias pueden discrepar.

    Dos brazos sopesados, o ninguno, es una negativa y no una eleccion: una
    figura rotulada con el brazo equivocado dibuja el `w` de un brazo bajo el
    nombre de otro, y nada falla.
    """
    sopesados = set()
    # La llave se toma ENTERA y se rotula con `join`, no se desempaca.
    #
    # Desempacarla como `(fam, rho)` reventaba con `too many values to unpack`
    # en cuanto el productor paso a llaves de tres --- `(familia,
    # perturbacion, rho)` --- y costo el plot entero despues de 23 minutos de
    # generador ya pagados. El bucle no necesita las partes: usa `celda`, y la
    # llave solo aparece en el mensaje de la negativa. Asi que el acoplamiento
    # a la aridad era gratuito, y cualquier aridad futura entra sin tocar esto.
    for clave_celda, celda in sorted(CONTAM['celdas'].items(),
                                     key=lambda kv: tuple(map(str, kv[0]))):
        _rotulo_celda = ' '.join(str(parte) for parte in clave_celda)
        aca = {b for b, filas in celda['brazos'].items()
               if filas and all(f.get('sopesado') is True for f in filas)}
        # Esta negativa esta SUBSUMIDA por la de abajo y se mide asi: todo
        # estado que refuse --- cero brazos sopesados, o dos --- deja el
        # conjunto cruzado con un tamanio distinto de uno, asi que la de abajo
        # tambien lo refusaria. Apagar solo esta no vuelve alcanzable ningun
        # defecto, y la probe lo confirmo: el test siguio verde.
        #
        # Queda por el MENSAJE y se declara que es para eso: nombra el
        # escenario y los brazos concretos, que es lo que un lector necesita
        # para arreglarlo, y la de abajo no puede nombrarlos. Lo que NO se
        # puede decir es que sea la que atrapa el caso --- un guardia
        # presentado como la red cuando la red es otro es exactamente lo que
        # hace que nadie note cuando se cae.
        if len(aca) != 1:
            raise ValueError(
                f'{_rotulo_celda}: {len(aca)} brazos declaran `sopesado: '
                f'True` ({sorted(aca)}); el artefacto no dice de cual brazo '
                'es el `w` que guardo')
        sopesados |= aca
    if len(sopesados) != 1:
        raise ValueError(
            f'los escenarios no coinciden en cual es el brazo sopesado: '
            f'{sorted(sopesados)}')
    return sopesados.pop()


def medias_w(w, idx):
    """La media de `w` sobre los contaminados y sobre su COMPLEMENTO.

    Este par es la figura entera. Lo predicho por escrito antes de la corrida
    es que la media de los contaminados NO cae debajo de la de los limpios,
    porque `w = 1 - H2(g)/ln C` mide incertidumbre y una equivocacion
    confiada es confiada. Calcularlo sobre la particion equivocada ---
    contra TODOS los puntos en vez de contra el complemento --- contesta otra
    pregunta con la misma forma, y la media de todos incluye a los
    contaminados, asi que la diferencia sale diluida en vez de equivocada.

    Sin contaminados no hay numero, y devuelve `None` y no un cero: un cero se
    dibuja, y una fila sin contaminar quedaria con una curva en el piso que
    nadie midio. Es la misma convencion que `_resumen_w`.
    """
    w = np.asarray(w, dtype=float)
    idx = np.asarray(idx, dtype=int)
    mask = np.zeros(len(w), dtype=bool)
    if len(idx):
        mask[idx] = True
    sucio = float(w[mask].mean()) if mask.any() else None
    assert (~mask).any(), 'no quedo ningun punto limpio contra el que comparar'
    return sucio, float(w[~mask].mean())


def rotulo_fila_contaminacion(escenario, f, j):
    """El rotulo de la fila: su escenario, el cuadro en pantalla y SU parada.

    El escenario es el PAR `(perturbacion, rho)`: a un mismo rho corren dos
    perturbaciones que predicen lo contrario, asi que un rotulo con el rho
    solo no dice que escenario se esta mirando.

    `mejor_paso` es el de ESTA fila y nunca el de otra. Medido en `lunas` bajo
    el techo en PASOS --- la cita se deja fechada, no se actualiza a ojo: es de
    antes del 2026-10-19 --- la escalera eligio en 52, 44, 52 y 102, cuatro
    puntos distintos. Un rotulo que lleva el numero de una fila sobre los
    paneles de otra afirma que el criterio eligio donde no eligio, y nada
    revienta.

    **Lo que el rotulo IMPRIME son epocas**, que es la unidad en la que este
    paquete decide; los pasos de arriba son el indice del arreglo y no salen de
    esta funcion. El cuadro en pantalla va al lado de la eleccion a proposito:
    sin los dos juntos no se puede ver CUANDO eligio el criterio respecto de lo
    que se esta mirando.
    """
    perturbacion, rho = escenario
    pasos = [int(v) for v in f['pasos']]
    por_epoca = pasos_por_epoca_del_cuadro(pasos)
    mejor = int(f['mejor_paso'])
    paso = pasos[j]
    estado = ('el criterio ya eligió' if paso >= mejor
              else 'el criterio todavía no eligió')
    # **La comparacion de arriba es en PASOS y los dos numeros de abajo en
    # EPOCAS.** «ya eligió» es una pregunta sobre el orden de dos puntos de la
    # traza, que el arreglo indexa por paso; lo que el lector lee es la epoca,
    # que es la unidad en la que este paquete decide. Convertir antes de
    # comparar no cambiaria el resultado --- la division es exacta --- pero
    # haria que la guarda dependiera de una conversion en vez del indice.
    return (f'{rotulo_perturbacion(perturbacion)}' + chr(10)
            + f'{chr(961)} = {float(rho):.2f}' + chr(10)
            + f'{len(np.asarray(f["idx"]))} perturbados del {SPLIT_DIBUJADO}'
            + chr(10) + f'época {paso // por_epoca} de la fase 2' + chr(10)
            + f'eligió la época {mejor // por_epoca} {chr(183)} {estado}')


def panel_medias_w(a, pasos, W, idx, j):
    """El par de medias, PROGRESIVO hasta el cuadro `j`, y su orden MEDIDO.

    El eje es el 0-1 FIJO del propio `w`: `w = 1 - H2(g)/ln C` vive ahi por
    construccion, igual que los dos paneles de perdida de al lado, y asi los
    cuatro se leen en la misma escala. El rango REALIZADO va en el titulo por
    el mismo motivo que en `panel_fase1`: con las medias medidas entre 0.77 y
    0.90 el par ocupa el 13% del eje y sale casi plano, y sin ese numero eso se
    lee como un grafico roto en vez de como lo que es.

    Y el titulo dice, MEDIDO en este cuadro, cual de las dos esta arriba. El
    pie afirma la prediccion; esto la comprueba. Una afirmacion que solo vive
    en una oracion envejece sin que nada falle, y esta es exactamente la
    oracion que la figura existe para sostener.
    """
    # El eje en EPOCAS, que es la unidad de este paquete. `pasos` es la
    # rejilla de cuadros en pasos --- `corrida` la escribe multiplicando la
    # rejilla declarada en epocas por el pase de la celda --- y dividir por ese
    # mismo pase la deshace exactamente.
    pasos = [int(v) for v in pasos]
    _ppe = pasos_por_epoca_del_cuadro(pasos)
    epocas = [v // _ppe for v in pasos]
    s = epocas[:j+1]
    pares = [medias_w(W[t], idx) for t in range(j+1)]
    sucios = [p[0] for p in pares]
    limpios = [p[1] for p in pares]
    hay = all(v is not None for v in sucios)
    if hay:
        a.plot(s, sucios, color=COL_SUCIO, lw=2.0, marker='o', ms=3.5,
               label='contaminados (media de w)')
        a.plot([s[-1]], [sucios[-1]], 'o', ms=9, color=COL_SUCIO)
    a.plot(s, limpios, color='#8c564b', lw=1.8, ls='--', marker='o', ms=3.5,
           label='limpios (media de w)')
    a.plot([s[-1]], [limpios[-1]], 'o', ms=9, color='#8c564b')
    a.set_ylim(0.0, 1.0)
    a.set_xlim(0, max(epocas[-1], 1))
    a.grid(alpha=.25); a.tick_params(labelsize=8)
    a.set_xlabel('época de la fase 2', fontsize=9)
    a.legend(loc='lower right', fontsize=8, framealpha=.9)
    if not hay:
        a.set_title(f'{SPLIT_DIBUJADO} {chr(183)} sin contaminación: '
                    f'no hay par de medias de w', fontsize=9.5)
        return
    _todo = [v for v in sucios + limpios]
    _lo, _hi = float(min(_todo)), float(max(_todo))
    orden = ('ARRIBA' if sucios[-1] > limpios[-1] else
             'DEBAJO' if sucios[-1] < limpios[-1] else 'IGUAL que')
    a.set_title(
        f'media de w {chr(183)} {SPLIT_DIBUJADO}' + chr(10)
        + f'contaminados {sucios[-1]:.4f} {orden} limpios {limpios[-1]:.4f} '
          f'({sucios[-1]-limpios[-1]:+.4f})' + chr(10)
        + f'(realizado: [{_lo:.4f}, {_hi:.4f}] = {100*(_hi-_lo):.1f}% del eje)',
        fontsize=9, linespacing=1.35)


def panel_aciertos(a, pasos, acc_va, acc_te, j, mejor_paso):
    """Los aciertos del blanco, PROGRESIVOS, con el azar y el paso elegido.

    Valid y test juntos y rotulados: valid es el split que la busqueda y la
    parada MIRAN, test es el que se reporta y no entra en ninguna decision. Son
    la misma cantidad sobre dos conjuntos, asi que comparten eje; lo que no
    comparten es para que sirve cada uno, y eso lo dice el rotulo.

    De UNA semilla --- la del cuadro guardado --- y por eso no lleva dispersion
    ni verdicto: lo que compara los brazos es la tabla de al lado, sobre las
    semillas que el barrido corrio.
    """
    # El eje en EPOCAS, por la misma via que `panel_medias_w`: la rejilla de
    # cuadros sale de la rejilla declarada en epocas, asi que dividir por el
    # pase de la celda la devuelve sin aproximar nada.
    pasos = [int(v) for v in pasos]
    _ppe = pasos_por_epoca_del_cuadro(pasos)
    epocas = [v // _ppe for v in pasos]
    s = epocas[:j+1]
    va = [float(v) for v in acc_va[:j+1]]
    te = [float(v) for v in acc_te[:j+1]]
    a.plot(s, va, color='#1f77b4', lw=1.6, ls='--', marker='o', ms=3.5,
           label='acierto VALID (lo que el criterio mira)')
    a.plot(s, te, color='#d62728', lw=2.0, marker='o', ms=3.5,
           label='acierto TEST (lo que se reporta)')
    a.plot([s[-1]], [te[-1]], 'o', ms=9, color='#d62728')
    a.axhline(1/CLASES, color='#555555', lw=.9, ls=':', alpha=.8,
              label=f'azar {1/CLASES:.3f}')
    # La guarda compara PASOS --- `mejor_paso` y la rejilla son pasos --- y la
    # linea se dibuja en EPOCAS, que es el eje. Mezclar las dos pondria la
    # marca cuadros antes de donde va.
    if int(mejor_paso) <= pasos[-1]:
        a.axvline(int(mejor_paso) // _ppe, color='#2ca02c', lw=1.0, alpha=.7)
    a.set_ylim(0.0, 1.0)
    a.set_xlim(0, max(epocas[-1], 1))
    a.grid(alpha=.25); a.tick_params(labelsize=8)
    a.set_xlabel('época de la fase 2', fontsize=9)
    a.legend(loc='lower right', fontsize=7.5, framealpha=.9)
    a.set_title(f'acierto del blanco {chr(183)} una semilla' + chr(10)
                + f'valid {va[-1]:.4f} {chr(183)} test {te[-1]:.4f}',
                fontsize=9.5, linespacing=1.35)


def _cu_del_cuadro(f):
    """El `cu` que `panel_embedding` espera, rearmado del cuadro guardado.

    Los campos EN EL ORDEN que `CAMPOS_CU_NOMBRES` declara, y la aridad
    comprobada: este unpack ya se rompio cinco veces, cada una al agregarse un
    campo nuevo al registro aunque fuera al final.
    """
    cu = [(f['pasos'][t], np.asarray(f['Pt'][t], dtype=float),
           f['acc_va'][t], f['pok'][t], np.asarray(f['rut'][t]),
           f['acc_s'][t], f['u_t'][t], f['acc_te'][t])
          for t in range(len(f['pasos']))]
    assert all(len(c) == CAMPOS_CU for c in cu), (len(cu[0]), CAMPOS_CU)
    return cu


def figura_cuadro_contaminacion(nombre, escenario, f, j, lim, cuadrada, brazo,
                                pos, banderas):
    """UN cuadro de la animacion, devuelto como figura para poder LEERLO.

    Devuelve la figura y no el arreglo de pixeles porque los rotulos de un
    arreglo RGB no se pueden comprobar: `cuadros_comunes` evita el
    `IndexError` y nada evitaria un rotulo que dice el paso de otra fila.
    `cuadro_contaminacion` es el que renderiza.

    Las columnas, en el orden en que hay que leerlas. Las cuatro primeras son
    las acordadas para la figura estatica, sin cambiar; las dos ultimas son lo
    que solo tiene sentido con el tiempo como eje:

      1. el train del blanco, coloreado por la clase REAL
      2. los limpios en gris de fondo y los contaminados en negro, con el
         tamanio de cada marca por `tamano_por_w` sobre el `w` de ESTE cuadro
      3. la fase 1, entera: termino antes de que empezara la fase 2, asi que
         no tiene nada que avanzar cuadro a cuadro --- es la misma forma que
         `cuadro_config` ya usa en el GIF de trayectorias
      4. la fase 2, hasta el paso de este cuadro
      5. el par de medias de `w`, hasta este cuadro
      6. los aciertos del blanco, hasta este cuadro
    """
    perturbacion, rho = escenario
    pasos = [int(v) for v in f['pasos']]
    cu = _cu_del_cuadro(f)
    D = {'ys_tr': np.asarray(f['ys_tr']), 'yt_tr': np.asarray(f['yt_tr'])}
    fig, ax = plt.subplots(1, 6, squeeze=False, figsize=(34.0, 6.6))
    a = ax[0]
    panel_embedding(a[0], np.asarray(f['PS_t'][j], dtype=float), cu, j, lim, D,
                    np.asarray(f['pred_s']), False, cuadrada)
    a[0].set_title(f'{SPLIT_DIBUJADO} {chr(183)} color = clase REAL' + chr(10)
                   + a[0].get_title(), fontsize=10, linespacing=1.4)
    panel_contaminacion(a[1], f['Pt'][j], f['w'][j], f['idx'], lim, cuadrada)
    panel_fase1(a[2], [tuple(x) for x in f['hist_s']],
                'fase 1 ' + chr(183) + ' Ec. 21, eje fijo 0-1', int(f['ep_s']))
    panel_fase2(a[3], [tuple(x) for x in f['hist_t']], pasos[j],
                'fase 2 ' + chr(183) + ' eje fijo 0-1', int(f['mejor_paso']),
                pasos_por_epoca_del_cuadro(pasos))
    panel_medias_w(a[4], pasos, f['w'], f['idx'], j)
    panel_aciertos(a[5], pasos, f['acc_va'], f['acc_te'], j,
                   int(f['mejor_paso']))
    a[0].set_ylabel(rotulo_fila_contaminacion(escenario, f, j), fontsize=11.5,
                    labelpad=10, linespacing=1.4)
    fig.suptitle(
        f"{nombre} {chr(183)} sopesado bajo contaminación {chr(183)} "
        f"{NOMBRE_FILA[pos]} {chr(183)} "
        f"brazo «{brazo}» {chr(183)} semilla {int(f['semilla'])} {chr(183)} "
        f"perturbación {rotulo_perturbacion(perturbacion)} {chr(183)} "
        f"{chr(961)} = {float(rho):.2f} {chr(183)} época "
        f"{pasos[j] // pasos_por_epoca_del_cuadro(pasos)} de "
        f"{pasos[-1] // pasos_por_epoca_del_cuadro(pasos)} "
        f"(cuadro {j+1} de {len(pasos)}) {chr(183)} "
        f"{SPLIT_DIBUJADO} {chr(183)} azar {1/CLASES:.3f} {chr(183)} "
        f"{etiqueta_balance(banderas)}", fontsize=13, y=0.985)
    # El pie lleva lo MEDIDO en este cuadro, no una afirmacion fija: con dos
    # perturbaciones que predicen lo contrario, un pie constante estaria
    # equivocado en la mitad de la seccion.
    _sucio, _limpio = medias_w(f['w'][j], f['idx'])
    fig.subplots_adjust(left=0.045, right=0.995, top=0.855, bottom=0.235,
                        wspace=0.23)
    # `wrap=False`: el texto ya viene partido por `pie_del_gif`, y dejar que
    # matplotlib lo re-envuelva deshace esos saltos.
    fig.text(0.5, 0.015, pie_del_gif(perturbacion, _sucio, _limpio),
             ha='center', va='bottom', fontsize=8.5, linespacing=1.5,
             wrap=False)
    return fig


def cuadro_contaminacion(nombre, escenario, f, j, lim, cuadrada, brazo, pos,
                         banderas):
    """El cuadro, renderizado a RGB para que `imageio` lo codifique."""
    fig = figura_cuadro_contaminacion(nombre, escenario, f, j, lim, cuadrada,
                                      brazo, pos, banderas)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def animar_contaminacion(nombre, CONTAM, banderas, destino, *,
                         perturbacion=None, al_paso=print):
    """El GIF de UN escenario: los cuadros son los puntos de control.

    NO corre nada: lee el artefacto que `guardar_contaminacion` dejo.

    El largo sale de `cuadros_comunes` sobre TODAS las series que un cuadro
    indexa --- los pasos, las proyecciones, los pesos, las dos de acierto ---
    y no solo sobre los pasos. Deberian medir lo mismo; indexar una con el
    largo de otra es exactamente el defecto que ya mordio tres veces, y
    comprobarlo cuesta una linea.
    """
    escenario = escenario_del_gif(CONTAM, perturbacion)
    perturbacion, rho = escenario
    brazo = brazo_de_la_figura(CONTAM)
    celda = _celda_escenario(CONTAM, nombre, perturbacion, rho)
    f = celda.get('figura')
    if f is None:
        raise ValueError(
            f'{nombre} {rotulo_perturbacion(perturbacion)} rho={rho:.2f}: el '
            'escenario no tiene cuadro de figura en el artefacto; corré el '
            'generador')
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    n = cuadros_comunes([len(f['pasos']), len(f['Pt']), len(f['PS_t']),
                         len(f['w']), len(f['acc_va']), len(f['acc_te'])])
    assert n > 1, (
        n, 'un GIF de un cuadro es un PNG con otro sufijo: la animacion '
           'existe para mostrar la evolucion de w, y con un cuadro no hay '
           'ninguna evolucion que mostrar')
    lim, cuadrada = caja_contaminacion([f])
    # La celda sale del ARTEFACTO (`CONTAM['pos']`), no de una bandera ni de
    # un valor por defecto: un titulo que afirma una normalizacion que la
    # corrida no uso es exactamente el defecto que `etiqueta_balance` existe
    # para cerrar del otro lado.
    pos = CONTAM['pos']
    if pos not in NOMBRE_FILA:
        raise ValueError(
            f'el artefacto declara la celda {pos!r} y no hay rotulo para '
            f'ella; conocidas: {sorted(NOMBRE_FILA)}')
    cuadros = [cuadro_contaminacion(nombre, escenario, f, j, lim, cuadrada,
                                    brazo, pos, banderas) for j in range(n)]
    # La perturbacion va SIEMPRE en el nombre, incluso cuando el artefacto no
    # la declara: un GIF de un artefacto que no dice que perturbo no puede
    # llamarse igual que el de uno que si lo dice, o el segundo pisaria al
    # primero y quedaria un archivo de menos que nadie mira.
    ranura = RANURA_SIN_PERTURBACION if perturbacion is None else perturbacion
    return _escribir_gif(
        destino/f'contaminacion_paso_{nombre}_{ranura}.gif', cuadros, al_paso)


def tabla_aciertos_contaminacion(nombre, escenario, CONTAM, banderas):
    """Los aciertos de TODOS los brazos de UN escenario, en escalera.

    La forma es la de `tabla_dataset` y el emisor de markdown es el mismo. Lo
    que NO es igual es el registro: el de contaminacion es mas angosto que el
    del barrido principal, y las columnas que no puede llenar se MIDEN
    ausentes y se nombran en el pie. Llenarlas con otra cosa daria una tabla
    que se lee como la de la primera seccion y contesta otra pregunta.

    Es DESCRIPTIVA y no da ningun verdicto: quien compara los brazos es
    `tabla_cuatro_brazos`, al lado, con Friedman sobre los cuatro METODOS y Holm
    dentro del conjunto DECLARADO de pares de este escenario. Dos medias
    desnudas no declaran un ganador.
    """
    perturbacion, rho = escenario
    celda = _celda_escenario(CONTAM, nombre, perturbacion, float(rho))
    brazos = celda['brazos']
    # El padron sale del ARTEFACTO y no de una lista de cuatro escrita aca: la
    # escalera crecio de dos brazos de CREDA a tres --- `w` apagado, declarado
    # y oraculo --- y una lista fija habria dejado el tercero fuera de la tabla
    # sin que nada fallara. Esta tabla es DESCRIPTIVA, de modo que un brazo mas
    # le agrega una fila y no un tratamiento; la de al lado no puede hacer lo
    # mismo, y por eso `BRAZOS_CUATRO` esta severado del padron y declara sus
    # pares.
    padron = [str(b) for b in CONTAM['brazos']]
    assert padron, 'el artefacto no declara ningún brazo'
    faltan = [b for b in padron if b not in brazos]
    if faltan:
        raise ValueError(
            f'{nombre} {rotulo_perturbacion(perturbacion)} rho={rho}: el '
            f'artefacto declara los brazos {padron} y la celda no tiene '
            f'{faltan}')
    n = _semillas([len(brazos[b]) for b in padron], banderas,
                  f'{nombre} rho={rho}: la tabla de aciertos')
    enc = ('método', 'época que eligió', 'fuente · test', 'blanco · test',
           'vigía: fuente conservada', 'blanco pico · test (oráculo)')
    _col = lambda filas, k: (_m([float(r[k]) for r in filas])
                             if all(k in r for r in filas) else _NADA)
    _med = lambda filas, k: (f'{int(np.median([r[k] for r in filas]))}'
                             if all(k in r for r in filas) else _NADA)
    techos = brazos_techo(CONTAM)
    fil = []
    for b in padron:
        filas = brazos[b]
        assert filas, (nombre, rho, b, 'el brazo no registro ninguna semilla')
        for r in filas:
            if r.get('split') != 'test':
                raise ValueError(
                    f'{nombre} rho={rho}: el brazo {b!r} registra el split '
                    f'{r.get("split")!r} y esta tabla reporta test')
        rotulo = f'`{b}`' + (f' {MARCA_TECHO}' if b in techos else '')
        # La EPOCA, derivada por `epoca_elegida_del_brazo` porque el registro
        # de contaminacion no la guarda. `_med` toma un campo del brazo y este
        # no es un campo, asi que la mediana se arma aca con la misma forma:
        # una raya cuando a algun brazo le falta la traza de la que sale.
        _epoca = lambda fs: (
            f"{int(np.median([epoca_elegida_del_brazo(r) for r in fs]))}"
            if all('mejor_paso' in r and 'paso_fin' in r for r in fs)
            else _NADA)
        fil.append((rotulo, _epoca(filas),
                    _col(filas, 'acc_s_te'), _col(filas, 'acc_te'),
                    _col(filas, 'vigia_fin'), _col(filas, 'pico_te')))
    # Lo ausente, MEDIDO contra lo que `tabla_dataset` lee, y no una nota.
    creda = brazos[BRAZO_CON_SOP]
    ausentes = [c for c in CAMPOS_TABLA_DATASET
                if not all(c in r for r in creda)]
    nota = ' {} '.format(chr(183)).join(
        f'`{c}`: {RAZON_CAMPO_AUSENTE.get(c, "no está en el registro")}'
        for c in ausentes) or 'el registro llena todas las columnas'
    # Cuantos techos marco, DICHO: un techo sin marcar se lee como un metodo
    # que gana, y un registro que no trae la bandera deja la tabla sin marcas
    # sin que nada falle. El numero medido es lo que hace visible la omision.
    marcados = [b for b in padron if b in techos]
    nota_techo = (
        f'{len(marcados)} brazo(s) marcados **TECHO** por el registro '
        f'({", ".join(f"`{b}`" for b in marcados)})' if marcados else
        '**el registro no marca ningún brazo como TECHO**: si alguno lee '
        '`y_t`, esta tabla lo está imprimiendo como un método que compite. La '
        'marca se lee de `techo: True` en las filas del brazo, igual que '
        '`sopesado`; sin ese campo no hay nada que leer')
    pie = (f'Descriptiva: **no declara ningún ganador** {chr(183)} el verdicto '
           f'de este escenario lo da la tabla de cuatro brazos de al lado, con '
           f'Friedman sobre sus cuatro métodos y Holm dentro de sus '
           f'{len(PARES_CUATRO_BRAZOS)} pares DECLARADOS {chr(183)} todo '
           f'sobre TEST, que '
           f'no entra en ninguna decisión {chr(183)} la contaminación se '
           f'inyecta en el {SPLIT_DIBUJADO} y el test no se toca, así que el '
           f'acierto es comparable a lo largo de la escalera {chr(183)} '
           f'columnas que el registro de esta sección NO tiene, medidas '
           f'ausentes contra las que `tabla_dataset` lee: {nota} '
           f'{chr(183)} {nota_techo}')
    return Markdown(
        f'**{nombre.upper()} {chr(183)} '
        f'{rotulo_perturbacion(perturbacion)} {chr(183)} '
        f'{chr(961)} = {float(rho):.2f} '
        f'({len(celda["indices"][sorted(celda["indices"])[0]])} puntos del '
        f'{SPLIT_DIBUJADO} reubicados) {chr(183)} {n} semillas {chr(183)} '
        f'media {chr(177)} desvío {chr(183)} azar {1/CLASES:.4f} {chr(183)} '
        f'test, sólo reporte {chr(183)} {etiqueta_balance(banderas)}**'
        + chr(10)*2 + _md(enc, fil) + chr(10)*2 + pie)


# ===========================================================================
# La ESCALERA de la seccion 2: una figura de entradas, un GIF y cuatro tablas
# ---------------------------------------------------------------------------
# Lo que la seccion compara a lo largo de la escalera de ruido, y lo que NO:
#
# - las ENTRADAS, nueve nubes en una figura, para ver de un vistazo que el
#   ruido le hace al dato antes de que nada entrene;
# - la TRAYECTORIA, un GIF cuyas filas son los niveles y cuyas CUATRO columnas
#   son exactamente las de `cuadro_config`. El operador fijo esas cuatro: una
#   quinta columna es otra figura con el mismo nombre;
# - el ACIERTO por nivel, descriptivo;
# - el SOPESADO contra el no sopesado, con test, sobre el acierto y sobre el
#   agrupamiento.
#
# **La correccion de Holm vive DENTRO del nivel y nunca a lo largo de la
# escalera.** Los pares de los cuatro niveles son una familia que nadie
# declaro, y corregir sobre ella corrige de mas; es la misma regla que
# `tabla_cuatro_brazos` ya escribe para su escenario. Con DOS brazos hay
# exactamente UN par por nivel, asi que Holm dentro de un nivel es la p cruda
# --- y eso la tabla lo DICE en vez de dejarlo como una coincidencia.
#
# **El agrupamiento son DOS columnas y nunca una.** `tr_e` cayendo son clases
# que se FUSIONAN y `tr_d` creciendo son nubes que se DESPARRAMAN: dos
# fenomenos opuestos. Medido en el barrido del sopesado, `tr_e` cae 4.6%
# (p<0.001) mientras `tr_d` no se mueve (-0.7%, p=0.052) --- un solo numero
# habria reportado una cosa y la lectura habria sido otra. Ver
# `NOTA_SIN_COCIENTE`, que es lo que la tabla imprime, y `_fila_brazo`, que ya
# se niega a GUARDARLO por la misma razon.
# ===========================================================================

#: El campo del ACIERTO en una fila del registro, escrito una vez y leido por
#: las dos tablas que lo comparan.
CAMPO_ACIERTO = 'acc_te'

#: El agrupamiento que las tablas reportan: SOLO `tr_d`.
#:
#: `tr_e` salio del todo el 2026-10-12, por decision del operador y con una
#: razon de fondo: **CREDA no tiene ningun termino que separe clases.** Tiene
#: un termino que alinea y compacta, asi que `tr_d` --- la dispersion DENTRO
#: de clase --- es su criterio y `tr_e` no. Que la separacion entre clases
#: caiga es una consecuencia, no un incumplimiento, y reportarla al lado
#: invitaba a leerla como si el metodo hubiera prometido algo ahi.
#:
#: Sigue siendo una tupla, no un escalar: `_fila_brazo` registra las dos ---
#: el dato no se pierde --- y la forma de tupla deja volver a abrirla sin
#: tocar a los consumidores.
CAMPOS_AGRUPAMIENTO = ('tr_d',)
assert CAMPOS_AGRUPAMIENTO == ('tr_d',), CAMPOS_AGRUPAMIENTO

#: Como se rotula cada cantidad en una tabla. El nombre del campo se imprime
#: COMO ESTA --- es un contrato de datos con la mitad que escribe el registro
#: --- y la glosa va al lado, para el lector.
GLOSA_CAMPO = {
    'tr_e': 'traza ENTRE clases (sube = clases separadas, baja = se fusionan)',
    'tr_d': 'traza DENTRO de clase (baja = nubes compactas, sube = se desparraman)',
    CAMPO_ACIERTO: 'acierto del blanco sobre test (sube = mejor)',
}

#: En QUE direccion se gana, por cantidad. Es lo que el conteo de victorias
#: tiene que leer, y lo que `_par` contaba mal: `int((dif > 0).sum())` para
#: todo, polaridad unica.
#:
#: Para `tr_e` eso esta bien --- mas dispersion ENTRE clases es mas separacion
#: --- pero para `tr_d` esta AL REVES: mas dispersion DENTRO de clase son nubes
#: mas desparramadas. Medido en la tabla que se publico antes de esto: `rollo`,
#: frontera, rho=0.10, `tr_d` con delta **+0.8287** salia rotulado **"gana
#: 6/10"**, cuando +0.83 de dispersion intraclase es la direccion mala.
#:
#: Un conteo de victorias con el signo invertido es PEOR que no tenerlo, porque
#: se lee con confianza. El mismo defecto que `_par` documenta para el orden de
#: los brazos, una vuelta mas arriba.
MEJOR_SI = {'tr_e': 'sube', 'tr_d': 'baja', CAMPO_ACIERTO: 'sube'}
# `tr_e` sigue declarado aunque las tablas no lo reporten: el registro lo
# guarda y una polaridad declarada no cuesta nada, mientras que una cantidad
# que reaparece sin direccion cuenta las derrotas como victorias.
assert set(CAMPOS_AGRUPAMIENTO) | {CAMPO_ACIERTO} <= set(MEJOR_SI), MEJOR_SI
assert all(v in ('sube', 'baja') for v in MEJOR_SI.values()), MEJOR_SI
#: Y la direccion tiene que estar DICHA en la glosa que el encabezado imprime:
#: un conteo sin su direccion al lado se lee con el signo de quien lo lee.
assert all(MEJOR_SI[c] in GLOSA_CAMPO[c] for c in MEJOR_SI), (
    'una glosa no nombra la direccion en que su cantidad gana')
assert all(c in GLOSA_CAMPO for c in CAMPOS_AGRUPAMIENTO + (CAMPO_ACIERTO,))

#: Por que no hay una sola columna, con la medicion que lo decidio. Se imprime
#: en las dos tablas de agrupamiento: la decision vive donde alguien la leeria,
#: no en un comentario que nadie abre.
NOTA_SIN_COCIENTE = (
    f'`tr_e` y `tr_d` van en **dos columnas separadas** y nunca combinadas en '
    f'un solo número: `tr_e` bajando son clases que se **FUSIONAN** y '
    f'`tr_d` subiendo son nubes que se **DESPARRAMAN**, dos fenómenos '
    f'opuestos. Medido en el barrido del sopesado, `tr_e` cae **4.6%** '
    f'(p<0.001) mientras `tr_d` **no se mueve** (-0.7%, p=0.052) --- un solo '
    f'número habría reportado una cosa y la lectura habría sido '
    f'otra. Las dos leen `y_t`, así que son **DIAGNÓSTICO**: ni '
    f'parada, ni búsqueda, ni término')

#: El rotulo de la fila que agrupa los niveles, y el eje que colapsa. Dicho
#: donde se colapsa: una agregacion que no nombra su eje produce algo con la
#: forma correcta y otro significado. Es la misma convencion que
#: `ROTULO_PROMEDIO` / `EJE_PROMEDIO` de la primera seccion.
ROTULO_AGRUPADO = 'AGRUPADO'
EJE_AGRUPADO = 'el nivel de ruido, DENTRO de cada semilla'

#: Como se marca en la tabla un nivel DECLARADO que el barrido no corrio. Se
#: imprime en la propia fila y no solo en el pie: una fila ausente y una fila
#: medida en cero se leen igual desde el cuerpo de la tabla.
RAYA_NIVEL_AUSENTE = '*(el barrido no corrió este nivel)*'

#: Los DOS pesos que esta seccion enfrenta, y el orden en que se nombran: el
#: sopesado PRIMERO, de modo que `delta > 0` quiera decir que gana sopesar.
#:
#: Se declaran como PESOS y no como nombres de brazo a proposito. Lo que la
#: mitad que escribe el registro promete es el campo `peso` de cada fila; el
#: nombre del brazo es un rotulo. Derivar el brazo del nombre es la misma clase
#: de defecto que adivinar un techo por como se llama, y este archivo ya tiene
#: `brazos_techo` escrito para no hacerlo.
PESOS_SOPESADO = (PESO_DECLARADO, PESO_APAGADO)
assert len(set(PESOS_SOPESADO)) == 2, PESOS_SOPESADO


def _brazos_por_peso(celda, rotulo):
    """El brazo de cada peso de `PESOS_SOPESADO`, LEIDO del `peso` que las
    filas registran --- nunca del nombre del brazo.

    Un brazo rotulado «con sopesado» no esta sopesado si sus filas no lo dicen,
    y uno rotulado de cualquier otra forma lo esta si lo dicen. El campo `peso`
    es lo unico que `_fila_brazo` promete, asi que es lo unico que se lee.

    Cero brazos o dos con el mismo peso son una NEGATIVA y no una eleccion: con
    dos, nada en el registro dice cual de los dos la tabla deberia leer, y
    elegir el primero publicaria una comparacion de un brazo contra si mismo
    bajo el rotulo de otra. Las filas sin campo `peso` --- el piso y AdaBN, que
    no corren CREDA --- no son candidatas, que es por que se exige el campo en
    TODAS las filas del brazo y no en alguna.
    """
    salida = {}
    for peso in PESOS_SOPESADO:
        aca = sorted(b for b, filas in celda['brazos'].items()
                     if filas and all(f.get('peso') == peso for f in filas))
        if len(aca) != 1:
            raise ValueError(
                f'{rotulo}: {len(aca)} brazos registran `peso` == {peso!r} '
                f'({aca}); el registro no dice cual brazo corrio ese peso, y '
                'leer el nombre del brazo en su lugar es adivinar la mitad '
                'que el artefacto ya sabe')
        salida[peso] = aca[0]
    return salida


def _celdas_de_la_escalera(CONTAM, nombre, perturbacion=None):
    """Un par `(rho, celda-o-None)` por cada nivel DECLARADO en `RHOS`.

    **Las filas salen de `RHOS` y no de `CONTAM['rhos']`, y eso es la razon de
    que esta funcion exista.** El cero es el nivel que dice que el arnes de
    perturbacion no movio nada por su cuenta; sin el no hay contra que leer el
    dano. Si las filas salieran de lo que el barrido corrio, una escalera que
    arranca en 0.10 produciria una tabla IDENTICA en forma a una que arranca en
    cero, y nada lo diria. Con `RHOS` el nivel que falta sale como una fila
    marcada, que es un hecho que se puede ver.

    Un nivel que el artefacto trae y `RHOS` no declara es una negativa: el
    barrido mismo lo comprueba al correr, y una tabla que lo imprimiera estaria
    rotulando un escalon que la escalera declarada no tiene.

    `perturbacion=None` NO quiere decir `reubicado`: quiere decir «la que el
    artefacto declare, si declara exactamente una», igual que en
    `_celda_contaminacion` y por la misma razon --- las declaradas predicen lo
    CONTRARIO sobre el par de medias, asi que un defecto elegiria la
    conclusion.
    """
    if perturbacion is None:
        declaradas = perturbaciones_del_artefacto(CONTAM)
        if len(declaradas) != 1:
            raise ValueError(
                f'{nombre}: el artefacto declara {len(declaradas)} '
                f'perturbaciones ({declaradas}); nombrá una con '
                '`perturbacion=`. No se elige por defecto: las declaradas '
                'predicen lo contrario sobre el par de medias')
        perturbacion = declaradas[0]
    corridos = {}
    for k in CONTAM['celdas']:
        if k[0] != nombre:
            continue
        if _perturbacion_de_clave(k) != perturbacion:
            continue
        corridos.setdefault(float(k[-1]), []).append(k)
    repetidos = {r: [str(k) for k in v] for r, v in corridos.items()
                 if len(v) > 1}
    if repetidos:
        raise ValueError(
            f'{nombre} {rotulo_perturbacion(perturbacion)}: el artefacto trae '
            f'mas de una celda al mismo nivel ({repetidos}); dos escenarios '
            'bajo una sola fila dibujan uno bajo el pie del otro')
    declarados = {float(r) for r in RHOS}
    sobran = sorted(r for r in corridos if r not in declarados)
    if sobran:
        raise ValueError(
            f'{nombre} {rotulo_perturbacion(perturbacion)}: el artefacto trae '
            f'los niveles {sobran} y la escalera declarada es '
            f'{sorted(declarados)}; una fila sobre un escalón que la '
            'escalera no declara rotula un nivel que nadie fijó')
    if not corridos:
        raise ValueError(
            f'{nombre} {rotulo_perturbacion(perturbacion)}: el artefacto no '
            f'trae ningún nivel de este toyset; las claves que hay son '
            f'{sorted(str(k) for k in CONTAM["celdas"])[:8]}')
    # Las filas salen de `ESCALERA_CONTAMINACION` --- la escalera que esta
    # seccion DECLARA correr --- y no de `RHOS`, que declara los escalones que
    # existen. Salian de `RHOS`, y con la seccion en un nivel desde el
    # 2026-10-19 eso ponia tres filas diciendo «el barrido no corrio este
    # nivel»: cierto, y ruido en un reporte de un nivel. Lo que esa eleccion
    # protegia --- que un nivel declarado y NO corrido salga como fila ausente
    # en vez de desaparecer --- se conserva entero: sigue siendo la escalera
    # declarada la que manda, solo que la de esta seccion.
    return perturbacion, [
        (float(r), CONTAM['celdas'][corridos[float(r)][0]]
         if float(r) in corridos else None)
        for r in ESCALERA_CONTAMINACION]


# -------------------------------------------- la figura de entradas en escalera


#: El pie del GIF de la escalera. Vivia adentro de
#: `figura_entradas_escalera` y se fue cuando reescribi esa funcion; el gif lo
#: lee y reventaba con `NameError` en una rama que ningun test de columnas
#: ejercitaba. Vive afuera para que no vuelva a irse con una reescritura.
PIE_ESCALERA = (
    'cada fila es un nivel de la escalera de ruido declarada, con su propio '
    'paso de parada: el rótulo de la fila dice en qué paso paró ESA fila, no '
    'la más larga. Las cuatro columnas son las mismas de la sección 1 y en el '
    'mismo orden --- embedding por clase real, embedding por predicción, '
    'curva de fase 1, curva de fase 2 --- y un test compara las dos secuencias '
    'de paneles para que sigan siéndolo.')


def figura_entradas_escalera(CONTAM, ENTR, banderas, destino=None,
                             perturbacion=None, rhos=None, nombres=None):
    """Las nubes de ENTRADA en UNA figura: una fila por nivel, una columna por toyset.

    **Las filas son los niveles de ruido y las columnas los toysets.** La
    figura transpuesta dibuja las mismas nueve nubes y se renderiza igual, asi
    que lo unico que las distingue es que rotulo cae en que panel: cada panel
    nombra en su PROPIO titulo su toyset y su rho, y no en un rotulo de fila
    que un lector tiene que cruzar con la posicion.

    **Se dibuja el BLANCO y no la fuente**, y no es una omision: la
    perturbacion entra solo en el train del blanco, asi que la nube de la
    fuente es la MISMA en las tres filas de una columna. Gastar un tercio de la
    figura en repetirla tres veces no agrega nada que la figura de dos columnas
    --- `figura_entradas`, que sigue ahi --- no muestre mejor, y el panel es
    literalmente el mismo: `_panel_entradas` lo dibuja para las dos.

    **Quienes son los atipicos.** Los PERTURBADOS, nombrados por el npz sin
    ambiguedad: `rut[k]` es la clase predicha en el punto de control donde la
    parada eligio y `yt_tr` la real, de modo que el conjunto es
    `rut[k] != yt_tr`. No se estima nada.

    `rhos` y `nombres` por default salen del ARTEFACTO --- la escalera que
    corrio y los toysets que trae --- y no de una lista de tres escrita aca:
    una rejilla clavada en 3x3 queda falsa en cuanto el barrido corre otro
    nivel, y la que queda falsa en silencio es la que dibuja ocho nubes en
    nueve casillas.
    """
    if nombres is None:
        presentes = {k[0] for k in CONTAM['celdas']}
        declarados = [f['nombre'] for f in FAMILIAS]
        nombres = ([n for n in declarados if n in presentes]
                   + sorted(presentes - set(declarados)))
    nombres = list(nombres)
    rhos = sorted(float(r) for r in (CONTAM['rhos'] if rhos is None else rhos))
    assert nombres, 'el artefacto no trae ningun toyset que dibujar'
    assert rhos, 'no hay ningun nivel que dibujar'

    # La que se DIBUJO, no la que se pidio, igual que en `figura_contaminacion`:
    # el titulo tiene que nombrar lo que hay en el dibujo, y con una sola
    # declarada `None` la resuelve.
    declaradas = perturbaciones_del_artefacto(CONTAM)
    dibujada = perturbacion if perturbacion is not None else (
        declaradas[0] if len(declaradas) == 1 else None)

    # DOS columnas por toyset --- FUENTE y BLANCO --- igual que
    # `figura_entradas` de la seccion 1, y por la misma razon que esa funcion
    # declara: estos toysets SON el corrimiento, asi que una nube del blanco
    # sola muestra una forma y no una transferencia. Sin la fuente al lado no
    # se puede leer la MAGNITUD del desplazamiento, que es justo lo que la
    # escalera de ruido existe para mostrar.
    # El MISMO tamanio por panel que la seccion 1 --- 6.75 x 6.2 --- para que
    # los dos dibujos se vean iguales y no uno estirado.
    # EL MISMO DIBUJO que la figura `familias_duras` de la seccion 1, que es
    # la que el operador pidio: UN panel por casilla con la FUENTE y el BLANCO
    # JUNTOS, `COL_DURA` --- un color por CLASE, el mismo para los dos dominios
    # --- `x` para la fuente y `o` para el blanco, la vista de `VISTA_DURA` y
    # los limites sobre la union de las dos nubes.
    #
    # Lo copie de `figura_entradas` seis veces, que es OTRA figura: dos paneles
    # por toyset, paleta por defecto. Mientras la copia fue de la figura
    # equivocada, igualar el estilo no podia salir bien.
    fig = plt.figure(figsize=(6.5*len(nombres), 6.0*len(rhos)))
    semillas, atipicos = set(), {}
    for r, rho in enumerate(rhos):
        for c, nombre in enumerate(nombres):
            f = _celda_contaminacion(CONTAM, nombre, rho,
                                     perturbacion)['figura']
            if f is None:
                raise ValueError(
                    f'{nombre} {chr(961)}={rho:.2f}: el escenario no tiene '
                    'cuadro de figura en el artefacto; corré el generador')
            yt = np.asarray(f['yt_tr']).ravel()
            sem = int(np.asarray(f['semilla']))
            semillas.add(sem)
            faltan = [cl for cl in ('Xs_tr', 'Xt_tr', 'ys_tr')
                      if f'{nombre}|{sem}|{cl}' not in ENTR]
            if faltan:
                raise ValueError(
                    f'{nombre} semilla {sem}: faltan {faltan} en el npz de '
                    'entradas; corré `guardar_entradas` con esa semilla')
            Xs = np.asarray(ENTR[f'{nombre}|{sem}|Xs_tr'], dtype=float)
            Xt = np.asarray(ENTR[f'{nombre}|{sem}|Xt_tr'], dtype=float)
            ys = np.asarray(ENTR[f'{nombre}|{sem}|ys_tr']).ravel()
            if not (len(Xt) == len(yt)):
                raise ValueError(
                    f'{nombre} {chr(961)}={rho:.2f}: la nube tiene {len(Xt)} '
                    f'puntos y las etiquetas {len(yt)}; emparejarlos por '
                    'posición marcaría los puntos equivocados')
            # Lo marcado son los PERTURBADOS, del registro del escenario. A
            # `rho=0` sale vacio, que es la fila de control.
            idx = np.asarray(f['idx']).ravel().astype(int)
            mal = np.zeros(len(yt), dtype=bool)
            if len(idx):
                mal[idx] = True
            atipicos[(nombre, rho)] = int(mal.sum())
            a = fig.add_subplot(len(rhos), len(nombres),
                                r*len(nombres) + c + 1, projection='3d')
            a.set_box_aspect(None, zoom=1.18)
            primero = (r == 0 and c == 0)
            for cl in range(CLASES):
                Pm = Xs[ys == cl]
                a.scatter(Pm[:, 0], Pm[:, 1], Pm[:, 2], c=COL_DURA[cl],
                          marker='x', s=11, linewidths=.7,
                          label=f'clase {cl}' if primero else None)
                Qm = Xt[yt == cl]
                a.scatter(Qm[:, 0], Qm[:, 1], Qm[:, 2], c=COL_DURA[cl],
                          marker='o', s=7, alpha=1.0, edgecolors='none')
            if mal.any():
                a.scatter(Xt[mal, 0], Xt[mal, 1], Xt[mal, 2], s=55,
                          facecolors='none', edgecolors='k', linewidths=0.8)
            todo = np.vstack([Xs, Xt]); pad = 0.05*(todo.max(0) - todo.min(0))
            a.set_xlim(todo[:, 0].min()-pad[0], todo[:, 0].max()+pad[0])
            a.set_ylim(todo[:, 1].min()-pad[1], todo[:, 1].max()+pad[1])
            a.set_zlim(todo[:, 2].min()-pad[2], todo[:, 2].max()+pad[2])
            a.view_init(*VISTA_DURA[nombre])
            a.set_xticklabels([]); a.set_yticklabels([]); a.set_zticklabels([])
            a.set_title(f'{nombre} {chr(183)} {chr(961)} = {rho:.2f}'
                        + chr(10) + f'{int(mal.sum())} de {len(mal)} '
                        f'perturbados ({100*mal.mean():.0f}%)',
                        fontsize=10.5, linespacing=1.5)
            if primero:
                a.legend(loc='upper left', fontsize=9, markerscale=2.2)

    sem = sorted(semillas)
    fig.suptitle(
        f"el BLANCO en escalera {chr(183)} filas = nivel de ruido, "
        f"columnas = toyset {chr(183)} `x` fuente, `o` blanco {chr(183)} "
        f"perturbación {rotulo_perturbacion(perturbacion)} {chr(183)} "
        f"semilla {sem[0] if len(sem) == 1 else sem} {chr(183)} "
        f"{etiqueta_balance(banderas)}", fontsize=12.5, y=0.995)
    detalle = ' {} '.format(chr(183)).join(
        f'`{n}` {chr(961)}={r:.2f}: {atipicos[(n, r)]}'
        for r in rhos for n in nombres)
    fig.text(0.5, 0.004,
             chr(10).join(textwrap.wrap(
                 'círculo hueco = PERTURBADO, nombrado por los índices que el '
                 'escenario registró al contaminar --- el dato, no una '
                 f're-derivación {chr(183)} perturbados por panel: {detalle}',
                 ANCHO_PIE)),
             ha='center', va='bottom', fontsize=8.5, linespacing=1.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.06,
                        wspace=0.03, hspace=0.14)
    if destino is None:
        return fig
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    ranura = RANURA_SIN_PERTURBACION if perturbacion is None else perturbacion
    ruta = destino/f'escalera_entradas_{ranura}.png'
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    return ruta


def cuadro_escalera_panel(a, f, j, lim, cuadrada):
    """Las CUATRO columnas de UNA fila de la escalera, en los ejes que recibe.

    **Las mismas cuatro que `cuadro_config`, en el mismo orden**, y por eso
    estan aca y no repartidas en el cuerpo del cuadro: el acuerdo del operador
    es que estas columnas no cambian, y una funcion cuyo cuerpo ES la lista de
    columnas se puede comparar contra la otra.
    `tests/test_toy_seccion2.py` compara las dos secuencias de llamadas.

    La fase 1 va ENTERA y no hasta el cuadro: termino antes de que la fase 2
    empezara, asi que no tiene nada que avanzar cuadro a cuadro. Es la misma
    forma que `cuadro_config` ya usa.
    """
    cu = _cu_del_cuadro(f)
    D = {'ys_tr': np.asarray(f['ys_tr']), 'yt_tr': np.asarray(f['yt_tr'])}
    pasos = [int(v) for v in f['pasos']]
    panel_embedding(a[0], np.asarray(f['PS_t'][j], dtype=float), cu, j, lim, D,
                    np.asarray(f['pred_s']), False, cuadrada)
    a[0].set_title(COL_EMB_REAL + chr(10) + a[0].get_title(),
                   fontsize=10, linespacing=1.4)
    panel_embedding(a[1], np.asarray(f['PS_t'][j], dtype=float), cu, j, lim, D,
                    np.asarray(f['pred_s']), True, cuadrada)
    a[1].set_title(COL_EMB_PRED + chr(10) + a[1].get_title(),
                   fontsize=10, linespacing=1.4)
    panel_fase1(a[2], [tuple(x) for x in f['hist_s']],
                'fase 1 ' + chr(183) + ' Ec. 21, eje fijo 0-1', int(f['ep_s']))
    panel_fase2(a[3], [tuple(x) for x in f['hist_t']], pasos[j],
                'fase 2 ' + chr(183) + ' eje fijo 0-1', int(f['mejor_paso']),
                pasos_por_epoca_del_cuadro(pasos))


def figura_cuadro_escalera(nombre, escenarios, figs, j, lim, cuadrada, pos,
                           banderas):
    """UN cuadro de la animacion de la escalera, devuelto como figura para LEERLO.

    Una fila por nivel de ruido, cuatro columnas. Devuelve la figura y no el
    arreglo de pixeles por la misma razon que `figura_cuadro_contaminacion`:
    los rotulos de un arreglo RGB no se pueden comprobar, y `cuadros_comunes`
    evita el `IndexError` mientras nada evitaria un rotulo que dice el paso de
    otra fila.
    """
    assert len(escenarios) == len(figs), (len(escenarios), len(figs))
    assert figs, 'no hay ninguna fila que dibujar'
    fig, ax = plt.subplots(len(figs), 4, squeeze=False,
                           figsize=(23.5, 5.8*len(figs)))
    for r, (esc, f) in enumerate(zip(escenarios, figs)):
        cuadro_escalera_panel(ax[r], f, j, lim, cuadrada)
        # El rotulo de la fila es el de ESTA fila: su escenario, el cuadro en
        # pantalla y SU paso de parada. Medido en `lunas`, la escalera paro en
        # 52, 44, 52 y 102, asi que un rotulo con el numero de otra fila afirma
        # que el criterio disparo donde no disparo y nada revienta. Es el mismo
        # rotulo que el GIF de contaminacion usa.
        ax[r, 0].set_ylabel(rotulo_fila_contaminacion(esc, f, j),
                            fontsize=11.5, labelpad=10, linespacing=1.4)
    pasos = [int(v) for v in figs[0]['pasos']]
    perturbaciones = sorted({p for p, _ in escenarios}, key=str)
    assert len(perturbaciones) == 1, (
        perturbaciones, 'las filas no son de una sola perturbacion, asi que un '
        'solo pie afirmaria la prediccion de una sobre las dos')
    fig.suptitle(
        f"{nombre} {chr(183)} la escalera de ruido {chr(183)} "
        f"{NOMBRE_FILA[pos]} {chr(183)} perturbación "
        f"{rotulo_perturbacion(perturbaciones[0])} {chr(183)} semilla "
        f"{int(figs[0]['semilla'])} {chr(183)} época "
        f"{pasos[j] // pasos_por_epoca_del_cuadro(pasos)} de "
        f"{pasos[-1] // pasos_por_epoca_del_cuadro(pasos)} "
        f"(cuadro {j+1} de {len(pasos)}) {chr(183)} "
        f"azar {1/CLASES:.3f} {chr(183)} {etiqueta_balance(banderas)}",
        fontsize=13, y=0.985)
    # `wrap=False`: el texto ya viene partido, y dejar que matplotlib lo
    # re-envuelva deshace esos saltos y lo deja en una sola linea de mil
    # caracteres pisando los rotulos del eje.
    fig.text(0.5, 0.012,
             chr(10).join(textwrap.wrap(PIE_ESCALERA, ANCHO_PIE)),
             ha='center', va='bottom', fontsize=8.5, linespacing=1.5,
             wrap=False)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.115,
                        hspace=0.3, wspace=0.21)
    return fig


def cuadro_escalera(nombre, escenarios, figs, j, lim, cuadrada, pos, banderas):
    """El cuadro, renderizado a RGB para que `imageio` lo codifique."""
    fig = figura_cuadro_escalera(nombre, escenarios, figs, j, lim, cuadrada,
                                 pos, banderas)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def escalera_del_gif(CONTAM, nombre, perturbacion=None):
    """La perturbacion y los niveles CORRIDOS de un toyset, en orden ascendente.

    A diferencia de `escenario_del_gif`, que elige el escalon MAS ALTO porque
    anima uno solo, esto devuelve TODOS los que corrieron: la escalera es el
    eje de las filas de este GIF. Los niveles que la escalera declara y el
    barrido no corrio no son filas --- un panel vacio no dibuja nada --- y eso
    es lo que separa esta figura de las tablas, donde el nivel ausente SI
    tiene fila porque una raya se puede leer.

    `perturbacion=None` es «la que el artefacto declare, si declara exactamente
    una», nunca un valor por defecto, por la misma razon que en todo el resto
    de esta seccion.
    """
    perturbacion, niveles = _celdas_de_la_escalera(CONTAM, nombre, perturbacion)
    corridos = [r for r, celda in niveles if celda is not None]
    assert corridos, (nombre, perturbacion, 'ningun nivel corrido')
    return perturbacion, corridos


#: Las CUATRO columnas del gif del SOPESADO, declaradas por el operador el
#: 2026-10-12. NO son las de la seccion 1: aquellas comparan clase real contra
#: prediccion de UN brazo, estas comparan los DOS brazos entre si.
#:
#: El orden importa y por eso vive en una constante: con, sin, con, sin.
#: Invertido, un lector compara el brazo equivocado y la figura se lee con
#: confianza. `tests/test_toy_gif_sopesado.py` lo fija.
COLUMNAS_SOPESADO = ('embedding ' + chr(183) + ' CON sopesado',
                     'embedding ' + chr(183) + ' SIN sopesado',
                     'loss de fase 2 ' + chr(183) + ' CON sopesado',
                     'loss de fase 2 ' + chr(183) + ' SIN sopesado')


def _caminos_de_frontera(H, ys, cab, pasos):
    """Los tres caminos y la proyeccion, calculados UNA vez.

    Lo comparten la figura estatica y el GIF: dos copias del mismo calculo
    divergen, y la que divergiera seria la que nadie volvio a mirar --- este
    archivo ya tuvo dos listas de brazos escritas aparte.
    """
    H = np.asarray(H, dtype=float); ys = np.asarray(ys).ravel()
    centros = {c: H[ys == c].mean(0) for c in range(CLASES)}
    arranques = [((0, 1), 0), ((1, 2), 2), (tuple(range(CLASES)), 1)]
    caminos = []
    for clases, dst in arranques:
        ini = puntos_de_frontera(cab, clases, latente=H.shape[1])
        t, w = caminata_al_centro(cab, ini, centros[dst], pasos=pasos)
        Z = ini[None, :] + t[:, None] * (centros[dst] - ini)[None, :]
        caminos.append(dict(clases=clases, dst=dst, t=t, w=w, Z=Z))
    mu = H.mean(0); _, _, Vt = np.linalg.svd(H - mu, full_matrices=False)
    P2 = Vt[:2].T
    return caminos, (lambda Z: (np.atleast_2d(Z) - mu) @ P2)


def figura_cuadro_fronteras(nombre, H, ys, cab, j, pasos=60):
    """UN cuadro: el punto en el paso `j` y la curva dibujada HASTA `j`.

    La curva PARCIAL es lo que hace que la animacion sea una animacion: con la
    curva entera en cada cuadro el gif seria un PNG repetido, y lo que esto
    muestra es que `w` sube MIENTRAS el punto camina --- no que al final valga
    uno.
    """
    caminos, proy = _caminos_de_frontera(H, ys, cab, pasos)
    H = np.asarray(H, dtype=float); ys = np.asarray(ys).ravel()
    j = int(np.clip(j, 0, pasos - 1))
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.2))
    Hp = proy(H)
    for c in range(CLASES):
        m = ys == c
        ax[0].scatter(Hp[m, 0], Hp[m, 1], c=COL_DURA[c], marker='o', s=6,
                      alpha=.30, edgecolors='none', label=f'clase {c}')
    for i, k in enumerate(caminos):
        Zp = proy(k['Z'])
        ax[0].plot(Zp[:, 0], Zp[:, 1], color='0.6', lw=1.0, ls=':', zorder=4)
        ax[0].plot(Zp[:j+1, 0], Zp[:j+1, 1], color='k', lw=1.8, zorder=5)
        ax[0].scatter(Zp[-1, 0], Zp[-1, 1], s=120, marker='*',
                      c=COL_DURA[k['dst']], edgecolors='k', linewidths=.8,
                      zorder=6)
        ax[0].scatter(Zp[j, 0], Zp[j, 1], s=150, facecolors='none',
                      edgecolors='k', linewidths=2.0, zorder=7)
        ax[0].annotate(f"{i+1} {chr(183)} w={k['w'][j]:.3f}",
                       (Zp[j, 0], Zp[j, 1]), fontsize=10, fontweight='bold',
                       xytext=(8, 8), textcoords='offset points', zorder=8)
    ax[0].set_xticklabels([]); ax[0].set_yticklabels([])
    ax[0].legend(loc='best', fontsize=9, markerscale=2.0)
    ax[0].set_title(f'{nombre} {chr(183)} el latente de la FUENTE' + chr(10)
                    + 'el punto camina de la frontera al centro de su clase',
                    fontsize=10.5, linespacing=1.4)
    for i, k in enumerate(caminos):
        rot = ('frontera ' + '|'.join(str(c) for c in k['clases'])
               + f" {chr(8594)} centro {k['dst']}"
               + ('  (TRIPLE)' if len(k['clases']) == CLASES else ''))
        ax[1].plot(k['t'][:j+1], k['w'][:j+1], lw=2.2, color=COL_DURA[k['dst']],
                   ls='-' if len(k['clases']) < CLASES else '--',
                   label=f"{i+1}. {rot} {chr(183)} w = {k['w'][j]:.3f}")
        ax[1].scatter([k['t'][j]], [k['w'][j]], s=50, color=COL_DURA[k['dst']],
                      edgecolors='k', linewidths=.8, zorder=5)
    ax[1].axhline(1.0, color='k', lw=.8, ls=':')
    ax[1].set_ylim(-0.03, 1.05); ax[1].set_xlim(0, 1)
    ax[1].set_xlabel('recorrido de la frontera al centro de clase')
    ax[1].set_ylabel('$w = 1 - \\hat{H}_2(g)/\\ln C$')
    ax[1].grid(alpha=.25); ax[1].legend(loc='lower right', fontsize=8.5)
    ax[1].set_title('el peso MIENTRAS el punto camina' + chr(10)
                    + 'los tres llegan a 1 sin que nadie mire la etiqueta',
                    fontsize=10.5, linespacing=1.4)
    fig.suptitle(
        f'{nombre.upper()} {chr(183)} qué hace el peso de la Ec. 24 '
        # **Ni «paso» ni «época»: es un BARRIDO.** Los `pasos` de esta figura
        # no son actualizaciones del optimizador ni pasadas por los datos ---
        # son los puntos de una caminata CONSTRUIDA sobre la frontera del peso
        # de la Ec. 24, como el propio subtitulo dice dos palabras antes.
        # Rotularlos «época» para igualar al resto seria la misma mentira que
        # «paso», en la otra direccion. **Decidido por el operador el
        # 2026-10-19: «ese sí es un barrido, colocá así porque no es ni paso ni
        # época».**
        f'{chr(183)} construido, no medido {chr(183)} barrido {j+1} de {pasos}',
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def animar_fronteras_w(nombre, H, ys, cab, destino, *, pasos=60,
                       al_paso=print):
    """El GIF de la caminata. No corre nada: es geometria del cabezal."""
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    cuadros = []
    for j in range(int(pasos)):
        fig = figura_cuadro_fronteras(nombre, H, ys, cab, j, pasos=pasos)
        fig.canvas.draw()
        cuadros.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return _escribir_gif(destino/f'fronteras_w_{nombre}.gif', cuadros, al_paso)


def figura_fronteras_w(nombre, H, ys, cab, destino=None, pasos=60):
    """QUE HACE el peso, por construccion: tres puntos que caminan a un centro.

    Todo lo demas que tenemos del sopesado es estadistica de RESULTADOS. Esta
    figura muestra el MECANISMO, y no necesita semillas ni test porque no es
    una medicion con varianza: `w = 1 - H2(g)/ln C` es funcion del posterior y
    nada mas, asi que la curva es una propiedad de la Ec. 24 y del cabezal ya
    entrenado --- exacta.

    **Los tres puntos arrancan en lugares DISTINTOS a proposito.** Medido
    sobre el cabezal real de `lunas`:

        frontera de DOS clases (empate 0.5/0.5)    w = 0.369
        frontera TRIPLE (las tres en 1/3)          w = 0.000003
        centro de clase (one-hot)                  w = 1.000001

    Tres puntos en fronteras DOBLES arrancarian los tres en 0.369 y las tres
    curvas serian la misma. El triple es el unico lugar donde `w` se anula de
    verdad, y es donde la premisa se ve literal: desde ahi el punto puede irse
    a cualquiera de las tres clases y `w` sube igual --- **nadie le pregunto su
    etiqueta**. Eso es lo que la figura demuestra y lo unico que demuestra:
    que `w` mide CONFIANZA y no acierto.

    Columna 1: el latente con las clases y las tres trayectorias.
    Columna 2: `w` a lo largo de cada una, las tres llegando a 1.
    """
    # Recibe el EMBEDDING ya calculado y no el encoder: este modulo no
    # importa torch --- es una invariante con test --- y calcularlo aca lo
    # obligaria a hacerlo. El llamador tiene el encoder y lo evalua.
    H = np.asarray(H, dtype=float)
    ys = np.asarray(ys).ravel()
    # Los centros de clase en el LATENTE, que es donde vive el cabezal.
    centros = {c: H[ys == c].mean(0) for c in range(CLASES)}
    # Los tres arranques: dos fronteras dobles y la triple.
    arranques = [((0, 1), 0), ((1, 2), 2), (tuple(range(CLASES)), 1)]
    caminos = []
    for clases, destino_cl in arranques:
        ini = puntos_de_frontera(cab, clases, latente=H.shape[1])
        t, w = caminata_al_centro(cab, ini, centros[destino_cl], pasos=pasos)
        caminos.append((clases, destino_cl, ini, centros[destino_cl], t, w))

    # La proyeccion: la MISMA que las otras figuras, clavada a la fuente.
    mu = H.mean(0); _, _, Vt = np.linalg.svd(H - mu, full_matrices=False)
    P2 = Vt[:2].T
    proy = lambda Z: (np.atleast_2d(Z) - mu) @ P2

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.2))
    Hp = proy(H)
    for c in range(CLASES):
        m = ys == c
        ax[0].scatter(Hp[m, 0], Hp[m, 1], c=COL_DURA[c], marker='o', s=6,
                      alpha=.35, edgecolors='none', label=f'clase {c}')
    for i, (clases, dst, ini, ctr, t, w) in enumerate(caminos):
        Z = ini[None, :] + t[:, None] * (ctr - ini)[None, :]
        Zp = proy(Z)
        ax[0].plot(Zp[:, 0], Zp[:, 1], color='k', lw=1.6, ls='--', zorder=5)
        ax[0].scatter(Zp[0, 0], Zp[0, 1], s=130, facecolors='none',
                      edgecolors='k', linewidths=1.8, zorder=6)
        ax[0].annotate(f'{i+1}', (Zp[0, 0], Zp[0, 1]), fontsize=11,
                       fontweight='bold', xytext=(6, 6),
                       textcoords='offset points', zorder=7)
        ax[0].scatter(Zp[-1, 0], Zp[-1, 1], s=130, marker='*',
                      c=COL_DURA[dst], edgecolors='k', linewidths=.8, zorder=6)
    ax[0].set_xticklabels([]); ax[0].set_yticklabels([])
    ax[0].legend(loc='best', fontsize=9, markerscale=2.0)
    ax[0].set_title(f'{nombre} {chr(183)} el latente de la FUENTE' + chr(10)
                    + 'círculo = arranque en la frontera, estrella = centro '
                      'de la clase destino', fontsize=10.5, linespacing=1.4)
    for i, (clases, dst, _, _, t, w) in enumerate(caminos):
        rot = ('frontera ' + '|'.join(str(c) for c in clases)
               + f' {chr(8594)} centro {dst}'
               + ('  (TRIPLE)' if len(clases) == CLASES else ''))
        ax[1].plot(t, w, lw=2.0, color=COL_DURA[dst],
                   ls='-' if len(clases) < CLASES else '--',
                   label=f'{i+1}. {rot} {chr(183)} w: {w[0]:.3f} '
                         f'{chr(8594)} {w[-1]:.3f}')
    ax[1].axhline(1.0, color='k', lw=.8, ls=':')
    ax[1].set_ylim(-0.03, 1.05); ax[1].set_xlim(0, 1)
    ax[1].set_xlabel('recorrido de la frontera al centro de clase')
    ax[1].set_ylabel('$w = 1 - \\hat{H}_2(g)/\\ln C$')
    ax[1].grid(alpha=.25)
    ax[1].legend(loc='lower right', fontsize=8.5)
    ax[1].set_title('el peso a lo largo de cada camino' + chr(10)
                    + 'los tres llegan a 1 sin que nadie mire la etiqueta',
                    fontsize=10.5, linespacing=1.4)
    fig.suptitle(
        f'{nombre.upper()} {chr(183)} qué hace el peso de la Ec. 24 '
        f'{chr(183)} construido, no medido: `w` es función del posterior y '
        f'nada más', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if destino is None:
        return fig
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    ruta = destino/f'fronteras_w_{nombre}.png'
    fig.savefig(ruta, dpi=110); plt.close(fig)
    return ruta


def _panel_blanco(a, Pt, yt, rut, rotulo):
    """SOLO el blanco, con la misma gramatica visual que `_panel_entradas`.

    Color por CLASE REAL, circulo hueco para el mal ruteado, sin numeros en
    los ejes. Es el panel de la seccion 1 llevado a la proyeccion 2-D del
    embedding, y existe porque `panel_embedding` --- que es de otra figura ---
    traia dos problemas medidos:

      ESCALA  dibuja la FUENTE y el blanco juntos. La fuente esta CONGELADA
              (`mover_fuente=False`), asi que es identica en los dos brazos y
              ocupa 58.6 de ancho; el blanco, que es lo que se compara,
              difiere 2.10 entre brazos --- el 3.6% del panel. Las dos
              columnas mostraban dato distinto y se veian iguales.
      ESTILO  otra paleta y otra gramatica que la seccion 1: ejes numerados,
              verde/azul/rombos.

    Sin la fuente, la caja la fija el blanco y la diferencia entre brazos pasa
    de ser el 3.6% del panel a ser lo unico que hay.
    """
    # `COL_DURA`: un color por CLASE, el mismo que la figura `familias_duras`
    # de la seccion 1. Y SIN circulos negros --- el operador los retiro el
    # 2026-10-12: lo que esta figura compara son los dos brazos, y una marca
    # que los dos llevan igual no distingue nada y tapa los puntos.
    Pt = np.asarray(Pt, dtype=float)
    yt = np.asarray(yt).ravel()
    rut = np.asarray(rut).ravel()
    for c in range(CLASES):
        m = yt == c
        a.scatter(Pt[m, 0], Pt[m, 1], c=COL_DURA[c], marker='o', s=7,
                  alpha=1.0, edgecolors='none', label=f'clase {c}')
    mal = rut != yt
    a.set_title(rotulo + chr(10) + f'{int(mal.sum())} de {len(mal)} '
                f'mal ruteados ({100*mal.mean():.0f}%)',
                fontsize=10, linespacing=1.4)
    a.set_xticklabels([]); a.set_yticklabels([])


def cuadro_sopesado_panel(a, f_con, f_sin, j, lim, cuadrada):
    """Las cuatro columnas del sopesado en los ejes que recibe.

    Recibe las DOS figuras --- la del brazo con sopesado y la del apagado ---
    porque las columnas 2 y 4 son del segundo. Mientras el registro guardo una
    sola, esta figura no se podia dibujar; `barrer_contaminacion` escribe
    `figuras` por brazo desde el 2026-10-12.

    **Los dos brazos paran en pasos distintos**, asi que sus trayectorias
    tienen distinto numero de cuadros --- medido, 9 contra 5 en lunas con
    rho=0.30. El cuadro `j` se acota al largo de CADA uno por separado: con un
    indice comun, la figura corta revienta con `IndexError` adentro del bucle.
    """
    for col, f in ((0, f_con), (1, f_sin)):
        k = min(j, len(f['Pt']) - 1)
        _panel_blanco(a[col], f['Pt'][k], f['yt_tr'], f['rut'][k],
                      COLUMNAS_SOPESADO[col])
    for col, f in ((2, f_con), (3, f_sin)):
        pasos = [int(v) for v in f['pasos']]
        k = min(j, len(pasos) - 1)
        panel_fase2(a[col], [tuple(x) for x in f['hist_t']], pasos[k],
                    COLUMNAS_SOPESADO[col], int(f['mejor_paso']),
                    pasos_por_epoca_del_cuadro(pasos))


def figura_cuadro_sopesado(nombre, escenarios, pares, j, lim, cuadrada, pos,
                           banderas):
    """UN cuadro del gif del sopesado: una fila por nivel, cuatro columnas."""
    assert len(escenarios) == len(pares), (len(escenarios), len(pares))
    assert pares, 'no hay ninguna fila que dibujar'
    fig, ax = plt.subplots(len(pares), 4, squeeze=False,
                           figsize=(23.5, 5.8*len(pares)))
    for r, (esc, (f_con, f_sin)) in enumerate(zip(escenarios, pares)):
        cuadro_sopesado_panel(ax[r], f_con, f_sin, j, lim, cuadrada)
        ax[r, 0].set_ylabel(rotulo_fila_contaminacion(esc, f_con, j),
                            fontsize=11.5, labelpad=10, linespacing=1.4)
    pasos = [int(v) for v in pares[0][0]['pasos']]
    k = min(j, len(pasos) - 1)
    perturbaciones = sorted({p for p, _ in escenarios}, key=str)
    assert len(perturbaciones) == 1, perturbaciones
    fig.suptitle(
        f"{nombre} {chr(183)} sopesado contra sin sopesar {chr(183)} "
        f"{NOMBRE_FILA[pos]} {chr(183)} perturbación "
        f"{rotulo_perturbacion(perturbaciones[0])} {chr(183)} semilla "
        f"{int(pares[0][0]['semilla'])} {chr(183)} época "
        f"{pasos[k] // pasos_por_epoca_del_cuadro(pasos)} "
        f"{chr(183)} azar {1/CLASES:.3f} {chr(183)} "
        f"{etiqueta_balance(banderas)}", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.05,
                        hspace=0.30, wspace=0.22)
    return fig


def cuadro_sopesado(nombre, escenarios, pares, j, lim, cuadrada, pos, banderas):
    """El mismo cuadro como arreglo de pixeles, para el GIF."""
    fig = figura_cuadro_sopesado(nombre, escenarios, pares, j, lim, cuadrada,
                                 pos, banderas)
    # La misma conversion que `cuadro_contaminacion`, escrita igual: inventé
    # un `_a_pixeles` que no existe y el cuaderno reventó al dibujar.
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def animar_sopesado(nombre, CONTAM, banderas, destino, *, perturbacion=None,
                    al_paso=print):
    """El GIF del SOPESADO de UN toyset: filas = niveles, columnas = los brazos.

    NO corre nada: lee las DOS figuras que el escenario registra por brazo.
    """
    perturbacion, rhos = escalera_del_gif(CONTAM, nombre, perturbacion)
    pares, escenarios, planas = [], [], []
    for rho in rhos:
        celda = _celda_escenario(CONTAM, nombre, perturbacion, rho)
        figs = celda.get('figuras')
        if not figs:
            raise ValueError(
                f'{nombre} {rotulo_perturbacion(perturbacion)} '
                f'rho={rho:.2f}: el escenario no guarda figuras POR BRAZO; '
                'corré el generador --- el gif del sopesado necesita el brazo '
                'apagado y el registro viejo sólo traía el sopesado')
        for b in (BRAZO_CON_SOP, BRAZO_SIN_SOP):
            if b not in figs:
                raise ValueError(
                    f'{nombre} rho={rho:.2f}: falta la figura de {b!r}')
        pares.append((figs[BRAZO_CON_SOP], figs[BRAZO_SIN_SOP]))
        escenarios.append((perturbacion, rho))
        planas.extend((figs[BRAZO_CON_SOP], figs[BRAZO_SIN_SOP]))
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    # El largo sale del MINIMO sobre las dos figuras de todas las filas: los
    # brazos paran en pasos distintos, medido 9 contra 5.
    n = cuadros_comunes([len(f[c]) for f in planas
                         for c in ('pasos', 'Pt', 'PS_t', 'acc_va', 'acc_te')])
    assert n > 1, (n, 'un GIF de un cuadro es un PNG con otro sufijo')
    lim, cuadrada = caja_contaminacion(planas)
    pos = CONTAM['pos']
    if pos not in NOMBRE_FILA:
        raise ValueError(f'el artefacto declara la celda {pos!r} y no hay '
                         f'rótulo; conocidas: {sorted(NOMBRE_FILA)}')
    cuadros = [cuadro_sopesado(nombre, escenarios, pares, j, lim, cuadrada,
                               pos, banderas) for j in range(n)]
    ranura = RANURA_SIN_PERTURBACION if perturbacion is None else perturbacion
    return _escribir_gif(destino/f'sopesado_paso_{nombre}_{ranura}.gif',
                         cuadros, al_paso)


def animar_escalera(nombre, CONTAM, banderas, destino, *, perturbacion=None,
                    al_paso=print):
    """El GIF de la ESCALERA de UN toyset: las filas son los niveles de ruido.

    NO corre nada: lee el artefacto que `guardar_contaminacion` dejo.

    El largo sale de `cuadros_comunes` sobre TODAS las series que un cuadro
    indexa, de TODAS las filas --- los pasos, las dos proyecciones y las dos de
    acierto --- y nunca del conteo de la primera fila. Con la parada temprana
    cada nivel corta en un paso distinto, asi que indexar la fila corta con el
    conteo de la larga revienta con `IndexError` adentro de ella. Paso de
    verdad tres veces en este cuaderno, y `nbconvert` salio 0 igual.
    """
    perturbacion, rhos = escalera_del_gif(CONTAM, nombre, perturbacion)
    figs, escenarios = [], []
    for rho in rhos:
        celda = _celda_escenario(CONTAM, nombre, perturbacion, rho)
        f = celda.get('figura')
        if f is None:
            raise ValueError(
                f'{nombre} {rotulo_perturbacion(perturbacion)} '
                f'rho={rho:.2f}: el escenario no tiene cuadro de figura en el '
                'artefacto; corré el generador')
        figs.append(f)
        escenarios.append((perturbacion, rho))
    destino = Path(destino); destino.mkdir(parents=True, exist_ok=True)
    n = cuadros_comunes([len(f[c]) for f in figs
                         for c in ('pasos', 'Pt', 'PS_t', 'acc_va', 'acc_te')])
    assert n > 1, (
        n, 'un GIF de un cuadro es un PNG con otro sufijo: la animacion existe '
           'para mostrar como avanza la fase 2, y con un cuadro no hay ningun '
           'avance que mostrar')
    # La caja es COMUN a todas las filas y se calcula por el MISMO lector que
    # la figura estatica: dos cajas calculadas aparte divergen, y entonces el
    # PNG y el GIF del mismo escenario no son el mismo dibujo en dos formatos.
    lim, cuadrada = caja_contaminacion(figs)
    # La celda sale del ARTEFACTO y no de una bandera ni de un valor por
    # default: un titulo que afirma una normalizacion que la corrida no uso es
    # exactamente el defecto que `etiqueta_balance` cierra del otro lado.
    pos = CONTAM['pos']
    if pos not in NOMBRE_FILA:
        raise ValueError(
            f'el artefacto declara la celda {pos!r} y no hay rótulo para '
            f'ella; conocidas: {sorted(NOMBRE_FILA)}')
    cuadros = [cuadro_escalera(nombre, escenarios, figs, j, lim, cuadrada, pos,
                               banderas) for j in range(n)]
    ranura = RANURA_SIN_PERTURBACION if perturbacion is None else perturbacion
    return _escribir_gif(destino/f'escalera_paso_{nombre}_{ranura}.gif',
                         cuadros, al_paso)


# ---------------------------------------------- la tabla de acierto por nivel


def tabla_acc_escalera(nombre, CONTAM, banderas, perturbacion=None):
    """El acierto de cada brazo, UNA FILA POR NIVEL de la escalera declarada.

    **Las filas son los niveles de `RHOS`, el cero incluido**, y un nivel que
    el barrido no corrio sale como una fila marcada en vez de desaparecer: sin
    el cero no hay contra que leer el dano, y una tabla que arranca en 0.10 se
    lee igual que una que arranca en cero. Ver `_celdas_de_la_escalera`.

    El padron de columnas sale del ARTEFACTO y no de una lista escrita aca: la
    escalera crecio de dos brazos de CREDA a tres --- `w` apagado, declarado y
    oraculo --- y una lista fija habria dejado el tercero fuera sin que nada
    fallara. Esta tabla es DESCRIPTIVA, de modo que un brazo mas le agrega una
    columna y no un tratamiento.

    Es DESCRIPTIVA y no da ningun verdicto: quien compara el sopesado contra el
    no sopesado es `tabla_sopesado_escalera`, al lado, con el t pareado y
    Wilcoxon por nivel. Dos medias desnudas no declaran un ganador.

    **La MEDIANA va al lado de la media en cada celda.** Una metrica de este
    proyecto leyo media -0.0254, que se ve como una mejora, mientras la mediana
    era +0.0119 y solo 28 de 90 pares mejoraban: era una cola y no una
    tendencia. Esta tabla no corre ningun test, asi que lo que puede dar es las
    dos lecturas de nivel; las victorias las da la tabla de al lado, que es la
    que resta pares.
    """
    perturbacion, niveles = _celdas_de_la_escalera(CONTAM, nombre, perturbacion)
    padron = [str(b) for b in CONTAM['brazos']]
    assert padron, 'el artefacto no declara ningun brazo'
    presentes = [(r, c) for r, c in niveles if c is not None]
    for rho, celda in presentes:
        faltan = [b for b in padron if b not in celda['brazos']]
        if faltan:
            raise ValueError(
                f'{nombre} {rotulo_perturbacion(perturbacion)} rho={rho}: el '
                f'artefacto declara los brazos {padron} y la celda no tiene '
                f'{faltan}')
        for b in padron:
            for r in celda['brazos'][b]:
                if r.get('split') != 'test':
                    raise ValueError(
                        f'{nombre} rho={rho}: el brazo {b!r} registra el '
                        f'split {r.get("split")!r} y esta tabla reporta test')
    n = _semillas([len(celda['brazos'][b]) for _, celda in presentes
                   for b in padron], banderas,
                  f'{nombre} {rotulo_perturbacion(perturbacion)}: la tabla de '
                  'acierto por nivel')
    # La marca de techo se LEE del registro --- `brazos_techo` la deriva de un
    # campo cuyo nombre lleva `PALABRA_TECHO` y cuyo valor es booleano --- y
    # nunca se adivina del nombre del brazo: un oraculo llamado de otra forma
    # quedaria sin marcar y la tabla lo imprimiria como un metodo que gana.
    techos = brazos_techo(CONTAM)
    marcados = [b for b in padron if b in techos]
    enc = (chr(961), 'perturbados') + tuple(
        f'`{b}`' + (f' {MARCA_TECHO_PAR}' if b in techos else '')
        for b in padron)
    fil = []
    for rho, celda in niveles:
        if celda is None:
            fil.append((f'{rho:.2f} {RAYA_NIVEL_AUSENTE}', _NADA)
                       + tuple(_NADA for _ in padron))
            continue
        sucios = sorted({len(v) for v in celda['indices'].values()})
        fil.append(
            (f'{rho:.2f}',
             str(sucios[0]) if len(sucios) == 1 else str(sucios))
            + tuple(
                f'{np.mean([float(r[CAMPO_ACIERTO]) for r in celda["brazos"][b]]):.4f} '
                f'{chr(177)} '
                f'{np.std([float(r[CAMPO_ACIERTO]) for r in celda["brazos"][b]]):.4f} '
                f'{chr(183)} med '
                f'{np.median([float(r[CAMPO_ACIERTO]) for r in celda["brazos"][b]]):.4f}'
                for b in padron))
    ausentes = [r for r, c in niveles if c is None]
    nota_ausentes = (
        f'**{len(ausentes)} de los {len(ESCALERA_CONTAMINACION)} niveles declarados no '
        f'corrieron** ({", ".join(f"{r:.2f}" for r in ausentes)}) y salen como '
        f'raya: una fila ausente y una fila medida en cero se leen igual si la '
        f'ausente desaparece' if ausentes else
        f'los {len(ESCALERA_CONTAMINACION)} niveles declarados corrieron')
    # Cuantos techos marco el registro, DICHO. Un techo sin marcar se lee como
    # un metodo que gana, y un registro que no trae la bandera deja la tabla
    # sin marcas sin que nada falle: el numero medido es lo que hace visible la
    # omision.
    nota_techo = (
        f'{len(marcados)} brazo(s) marcados {MARCA_TECHO} por el registro '
        f'({", ".join(f"`{b}`" for b in marcados)})' if marcados else
        f'**el registro no marca ningún brazo como {PALABRA_TECHO.upper()}**: '
        f'si alguno lee `y_t`, esta tabla lo está imprimiendo como un '
        f'método que compite. La marca se lee de un campo cuyo nombre '
        f'lleva `{PALABRA_TECHO}` como palabra y cuyo valor es booleano; sin '
        'ese campo no hay nada que leer')
    pie = (f'Descriptiva: **no declara ningún ganador** {chr(183)} el '
           f'verdicto del sopesado lo da `tabla_sopesado_escalera`, al lado, '
           f'con el t pareado y Wilcoxon DENTRO de cada nivel {chr(183)} cada '
           f'celda es la MEDIANA, una sola estadística por cantidad: la '
           f'media sola ya mintió una vez en este proyecto (-0.0254 de '
           f'media con mediana +0.0119 y 28 de 90 pares mejorando --- una cola, '
           f'no una tendencia) {chr(183)} todo sobre TEST, que no entra en '
           f'ninguna decisión {chr(183)} la perturbación se inyecta '
           f'en el {SPLIT_DIBUJADO} y el test no se toca, así que el '
           f'acierto es comparable a lo largo de la escalera {chr(183)} '
           f'{nota_ausentes} {chr(183)} {nota_techo}')
    return Markdown(
        f'**{nombre.upper()} {chr(183)} acierto por nivel de ruido {chr(183)} '
        f'{rotulo_perturbacion(perturbacion)} {chr(183)} '
        f'{NOMBRE_FILA[CONTAM["pos"]]} {chr(183)} {n} semillas {chr(183)} '
        f'azar {1/CLASES:.4f} {chr(183)} test, sólo reporte {chr(183)} '
        f'{etiqueta_balance(banderas)}**'
        + chr(10)*2 + _md(enc, fil) + chr(10)*2 + pie)


# ------------------------------- el sopesado contra el no sopesado, por nivel


def comparar_sopesado_escalera(nombre, CONTAM, banderas, campo,
                               perturbacion=None):
    """Un campo del registro, SOPESADO contra NO SOPESADO, agrupado y por nivel.

    Dos brazos, asi que **t pareado** --- con Shapiro-Wilk sobre las diferencias
    pareadas al lado, porque la normalidad es lo que LICENCIA el t en vez de
    algo que se asume --- y **Wilcoxon reportado tambien**, para que se vea si
    los dos coinciden. Nada de Friedman: Friedman es para tres o mas, y eso lo
    dice el metodo declarado de este archivo.

    Los dos brazos salen del `peso` que cada fila REGISTRA y no del nombre del
    brazo; ver `_brazos_por_peso`. Y tienen que ser los MISMOS en todos los
    niveles: si un nivel corrio el peso declarado en un brazo y otro nivel en
    otro, la fila agrupada estaria restando dos cosas distintas.

    **La fila AGRUPADA promedia los niveles DENTRO de cada semilla**, lo que da
    `n` pares, y NUNCA apila los `n * niveles` de nivel-semilla. Esa es toda la
    razon de que se calcule asi: la semilla 0 aparece en todos los niveles, de
    modo que los pares apilados no son observaciones independientes y apilarlos
    es pseudo-replicacion. El promedio puntual sale igual de las dos formas ---
    es el denominador el que cambia --- de modo que apilar no mueve la columna
    del delta y multiplica el estadistico, que es la forma mas silenciosa que
    tiene un numero de quedar mal. Es la misma decision, por la misma razon,
    que `comparar_sin_contra_con` toma sobre las normalizaciones.

    **Holm corrige DENTRO del nivel y nunca a lo largo de la escalera.** Los
    pares de los niveles son una familia que nadie declaro. Con dos brazos hay
    exactamente un par por nivel, asi que `familia_de` es 1 y el ajustado es la
    p cruda: la tabla lo imprime con ese numero al lado en vez de dejarlo como
    una coincidencia que un lector tiene que deducir.

    `delta = sopesado - sin sopesar`, de modo que `delta > 0` quiere decir que
    gana sopesar. El orden del par es fijo y no sigue al ganador.
    """
    perturbacion, niveles = _celdas_de_la_escalera(CONTAM, nombre, perturbacion)
    presentes = [(r, c) for r, c in niveles if c is not None]
    rotulo = f'{nombre} {rotulo_perturbacion(perturbacion)}'
    pares = {}
    for rho, celda in presentes:
        pares[rho] = _brazos_por_peso(celda, f'{rotulo} rho={rho:.2f}')
    distintos = {peso: sorted({p[peso] for p in pares.values()})
                 for peso in PESOS_SOPESADO}
    for peso, brazos in distintos.items():
        if len(brazos) != 1:
            raise ValueError(
                f'{rotulo}: el peso {peso!r} lo corrieron brazos distintos en '
                f'distintos niveles ({brazos}); la fila agrupada estaría '
                'restando dos cosas distintas bajo un solo rótulo')
    brazo_con = distintos[PESO_DECLARADO][0]
    brazo_sin = distintos[PESO_APAGADO][0]
    if brazo_con == brazo_sin:
        raise ValueError(
            f'{rotulo}: un solo brazo ({brazo_con!r}) registra los dos pesos; '
            'una comparación de un brazo contra sí mismo no es la '
            'pregunta de la sección')
    # El campo tiene que estar en TODAS las filas de los dos brazos. Faltando
    # en algunas, promediar las que lo tienen daria una media sobre un
    # subconjunto de semillas bajo el rotulo del total.
    for rho, celda in presentes:
        for b in (brazo_con, brazo_sin):
            if not all(campo in r for r in celda['brazos'][b]):
                raise ValueError(
                    f'{rotulo} rho={rho:.2f}: el brazo {b!r} no registra '
                    f'{campo!r} en todas sus semillas; promediar las que lo '
                    'tienen da una media sobre un subconjunto bajo el '
                    'rótulo del total')
    n = _semillas([len(celda['brazos'][b]) for _, celda in presentes
                   for b in (brazo_con, brazo_sin)], banderas,
                  f'{rotulo}: {campo} sopesado contra no sopesado')

    def _fila(rotulo_fila, con, sin):
        # El CAMPO va: el conteo de victorias se cuenta en la direccion que
        # esa cantidad declara, no con el signo del delta. `tr_d` gana BAJANDO.
        f = _par(brazo_con, brazo_sin, con, sin, n, campo=campo)
        _t_pareado(f, con, sin)
        f['rotulo'] = rotulo_fila
        f['con'] = float(np.mean(con))
        f['sin'] = float(np.mean(sin))
        # Holm DENTRO de esta familia, y la familia es este nivel: UN par. El
        # ajustado sale de `_holm`, el mismo que corrigen las otras tablas, en
        # vez de escribirse como «la p cruda» --- un atajo que seria correcto
        # hoy y falso el dia que el nivel tenga dos pares.
        f['holm'] = _holm([f['wilcoxon']])[0]
        f['familia_de'] = 0 if f['wilcoxon'] is None else 1
        return f

    por_nivel, pila_con, pila_sin = [], [], []
    for rho, celda in niveles:
        if celda is None:
            por_nivel.append(dict(rho=rho, fila=None))
            continue
        con = np.array([float(r[campo]) for r in celda['brazos'][brazo_con]])
        sin = np.array([float(r[campo]) for r in celda['brazos'][brazo_sin]])
        pila_con.append(con); pila_sin.append(sin)
        por_nivel.append(dict(rho=rho, fila=_fila(f'{rho:.2f}', con, sin)))
    # DENTRO de la semilla: un vector de n, no una pila de n*niveles.
    # La fila AGRUPADO se retiro el 2026-10-12 por decision del operador: un
    # promedio sobre los niveles de ruido DENTRO de cada semilla mezcla cuatro
    # regimenes en un numero, y lo que la seccion compara es el efecto POR
    # NIVEL. Se sigue calculando porque el pie la nombra para decir que no esta.
    general = _fila(ROTULO_AGRUPADO, np.mean(pila_con, axis=0),
                    np.mean(pila_sin, axis=0))
    return {
        'familia': nombre,
        'campo': campo,
        'perturbacion': perturbacion,
        'normalizacion': CONTAM['pos'],
        'n': n,
        'brazo_con': brazo_con,
        'brazo_sin': brazo_sin,
        'niveles': por_nivel,
        'general': general,
        'rhos_presentes': [r for r, c in niveles if c is not None],
        'rhos_ausentes': [r for r, c in niveles if c is None],
        'eje_agrupado': EJE_AGRUPADO,
        # Lo que la fila agrupada se NIEGA a ser, con el numero adentro: es la
        # mitad de la afirmacion que un lector no puede deducir de la tabla.
        'pares_apilados_rechazados': n * max(
            len([r for r, c in niveles if c is not None]), 1),
    }


def _md_sopesado(res_por_campo):
    """Las DOS tablas del sopesado: la agrupada y la abierta por nivel.

    Un solo emisor para las dos y para los dos campos, igual que
    `tabla_n_brazos` es uno solo: dos renderizadores del mismo cuadro divergen
    y el signo se lee en el que nadie reviso. La columna de la cantidad existe
    porque la tabla del agrupamiento reporta DOS --- `tr_e` y `tr_d`, sueltas
    --- y la del acierto una; con una sola cantidad la columna dice siempre lo
    mismo, que es exactamente lo que un lector necesita para saber que no falta
    nada.
    """
    primero = res_por_campo[0]
    gl = primero['general']['gl']
    enc = ('nivel', 'cantidad', 'sopesado', 'sin sopesar',
           f'{_DELTA} = sopesado {chr(8722)} sin',
           f'mediana de {_DELTA}', f'IC95 de {_DELTA}', f't({gl})', 'p (t)',
           'Shapiro p', 'p (Wilcoxon)', 'p Holm', 'gana')

    def _linea(nivel, campo, f):
        if f is None:
            return (nivel, f'`{campo}`') + tuple(_NADA for _ in enc[2:])
        return (nivel, f'`{campo}`',
                f'{f["con"]:.4f}', f'{f["sin"]:.4f}',
                f'{f["delta"]:+.4f}', f'{f["mediana"]:+.4f}',
                f'[{f["ic"][0]:+.4f}, {f["ic"][1]:+.4f}]',
                _num(f['t'], '.2f'), _num(f['p'], '.5f'),
                _num(f['shapiro'], '.3f'), _num(f['wilcoxon'], '.4f'),
                _num(f['holm'], '.4f'), f'{f["gana"]}/{f["n"]}')

    # La fila AGRUPADO se retiro el 2026-10-12 por decision del operador:
    # promediar los niveles de ruido DENTRO de cada semilla mezcla cuatro
    # regimenes en un numero, y lo que la seccion compara es el efecto POR
    # NIVEL. `comparar_sopesado_escalera` la sigue calculando --- su `general`
    # es parte del contrato que los tests fijan --- pero no se imprime.
    por_nivel = []
    for nivel in range(len(primero['niveles'])):
        for r in res_por_campo:
            e = r['niveles'][nivel]
            rotulo = (f'{chr(961)} = {e["rho"]:.2f}' if e['fila'] is not None
                      else f'{chr(961)} = {e["rho"]:.2f} {RAYA_NIVEL_AUSENTE}')
            por_nivel.append(_linea(rotulo, r['campo'], e['fila']))
    return (_md(enc, por_nivel),)


def _tablas_sopesado(nombre, CONTAM, banderas, campos, asunto, nota_extra='',
                     perturbacion=None):
    """Las dos tablas del sopesado sobre `campos`, en markdown.

    Una sola composicion para el acierto y para el agrupamiento: lo unico que
    cambia entre las dos tablas publicas son los CAMPOS que compara y la nota
    que explica de que hablan. El test, el emparejamiento, el eje que la fila
    agrupada colapsa y el alcance de Holm son los mismos, y tienen que serlo:
    dos copias del mismo cuadro divergen.
    """
    res = [comparar_sopesado_escalera(nombre, CONTAM, banderas, c,
                                      perturbacion)
           for c in campos]
    uno = res[0]
    (por_nivel,) = _md_sopesado(res)
    niveles = len(uno['rhos_presentes'])
    glosa = ' {} '.format(chr(183)).join(
        f'`{c}`: {GLOSA_CAMPO[c]}' for c in campos)
    nota_ausentes = (
        f'**{len(uno["rhos_ausentes"])} de los {len(ESCALERA_CONTAMINACION)} niveles declarados '
        f'no corrieron** '
        f'({", ".join(f"{r:.2f}" for r in uno["rhos_ausentes"])}) y salen como '
        f'raya: sin el nivel más bajo no hay contra qué leer el '
        f'daño' if uno['rhos_ausentes'] else
        f'los {len(ESCALERA_CONTAMINACION)} niveles declarados corrieron')
    cabeza = (
        f'**{nombre.upper()} {chr(183)} {asunto} {chr(183)} sopesado '
        f'(`{PESO_DECLARADO}`) contra sin sopesar (`{PESO_APAGADO}`) '
        f'{chr(183)} {rotulo_perturbacion(uno["perturbacion"])} {chr(183)} '
        f'{NOMBRE_FILA[uno["normalizacion"]]} {chr(183)} {uno["n"]} semillas '
        f'{chr(183)} {etiqueta_balance(banderas)}**')
    # Los dos brazos, nombrados, y de donde salio cada uno: el nombre del brazo
    # es un rotulo y el `peso` es el contrato. Quien lea la tabla tiene que
    # poder ver que no se adivino por el nombre.
    nota_brazos = (
        f'los dos brazos salen del campo `peso` que cada fila REGISTRA y no de '
        f'su nombre: `{uno["brazo_con"]}` corrió `{PESO_DECLARADO}` y '
        f'`{uno["brazo_sin"]}` corrió `{PESO_APAGADO}`')
    pie = (
        f'`{_DELTA} = sopesado {chr(8722)} sin sopesar`, así que '
        f'**{_DELTA}>0 quiere decir que gana SOPESAR** {chr(183)} el orden del '
        f'par es fijo y no sigue al ganador {chr(183)} **la mediana de '
        f'{_DELTA} y las victorias van al lado de la media**: la media sola ya '
        f'mintió una vez en este proyecto --- -0.0254 de media con mediana '
        f'+0.0119 y 28 de 90 pares mejorando, que es una cola y no una '
        f'tendencia {chr(183)} t pareado, Shapiro-Wilk sobre las diferencias '
        f'pareadas, Wilcoxon al lado {chr(183)} **Holm corrige DENTRO de cada '
        f'nivel y nunca a lo largo de la escalera**: los pares de los niveles '
        f'son una familia que nadie declaró. Con dos brazos hay '
        f'exactamente UN par por nivel, así que la familia es de 1 y el '
        f'ajustado es la p cruda {chr(183)} cada fila son {uno["n"]} pares de '
        f'UN nivel, y **nunca los {uno["pares_apilados_rechazados"]}** de '
        f'nivel-semilla juntos: la semilla 0 está en los {niveles} niveles, '
        f'así que apilarlos es pseudo-replicación --- el delta sale igual y el '
        f'estadístico se multiplica {chr(183)} **no hay fila agrupada**: '
        f'promediar los niveles dentro de cada semilla mezcla {niveles} '
        f'regímenes en un número, y el efecto que esta sección compara es POR '
        f'NIVEL {chr(183)} {nota_brazos} {chr(183)} {glosa} '
        f'{chr(183)} {nota_ausentes}')
    if nota_extra:
        pie += f' {chr(183)} {nota_extra}'
    # UNA sola tabla, abierta por nivel. La agrupada se retiró el
    # 2026-10-12: promediar los niveles de ruido DENTRO de cada semilla mezcla
    # cuatro regímenes en un número, y el efecto que la sección compara es POR
    # NIVEL.
    return Markdown(
        cabeza + chr(10)*2
        + f'Abierto POR NIVEL --- una fila por escalón de la escalera '
          f'declarada, el {chr(961)} = 0 incluido' + chr(10)*2
        + por_nivel + chr(10)*2 + pie)


def tabla_sopesado_escalera(nombre, CONTAM, banderas, perturbacion=None):
    """El sopesado sobre el ACIERTO, abierto por nivel. UNA tabla.

    Agrupada y abierta son la misma comparacion leida en dos resoluciones, no
    dos comparaciones: la de arriba contesta «sopesar sirve» con `n` pares
    independientes, la de abajo contesta «en que nivel» sin pedirle a Holm que
    corrija sobre una familia que nadie declaro.
    """
    return _tablas_sopesado(nombre, CONTAM, banderas, (CAMPO_ACIERTO,),
                            'acierto del blanco sobre test',
                            perturbacion=perturbacion)


def tabla_sopesado_agrupamiento(nombre, CONTAM, banderas, perturbacion=None):
    """Las mismas dos tablas sobre el AGRUPAMIENTO: `tr_e` y `tr_d`, sueltas.

    DOS cantidades y nunca una: cada una tiene su propia fila en las dos
    tablas, con su propio delta, su propia mediana y sus propias victorias. Ver
    `NOTA_SIN_COCIENTE`, que es lo que la tabla imprime.
    """
    # UNA TABLA POR CANTIDAD, no una con las dos apiladas: `tr_e` y `tr_d`
    # ganan en direcciones OPUESTAS, asi que alternarlas fila por fila obliga a
    # leer la columna `cantidad` antes de cada signo. Decision del operador el
    # 2026-10-12.
    return tuple(
        _tablas_sopesado(nombre, CONTAM, banderas, (campo,),
                         f'agrupamiento del embedding del blanco '
                         f'{chr(183)} `{campo}`',
                         nota_extra=NOTA_SIN_COCIENTE,
                         perturbacion=perturbacion)
        for campo in CAMPOS_AGRUPAMIENTO)


# ------------------------------------------------- la tabla del agrupamiento


def tabla_agrupamiento(nombre, CONTAM, banderas, perturbacion=None,
                       brazo=None):
    """`tr_e` y `tr_d` en DOS COLUMNAS SEPARADAS, una fila por nivel de ruido.

    **El cociente no aparece: ni como columna, ni como valor derivado.** `tr_e`
    cayendo son clases que se FUSIONAN y `tr_d` creciendo son nubes que se
    DESPARRAMAN, dos fenomenos opuestos. Medido en el barrido del sopesado,
    `tr_e` cae 4.6% (p<0.001) mientras `tr_d` no se mueve (-0.7%, p=0.052): un
    solo numero habria reportado una cosa y la lectura habria sido otra.
    `_fila_brazo` ya se niega a GUARDARLO por la misma razon --- tenerlo al lado
    invita a leerlo --- y esta tabla no lo reconstruye.

    Las dos leen `y_t`, asi que son DIAGNOSTICO: ni parada, ni busqueda, ni
    termino. Es el mismo estatuto que `acc_te`.

    `brazo=None` se DERIVA del registro con `brazo_de_la_figura`: el brazo
    sopesado es el que declara `sopesado: True` en sus filas, y afirmarlo con
    una constante de este archivo seria escribir por segunda vez algo que el
    artefacto ya sabe. Dos brazos sopesados, o ninguno, es una negativa.
    """
    perturbacion, niveles = _celdas_de_la_escalera(CONTAM, nombre, perturbacion)
    brazo = brazo_de_la_figura(CONTAM) if brazo is None else str(brazo)
    presentes = [(r, c) for r, c in niveles if c is not None]
    for rho, celda in presentes:
        if brazo not in celda['brazos']:
            raise ValueError(
                f'{nombre} {rotulo_perturbacion(perturbacion)} '
                f'rho={rho:.2f}: la celda no tiene el brazo {brazo!r}')
        for campo in CAMPOS_AGRUPAMIENTO:
            if not all(campo in r for r in celda['brazos'][brazo]):
                raise ValueError(
                    f'{nombre} rho={rho:.2f}: el brazo {brazo!r} no registra '
                    f'{campo!r} en todas sus semillas; una media sobre las '
                    'que lo tienen sale bajo el rotulo del total')
    n = _semillas([len(celda['brazos'][brazo]) for _, celda in presentes],
                  banderas,
                  f'{nombre} {rotulo_perturbacion(perturbacion)}: la tabla de '
                  'agrupamiento')
    techos = brazos_techo(CONTAM)
    # UNA estadistica por cantidad, la MEDIANA, y el nombre de la traza en
    # el encabezado en vez de la sigla sola. La media se fue: en este proyecto
    # ya mintio una vez --- leia -0.0254, que se ve como una mejora, mientras
    # la mediana era +0.0119 y solo 28 de 90 pares mejoraban. Era cola.
    enc = (chr(961),) + tuple(
        f'`{campo}` {chr(183)} {GLOSA_CAMPO[campo]} {chr(183)} mediana'
        for campo in CAMPOS_AGRUPAMIENTO)
    fil = []
    for rho, celda in niveles:
        if celda is None:
            fil.append((f'{rho:.2f} {RAYA_NIVEL_AUSENTE}',)
                       + tuple(_NADA for _ in enc[1:]))
            continue
        celdas = []
        for campo in CAMPOS_AGRUPAMIENTO:
            v = np.array([float(r[campo]) for r in celda['brazos'][brazo]])
            celdas.append(f'{np.median(v):.4f}')
        fil.append((f'{rho:.2f}',) + tuple(celdas))
    ausentes = [r for r, c in niveles if c is None]
    nota_ausentes = (
        f'**{len(ausentes)} de los {len(ESCALERA_CONTAMINACION)} niveles declarados no '
        f'corrieron** ({", ".join(f"{r:.2f}" for r in ausentes)}) y salen como '
        f'raya: sin el nivel más bajo no hay contra qué leer el '
        f'daño' if ausentes else
        f'los {len(ESCALERA_CONTAMINACION)} niveles declarados corrieron')
    nota_brazo = (
        f'el brazo se DERIVA del registro: `{brazo}` es el único que '
        f'declara `sopesado: True` en sus filas'
        + (f' {chr(183)} y el registro lo marca {MARCA_TECHO}'
           if brazo in techos else ''))
    pie = (f'Descriptiva: **no declara ningún ganador** {chr(183)} el '
           f'verdicto lo da `tabla_sopesado_agrupamiento`, al lado, con el t '
           f'pareado y Wilcoxon por nivel sobre las dos cantidades por '
           f'separado {chr(183)} {NOTA_SIN_COCIENTE} {chr(183)} la '
           f'**mediana es la única estadística** de cada cantidad: la media '
           f'se retiró porque ya mintió una vez en este proyecto --- leía '
           f'-0.0254, que se ve como una mejora, mientras la mediana era '
           f'+0.0119 y sólo 28 de 90 pares mejoraban {chr(183)} '
           f'{nota_brazo} {chr(183)} {nota_ausentes}')
    return Markdown(
        f'**{nombre.upper()} {chr(183)} agrupamiento del embedding del blanco '
        f'{chr(183)} brazo «{brazo}» {chr(183)} '
        f'{rotulo_perturbacion(perturbacion)} {chr(183)} '
        f'{NOMBRE_FILA[CONTAM["pos"]]} {chr(183)} {n} semillas {chr(183)} '
        f'mediana {chr(183)} '
        f'{etiqueta_balance(banderas)}**'
        + chr(10)*2 + _md(enc, fil) + chr(10)*2 + pie)
