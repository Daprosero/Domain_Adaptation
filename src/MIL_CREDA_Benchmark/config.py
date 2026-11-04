"""Every number the arms and the bag draw are decided with, in one place.

What remains is what `wiring`, `bags`, `harness`, `latent`, `mlp` and `toy_run`
read: the bag material, label noise, the network, the method's
hyperparameters and their declared ranges, the arms, and the paths. The digit
benchmark's transfers, scales, searches, records and report surfaces were
retired with the image datasets. `toy_run` rebinds several of these globals
while a family is bound and restores them on the way out (`BENT_GLOBALS`).
"""

from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------- material

REVISION = "research-concept-r35.md"

#: The domain codes a run has bound, and what each is called. Empty at rest:
#: `toy_run.material_bound` adds the two codes of the family it binds and
#: removes them on the way out, so a code nobody bound has no name here.
DOMAINS: dict[str, str] = {}


CLASSES = 10

#: A bag is a subject: 30 instances of one class, drawn at random inside it. No
#: instance carries a label of its own, and no instance comes from another class.
INSTANCES_PER_BAG = 30

#: Twelve per class, so the draw is stratified and balanced rather than
#: proportional. The local correspondence is undefined for a class with no source
#: bag, so class coverage is a requirement of the formulation and not a
#: convenience.
#:
#: It was ten until the ceiling search needed a role of its own. The two extra
#: bags a class contributes fund that role outright, so nothing was taken from
#: training or from the verdict: 64 / 20 / 36 where it used to be 64 / — / 36.
#: The material allows it — the domains hold far more than they contribute, and
#: USPS is the one that binds at 542 images in its smallest class, which is 18
#: bags of thirty. Twelve sits under that with room, and the other two domains
#: are not close to a limit.
BAGS_PER_CLASS = 12

#: Three roles, drawn identically in both domains, and disjoint.
#:
#: Training fits. Selection is where the ceiling search looks, and it exists
#: because the search chooses by outcome: a coefficient picked on the material the
#: verdict is read from makes the verdict read a decision it already made. The
#: evaluation role is never seen before the verdict.
#:
#: The 36 evaluation bags are what let 30 seeds resolve three points, and they are
#: untouched — the selection role is funded by the two extra bags per class, not
#: taken from anywhere. An 80/20 would have left 20 and resolved five.
#:
#: Twenty selection bags across two search transfers and 30 seeds is 60
#: measurements per ceiling, which separates about three points between grid
#: points. That is enough to pick one scalar out of five and not enough to be
#: read as a result, which is exactly what it is for.
TRAIN_BAGS = 64
VALID_BAGS = 20
EVAL_BAGS = 36

# ---------------------------------------------------------------- label noise

#: The fraction of a training bag's instances replaced by images of another
#: class. The bag's label never changes: bags are pure and no instance carries a
#: label of its own, so there is nothing to flip. What contamination corrupts is
#: the evidence, not the answer.
#:
#: The same replacement is two different perturbations, and that asymmetry is the
#: experiment rather than a wrinkle in it. `wiring.py` broadcasts the bag's label
#: to all `INSTANCES_PER_BAG` instances, so for an instance-unit arm those
#: replaced instances carry a genuinely wrong label; for a bag-unit arm the label
#: stays at the bag and the replacements are witnesses the attention may learn to
#: downweight.
NOISE = 0.0


#: The one donor class each label's bags are contaminated from. An involution
#: on the ten classes with no fixed point -- five mutual pairs -- and DERIVED
#: from confusability in the data rather than chosen from a prior about which
#: digits look alike, and never from a trained model, because a declaration
#: must not depend on a run.
#:
#: Two independent class-centroid cosine proxies decided it: raw pixels
#: exactly as the bags hand them to the encoder (no model needed at all), and
#: the declared backbone's UNTRAINED ImageNet features (closer to what the
#: encoder actually sees, at the cost of inheriting the backbone choice). Each
#: proxy is normalized per domain and averaged over the three domains, and the
#: matching itself is solved exactly rather than approximated -- all 945
#: perfect matchings on ten classes, enumerated.
#:
#: The two proxies disagree on their own optimum, so neither alone is
#: defensible: the backbone proxy's own winner falls to rank 754 of 945 under
#: pixels, and declaring it would have declared the proxy rather than the
#: confusability. The averaged optimum below is the only matching near-optimal
#: under BOTH -- ranks 2 and 19 of 945 -- and it stays rank 2 under pixels
#: alone, which matters because that proxy needs no backbone while three are
#: declared. It carries the two most confusable pairs of the averaged matrix,
#: 3-5 (cosine distance 0.0546) and 7-9.
#:
#:     0-6   1-4   2-8   3-5   7-9
NOISE_PAIRING = [6, 4, 8, 5, 1, 3, 0, 9, 2, 7]

#: Past this the donor class would outnumber the bag's own and the label would
#: stop being defensible. Under the single donor `NOISE_PAIRING` declares for
#: each label, the two counts a composition reader faces are exact:
#: `rate * INSTANCES_PER_BAG` for the donor against
#: `(1 - rate) * INSTANCES_PER_BAG` for the bag's own class, so plurality
#: breaks exactly at one half rather than approximately -- a claim this
#: docstring made too early once already, back when the donor was still spread
#: across the other nine classes and the plurality in fact only broke at rho
#: 0.90. `rate <= NOISE_CAP` admits the tie itself: at 0.5 the two counts are
#: equal and the bag's own class is still half the evidence, not yet the
#: minority. Declared as a cap and enforced, not left to whoever edits
#: `NOISE_LEVELS` next.
NOISE_CAP = 0.5

#: Which roles the noise reaches: `train` and `eval`, in both domains, with ONE
#: shared draw across every arm -- the same instances are replaced for `B`, `E`,
#: `F` and `G` alike, because the draw is a property of the material
#: (`bags.build` is called once per domain/seed and the same `BagSet` is handed
#: to every arm) and never a property of which arm is training.
#:
#: `valid` stays clean, and the reason is not that contaminating it would corrupt
#: a measurement: it is that nothing reads it under noise. The search runs always
#: on clean material and its values are used unchanged when the material is
#: corrupted, so a contaminated `valid` would be drawn and never opened. `eval`
#: IS contaminated, because the sharpest half of this question is whether the
#: method still decides when the evidence it is judged on is corrupted too.
#:
#: One rate for source and target alike. The two are not the same perturbation --
#: the target trains unsupervised, through `pseudolabel` (Eq. 22) and
#: `confidences` (Eq. 24), so contaminating it corrupts the conditional the
#: adaptation term aligns to rather than any label -- but separating the rates
#: would make the sweep two-dimensional and multiply a campaign that already
#: costs `len(ARMS) * len(TRANSFERS) * len(FULL_SEEDS)` runs. Which of the two
#: domains hurts more is a rung of its own, later, on one transfer.
NOISE_ROLES = ("train", "eval")


def noise_instances(rate: float) -> int:
    """How many of a bag's instances a rate replaces.

    Exact at every declared level: the levels are tenths and a half, and
    `INSTANCES_PER_BAG` is thirty, so nothing rounds. Refuses past the cap
    rather than clamping, because a run silently held at 0.5 while its record
    says 0.7 is a table nobody can attribute. The cap itself is admitted --
    `<=`, not `<` -- because at the tie the bag's own class is still half the
    evidence, which is defensible; one step past it the donor would be the
    majority and it would stop being.
    """
    if not 0.0 <= rate <= NOISE_CAP:
        raise ValueError(
            f"noise rate {rate} is outside [0, {NOISE_CAP}]; past the cap the "
            f"donor class would outnumber the bag's own and its label would "
            f"stop being defensible"
        )
    return round(rate * INSTANCES_PER_BAG)


# -------------------------------------------------------------------- network

#: `"mlp"`/`"mlp-nobn"` (`mlp.py`, read by `wiring._resolve_encoder`) build a
#: small plain MLP for the toy domains; any other name reaches
#: `wiring.FeatureExtractor`, which refuses it now that the image backbones are
#: retired (tests replace that seam with a stub encoder). The default below is
#: the value `toy_run` snapshots and restores, never one it builds -- the two differ in EXACTLY their `BatchNorm1d` layers and
#: nothing else, which is the whole instrument this pair of backbones exists
#: to be: if a normalization statistic alone already recovers a translated
#: target, `"mlp"` should and `"mlp-nobn"` should not.
BACKBONE = "resnet18"
PRETRAINED = True          # ImageNet weights, identical on both sides

#: The MLP backbones' hidden width. Deliberately small -- toy domains are meant
#: to be solved by the method's own kernels, never brute-forced by backbone
#: capacity the way an 11M-parameter ImageNet resnet can.
#:
#: Thirty-two, and the number is MEASURED rather than chosen. Swept 8/16/32/64/128
#: over three seeds, reading source-against-source accuracy for BOTH backbone
#: variants, since what makes the pair a fair instrument is that NEITHER variant
#: is left unable to learn its own domain. Criterion: maximise the WORSE of the
#: two, breaking ties on the smaller gap. Measured (worse-of-two, by width):
#:   toroides  0.852 / 0.917 / 0.926 / 0.907 / 0.917
#:   rollo     0.806 / 0.889 / 0.880 / 0.870 / 0.815
#:   lunas     0.806 / 0.815 / 0.824 / 0.796 / 0.787
#: So 8 is too small and 128 too large in all three, while 16, 32 and 64 are
#: indistinguishable inside a dispersion of +-0.013 to +-0.082. One width for all
#: three columns rather than three per-column values: nothing in the measurement
#: justifies differentiating them, and a single value removes a free parameter
#: instead of adding three decisions to defend. Thirty-two is the toroides' own
#: best and the other two accept it at no measurable cost.
#:
#: The gap between the two variants runs -0.065 to +0.074 across all fifteen
#: (dataset, width) cells with no systematic direction, which is the evidence
#: that giving both the identical architecture is fair: the variant without
#: BatchNorm is not handicapped at LEARNING, only at transferring.
MLP_HIDDEN_WIDTH = 32

#: The MLP backbones' `output_dim`, read by every caller that already reads
#: `FeatureExtractor.output_dim` (`Arm.head`, `self.creda`'s instance reshape,
#: and `Arm.encoder.output_dim` elsewhere) -- half the hidden width, for the same
#: reason `MLP_HIDDEN_WIDTH` is small.
MLP_OUTPUT_DIM = 16

#: Cada cuantos pasos del optimizador los parametros de `theta_s` se transmiten
#: a `theta_t`. Procedimiento de ENTRENAMIENTO, no matematica: la Ec. (13) sigue
#: declarando dos codificadores que no comparten ningun tensor, y despues de
#: cada transmision vuelven a divergir. La decision y sus mediciones estan en
#: `AGREED.md`; esto es solo el intervalo.
#:
#: **Declarado fijo en vez de buscado, y la razon es medida** --- pero lo que se
#: midio es el NUMERO DE TRANSMISIONES, no el intervalo, y la diferencia importa:
#: un intervalo cuenta pasos, y cuantos pasos tiene una corrida depende de su
#: escala. A 20 epocas hay 140 pasos y a 3 hay 21, con
#: `TRAIN_BAGS`/`BAGS_PER_STEP` de hoy.
#:
#: A 20 epocas los intervalos 1, 7, 35 y 70 midieron 1.0000 los cuatro (2 a 140
#: transmisiones) contra 0.2500 sin transmitir. A 3 epocas, barrido sobre los
#: cuatro intervalos que ALCANZAN a disparar:
#:
#:     transmisiones    U->M      M->S
#:               21    0.9444    0.5278
#:                7    0.9167    0.4722
#:                3    0.8889    0.5000
#:                1    0.2500    0.0833   <- una sola, y en el ultimo paso
#:                0    0.1111    0.0833   <- control
#:
#: De 3 a 21 transmisiones el resultado es plano y ni siquiera monotono (en M->S
#: tres le gana a siete), asi que el intervalo exacto no es lo que decide. Lo que
#: decide es que haya MAS DE UNA y que ocurran DURANTE el entrenamiento: una sola
#: en el ultimo paso no compra nada en M->S y casi nada en U->M. El umbral esta
#: entre 1 y 3 transmisiones.
#:
#: Siete es un paso por epoca con `TRAIN_BAGS`/`BAGS_PER_STEP` de hoy, elegido
#: por ser el mas legible de la region plana, nunca por ser mejor. **Y es el
#: borde de esa region a escala de piloto**: tres transmisiones. Bajar `EPOCHS`
#: por debajo de 3 dejaria menos, y un intervalo mayor a los pasos totales no
#: transmite NUNCA sin que nada avise.
#:
#: La condicion que el mecanismo necesita es que los dos codificadores tengan la
#: MISMA ARQUITECTURA, no que reciban la misma entrada: medido sobre `resnet18`,
#: las sesenta formas de parametros son identicas con entradas de 32, 64 y 96
#: pixeles, porque la red cierra con `AdaptiveAvgPool2d`. Un `vit_tiny` cambiaria
#: un solo tensor, `pos_embed`, que depende de la cantidad de parches.
SOURCE_TO_TARGET_SYNC_STEPS = 7

#: Hidden width of the relevance selector's R_phi of Eq. (15). It has no
#: counterpart on the other side, so it is declared and never tuned.
ATTENTION_WIDTH = 128

#: gamma of Eq. (15): the weight of the within-bag consensus term against the
#: learned relevance R_phi. The proposal fixes it as a hyperparameter during
#: training but gives it no value. Zero sits at the neutral where Eq. (15)
#: reduces to R_phi with the l1-normalized v_R (r25 Eq. (15)) -- not to a bare,
#: unconstrained v_R^T tanh(...), which is what a prior revision's attention
#: computed and which this one no longer does at any value of gamma. The
#: l1-ball reparametrization is not gated by gamma; only the consensus term
#: is. It stays tunable and is not asserted as the last word on it.
ATTENTION_GAMMA = 0.0

#: tau_att of Eq. (16): the attention temperature. The proposal fixes it as a
#: hyperparameter during training but gives it no value. One sits at the
#: neutral where Eq. (16) reduces to a plain softmax of Eq. (15)'s logit (no
#: temperature scaling at all) -- and, at ATTENTION_GAMMA = 0.0 as well, of
#: R_phi with the l1-normalized v_R alone, per the note above. It stays
#: tunable and is not asserted as the last word on it.
ATTENTION_TEMPERATURE = 1.0

#: The local temperature of Eq. (34). Also without a counterpart in CREDA, so it
#: is fixed at one and reported as fixed rather than chosen.
TAU_LOCAL = 1.0

#: sigma of Eq. (14): the one bandwidth of the one instance kernel. r25
#: Eq. (23) is explicit that a single sigma governs the three blocks of a
#: class's mixed matrix "ya que los tres derivan del mismo kernel de
#: instancia", and r25 Eq. (15) says the attention consensus reuses that
#: same kernel and its bandwidth rather than a second one of its own -- so
#: one constant serves the consensus term inside the attention logit (Eq. 15),
#: the top-k selection ranking, every bag kernel block Eq. (23) builds
#: (K_ss, K_st, K_tt) -- the three blocks of a class's mixed matrix are
#: defined there, not at Eqs.
#: (17)-(18), which define the bag kernel and its representation in general --
#: and the local correspondence's own kernel evaluations (Eq. 34, 37). Every
#: caller passes this explicitly, with no default anywhere in the call
#: chain -- a constant carries no gradient, so there is no question of it
#: being learned, and the value below is not asserted as final.
#:
#: Measured once, by the median heuristic prior work already applies per
#: batch (`CREDALoss._compute_sigma`, and MIL-CREDA's own former per-call
#: rule before this constant replaced it): sqrt(median(||h_a - h_a'||^2)
#: + 1e-6) over the off-diagonal pairwise squared distances of embeddings
#: h = F_theta(x), computed on the pilot configuration's source training
#: material -- domain M (MNIST), seed SEEDS[0] = 0, the 64 TRAIN_BAGS x 30
#: INSTANCES_PER_BAG = 1920 images that role draws -- in the regime training
#: actually uses: the encoder in TRAINING mode (so BatchNorm normalizes with
#: each chunk's own batch statistics rather than accumulated running ones),
#: passed through in BAGS_PER_STEP = 10-bag / 300-image chunks, walked in
#: `train_idx`'s own sequential order -- not the class-stratified shuffle
#: `training_step` actually draws its batches in via `balanced_batches`, which
#: this measurement did not reproduce. An earlier measurement of this same
#: constant ran the encoder in
#: eval mode instead -- accumulated running statistics rather than the batch
#: statistics training actually normalizes with -- and got 8.709090275783575
#: from a median squared distance of 75.848252431748; measured in the regime
#: above, the median squared distance is 1305.7392578125 and the constant
#: below follows from it. This is a placeholder to be tuned with Optuna
#: alongside the ceiling search -- not added to that search by this stretch
#: of work, which was not asked to make that change.
#:
#: `tools/measure_kernel_sigma.py` is this procedure, committed, not only
#: described: seed the global generator with `SEEDS[0]`, build the clean
#: material for domain M, build any arm (the encoder is constructed first,
#: before any arm-specific parameter, so which arm is irrelevant), leave the
#: model in training mode, walk the 64 training bags through
#: `instance_embeddings` in `BAGS_PER_STEP`-bag chunks in `train_idx`'s own
#: order under `torch.no_grad()`, concatenate the 1920 embeddings, and take
#: `sqrt(median(off-diagonal squared distance) + 1e-6)` directly over that set
#: -- every tensor left in the float32 dtype `bags.build` and an ordinary
#: campaign both already use, never upcast to this package's own
#: `MIL_CREDA_Benchmark.DTYPE`, which governs a different module's internal
#: tensors and not what a campaign trains with. Run today it measures
#: 36.135013580322266, 7.245e-07 away from the value below; stable across
#: repeated runs on this machine, so the gap is not run-to-run noise here, but
#: a gap this small is exactly what unordered floating-point summation inside
#: a multi-threaded BLAS matrix multiply can produce between environments
#: without the procedure itself differing. The script prints both numbers
#: rather than asserting they must agree.
KERNEL_SIGMA = 36.135014304860874

#: Whether the instance kernel kappa^I (Eq. 14) evaluates on the L2-normalized
#: instance embedding rather than the raw one `Arm.instance_embeddings` hands
#: it. `KERNEL_SIGMA` above is a constant CHOSEN BEFORE TRAINING, and the
#: kernel's regime is set by the ratio of the embedding's own scale to that
#: constant -- a ratio training is free to move. Measured on toy datasets this
#: session: the trained encoder's median instance-embedding distance ended at
#: 7.03 with BatchNorm and 38.97 without -- same architecture, same data, a
#: 5.5x drift -- and two of six cells ended with a dead or saturated kernel
#: because of it.
#:
#: Normalizing bounds ||h - h'||^2 to [0, 4] by construction, which pins the
#: scale this sigma governs. But a global normalization would also change
#: what the classification head eats (Eq. 19's `z = sum_a beta_a h_a`, read
#: by `self.head`), and the head is the one consumer that wants the scale
#: free -- so this flag governs only the kernel's own view of the embedding;
#: see `Arm._kernel_embedding` for the fork and every routed call site.
#:
#: Run-level, never per-arm: a per-arm flag would let two adjacent rungs of
#: the ablation ladder differ in the component under test AND in the
#: normalization, so no difference between them could be attributed to
#: either one -- the same confound this repository already walked into once
#: with the learning rate.
#:
#: Defaulted to the CURRENT behaviour (off, not normalized) so every existing
#: number stays reproducible and the full suite stays green with no flag set;
#: `test_kernel_normalization.py`'s own bit-identical-with-the-flag-off test
#: is what a later change to this default has to keep passing.
NORMALIZE_KERNEL_EMBEDDING = False

#: The stabilizer inside a logarithm, shared by two call sites rather than
#: private to either: `wiring.py` passes it as eps_src, `source_loss`'s (and
#: `source_bound`'s) own stabilizer for Eq. (21) -- what fixes B_src's scale --
#: and, for an instance-unit arm's `forward`, it also keeps the averaged
#: instance distribution off zero before it is logged. A prior version of this
#: comment said Eq. (21) "normalizes its own", which read as though this
#: constant played no part in it; it does, at the one call site that matters:
#: `wiring.py`'s `source_loss(..., config.EPSILON)`.
EPSILON = 1e-8

#: The stabilizer of the DISPERSION probe's divisor, and nothing else.
#:
#: **It was `eps_loc`, Eq. (38)'s own stabilizer, until 2026-10-18.** r32
#: replaced that equation's confidence-weighted quotient with the uniform mean
#: `(1/N_t) sum_j l_loc,j`, so there is no `sum_j w_j^t` left to keep away from
#: zero and `eps_loc` left the formulation with it. `local_loss` no longer
#: takes an epsilon at all, which is why no call site passes this any more.
#:
#: What still reads it is `wiring._dispersion_normalized_local_term`, where the
#: divisor is a MEASURED dispersion that genuinely can reach zero -- that is
#: the regime the probe exists to visit. Kept as its own name rather than
#: folded into `EPSILON`: that one stabilizes Eq. (21)'s logarithm, this one a
#: probe's denominator, and nothing requires them to move together.
EPSILON_LOCAL = 1e-8

#: Whether the local term's first half (Eq. (38)'s `l_loc,j`) normalizes
#: `d_j^2 = ||Psi(B_j^t) - mu_j^s||^2` by the target representations' own
#: MEASURED DISPERSION instead of the revision's constant 2.
#:
#: r25:1047 proves `<Psi, mu> > 0` (every inner product in H_I is strictly
#: positive and both are convex combinations with non-negative weights),
#: hence `d_j^2 < 2` -- the tight bound `normalized_local_distance` divides
#: by. That bound stops the term from DIVERGING; it does nothing to stop it
#: from DEGENERATING. Minimizing `d_j^2` alone has a degenerate global
#: minimum: collapse every bag representation to one point in H_I and
#: `d_j^2 -> 0` for every bag, a perfect local score carrying zero
#: information. Dividing by a constant does not move where that minimum
#: sits -- only WHAT is in the denominator can.
#:
#: Measured, without BatchNorm, on the target embedding's covariance
#: participation ratio (1 = all variance in one direction, 16 = isotropic):
#:
#:              B      E      F      G
#:     rollo    2.09   2.09   2.09   1.74
#:     toroides 2.55   2.56   2.56   1.07
#:     lunas    1.32   1.32   1.32   1.15
#:
#: `E` and `F` leave it bit-identical to the floor; only `G` -- the arm
#: carrying the local term -- lowers it, and on the rollo it does so while
#: LOSING transfer accuracy against the floor (0.4722 -> 0.2222). The
#: remedy this flag probes: normalize `d_j^2` by the bags' actual spread
#: rather than the constant bound, so collapse shrinks the numerator and
#: the denominator together instead of paying for free.
#:
#: **The denominator carries gradient. It must.** `wiring._target_dispersion`
#: is NOT detached: with `L = d_j^2 / D` and `D` detached, descent on `d_j^2`
#: alone is scaled by `1/D`, which EXPLODES as the cloud collapses (`D -> 0`)
#: -- a detached denominator is a positive feedback loop into the exact
#: collapse this flag exists to resist, not merely an inert rescaling. Live,
#: a second gradient term appears that pushes to INCREASE `D`; that term is
#: the entire remedy. See `wiring._target_dispersion`'s own docstring for the
#: full derivation, and `tests/test_local_dispersion.py`'s own
#: `test_the_dispersion_denominator_carries_gradient_and_resists_collapse`
#: for the measurement.
#:
#: **This is a PROBE on the harness side, not a change to the mathematics.**
#: `MIL_CREDA.local_term.normalized_local_distance` is never touched, stays
#: bound to Eq. (38), and keeps running by default (flag off). And the cost
#: is paid in the open, not hidden behind the probe succeeding: dividing by
#: a MEASURED dispersion instead of the declared constant 2 LOSES Eq. (38)'s
#: `[0, 1]` guarantee -- the ratio is unbounded above whenever the measured
#: dispersion is small, which is exactly the regime a probe of collapse has
#: to visit. `wiring._milcreda_term` restates this at the call site and
#: applies no clamp; if this probe ever earned a remedy, the lost bound is
#: part of what a deliberation would have to decide about, not a detail to
#: paper over here.
#:
#: Run-level, never per-arm, for the same reason `NORMALIZE_KERNEL_EMBEDDING`
#: above is one: a per-arm flag would make two adjacent rungs of the ladder
#: differ in the component under test AND in the normalization, so no
#: difference between them could be attributed to either one.
#:
#: Defaulted to the CURRENT behaviour (off, divide by the constant 2) so
#: every number measured so far stays reproducible and the full suite stays
#: green with no flag set; the bit-identical-with-the-flag-off invariant in
#: `tests/test_local_dispersion.py` is what a later change to this default
#: has to keep passing.
NORMALIZE_LOCAL_BY_DISPERSION = False

# ------------------------------------------------------- schedules, shared by all

#: The adaptation coefficient is one object with two knobs — how fast it grows
#: and how far — and `CREDA.schedules.creda_ramp` is where both live. Both families
#: drive it from there, so the two sides of the comparison share one
#: implementation rather than two copies of the same formula.
#:
#: Nothing multiplies the schedule afterwards. Three factors, two of them pinned
#: at one and therefore invisible, is how a scale error hides: the supervised
#: term of Eq. (21) spent a revision at 18.42 times its stated weight because the
#: coefficient was spread across places nobody read together.

#: How far. One is the neutral of Eq. (39), and from r17 that is a statement about
#: the objective rather than about this setting. Eq. (32) normalizes the global
#: score by the conservative bounds, Eq. (38) bounds the local term, and Eq. (21)
#: is divided by its own supremum B_src — so all three terms live in [0, 1) and a
#: coefficient of one weighs them equally, which is what the normalization is for.
#:
#: It is identical for every arm that has an adaptation term, and it is not chosen
#: by looking at outcomes. Tuning it on one side while the other keeps whatever
#: its author chose would make the two arms differ in two things; held identical,
#: the normalization shows up in the results instead of being compensated for.
#:
#: CREDA's own published ceiling for these domains is `creda_lambda_special`
#: = 1e-4, and running it there was measured to be inert: at 1e-4, 1e-2 and 1e-1
#: every adapted arm scored exactly what its own floor scored. That measurement
#: was taken against the UN-NORMALIZED objective — before Eq. (21) was divided
#: by its own supremum B_src and the three terms of Eq. (39) were brought onto a
#: common scale — so it is historical record and not a reading of the objective
#: this comment describes above. What it established still stands: a comparison
#: against a CREDA whose term changes no decision is a comparison against the
#: source-only floor with extra wall time.
#:
#: What it no longer supports is the closing clause it used to carry, which had
#: the two families sharing this one ceiling. They share none: each searches its
#: own and its derivations inherit it, and
#: `harness.ceiling_for(reduction, family, transfer)` is where any arm's
#: coefficient is read from. This constant is the neutral that search is read
#: against, not the value either family runs at. The report prints the ceiling
#: each family actually found rather than leaving it to whoever knows CREDA
#: to notice.
RAMP_CEILING = 1.0

# --------------------------------------------------------- the ceiling search
#
# Each family looks for its own ceiling and keeps it for its derivations. A
# shared ceiling equalizes the coefficient and unequalizes the balance: the two
# objectives sit a factor of B_src apart, so the same number puts adaptation at
# about 85% of one objective and 10% of the other. Searching per family
# equalizes what actually matters, which is where each method operates.
#
# One search per family, inherited. If every arm found its own, B->E would
# differ in two things and no rung would be attributable. The consequence is
# declared rather than hidden: E and F carry no local term, so the ceiling found
# on the complete method is not necessarily theirs, and that is the price of a
# ladder that can be read.


#: El rango sobre el que se busca, en escala logarítmica. Son los extremos de la
#: rejilla que reemplaza: lo que cambia es que adentro ya no hay cinco puntos
#: sino un continuo.
#:
#: `lambda_glob` de la Ec. (39) -- Decision: leído por `ceiling_for`/
#: `reduction.ceilings`. Ver `CEILING_LOCAL_RANGE` para el segundo coeficiente
#: que esta constante ya no tiene que cubrir sola.
CEILING_RANGE = (1e-4, 1.0)

#: `lambda_loc` de la Ec. (39): el segundo coeficiente que este mismo `objective.
#: total_objective` siempre aceptó por separado (su propia firma y docstring lo
#: dicen) y que hasta esta etapa el harness financiaba con el MISMO número que
#: `lambda_glob` -- una rampa, un techo, pasado dos veces. Ese acoplamiento era el
#: defecto, no una decisión de la Ec. (39): el comentario que solía vivir en esta
#: sección, más abajo, argumentaba lo contrario y quedó falsificado por este mismo
#: cambio; ver la nota nueva donde estaba.
#:
#: **Su propio rango, no un alias del global, y la diferencia es la decisión.**
#: Escrito `= CEILING_RANGE` quedaba acoplado a un número que tiene un paper
#: detrás: el piso de `CEILING_RANGE` es el techo PUBLICADO de CREDA. Mover ese
#: piso --legítimo, porque responde a CREDA-- arrastraba en silencio al
#: coeficiente local, que es exactamente lo que este cambio acaba de declarar
#: independiente. Dos números iguales hoy y una sola decisión mañana.
#:
#: **El tope, 1.0, no se presta del global: es el neutro de la Ec. (39) misma**
#: (`MILCREDA_CEILING`). Peso uno significa "el término entra tal como la
#: ecuación lo escribe"; pasar de ahí afirmaría que el término local necesita más
#: peso que el supervisado, y esa es una decisión que nadie tomó.
#:
#: **El piso, 1e-4, es una ELECCIÓN declarada y no una medición**, y se dice así
#: de frente: CREDA no tiene término local, así que no hay ningún valor publicado
#: del que partir --- ni acá ni en ninguna parte. Lo que el número dice es
#: "suficientemente chico para que el término esté apagado", y las cuatro décadas
#: que abre en escala logarítmica son el espacio donde la búsqueda decide.
#:
#: **Y la búsqueda avisa si la ventana quedó mal puesta**: un ganador pegado a
#: cualquiera de los dos bordes es la señal de que el rango, y no el criterio,
#: eligió. Ese es el control que reemplaza al piso medido que este término no
#: puede tener.
CEILING_LOCAL_RANGE = (1e-4, 1.0)

# ---------------------------------------------- las otras siete dimensiones
#
# The search covers eight dimensions now, not six: the two ramp ceilings above
# (`CEILING_RANGE`/`CEILING_LOCAL_RANGE`), the two ramp growth rates below
# (`RAMP_DELTA_RANGE`/`RAMP_DELTA_LOCAL_RANGE`), and the four that follow them.
# Every one is an Eq. (39)/Eq. (15)/Eq. (16)/Eq. (34) constant this file
# otherwise declares fixed.
#
# `lambda_glob`/`lambda_loc` used NOT to be among them, on the premise that both
# came out of the one shared ramp (`harness.ramp`, then `total_objective(...,
# coefficient, coefficient)`) and a second, independent coefficient for each
# would change Eq. (39) itself rather than search over it. That premise was
# backwards: Eq. (39) already names two coefficients -- `MIL_CREDA.objective.
# total_objective`'s own signature and docstring always took `lambda_global`
# and `lambda_local` apart -- so giving each its own schedule is not a change to
# the equation, it is the harness finally matching an equation it had been
# under-applying. `ATTENTION_WIDTH` stays out and fixed for the reason this
# section always gave: it sizes `R_phi`'s hidden layer, an architectural choice
# with no equation attached, never a hyperparameter Eq. (15) itself names.
#
# Each range is declared beside where it came from, the same discipline
# `CEILING_RANGE`'s own comment already applies -- log scale where the constant
# is inherently a ratio or a bandwidth, linear where it is a bounded mixing
# weight.

#: How fast the ramp climbs. CREDA's own `delta = 20` is the interior point this
#: range is built around, not its edge: the existing note on `RAMP_DELTA` below
#: already measures what happens at the two ends of a wide range -- at three
#: epochs `delta = 20` reaches 0.9975 by the second, i.e. a near-instant step
#: function, and a much smaller delta would still be ramping past the twentieth
#: epoch of a full run. `(1.0, 100.0)`, log scale, spans "barely past the
#: neutral by the end of a full run" to "a step function in the first epoch" --
#: the two qualitative regimes a growth-rate search has to be able to reach.
RAMP_DELTA_RANGE = (1.0, 100.0)

#: How fast the LOCAL term's ramp climbs -- `lambda_loc`'s own growth rate,
#: independent of `lambda_glob`'s above. Reuses `RAMP_DELTA_RANGE`'s bracket
#: outright rather than re-deriving one, and unlike `CEILING_LOCAL_RANGE` this
#: reuse is not a weaker stand-in: growth SPEED is a property of a schedule
#: climbing from zero to whatever ceiling it was given, over the fraction of
#: training elapsed (`CREDA.schedules.creda_ramp`'s own math), and nothing in
#: that shape is specific to which term or which method the coefficient belongs
#: to. The same "barely past neutral" to "step function in the first epoch"
#: argument `RAMP_DELTA_RANGE` already makes applies unchanged.
RAMP_DELTA_LOCAL_RANGE = RAMP_DELTA_RANGE

#: Decision 1's one bandwidth. `KERNEL_SIGMA`'s own docstring already declares it
#: "a placeholder to be tuned with Optuna alongside the ceiling search", measured
#: once by the median heuristic at 36.135014304860874. The range is one order of
#: magnitude either side of that measured value, log scale, because a kernel
#: bandwidth is a ratio quantity (it rescales a squared distance) and an
#: order-of-magnitude sweep is the standard way to bracket a median-heuristic
#: estimate without asserting the heuristic itself is exactly right.
KERNEL_SIGMA_RANGE = (KERNEL_SIGMA / 10.0, KERNEL_SIGMA * 10.0)

#: gamma of Eq. (15): the weight of the within-bag consensus term against the
#: learned relevance R_phi. `ATTENTION_GAMMA`'s own docstring already declares
#: zero the neutral (R_phi alone, l1-normalized); the mixing weight is not
#: declared with an upper bound past one anywhere in the revision, and `(0.0,
#: 1.0)` is the natural closed range for a term that never exceeds giving the
#: consensus term the whole logit.
ATTENTION_GAMMA_RANGE = (0.0, 1.0)

#: tau_att of Eq. (16): the attention temperature. `ATTENTION_TEMPERATURE`'s own
#: docstring declares one the neutral (a plain softmax, no scaling). `(0.1,
#: 10.0)`, log scale, is the ordinary bracket for a softmax temperature: at the
#: low end the distribution is far sharper than the neutral, at the high end far
#: flatter, and the neutral sits at the geometric range's own centre.
ATTENTION_TEMPERATURE_RANGE = (0.1, 10.0)

#: The local temperature of Eq. (34). `TAU_LOCAL`'s own docstring declares one
#: the fixed neutral, with no counterpart in CREDA to measure a range from --
#: it plays the identical role Eq. (16)'s temperature plays, scaling a softmax
#: rather than a kernel bandwidth or a mixing weight, so it is given the same
#: bracket for the same reason.
TAU_LOCAL_RANGE = (0.1, 10.0)


#: How fast. CREDA's own `delta`, the shape of `get_lambda`: zero at the first
#: epoch, approaching the ceiling afterwards. Applied to both sides so the
#: schedule is not one more difference.
#:
#: The curve runs on the fraction of training elapsed, so a short run is not a
#: slower version of a long one — at three epochs it is already at 0.9975 by the
#: second, while twenty epochs take about five to get there. The pilot therefore
#: exercises almost no warm-up, which is worth remembering before reading a pilot
#: as evidence about a schedule.
RAMP_DELTA = 20

#: One fixed, declared learning rate -- CREDA's own `creda_lr_special` -- for
#: every arm, with no decay. `LR_ALPHA`/`LR_BETA` and the decay of `get_eta` are
#: removed: this stretch's own decision is one fixed rate rather than a schedule,
#: and it is not searched -- the ceiling search's six dimensions
#: (`RAMP_CEILING`, `RAMP_DELTA`, `KERNEL_SIGMA`, `ATTENTION_GAMMA`,
#: `ATTENTION_TEMPERATURE`, `TAU_LOCAL`) are all Eq. (39)/Eq. (15)/Eq. (16)/
#: Eq. (34) constants; the optimizer's own rate is not one of them.
LR = 1e-3

#: Held equal across the two units. One bag per class covers every class in every
#: step, which the local correspondence requires; the instance arms take the same
#: number of images so neither side is measured against a different optimizer.
BAGS_PER_STEP = CLASSES                                   # 10

# ----------------------------------------------------------------------- arms
#
# unit        which statistical unit the arm trains on
# adaptation  None, "creda" or "milcreda"
# weighting   confidence weighting of the target blocks
# local       the subject-to-subject correspondence, which CREDA has no analogue of
# attention   how a bag becomes a representation, for bag-unit arms
#
# Normalization is part of the architecture and not a per-arm choice: every
# adapted arm's target forward runs through its OWN target encoder's
# normalization layers, in training mode, exactly as the source forward runs
# through the source encoder's -- each domain trains its own encoder's
# statistics (Eq. 13 splits the encoder; it does not split the treatment of
# normalization within one). There is no separate "frozen"/"live" knob here --
# the only normalization fact any arm declares is the floor's own (Decision
# 2): it never encodes a target image during training at all, so it never
# reaches a normalization layer with one either.

#: selection   which instances of a bag the arm is allowed to look at: None for
#:             all of them, or a rule that keeps `SELECT_K` of the `INSTANCES_PER_BAG`
#:
#: The display name is what every table and figure prints, and the asterisks in it
#: count what the arm lacks: `CREDA*` is CREDA without the confidence weighting,
#: `MIL-CREDA**` lacks the local term and the weighting, `MIL-CREDA*` lacks only
#: the local term, and an unmarked name is the complete method.

#: `normalization` is `"shared"` for every declared arm: the target forward
#: passes through the target encoder's own normalization layers, in training
#: mode, exactly as the source forward passes through the source encoder's --
#: each domain trains its own encoder's statistics, never a frozen or
#: per-domain-switched copy (the `target_embeddings` seam above `wiring.Arm`
#: states this; Eq. (13) is what splits the encoder itself into two).
#:
#: `GN` is retired, and the whole `"sourceBatch"` axis goes with it. It was the
#: one arm that declared it -- its target forward normalized with the CURRENT
#: SOURCE BATCH statistics of that same step, so its normalization layers never
#: learned from the target at all -- and it was the only thing that separated it
#: from `G`. `wiring.Arm._source_embeddings`, `_encode_with_frozen_stats` and
#: `_batchnorm_modules` existed to serve exactly that one arm and are retired
#: with it; nothing else reached any of the three. The `normalization` key stays
#: declared on the four remaining arms, at one value on all of them, and no code
#: reads it today: with the axis gone there is nothing left to branch on, and
#: whether the key itself should go is a decision about the arm declaration's
#: shape rather than part of retiring this arm.
ARMS = [
    {"id": "B", "name": "MIL-Baseline", "label": "source-only (bags)",
     "unit": "bag", "adaptation": None, "weighting": False, "local": False,
     "attention": "learned", "selection": None},
    {"id": "E", "name": "MIL-CREDA**", "label": "MIL-CREDA global, unweighted",
     "unit": "bag", "adaptation": "milcreda", "weighting": False, "local": False,
     "attention": "learned", "selection": None},
    {"id": "F", "name": "MIL-CREDA*", "label": "MIL-CREDA global, weighted",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": False,
     "attention": "learned", "selection": None},
    {"id": "G", "name": "MIL-CREDA", "label": "MIL-CREDA full (global + local)",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "learned", "selection": None},
]

ARMS_BY_ID = {arm["id"]: arm for arm in ARMS}

#: The display order of every table and figure: the order the arms are declared in.
ARM_ORDER = [arm["id"] for arm in ARMS]
NAME_OF = {arm["id"]: arm["name"] for arm in ARMS}

#: Selection arms (`SU`, `SA`, `SK`) and their budget (`SELECT_K`,
#: `SELECTION_SEED`) are removed: every remaining arm keeps every instance of a
#: bag. `wiring.Arm.select` still exists and still returns `H` unchanged for
#: every declared arm's `spec["selection"] is None`.
#:
#: What each rung of the ladder reads. A comparison is only attributable when its
#: two arms differ in one thing, so the pairs are written out rather than left to
#: whoever reads the table.
LADDER = [
    ("B", "E", "qué compra el término global, sin ponderar"),
    ("E", "F", "qué compra la ponderación por confianza en MIL-CREDA"),
    ("F", "G", "qué compra la correspondencia local"),
]


#: Which floor each adapted arm is read against: same unit, same everything, with
#: the adaptation term switched off.
FLOOR_OF = {"G": "B", "F": "B", "E": "B"}

# ---------------------------------------------------------------------- paths

#: This file sits at <repo>/src/MIL_CREDA_Benchmark/config.py, so the repository
#: is two levels up. The product folder is named with a hyphen, which is legal in
#: a directory and not in an identifier — hence the two spellings.
REPOSITORY = Path(__file__).resolve().parents[2]
PRODUCT = REPOSITORY / "MIL-CREDA"
