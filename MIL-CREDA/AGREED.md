# Agreed for the benchmark report — bound to research-concept-r17.md

Every item here was settled in deliberation, not derived from a plan of work. An
item that never reaches the code shows up as an unticked box rather than as a
silence. Nothing here is re-decided while implementing: a collision with any of
these is raised, not resolved in passing.

## Dispersion and scale

- [x] The `±` in every phase-1 table is the dispersion **across seeds**, not the `test_the_plus_minus_of_a_printed_cell_is_the_dispersion_across_seeds`
      batch-wise dispersion the reference paper reports. The seed dispersion is
      what the verdict rule consumes; a batch-wise `±` would stay plausible while
      the run grants no verdicts.
      ceiling is read against. The realized share of the objective is reported
      too, per arm, so a term scaled to irrelevance is visible rather than
      inferred. Renamed from LAMBDA_CONST when the coefficient became one
      object with two knobs; superseded as *the* value by the per-family search,
      and the Reversed section below says how.
- [x] Accuracy granularity is **1/36 = 2.78 points** with 36 evaluation bags, so `test_the_printed_precision_is_the_granularity_of_the_instrument`
      tables print one decimal and say so. Two decimals would be false precision.

## Nomenclature — asterisks count what is absent

| id | display name | what it lacks |
| --- | --- | --- |
| `A` | `Baseline` | everything |
| `C` | `CREDA*` | the confidence weighting |
| `D` | `CREDA` | nothing |
| `B` | `MIL-Baseline` | everything |
| `E` | `MIL-CREDA**` | the local term and the weighting |
| `F` | `MIL-CREDA*` | the local term |
| `G` | `MIL-CREDA` | nothing |
| `SU` | `MIL-CREDA-U` | 20 of its 30 instances, regular selection |
| `SA` | `MIL-CREDA-A` | 20 of its 30 instances, arbitrary fixed selection |
| `SK` | `MIL-CREDA-K` | 20 of its 30 instances, top-K by learned attention |

      attention over the selected ones, so the trio differs in one thing: the
      selection rule.
      generator would shift every later draw and the rung would credit the
      selection with what the offset did.

## Ladder

- [ ] The attention is compared on the full method alone, against ABMIL as published with its output vector unnormalized, ABMIL gated, max pooling and mean pooling, in source and target, clean and contaminated.
- [ ] Arm GN is the full method with its normalization layers not learning from target, normalized with the source batch statistics of the same step; every other adapted arm keeps normalization as part of the model.
- [ ] Every adapted arm passes source and unlabelled target through the same extractor, because the adaptation needs both domains; its normalization layers are part of the model and are neither frozen nor handled per domain. `test_an_adapted_arms_target_forward_updates_running_statistics`
- [ ] The floor B trains on source only with the normalized supervised loss and predicts on target: no target image passes through its encoder during training, and it consumes the random generator identically to the adapted arms. `test_a_floor_never_encodes_a_target_image_during_training`
      The first two hold the budget at 10 instances; the third reads what
      dropping from 30 to the top 10 costs.

## Tables (phase 1) — Table 6 of the reference, our names

- [ ] Each accuracy table carries a rank column per block, independent between the clean and the contaminated block, and arms whose difference does not exceed their combined standard error share a place rather than being ordered by noise.
- [x] Rows are arms by display name, columns are the six transfers, plus `Avg`. `test_the_table_is_arms_by_display_name_over_the_six_transfers_and_an_average`
- [x] **Two tables**: target accuracy as the headline, source accuracy as the `test_the_report_shows_the_target_table_and_its_source_complement`
      complement. A method that wins on target by wrecking source is the
      degenerate case, and one table cannot distinguish it from a success.
      peak is a number rather than a picture.
      rungs say which piece did the work.
- [x] Below the declared repetition floor, no verdicts are granted, the reason is `test_below_the_repetition_floor_the_reason_is_stamped_and_the_table_still_prints`
      stamped in the header, and the table is printed anyway.
      accuracy. It replaces the per-run listing, which repeated in prose what the
      tables say in a grid.
      transversal reading last.
- [x] **One table per cell**, each preceded by three short lines — what is `test_a_cell_shows_one_table_and_not_two`
      measured, why, and which direction is better — and followed by the most
      relevant conclusion.
- [x] That conclusion is **computed from the table**, never written by hand: a `test_the_conclusion_is_not_tied_to_nothing`
      hand-written conclusion is a second source of truth and goes stale in
      silence.
      not a name.
- [x] Notebook prose and table headings in Spanish; identifiers, JSON keys and `test_the_headings_are_spanish_and_the_keys_of_the_record_are_english`
      method names stay English, because they are a data contract rather than
      prose.
- [x] Progress during the run prints one line per transfer, not one per run. It is `test_progress_prints_one_line_per_cell_and_names_that_cells_slowest_arm`
      the only sign of life in a run measured in hours, so it is kept — but it is
      progress, not a report.

## Figures — phase 2

- [ ] For every target bag of the evaluation role, the five source training bags it is closest to, in order, ranked by the bag kernel over all source bags and marking which carry the true class; and the reverse table, how many target bags took each source bag into account, which is what names the source bags nobody uses.
- [ ] Two quantitative tables precede the grids: the distance ratio that reads whether classes group across domains, and domain separability read against chance, each in its clean and contaminated block.
- [ ] Bag figure highlights **the same bags in every panel**: the median bag of each class by correspondence mass, one colour each, and every other bag in its own class colour.
- [x] **Three transfers in every figure, not six.** Six rows at a legible panel size do not fit on a page and the tables already carry all six. Fixed in `FIGURE_TRANSFER_COUNT`; which three is the bullet below. `test_no_figure_draws_more_transfers_than_the_count_fixes`
- [x] **One display seed for every comparative grid**, chosen as the seed whose `test_the_display_seed_is_the_median_of_the_across_arm_mean`
      across-arm mean target accuracy is the median. Panels drawn from each arm's
      own median seed would differ in the method *and* in the draw, and — because
      the seed fixes the partition — would not even share their bags.
- [x] Artefacts are **median, never best**. The best of N grows with the arm's `test_median_seeds_is_the_selection_rule_on_its_own`
      dispersion, so best-vs-best flatters the noisiest arm by the most.
      size do not fit on a page and the tables already carry all six. Fixed in
      `FIGURE_TRANSFER_COUNT` and `FIGURE_TRANSFER_RULE` by a rule that looks at
      no outcome: each domain appears once as source and once as target.
- [x] **Which three is computed, not fixed**: the transfers where the methods `test_the_figure_transfer_rule_ranks_by_the_outcome_it_declares`
      reach the highest mean target accuracy. It is a choice made by the outcome,
      so it is declared in every caption. What makes it defensible here is that
      the latent space of a transfer where everything sits near chance is a
      picture of a model that did not learn. It never touches *which draw* is
      shown — that stays the display seed.
- [x] Latent grid is the **shared original space** and then one column per method: `test_the_latent_grid_is_the_shared_original_space_and_then_one_column_per_method`
      both floors, both CREDA, all three MIL-CREDA. `Original` is the images
      themselves before any model — shared because preprocessing already brings
      both domains to one tensor shape — sampled representatively and stratified
      by class.
      Drawn at the instance level on the pilot, `Baseline` and `MIL-Baseline`
      differ by up to 0.38 in distance ratio and 0.07 in domain separability.
      They train the same encoder through different objectives — per instance
      against per bag through the attention pooling — so their instance
      embeddings had no reason to agree. `latent.floors_agree` re-runs that check
      every campaign, and only it may retire one of the columns.
- [x] **Every panel of the grid is drawn at the instance level**, bag-unit arms `test_every_panel_of_the_grid_is_drawn_at_the_instance_level`
      included. Every arm encodes instances, so it is a space they all have and
      the only one where every panel carries the same number of points. One point
      per subject beside one point per instance made the instance-unit columns
      look like they covered the space and the bag-unit ones look sparse — the
      statistical unit drawn, not the alignment. The bag-level view stays in the
      phase-two tables, which measure each arm in its own unit.
- [x] Colour is the **class**; the marker is the domain — source circles, target `test_colour_is_the_class_and_the_marker_is_the_domain`
      triangles. Target markers larger with a dark edge, source smaller and
      semi-transparent, categorical 10-class palette.
- [x] Bag figure is **3 columns × 3 rows in one figure**, not one file per `test_the_bag_figure_is_one_figure_of_three_columns_by_three_rows`
      transfer: `MIL-Baseline` + `MIL-CREDA*` + `MIL-CREDA`, chosen by the
      mechanism rather than by the ranking — that is the rung the local
      correspondence lives on, so the figure can fail.
      each class by correspondence mass, one colour each, everything else grey.
- [x] The nearest source bag is found with the **bag kernel in the representation `test_the_nearest_source_bag_is_found_with_the_bag_kernel_in_the_representation_space`
      space**, never Euclidean and never in the 2-D projection, and the pair is
      joined by a line so the correspondence survives UMAP's distortion.
- [x] The measured correspondence hit rate is printed with the figure, so it `test_the_measured_correspondence_hit_rate_is_printed_with_the_figure`
      carries a number and not an impression.
- [x] **Every figure carries the same three lines as a table**: what is looked at, `test_every_figure_carries_the_same_three_lines_a_table_does`
      what is being sought, and the conclusion — and the conclusion is **computed
      from the results**. A sentence written by hand under a figure fixes itself:
      the figure is regenerated from other data and the sentence stays.
- [x] The bag figure keeps the projection rather than becoming a bipartite `test_the_bag_figure_keeps_the_projection_and_never_becomes_a_bipartite_diagram`
      diagram. Three columns instead of six separate files already makes the
      panels large enough to read; if it is still a knot, the bipartite is next.

## Removed on purpose

- [ ] Wall time and peak memory stop being measured, recorded and reported: no dimension, no column, no field, no figure. They answered a question this experiment does not ask.
- [x] The adaptation-term scale table (`min`/`max`/`width`). What it checked — `test_local_loss_in_unit_interval`
      that MIL-CREDA's terms stay in `[0, 1]` while the prior work's is bounded
      only by `ln n` — is guaranteed by the construction in Section 5 and already
      covered by the invariant suite. A table restating a proved bound is a third
      place for it to go stale.

## Figures — phase 1

- [ ] Loss curves for every declared arm, as the median across seeds with an interquartile band. `test_a_loss_curve_is_the_median_across_seeds_with_an_interquartile_band`
      median curve with an interquartile band across seeds.
      share ranging from 0.03 to 0.99 across arms at a fixed coefficient, which
      is the thing the curves exist to make visible.

## Checkpoints

- [x] Three per arm per cell, for every arm, so the top 3 can be selected after `test_three_checkpoints_are_kept_per_arm_per_cell_for_every_arm`
      the run instead of being guessed before it. Local, gitignored.

<!-- What follows was settled in the session of 16 August, against r17. It lived
in a separate `AGREEMENTS.md`, created by not having looked whether this file
already existed: they are two halves of one contract and they live in one. -->

## The objective and its coefficient

- [ ] `RAMP_CEILING` stays at 1.0 as the neutral the searched ceiling is read against: if the search lands there, the normalization argument is confirmed by measurement rather than by reasoning.
- [ ] The learning rate is one fixed declared value, without the inherited decay and outside the search, identical for every arm; the attention width stays fixed and declared, and the stabilizing epsilons are not tuned.
- [ ] One kernel bandwidth sigma is shared by the attention consensus and every kernel block, as r21 states; it is a constant hyperparameter (config.KERNEL_SIGMA, a measured placeholder) to be tuned with Optuna, never recomputed from data during training. `test_every_sigma_consumer_receives_the_one_declared_constant`
- [x] The supervised term of Eq. (18) is divided by its own supremum `B_src`, so the three terms of Eq. (39) live in [0, 1) and are read on a common scale. `test_source_loss_in_unit_interval`
- [x] The bag-unit arms call `source_loss` and `total_objective`. No supervised term written inline in the benchmark. `test_the_bag_unit_arms_assemble_the_objective_and_never_write_a_term_inline`
- [x] The 1.0 stops being the value and becomes **the neutral the searched value is read against**. If MIL-CREDA's ceiling lands there, the normalization argument is confirmed by measurement rather than by reasoning. `test_el_registro_escribe_el_neutro_al_lado_del_techo_que_encontro`
- [x] `RAMP_CEILING`'s comment states that argument, not r16's scale one. The old 1e-4 measurement stays as historical record, marked as taken against the un-normalized objective. `test_el_comentario_del_neutro_fecha_el_1e_4_y_no_promete_un_techo_comun`

## Prior work

- [x] `get_lambda` is untouched. DANN, ADDA and CDAN+E still read it at full strength: lowering their coefficient would switch CDAN+E's domain adversary off rather than attenuate it. `test_the_untouched_loops_still_see_the_schedule_they_saw_before`
- [x] The cost of moving the coefficient is measured and bounded, not denied: the product's reassociation stays within 2 ULP and the gradients come out bit-identical. `test_and_the_gradient_comes_out_bit_identical`

## The two ramps

- [x] Each method names its own ramp and carries its own default: `creda_ramp` at 1e-4, `milcreda_ramp` at 1.0. The defaults serve each method's own runs; the benchmark never uses them, it always passes explicit values. `test_each_family_keeps_the_default_its_own_method_was_defined_with`
- [x] A floor with no adaptation term gets a coefficient of zero, not the curve. It has no coefficient, and handing it one would suggest a term it does not carry. `test_a_floor_gets_no_coefficient_rather_than_one_it_does_not_carry`
- [x] The experiment hands both the same `delta` and the same `ceiling`, explicitly. `test_run_one_resolves_the_ceiling_of_the_transfer_it_was_given`
- [x] The curve is written once. `milcreda_ramp` binds to `creda_ramp` rather than copying it, and a test pins them to the same numbers. `test_the_two_families_get_the_same_curve_for_the_same_arguments`

## The record and the report

- [ ] The full method runs under per-transfer values, so its row averaged over transfers mixes configurations. Within a transfer every arm still shares them, which is what keeps each rung attributable; the report says which of the two readings it is giving. `test_the_per_transfer_conclusion_can_come_out_different`
- [ ] The record keeps the supervised term's magnitude and the ratio between terms. Without a denominator, "the term commanded nothing" and "the term was scaled to nothing" print alike. `test_the_distribution_declares_exactly_what_was_approved`
- [ ] The benchmark declares revision r21 and which sections each arm exercises. `test_the_benchmark_is_bound_to_the_same_revision_as_the_configuration`
- [ ] One results notebook carries every result, in this order: the noise sweep, source accuracy, target accuracy, the attention mechanisms, the representation, and the loss curves. The report and latent notebooks are gone: two documents over one campaign drift apart, and the reader had to hold both.
- [x] The benchmark declares `components` in its report contract. `test_the_benchmark_declares_the_components_its_objective_is_made_of`
- [x] `identicalAcrossShards` names the ceilings. They are the parameter the search just changed, so two shards straddling the search would merge into one table with adaptation inert on one half and not on the other, and nothing would refuse. `test_the_ceilings_are_what_has_to_agree_across_shards`


## The ceiling search

- [x] Three material roles, disjoint: 64 training / 20 selection / 36 evaluation. The search reads selection and the verdict reads evaluation, and `run_one` measures one **or** the other — a role the search cannot see is a stronger guarantee than one it agrees not to use. `test_the_three_roles_partition_the_bags_exactly`
- [x] The selection role is funded with **new material**, not taken from training: 12 bags per class instead of 10. Evaluation keeps its 36 and the three-point resolution the campaign was sized for. USPS is what binds, at 18 bags per class. `test_the_selection_role_is_funded_by_new_material_and_takes_nothing`
- [x] The search runs at 20 epochs and 3 seeds, and **never at pilot scale**. The ramp climbs on the fraction of training elapsed, so at three epochs it is saturated by the second and a ceiling found there describes a landscape the campaign never trains in. `test_the_required_search_scale_is_declared_apart_from_the_running_one`
- [x] Its required scale is declared apart from the one it runs at, and `atRequiredScale` goes into the record. With only one of them, a cheaply-found ceiling and the configuration agree with each other. `test_la_procedencia_dice_cual_y_a_que_escala`
- [x] The campaign **refuses** without ceilings, and with ceilings searched below scale. Funding your own coefficient out of the run you are about to report is choosing and judging in one pass. `test_la_campana_se_niega_sin_techos`
- [x] Ceilings are compared **paired** by cell (seed, transfer): every ceiling is measured on the same material, so the cell's difficulty cancels instead of drowning the effect. `test_the_pairing_survives_a_cell_that_is_simply_harder`
- [x] Whether each seed would have chosen the same on its own is recorded. Three seeds on three different ceilings and three on the same one produce the same winner and are not the same evidence. `test_el_registro_dice_si_cada_semilla_habria_elegido_lo_mismo`
- [x] The tie rule is written down: the **smallest** ceiling among the tied wins. Below some point a term is inert and everything ties, so there the tie-break is what actually chooses. `test_a_tie_goes_to_the_smallest_ceiling`
- [x] The grid runs between the two declared defaults, 1e-4 and 1.0, so nothing outside what was already defensible can come out. `test_the_search_grid_runs_between_the_two_declared_defaults`
- [x] The verdict is read over **all six** transfers. Withholding the two that funded the search bought nothing with the roles already disjoint by bag, and cost a third of the units the paired reading rests on. `test_the_campaign_runs_every_one_of_the_six_transfers_and_withholds_none`

## The full run

- [ ] Every artefact measured before this structure is deleted rather than kept: the pilot checkpoints, every pilot result and the unstamped ceiling record described another attention, another bandwidth and a floor that saw the target.
- [ ] Nothing runs (pilot, shard, notebook or search) until the record-provenance guard is closed for every record the decided experiments keep: stale ceilings reaching a run through import-time defaults, Reduction.from_record ignoring the revision, and the noise diagnostic re-stamping an unchecked sweep.
- [ ] The full grid — 30 seeds, 20 epochs — is not launched without an explicit authorization. Neither a clean verification nor a green pilot is permission.
- [ ] While the run stands at pilot scale, its numbers are not quoted as results: not in the report, not in the summary, not in conversation.


## The trials search

Agreed 2026-08-26/27, while replacing the grid engine. Every item here is
carried by code and by a test that dies when the code is mutated, except the one
marked open, which is open because the run has not happened.

- [ ] Only the full method is searched, `G`. The searched values are per transfer and never per arm, so a per-arm ceiling is not expressible. If it were, the term and the coefficient could not be told apart. `test_solo_busca_sobre_los_metodos_completos`
- [ ] The record carries one searched configuration per transfer and no pooled entry, so a run reading it cannot silently apply one transfer's values to another. `test_el_registro_sale_con_la_forma_que_los_lectores_esperan`
- [ ] Every transfer runs under the values searched on that transfer, by the paired rule and its tie-break. Nothing is inherited across transfers and no pooled fallback is reachable. `test_a_measured_transfer_keeps_its_own_pick_over_the_pooled_one`
- [ ] One search, on the full method, and every other arm inherits its values. If each arm searched its own, two arms of a rung would differ in two things and no rung would be attributable. Declared consequence: the arms carrying no local term inherit values found on a method that does.
- [ ] Six dimensions are searched on the full method, per transfer: the ramp ceiling, the ramp growth, the kernel bandwidth, the attention gamma, the attention temperature and the local tau. The two balance coefficients are not among them: they come from the ramp's own ceiling and growth.
- [ ] The search runs always on clean material and its values are used unchanged under noise, so whatever mitigates the contamination is the method and not a tuning fitted to it.
- [x] The search record writes its own wall time beside its declared scale. Without it nothing can project what the full search costs, and a gate is asked to authorize a run whose price nobody measured. `test_el_registro_se_lleva_cuanto_costo_la_busqueda_que_lo_escribio`
- [ ] The trials search has not run at full scale, and this declaration holds no ceiling record at all: the grid's was retired with the campaign it governed. No campaign is launched from this declaration until the search runs.
- [x] The search measures **every one of the six transfers**. Nothing inherits: the pooled fallback in `ceiling_for` is no longer reachable. The grid measured two and four inherited out of sample, and both of `MIL-CREDA`'s significant losses fell on inherited transfers. `test_busca_en_todas_las_transferencias`
- [x] Searching transfers the verdict also judges leaks nothing. What keeps the material disjoint is the **role** — the search reads `valid`, the verdict reads `eval` — and that holds identically on all six. `search_ceilings`' docstring claimed the transfer split did that work; it was false as configured and is corrected. `test_la_busqueda_y_el_veredicto_se_separan_por_rol_no_por_transferencia`
- [x] **One seed per trial, declared.** Two trials on different seeds would measure the ceiling and the draw at once. The material is drawn once per family and every trial of a transfer runs on it. `test_una_sola_semilla_declarada_en_todos_los_trials`
- [x] **A sampler seed per study**, derived from `(family, transfer)` with CRC32. Sharing one seed made all twelve studies visit the same four ceilings, and with a wide plateau the winner is the smallest *visited* point — so the record showed an agreement across transfers that was an artifact of the seed. Found by the local pilot, not by reading. `test_cada_estudio_explora_puntos_propios_y_no_los_del_vecino`
- [x] **The plateau is the instrument's resolution**, `1/VALID_BAGS`, and not the GP's own noise estimate. Two ceilings that differ by less than one bag are not distinguishable by the measurement, whatever the model thinks; using a fitted quantity would make the plateau's width depend on how well the model fitted. `test_la_meseta_es_la_resolucion_del_instrumento_y_no_el_ruido_del_gp`
- [x] **Within the plateau the smallest ceiling wins** — the same rule the tie-break had, and the same reason: the same outcome for less adaptation is the weaker claim. `test_dentro_de_la_meseta_gana_el_mas_chico`
- [x] The **grid engine stays reachable** by configuration. It wrote the record that governs the campaign of 1800 runs, and an engine that can no longer be run is a record that can no longer be reproduced. `test_a_resumed_ceiling_the_grid_no_longer_has_refuses_by_name`
- [x] The **pilot search writes its own record** and the full record always outranks it. With one file, a rehearsal's answer would have been consumed by a full campaign without a word. `test_el_registro_completo_le_gana_siempre_al_ensayo`
- [ ] **The trials search has not run at full scale.** The ceilings governing the campaign come from the grid search, so `probe` reports `search-first` and it is right: the current declaration has no record yet. No new campaign is launched from this declaration until it does.

## Reading the campaign

- [ ] The geometry uses uniform weights for every arm, so two arms differ in the space and never in the ruler. Attention keeps its own effect where it is already its own claim, in attentionSpread and correspondence.
- [x] Phase-two geometry is measured in the RKHS the method aligns in, not on the Euclidean embedding of Eq. (16): the class mean is Eq. (20) at uniform weights, its inner product Eq. (21) over the kernel of Eq. (19), and the reading is d = 1 - K_AB / sqrt(K_AA * K_BB), bounded to [0, 1] by Cauchy-Schwarz because the kernel is PSD. Raw distances stay in the record and are never rendered. `test_the_reading_is_the_kernel_distance_the_method_aligns_in`
- [x] The error of that mean is **between transfers, not over the pooled pairs**. A transfer is a setting and not a repetition; pooling claimed a stability across settings nobody measured, and understated the uncertainty by a factor of three. `test_the_error_is_between_transfers_not_over_the_pooled_pairs`
- [x] **Phase two rests on 3 repetitions, not 30.** `CHECKPOINTS` keeps three per cell, so every latent claim sits on the verdict floor by construction, however long phase one ran. `test_median_seeds_is_the_selection_rule_on_its_own`
- [x] Promotion never moves an arm's own median. Seeds added to keep a floor comparison possible, or to let a figure draw, are kept in separate fields and **never enter a marginal average**: a floor's extras were chosen by the dependent arms' orderings, so they are a biased sample of that floor. `test_an_arms_own_median_is_never_moved_to_match_its_floor`

## Label noise

- [ ] The sweep runs on M to U, fixed by the smallest domain gap and not by which transfer came out best. The rule is about the instrument: a transfer already near its floor at rho 0 has no room to fall and cannot show a curve, and the gap is a property of the material rather than of any measurement. It costs 1 arm x 1 transfer x 30 seeds x 5 levels = 150 runs. `test_el_barrido_lee_los_techos_una_vez_y_los_mantiene_en_los_cinco_niveles`
- [ ] Noise enters as a declared axis with 0 as its first level, inside the one results notebook: a clean block and a contaminated block of the same table, never a second notebook and never a second table.
- [ ] The target is the sharper half and not an afterthought. Target training is unsupervised -- `pseudolabel` and `confidences` -- so contaminating its bags is not label noise at all: it corrupts the conditional the adaptation term aligns to and poisons the pseudo-labels the weighting is computed from. F against E, and G against F, differ exactly in that weighting, and at rho 0 those rungs have almost nothing to separate them.
- [ ] One contamination draw, shared by every arm. Arms that saw differently corrupted material differ in the draw as well as in what they compute. The manifest records which instance slots were replaced and by which image indices, beside the `imageIndices` it already writes, so the material can be rebuilt without trusting a permutation to be stable.
- [ ] The contaminated block of every table is rho 0.2, fixed as the midpoint of the declared range before anything runs. Choosing it after seeing the sweep would put the noise level most favourable to the method in the headline table, chosen by outcome; the midpoint is arithmetic and nothing that comes out of the run can have decided it.
- [ ] The noisy condition contaminates the training and evaluation roles with one draw shared by every arm; validation stays clean because the search never reads it under noise, and the clean condition is clean throughout.
- [ ] The sweep opens the results notebook and runs the floor alone on one transfer, drawing accuracy against the noise level in source and in target. Measured on the arm that does not adapt, it says the contamination hurts without crediting the effect to any method.
- [x] The ceilings are searched at rho 0 and held fixed across all five levels. The curve is declared as the coefficient chosen clean, applied dirty — which is also the practical situation, since nobody recalibrates per noise level. Searching per level would multiply 2 families x 6 transfers x 30 trials x 20 epochs by five, another whole campaign before the campaign. `test_el_barrido_lee_los_techos_una_vez_y_los_mantiene_en_los_cinco_niveles`
- [x] The degradation notebook shows the whole curve over the five levels, so it chooses nothing. Accuracy against rho for every arm, and `adaptationShare` against rho for every arm that carries an adaptation term. Both instruments already exist and are already recorded: no new measurement is needed, only the axis. `test_the_degradation_figure_draws_the_share_against_rho_and_not_only_accuracy`
- [x] The levels are 0, 0.1, 0.2, 0.3 and 0.4, capped there. With 30 instances and contaminants drawn from the other nine classes the drawn class stays the plurality while rho is below 0.5; past that the bag label stops being defensible and the curve measures nothing. `test_every_declared_level_sits_under_the_cap_and_is_exact`
- [x] Contamination replaces k of the `INSTANCES_PER_BAG` instances of a bag with images of another class and never touches the bag label. Bags are pure and no instance carries a label, so there is no label to flip: what is corrupted is the evidence, not the answer. `test_every_training_bag_loses_exactly_the_declared_count`
The campaign as it stands runs on clean material, so nothing in it can show whether
the formulation mitigates the corruption it claims to. Noise enters as a declared
axis over the same notebooks rather than as a duplicated set of them.

## Reversed

What was agreed and later changed, and what replaced it. Written rather than deleted: an agreement that was turned over is part of the record, and removing it would lose exactly what this file exists to keep. No bullets, because the parser counts items and this is not one.

**"The ceiling is fixed at 1.0 and not chosen by looking at outcomes."** Agreed when the ceiling was shared and its value came from the neutral argument. Reversed on deciding that each method searches its own: a shared ceiling equalized the coefficient and unequalized the balance. The neutral argument did not die, it changed role — it is now the reference the searched value is read against.

**"The sweep is a sensitivity reading and not a selection, because there is no validation role."** Both halves changed. It is a selection now, and there is a validation role: it was carved, funded with new material so evaluation lost nothing.

**"The report says CREDA runs at the shared ceiling and not at its published 1e-4."** There is no shared ceiling any more. What the report has to say now is which ceiling each family found, on which role, over how many repetitions, and whether the seeds agreed.

**"No adaptive sampling. With 20 points the exhaustive grid is affordable and gives strictly more evidence: pruning biases against slow deltas, and sparse sampling destroys the tie structure, which here is a finding."** Reversed on 2026-08-27 for Optuna with `GPSampler`, 30 trials per `(family, transfer)` over a continuous range. Two halves of the old reason, and they did not fare the same. *Pruning biases against slow deltas* does not apply: nothing is pruned, every trial runs to its full epoch count. *Sparse sampling destroys the tie structure* was right, and the structure is not recovered — over a continuous range two evaluations never land on the same ceiling, so `tied` and `seedsAgree` cannot exist. What replaces them is the **plateau**: how many visited ceilings the criterion cannot tell apart. It carries the same finding — a wide plateau means the rule chose and not the criterion, which is what exposed CREDA's ceiling — but it is a different measurement and not a stronger one. The claim that the grid gives *strictly more evidence* was not tested before reversing it, and this file should say so rather than imply the trade was proven.

This reversal was made **without reading this file**, which is the failure the file exists to prevent. It is recorded here rather than quietly folded into the section above.

**"A `max` column beside `mean ± stdev`, so the peak is documented where a"** **"A `max` column beside `mean ± stdev`, so the peak is documented where a peak is a number rather than a picture."** Never built, and the reason is in `render`'s own docstring: the maximum of N repetitions grows with the arm's own dispersion, so printed beside a mean it rewards the noisiest arm -- the same reason the median artefact is kept and never the best. The value is still computed by `tables.table`; nothing reads it.

**"A `share` column: the realized contribution of the adaptation term."** **"A `share` column: the realized contribution of the adaptation term."** The column was never built. What stands in its place is a figure: `figures.contribution_curves` reports the realized share arm by arm, and that figure now carries a witness. A column beside a mean would have invited the comparison the curve makes honestly.

**"**The wall-time table comes first**, in the same shape, times instead of"** **"The wall-time table comes first, in the same shape, times instead of accuracy. It replaces the per-run listing."** Both halves are gone. The contract declares `seconds` as `perRun`, and pooling it prints a number that describes none of the runs behind it -- so the pooled table was removed and `render` now refuses any `perRun` dimension outright. The per-run listing is what stands, and it is the form the contract permits.

**"**Reading order**: time, source, source rungs, target, target rungs, and the"** **"Reading order: time, source, source rungs, target, target rungs, and the transversal reading last."** The first five hold. The sixth does not: the transversal reading was removed on 2026-08-14, `render_panorama` deleted with it, and only `summary["panorama"]` survives in the record, rendered nowhere. The removal was never written down here, which is the failure this section exists to prevent.

**"**Three transfers in every figure, not six.** Six rows at a legible panel"** **"Three transfers in every figure, not six. ...Fixed in `FIGURE_TRANSFER_COUNT` and `FIGURE_TRANSFER_RULE` by a rule that looks at no outcome: each domain appears once as source and once as target."** The count holds; the rule does not. The three are chosen by the outcome -- where the methods reach highest -- and declared in every figure's own caption, because a latent space in which every method sits near chance is a picture of a model that did not learn and says nothing about alignment. The no-outcome safeguard did not die, it moved: it now governs which seed is displayed, which is where a figure can still be made to look better than it is.

**"Bag figure highlights **the same bags in every panel**: the median bag of"** **"Bag figure highlights the same bags in every panel: the median bag of each class by correspondence mass, one colour each, everything else grey."** Three of the four clauses hold. The grey does not: every background bag now carries its own class colour, because in grey the correspondence was unreadable -- the colour is the only thing left to judge it by.

**"The notebooks under `CREDA/Notebooks/` remain functional, verified statically without executing them."** **"The notebooks under `CREDA/Notebooks/` remain functional, verified statically without executing them."** Out of this record's scope. `CREDA/` is prior work this project never edits, so nothing here can make the claim true or false, and an agreement nobody can act on is noise.

**"The report marks the three cross-family rungs — C→E, D→F, D→G — with whatever remains of the confound **after** the search. With per-family ceilings the balance is partly equalized, so the magnitude has to be measured again rather than repeating the `B_src` factor that came from the shared ceiling. The within-family rungs stay attributable."** Reversed on 2026-09-15, together with the comparison itself. CREDA and MIL-CREDA answer two different tasks, so arms A, C and D were undeclared and the three cross-family rungs no longer exist: there is no confound left to measure or to mark, because there is no rung that crosses. What replaces this item is not a smaller version of it — the six surviving rungs are all within MIL-CREDA and were already attributable, which this item said in its own last sentence. Nothing about prior work changed: `src/CREDA/` stays byte-identical and undeclared, never edited.

**"The same contamination is two perturbations, and the report says so. `wiring.py` broadcasts the bag label to all 30 instances, so for A, C and D those k instances carry a genuinely wrong label; for the bag-unit arms the label stays at the bag and the contaminants are witnesses the attention may downweight. Read as one perturbation the table would say "same noise, different robustness" when it says "one contamination, two consequences the formulations imply"."** Reversed on 2026-09-15. This item was the sharpest statement of what the two families made visible: one contamination landing as a wrong label on one side and as a downweightable witness on the other. With arms A, C and D undeclared there is no instance-unit side left, so the contamination is one perturbation with one consequence and the distinction has nothing to distinguish. This is a real loss and is recorded as one, not as housekeeping: what the noise sweep measured was partly this contrast, and the noise diagnostic notebook was built around it. What the sweep now measures, and whether that diagnostic still has a question, is undecided at the time of writing.

**"Every table framing states the equation its quantity comes from and what it means for the instance-unit family against the bag-unit family, before the table and never after."** Reversed on 2026-09-15. There is no instance-unit family: arms A, C and D were undeclared when the comparison was, so a framing contrasting the two families has no second family to name. The half of this item that survives — that every table states the equation its quantity comes from, before the table and never after — was never in dispute and is owed a placement of its own rather than being smuggled through as the remainder of a reversed item.

**"The report says which ceiling each family found, on which role, over how many repetitions, and whether the seeds agreed. That CREDA does not run at its published 1e-4 is a consequence of the search and has to be read there, not in a comment in `config.py`."** Reversed on 2026-09-15. One family searches now — `SEARCH_ARMS` is `{"milcreda": "G"}` — so "which ceiling each family found" names a plural that no longer exists, and CREDA runs nothing at all in this bench, at 1e-4 or at any other value. What the report still owes about the surviving search (which ceiling, on which role, over how many repetitions, whether the seeds agreed) is owed for one family and needs its own placement.

**"In every adapted arm except GN, unlabelled target passes through the encoder for the adaptation terms, but every layer with running statistics learns from source only."** Reversed on 2026-09-17, the day it was placed. Implemented as freezing normalization for the target forward, it normalized the target with accumulated statistics (mostly ImageNet's early on) while the source used its own batch statistics; measured on M->U at step 0 the two domains landed on different scales (median squared distance 1399 vs 117, off-diagonal bag kernel 0.004 vs 0.43), so the adaptation terms compared an artefact. The operator settled it more simply: the adapted arms pass source and unlabelled target through the same extractor, and whatever normalization layers it has are part of the model, not an effect this experiment isolates. The only normalization guarantee kept is the floor's: it never sees the target.

**"Arm GN is full MIL-CREDA identical to G except that its normalization layers also learn from target; the rung G to GN reads what that normalization adds, so no gain is credited to the adaptation loss that belongs to normalization."** Reversed on 2026-09-17, the day it was placed. GN existed only to isolate normalization learning from target against a G whose normalization was frozen. With that freeze withdrawn, GN would be identical to G, so the arm and its rung were removed: seven arms and six rungs remain. A rung from the floor reads the proposal's adaptation against a model trained on source only, not a mixture to decompose.

**"`SU`, `SA`, `SK` **select 10 instances** and then apply the same learned"** Reversed on 2026-09-17. The three selecting arms were retired: with CREDA gone, what the selection rungs validated — that the order the attention produces is informative — is answered instead by comparing aggregation mechanisms on the full method, and the arms cost a campaign each.

**"`SA`'s draw uses a **dedicated generator**. Consuming the training"** Reversed on 2026-09-17 with the arm it describes. SA no longer exists, so the generator it drew from has nothing left to protect.

**"The three attention rungs are replaced by `SU->SK`, `SA->SK` and `SK->G`."** Reversed on 2026-09-17. SU, SA and SK are retired, so those three rungs have no arms. What replaces them is a comparison of aggregation mechanisms on the full method: ours against ABMIL as published, ABMIL gated, max and mean.

**"The ladder table stays, directly below. Levels say who is ahead; only the"** Reversed on 2026-09-17. The rungs table goes with the rungs: the ladder is now three within-method steps and the results notebook reads position from a rank column inside each accuracy table, not from a separate ladder.

**"A rung is named `Baseline → MIL-Baseline`, never `A->B`. An identifier is"** Reversed on 2026-09-17. It named how a rung is written, and the rungs table no longer exists. Display names over identifiers survives where it still applies, in the arm column of every table.

**"**Both floors stay, because the measurement says they are not redundant.**"** Reversed on 2026-09-17. There is one floor: the instance-unit family went with CREDA, so no measurement can find two floors redundant or not.

**"A **contribution panel** beside them. The pilot already shows the realized"** Reversed on 2026-09-17. The loss curves stay for one purpose only — showing that the terms of Eq. (39) hold the common scale the B_src normalization gives them — and a contribution panel answers a different question that this section no longer asks.

**"The gains table is **paired within each transfer**, against each method's own floor, and reports four things because no single one of them is honest alone: the mean in points with its **between-transfer** error, the mean of the percentages, the span, and the agreement in words. The two means can disagree in sign on the same data — `+0.56` and `-3.61` — because the floors run from 23% to 81%."** Reversed on 2026-09-17. The gains table is retired with the rungs table. Which arm is ahead, and by how much over its floor, is read from the accuracy tables and their rank column; the pairing rules it carried survive wherever a difference is still read within a transfer.

**"Each family searches **its own** ceiling and passes it to its derivations. A shared ceiling equalizes the coefficient and unequalizes the balance: the two objectives sit a factor of `B_src` apart, so one number puts adaptation at 85% of one and 10% of the other."** Reversed on 2026-09-17. There is one family left, so there is nothing to equalize between two of them. One ceiling is searched, on the full method, per transfer.

**"CREDA is used as it is: per-instance cross-entropy, its own single-term objective, never edited to make the comparison work."** Reversed on 2026-09-17. CREDA is no longer an arm of this bench: the comparison against it was undeclared because the two methods answer different tasks. Its code stays on disk, untouched and unimported by any declared arm, and `priorWork` is what watches it now — not a test of an arm that no longer runs.

**"The 1e-4 leaves where it was and becomes the ceiling of CREDA's own ramp, with `creda_lambda_special` as the default. `train_creda` still reads it from the notebook's `cfg`, because it is not always 1e-4."** Reversed on 2026-09-17 with the arm it served. No declared arm carries CREDA's ramp, so its published coefficient governs nothing here.

**"The seconds dimension is never dumped run by run in the notebook. Inline it is the median with its min-max range per method and environment, collapsing the seed axis and naming it; every row stays in the written record, which is what a record is for."** Reversed on 2026-09-17. Time and memory stop being measured and recorded at all, so there is no seconds dimension left to collapse or print.

**"The **growth rate is not searched**: it stays at `RAMP_DELTA = 20`, CREDA's own, shared by both sides. Searching it too would have turned the run into a 2D grid of three to five hours."** Reversed on 2026-09-17. The growth rate is now searched. One of its two reasons died with the second family — a free dimension can no longer amplify an imbalance between two of them — and the other, that ceiling and growth are confounded, becomes a demand on the search rather than a reason to fix one of them: the search covers both and its record says what it explored.

**"The **growth rate is still not searched**, and the reason is new: the two families' objectives already sit an order of magnitude apart in `adaptationShare`, so a second free dimension amplifies that imbalance rather than resolving it. Ceiling and growth rate are also confounded — a high ceiling reached slowly and a low one reached fast give similar trajectories."** Reversed on 2026-09-17, for the same reason as the agreement it restated: the imbalance between two families it rested on no longer exists, and the search now covers six dimensions — ceiling, growth, kernel sigma, attention gamma, attention temperature and the local tau.

**"The report and latent notebooks show each table twice: at rho 0 and at one contaminated level, each with its own conclusion. A second conclusion that enumerates the numbers of its own table comes back as `restated`, which is the finding already open on `Benchmark_Latent_v1`."** Reversed on 2026-09-17. There is one results notebook and one table per quantity, with the clean and contaminated blocks inside it and a single conclusion below, so nothing is shown twice and the `restated` finding it warned about has nothing to fire on.

**"One rate, the same for source and target, applied only to `train_idx`. `valid_idx` and `eval_idx` are never contaminated: the first is where the search reads its criterion, the second is the answer key of the verdict. Train dirty, measure clean. Two separate rates would make the sweep two-dimensional and multiply a campaign that already costs 1800 runs; which of the two domains hurts more is a rung of its own, later, on one transfer."** Reversed on 2026-09-17. Measuring clean while training dirty answers whether learning survives contamination; what this method claims is that at prediction time the attention lowers the weight of contaminants in a bag, and that can only be read on contaminated evaluation bags. So the noisy condition contaminates training and evaluation with one shared draw across arms, and the clean condition stays clean throughout. Validation stays clean in both: the search runs only on clean material, so nothing reads it there, and contaminating it made the reserve outgrow USPS's smallest class at the declared cap.

**"The diagnostic runs at rho 0.4, the cap of the declared range, fixed now rather than after the curve. At the extreme the coefficient is under the most pressure: a re-searched ceiling that recovers nothing there recovers nothing anywhere, so the reading does not depend on where anyone chose to look. The cap is fixed by the range and not by a result."** Reversed on 2026-09-17 with the whole noise diagnostic. It existed to tell one failure from another when accuracy fell — the adaptation term failing, or the ceiling being too small for that level — and it did so by re-searching the ceiling under contamination. That contradicts the decision that the search always runs on clean material, so that its answer is the method's and not a tuning fitted to the noise. The cost is declared rather than hidden: when the sweep shows a fall, this bench can no longer separate those two causes. This item fixed the level it ran at.

**"That notebook is diagnostic and never a verdict. Its numbers do not enter the verdict tables, and what it decides is one thing: whether per-level ceilings are worth restructuring the code for."** Reversed on 2026-09-17 with the whole noise diagnostic. It existed to tell one failure from another when accuracy fell — the adaptation term failing, or the ceiling being too small for that level — and it did so by re-searching the ceiling under contamination. That contradicts the decision that the search always runs on clean material, so that its answer is the method's and not a tuning fitted to the noise. The cost is declared rather than hidden: when the sweep shows a fall, this bench can no longer separate those two causes. This item kept its numbers out of the verdict tables.

**"It needs three points and only pays for one. D and G at the chosen level under the clean ceiling come out of the campaign at no extra cost; what is run is the ceiling searched at that level, and D and G under it. A re-searched ceiling that recovers the loss says it was the coefficient; one that does not says it was the term."** Reversed on 2026-09-17 with the whole noise diagnostic. It existed to tell one failure from another when accuracy fell — the adaptation term failing, or the ceiling being too small for that level — and it did so by re-searching the ceiling under contamination. That contradicts the decision that the search always runs on clean material, so that its answer is the method's and not a tuning fitted to the noise. The cost is declared rather than hidden: when the sweep shows a fall, this bench can no longer separate those two causes. This item priced it at one measurement.

**"The diagnostic notebook runs on M to U — the same transfer as the degradation curve, so it reads against a curve that exists — with only D (CREDA) and G (MIL-CREDA), the two complete methods, one per family, and the only two carrying the coefficient. A and B have no adaptation term to re-search a ceiling for; C, E and F are ablations that would multiply the search without adding diagnosis."** Reversed on 2026-09-17 with the whole noise diagnostic. It existed to tell one failure from another when accuracy fell — the adaptation term failing, or the ceiling being too small for that level — and it did so by re-searching the ceiling under contamination. That contradicts the decision that the search always runs on clean material, so that its answer is the method's and not a tuning fitted to the noise. The cost is declared rather than hidden: when the sweep shows a fall, this bench can no longer separate those two causes. This item fixed its transfer and its two arms, one of which was CREDA.

**"What a fixed ceiling costs is that a degradation cannot be attributed: the term failing and the coefficient being too small look identical. That question is answered afterwards by a cheap diagnostic rung and not by paying for five searches up front."** Reversed on 2026-09-17 with the whole noise diagnostic. It existed to tell one failure from another when accuracy fell — the adaptation term failing, or the ceiling being too small for that level — and it did so by re-searching the ceiling under contamination. That contradicts the decision that the search always runs on clean material, so that its answer is the method's and not a tuning fitted to the noise. The cost is declared rather than hidden: when the sweep shows a fall, this bench can no longer separate those two causes. This item named the very ambiguity the diagnostic existed to resolve, which is now accepted as a declared limitation.

**"`RAMP_CEILING` stays at **1.0** as the neutral each family's searched"** The witness this line carried measured the contribution panel, and the panel is gone: nothing reports the realized share of the adaptation term any more. The claim about the neutral survives the panel and is re-placed without it.

**"Loss curves for `Baseline`, `CREDA`, `MIL-Baseline`, `MIL-CREDA`, as the"** Two of the four arms named here no longer exist in this benchmark: the CREDA family was undeclared, and the ladder is now one family of five arms.

**"One search per family, on the complete method — D and G — and inherited. If every arm searched its own, B→E would differ in two things and no rung would be attributable. Declared consequence: E and F carry no local term, so the ceiling found on the complete method is not necessarily theirs."** There is one family left, so there is no per-family search and no inheritance between families. What the line was protecting -- that no arm funds its own coefficient -- is kept, stated over the single ladder.

**"The benchmark declares revision r17 and which sections each arm exercises."** The benchmark was bound to r17; the implementation is now bound to r21, which is a different attention and a different bag embedding.

**"The record keeps the supervised term's magnitude and the ratio between terms, not only the contribution. Without a denominator, "the term commanded nothing" and "the term was scaled to nothing" print alike."** The line was written against a record that also carried the contribution, and the contribution is gone from every dimension, column and field.

**"On the two transfers the search measured, the ceiling in force is the one that won **on that transfer**, by the same paired rule and the same tie-break. On the remaining four it is the one that won pooled across the two searched transfers. The report states that rule where the ceilings are shown, because the scalar of those four was not chosen by looking at them."** The search measures every one of the six transfers, so there are no four transfers left to inherit a pooled value and no pooled rule to state.

**"The record carries the per-transfer picks beside the pooled one. A record holding only the pooled winner cannot express the rule above, and a run reading it would silently apply the pooled value everywhere."** The pooled winner it asked the record to carry beside the per-transfer picks no longer exists: with all six transfers measured, the pooled value is not a fallback for anything.

**"MIL-CREDA no longer runs at one coefficient across transfers, so its row averaged over transfers mixes two scalars. Within a transfer every arm still shares the ceiling, which is what keeps each rung attributable; the report says which of the two readings it is giving."** It named the two coefficients of the two families, and one of the families is gone.

**"Only the **full arms** are searched, `D` and `G`. The ceiling is per family and per transfer, **never per arm**: `ceiling_for(reduction, family, transfer)` takes no arm, so a per-arm ceiling is not expressible. If it were, the term and the coefficient could not be told apart."** It named one full arm per family, `D` and `G`, and `D` belonged to the family that was undeclared.

**"The contaminated level those two notebooks show is rho 0.2, fixed as the midpoint of the declared range before anything runs. Choosing it after seeing the degradation curve would put the noise level most favourable to the method in the headline table, chosen by outcome; the midpoint is arithmetic and nothing that comes out of the run can have decided it."** There are no longer two notebooks to show a contaminated level: every result lives in one.

**"The degradation notebook runs on M to U, fixed by the smallest domain gap and not by which transfer came out best. The rule is about the instrument: a transfer already near its floor at rho 0 has no room to fall and cannot show a curve, and the gap is a property of the material rather than of any measurement. It costs 10 arms x 1 transfer x 30 seeds x 5 levels = 1500 runs."** The sweep is now the floor alone on one transfer, so the arm count in its cost is wrong by an order of magnitude and the notebook it names no longer exists.

**"One contamination draw, shared by all ten arms. Arms that saw differently corrupted material differ in the draw as well as in what they compute. The manifest records which instance slots were replaced and by which image indices, beside the `imageIndices` it already writes, so the material can be rebuilt without trusting a permutation to be stable."** It counted ten arms, and the ladder is five; its witness was written against that count and no longer exists in the suite.

**"The target is the sharper half and not an afterthought. Target training is unsupervised — `pseudolabel` (Eq. 22) and `confidences` (Eq. 24) — so contaminating its bags is not label noise at all: it corrupts the conditional the adaptation term aligns to and poisons the pseudo-labels the weighting is computed from. D against C, and F and G against E, differ exactly in that weighting, and at rho 0 that rung has almost nothing to separate it."** It named the rungs of a ladder that had two families in it; the rungs that isolate the weighting are now F against E and G against F.

**"Noise enters as a declared axis with 0 as its first level, over the notebooks that already exist. No v2 duplicates: two copies of four notebooks differing in one parameter fork from the first day, and there is no clean campaign to protect — `Results/Benchmark/` is empty and the v1 grid was retired with its record."** It promised the axis would land on four existing notebooks; there is one notebook now, and the noise condition is a block inside each of its tables rather than a copy of anything.

<!-- position revision=research-concept-r21.md sha256=cc55237fd6f7d9c6418a66671b64b87573cb5103465b31894fa1f1dc460178c2 derivedAt=2026-09-17T05:07:03Z session=52829b86 target=none -->
- [ ] 1. The invariants hold against the tree as it stands: the suite is green and the verification notebook ran against this exact source. Two-state on purpose -- it runs here and nowhere else, and giving it a rung would be the position asserting a state it does not have. Nothing below is worth reading until this is ticked: every later step measures something, and a broken tree makes every measurement a description of the break. `@notebook Notebooks/verification.ipynb`
- [x] 2. A ceiling record exists at the scale the search declares for itself. Below that scale the search is a rehearsal: the ramp saturates by the second epoch, every ceiling is reached almost at once, and no value it chose is quotable. This is two-state because the record either meets its own declared scale or it does not. `@record:level ceilings`
- [ ] 3. The search's own report is current against this source, so what it says about how each ceiling was chosen -- by a difference in the criterion, or by the rule inside a plateau -- describes the code that is here. Its rung is the rung of the record behind it: a report is only ever as trustworthy about scale as the record it read. `@notebook:level Notebooks/Benchmark_Search_Report_v1.ipynb`
- [ ] 4. The degradation sweep ran and its notebook is current against this source. It sits here, between the search and the campaign, because it runs under the ceilings the search fixed and every reading of the campaign afterwards is read beside its curve -- not off to one side of the flow. Its rung is the rung of the run behind it: a sweep executed at pilot scale walks the same five levels with one seed, and what it draws is a shape rather than a measurement. `@notebook:level Notebooks/Benchmark_Noise_Sweep_v1.ipynb`
- [ ] 5. The sweep's own report is current against this source, so what it says about how much each method loses as the material is corrupted -- and whether the order between methods survives -- describes the code that is here. Its rung is the rung of the sweep behind it: a report is only ever as trustworthy about scale as the run it read. `@notebook:level Notebooks/Benchmark_Noise_Report_v1.ipynb`
- [x] 6. The campaign ran and left evidence, at whichever rung it reached. Read this one knowing two things it cannot see: a campaign run here at pilot scale leaves no shard at all, so this reads as the floor even when a pilot did run; and an arrived shard says nothing about which search governed it, so a shard from a retired engine still reads as the top rung. Both are gaps in the witness, not in the campaign. `@shard:level s00`
- [ ] 7. The report is current against this source and reads the merged record rather than a rehearsal's leftovers. This is the step the whole position exists for: a report executed against a pilot record once printed sixty runs beside an eighteen-hundred-run record with every other check green, and the rung is what makes that visible without anyone having to remember to doubt it. `@notebook:level Notebooks/Benchmark_Report_v1.ipynb`
- [ ] 8. The latent analysis is current and measured the checkpoints the campaign promoted, not an earlier run's. Its own claims rest on three repetitions per cell whatever the campaign ran, because only three are kept -- so its rung describes the record it read, never the power of what it says. `@notebook:level Notebooks/Benchmark_Latent_v1.ipynb`
- [ ] 9. The diagnostic paid its one measurement -- the ceiling re-searched under contamination, on one transfer and two arms -- and its notebook is current. It answers the one question the sweep cannot: a term failing under noise and a coefficient falling short draw the same curve. None of its numbers enter the verdict tables, and it governs no ceiling record; what it decides is whether restructuring to search per noise level is worth paying for. `@notebook:level Notebooks/Benchmark_Noise_Diagnostic_Search_v1.ipynb`
- [ ] 10. The diagnostic's report is current against this source and presents the `diagnostic.json` that already exists rather than computing anything. Last because it reads the last thing written: with this ticked the whole flow has been walked, and every earlier item says at which rung. `@notebook:level Notebooks/Benchmark_Noise_Diagnostic_Report_v1.ipynb`
<!-- /position -->

































