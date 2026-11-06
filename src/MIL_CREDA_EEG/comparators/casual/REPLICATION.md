# CASUAL controlled BALLADEER variant

## Code search — 2026-11-02

Route: reimplementation. No adopted repository, commit or license; no third-party code copied.

| Place searched | Answer |
|---|---|
| `guidance/experimental-papers/casual-2305.18458/casual-2305.18458.md` (whole ingested paper) | No code-availability URL found. |
| https://arxiv.org/html/2305.18458v2 and https://arxiv.org/abs/2305.18458 | No method repository link found; HTML includes Appendix G implementation details absent from ingested Markdown. |
| https://anh-tong.github.io/publications/ | CASUAL entry links only the AAAI PDF, unlike other entries with Code links. |
| https://ojs.aaai.org/index.php/AAAI/article/view/34166 | Fetch failed; not evidence of code absence. |
| Web queries `"CASUAL" "2305.18458" github code` and `"Conditional Support Alignment" Nguyen code AAAI 2025` | Paper and venue found; no author-linked implementation verified. |

This is a bounded search, not a claim that code cannot exist. Scratch HTML/text retained at `/var/folders/y9/9hfxsk810hv08cnh64bncjq80000gn/T/tmp.fACJ2Q6V9m/`.

## Plan and equation analysis (before coding)

Binding plan: `experiments-plan-experimental-mil-creda-sobre-balladeer-v57.md`, §§4.5, 7.1–7.4, 7.7 (rebound from v40 on 2026-11-03, rebound to v54 on 2026-11-04, to v55 on 2026-11-05 and to v56 on 2026-11-05 with no change of behaviour: v56 keeps its §7.1 row and the §§4.5, 7.2–7.4, 7.7 rules this record cites); T6/S1–S5. This is explicitly a controlled variant, not reproduction of published image experiments.

- `losses.py`: classification, conditional entropy, joint conditioning, discriminator loss, support loss and VAT, one function per computational equation.
- `training.py`: auxiliary discriminator, alternating updates, shared bridge and checkpoint reconstruction, full epoch budget and both validation selectors, explicit search registration and model-store wrappers.
- `__init__.py`: exports only, no implicit registration or mutation of sibling state.
- Tests: closed-form losses, unequal batches, conditioning, symmetry/support rather than distribution matching, finite gradients, VAT zero/nonzero/radius/mode isolation, frozen state, checkpoint identity and both channel directions, selection and search/store joins.

Boundary shapes: input is a list of B bags of `(m_b,C,512)` float32 windows, C=14 or 29. Reused EEGNet trunk emits `(m_b,256)`; shared CommonMILModel emits `(B,256)` float64 bag vectors and `(B,2)` probabilities. Outer product is `(B,256,2)`, flattened feature-major to `(B,512)`. Discriminator emits B scalar logits; sigmoid scores in `[0,1]` are used for support distances. Labels are B source bag labels, never window labels. Target training labels are unread.

| Paper equation | Function / disposition | Shapes and assumptions |
|---|---|---|
| 1–2, 7–10 | Theoretical uncertainty, discrepancy and risk bounds, not trainable estimators | Infinite hypothesis-space extrema and unknown target risks are not fabricated as runtime losses. |
| 3 | `support_alignment` empirical 1D SSD | B_s and B_t scores; independent directional means of nearest-neighbour distances. |
| 4–6 | Conditional divergence motivates the joint proxy | Not a separate oracle loss requiring true target labels. |
| 11 | `classification_loss` | `(B_s,K)` probabilities and B_s labels -> scalar unnormalized cross entropy. |
| 12 | `conditional_entropy` | `(B_t,K)` -> scalar mean entropy, natural log. |
| 13 | `virtual_adversarial_loss` | List of target bags -> scalar; detached clean predictions, power-iteration approximation, L2 norm per bag. |
| 14 | `discriminator_loss`, `joint_features`, `entropy_weights` | `(B,D)` and `(B,K)` -> B logits -> scalar; source=1, target=0; domain means added. |
| 15 | `support_alignment` | B_s/B_t sigmoid scores -> scalar; min over opposite support, not paired matching. |
| 16 | `adaptation_objective` | Weighted scalar sum; source CE constant in frozen-source variant; only target trunk differentiates. |
| 17 / Algorithm 1 | `alternating_step` | First discriminator update with detached joint inputs, then target update with discriminator frozen (input gradient retained). |

Cross-check: arXiv HTML confirms Eq. 17 is simply min_phi L_dis; Algorithm 1 updates discriminator before f,c; Eq. 13 is source VAT plus target VAT. The ingested display around Eq. 13 is also corrupted. Here source VAT is omitted as a frozen-source constant, and target VAT perturbs input windows per plan §7.4. Source CE is recorded, not made into a second trainable source path.

Lineage gaps: searched guidance filenames and available ingested papers; Tong et al. 2022 (ASA), Long et al. 2018 (CDAN entropy conditioning), and Miyato et al. 2018 (VAT) were not located as ingested papers. CASUAL gives the SSD/outer-product/VAT objectives, but lineage-specific implementation details below remain explicit assumptions.

## Provenance of choices

Every row is either a cited choice or explicitly unspecified. Published image settings are recorded as departures, not silently used as EEG settings.

| Choice | Value / implementation | Source or status |
|---|---|---|
| Encoder, embedding and head | Existing EEGNet-8,2 trunk, 256-d, one common binary head per bag | Plan §7.3; reuse eegnet.py, no new encoder. |
| MIL aggregation | Existing CommonMILModel tanh attention, gamma=0, temperature=1, width=128 | Plan §7.2 and existing mil_wrapper.py fixed choices. |
| Source/target independence and layer copy | Existing two_phase.bridge, including both 29↔14 directions | Plan §7.3. |
| Phase 1 | Existing common-group selected checkpoint of same transfer/rotation/seed/partition; no retraining | Plan §7.3. |
| Frozen state | Source trunk, attention, head parameters and buffers fixed; only target trunk plus auxiliary discriminator updates | Plan §§7.3–7.4; departure from paper Eq.16 trainable f,c. |
| Source supervision | Ordinary mean CE at bag level, diagnostic constant in phase 2 | Paper Eq.11; plan §7.2. Not MIL-CREDA normalized source loss. |
| Target entropy | Mean per-bag entropy | Paper Eq.12. |
| Conditional input | Soft predictions on both sides, outer product feature-major | Paper definition of G following Eq.15. |
| Support metric | Squared L2 on discriminator sigmoid outputs; L1 available | Paper Eq.15 allows either. |
| Alternation | One discriminator update before each target update | Paper Algorithm 1; Appendix G. |
| Discriminator architecture | 3 linear layers, 512 hidden units, leaky ReLU, sigmoid interpreted at loss boundary | Paper Appendix G USPS/STL architecture; selected smaller published architecture for EEG. |
| Discriminator negative slope and initialization | PyTorch Linear default and leaky slope 0.2 | unspecified — no slope/init in paper; conventional explicit fixed choice. |
| Entropy conditioning formula and scope | Detached weights 1+exp(-H), domain-normalized, discriminator BCE only; alignment remains exact Eq.15 | unspecified — paper cites Long et al. without writing formula or application scope; no gradient reversal through weights. |
| Probability log stabilizer | Clamp at dtype tiny | unspecified — numerical protection for logs without smoothing labels. |
| VAT domain | Target windows only | Plan §7.4; source term of paper Eq.13 constant under frozen source. |
| VAT norm unit | L2 over all windows/channels/time within each bag | unspecified — bag is prediction unit; no inherited window labels. Under plan v42 the radius is stated relative to a window («fracción de la norma L2 típica de una ventana», §7.7) while it still bounds the whole bag's perturbation, so across the m windows of a bag the per-window perturbation norm has root mean square radius/√m. |
| VAT finite difference and iterations | xi=0.01, one power iteration, Gaussian initial direction | unspecified — paper says power iteration but omits values; xi chosen above float32 roundoff. |
| VAT stochastic layers | Temporarily eval all model modules, restore exact modes; no clean-reference gradient | unspecified — isolates consistency from dropout/BN drift. |
| VAT radius | Searched as `vat_radius_fraction` on its own axis `search.VAT_RADIUS_FRACTION`, log-uniform in [0.01, 1]; the trainer uses radius = fraction × typical window L2 norm; each bag's whole (m, C, T) perturbation has that norm (`losses.virtual_adversarial_loss`, unchanged) | Plan v42 §7.7 «el radio de la perturbación VAT se busca como fracción de la norma L2 típica de una ventana, entre 0,01 y 1 en escala logarítmica … una elección de este plan y no del artículo» (task log L60); §7.4 |
| Typical window L2 norm | Median of the L2 norms (over channels × samples, float64) of every window of the rotation's **target-training** bags, measured once at the start of phase 2 (`training.typical_window_norm`); the median of an even count is the mean of the two middle values; recorded per epoch as `typical_window_norm` and `vat_radius` | unspecified — plan v42 §7.7 says «norma L2 típica de una ventana» without stating which windows or which statistic; reads windows only, never labels |
| Alignment, entropy, VAT weights | Required caller parameters; generic loss-weight search [1e-3,10] log | Plan §7.7; paper Appendix G dataset-specific weights/ramp not preserved. |
| Target optimizer | Adam using lr2/wd2, beta=(0.9,0.999), eps=1e-8, no LR scheduler | unspecified — shared existing trainer precedent; paper Appendix G varies SGD/Adam by image dataset. |
| Auxiliary optimizer | Adam, same lr2/wd2 and defaults as target, constant rate | unspecified — avoids unplanned additional search axes. |
| Epoch budget and batches | Required schedule; shuffled target pass and cyclic shuffled source batches; partial batches retained | unspecified — epochs/bagsPerStep supplied by common pilot; source sampling does not introduce target labels. |
| Alignment schedule | Fixed supplied weight, no ramp | unspecified — EEG pilot budget has no corresponding published image step count; departure from Appendix G ramps. |
| Selection | Entire epoch budget, last epoch kept (no early stopping, no restore); per-epoch target-validation predictions recorded for the search's pooled primary and label-free (trial, epoch) selection | Plan v56 §§4.5, 7.7, shared evaluate/balanced_accuracy/label_free_score. (Row corrected 2026-11-04: it described best-epoch states that v50 removed.) |
| Search | Explicit `register()` wires production search: shared lr2/wd2, three loss weights on the generic rule, `vat_radius_fraction` on `search.VAT_RADIUS_FRACTION` (admitted by the hook for CASUAL only) | Plan v42 §7.7. |
| Final persistence | Existing model_store plus comparator source digest in extra metadata; check both digests on load | Plan §7.3; existing store code_version excludes comparator files. |
| Seeds | Caller unit seed, torch and batch generator seeded from it | Plan §§4.6,7.7; reuse search seeds, no seed search. |

## Published image configurations not transferred to EEG

Read Appendix G in the HTML cross-check (not present in the ingested main-paper Markdown). These are paper settings, **not measured here**. Every entry below is replaced by the common plan/checkpoint or the explicit choices above; none is an EEG default or a claim of replicated accuracy.

| Published choice | Paper Appendix G value | Controlled-variant departure |
|---|---|---|
| USPS architecture | LeNet variant, 500 output features; discriminator 3 layers, width 512, leaky ReLU | Shared EEGNet/common MIL instead of LeNet; discriminator topology retained. |
| USPS optimizer/budget | 65000 SGD steps; lr .02; momentum .9; wd .0005; batch 64 | Caller epoch/bag budget; Adam lr2/wd2. |
| USPS LR schedule | After 30000 steps, linear anneal over next 30000 to .00002 | No scheduler. |
| USPS loss schedule | Alignment 0→1 in first 10000 steps; source VAT 1, target VAT .1 | Constant searched weights; no source update. |
| STL architecture | Deep CNN, 192 features; linear head; discriminator 3 layers, width 512, leaky ReLU | Shared EEGNet/common MIL; discriminator topology retained. |
| STL optimizer/budget | 40000 Adam steps; lr .001; beta1 .5; beta2 .999; wd 0; batch 64 | Caller epoch/bag budget and lr2/wd2; beta1 .9. |
| STL weights | Alignment .1, conditional entropy .1, source VAT 1, target VAT .1 | Searched EEG weights; source VAT constant omitted. |
| VisDA architecture | ResNet50, 256 bottleneck; pretrained except final layer; linear head; discriminator width 1024 | Shared EEGNet/common MIL and smaller published discriminator. |
| VisDA optimizer/budget | 25000 SGD steps; momentum .9; wd .01; batch 64; encoder lr .001, classifier .01, discriminator .005 | Caller epoch/bag budget; only target trunk plus auxiliary discriminator optimized using lr2/wd2. |
| VisDA schedules/weights | LR anneal factor .05 by end; alignment 0→.1 in first 5000; entropy .05; source VAT 0; target VAT .1 | No schedule, searched constant weights. |
| DomainNet architecture | Same as VisDA | Shared EEGNet/common MIL and smaller published discriminator. |
| DomainNet optimizer/budget | 20000 SGD steps; momentum .9; wd .0001; batch 64; encoder lr .01, classifier .1, discriminator .01; Garg et al. scheduler | Caller epoch/bag budget, Adam lr2/wd2, no scheduler. |
| DomainNet weights | Alignment 1, conditional entropy .02, VAT .1 | Searched EEG weights; target VAT only. |
| Image label-shift experiments | Dirichlet alpha=None,10,3,1,.5; five random runs; VisDA/DomainNet 85/15 target split | Not generated here; existing BALLADEER subject assignment/partition and plan search/report seeds govern. |

## Integration and remaining decisions

Import `MIL_CREDA_EEG.comparators.casual` by full path. No parent `comparators/__init__.py` is written. No shared module, partition or existing test is modified. `train_phase_two(data, rotation, seed, record, params, schedule, device)` implements the existing search trainer seam and checks the record's common group, phase, transfer, rotation, seed, partition and channel counts before loading. It calls the existing bridge and returns a CommonMILModel (no additional diagnostic head). It returns independent primary and label-free target states, first epoch winning a tie. Use `save_final_model` / `load_final_model` for primary model persistence: the existing shared store's code digest excludes comparator sources, so these wrappers add and verify `casual_code_digest`. No existing store logic is rewritten. The model-store roundtrip test mocks only the partition-file check (no partition file exists here); actual partition refusal remains covered by the shared BALLADEER tests.

`register()` is explicit (no import-time side effect) and takes no argument: plan v42 §7.7 settled the radius axis, so it registers the production search space. `vat_radius_fraction=.5` is only the synthetic smoke fixture.

Returned histories have the dashboard's epoch/seed/fold/model/transfer keys. The parent experiment runner owns placement under `MIL-CREDA/Results/` (S5); no official Results files were produced under these narrow edit surfaces.

## Verification — 2026-11-02

All commands below run **inside** `implementations/Domain_Adaptation-worktrees/casual`, with its `.venv/bin/python` (symlink supplied by orchestrator), and PYTHONPATH explicitly overriding the editable install's main-checkout source path.

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_balladeer_comparator_casual.py -q
# exit 0: 20 passed in 2.52s
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests -q -k balladeer
# exit 0: 270 passed, 2 skipped, 821 deselected, 4 Optuna experimental warnings in 25.31s
PYTHONPATH=$PWD/src .venv/bin/python -m MIL_CREDA_EEG.comparators.casual.smoke
# exit 0
```

Focused invariants exercise closed-form CE/entropy/BCE, entropy-conditioning weights, outer-product ordering and gradients, SSD symmetry and support-versus-mass distinction, unequal source/target batch sizes, L1 and squared L2, invalid hyperparameters/supports, VAT zero-radius and constant-predictor cases, nonzero VAT and input norm, untouched model modes and BN buffers during VAT, finite nonzero gradients through every trained target/discriminator parameter, detached auxiliary updates, frozen source/attention/head state, 29→14 / 14→29 / 14→14 shapes, max norm, checkpoint identity, unread target training labels, full-budget primary/secondary selection and distinct saved target states, explicit search registration and no import side effects, persistence through the actual shared store, comparator digest refusal, synthetic two-phase execution, and provenance/test correspondence for every new module.

Smoke printed (CPU, synthetic seed 20261102):

```json
{"best_epoch":0,"device":"cpu","phase1_epochs":1,"phase1_loss":[0.0373800347216141],"phase2_epochs":3,"phase2_loss":[0.7716913843521792,0.8555627339044003,1.0555905323291082],"phase2_vat":[1.1743343380418703e-7,1.1425795860695509e-7,1.1432421988545583e-7],"seed":20261102,"source_shape":[4,3,29,512],"target_shape":[4,3,14,512]}
```

The loss moved **up**, not down, in this stochastic alternating synthetic smoke; it establishes execution, not convergence or accuracy. There is no BALLADEER data in this worktree per orchestrator's isolation arrangement; no official-data toy result is claimed. No full unfiltered suite, GPU, remote worker, published benchmark, real-data search or publication-number reproduction was run. Needed lineage papers are not ingested (gap above). All 11 unspecified rows are unresolved implementation choices, not fidelity guarantees.

## Later parallelism assessment (not launch authorization)

Command, from the worktree, using the target interpreter:

```bash
PYTHONPATH=$PWD/src .venv/bin/python /Users/diego/Proyectos/papersmith-ai/skills/remote-parallelism/scripts/parallelism_cli.py assess
```

Exit 2; no verdict produced. Exact refusal:

> measured inputs are required, never guessed; missing: --units or --units-file, --unit-minutes, --owners, --sessions-per-account, --session-limit-hours, --quota-hours, --setup-minutes

Objective location: before `measured`; purpose is a measured split decision, arrival is a recorded verdict with reproducibility checklist satisfied. A synthetic CPU smoke does not supply GPU pilot duration, remote setup overhead, owner capacity or quota; no numbers substituted for them. Parent must measure these before assess can proceed. Later units must be seeded from `int(sha256(unit_id).hexdigest()[:8],16) % 2**31`; this trainer accepts the resulting seed and seeds torch and batch ordering from it (never from an account). Parent must reconcile that campaign unit-id mapping with the plan's fixed search/report seed definitions, rather than silently replace search seeds.

Inherited checklist: split only along declared fold/seed/subject axis (never arm/comparison); one published commit and pinned requirements; same pinned accelerator; identical environment stamps (Python, torch, CUDA/cuDNN, accelerator, commit, requirements digest, dataset checksum/preparation); deterministic torch/CUBLAS/cuDNN flags measured in pilot; same official prepared dataset; rehearsal/readiness per owner; timing kept per machine; one account/person/service and one service/campaign unless verified identical stamps. No jobs generated or launched.

## Plan v42 update — 2026-11-03

Run in `implementations/Domain_Adaptation-worktrees/v42cmp` (detached at 7db39f8). The VAT radius is now a searched fraction of the typical window norm (rows above); the typical norm's definition is this implementation's unspecified choice, owed to `/experimental-deliberation` if the operator wants it stated in the plan. Invariant witnesses: `casual_search_hook_uses_common_group_and_the_radius_fraction_axis` (renamed from `..._and_explicit_radius`), `casual_vat_radius_is_the_fraction_of_the_median_target_training_window_norm`.

Smoke (same seed 20261102, `vat_radius_fraction=.5`): typical window norm 84.65887687367304, VAT radius 42.32943843683652 (under v40 the radius was 0.5 absolute), `phase2_loss [0.7717759795983745, 0.8558351886599931, 1.0560732468618614]`, `phase2_vat [0.0008136540925804975, 0.0007753256761211273, 0.0007787624837322085]` (v40: about 1.1e-7), best epoch 0. Execution evidence only.
