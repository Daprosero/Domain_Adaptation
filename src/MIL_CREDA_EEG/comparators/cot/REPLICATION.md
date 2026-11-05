# COT — controlled BALLADEER variant

## Code search (2026-11-02) and route

Reimplementation; no upstream repository adopted, commit/license not applicable.
Searches performed before planning:

| Place searched | Answer |
|---|---|
| Ingested paper, guidance/experimental-papers/cot-cvpr2023/cot-cvpr2023.md, whole paper including footnotes | No code-availability link found. |
| CVF venue page, https://openaccess.thecvf.com/content/CVPR2023/html/Liu_COT_Unsupervised_Domain_Adaptation_With_Clustering_and_Optimal_Transport_CVPR_2023_paper.html | Related material lists PDF and BibTeX, no code or supplement link. |
| Web: exact title + code github | No official implementation located. |
| Web: Yang Liu Zhipeng Zhou COT code CVPR 2023 | Paper record, no verified code. |
| Web: ly261666 COT github; Baigui Sun COT github; COT Liu 2023 Clustering code author | No official implementation located; zhyu-lab/cot was a false positive. |
| https://github.com/zhyu-lab/cot README | Copy number Transformer for scDNA-seq, not this method; rejected. |
| https://china2024.gosim.org/speakers/baigui-sun and author's https://sunbaigui.github.io | Biography/FaceChain links; no COT implementation found. |
| Plan §7.1 | siddharthKatageri/COT is explicitly a different work; not used. |

Search failure is bounded to these places, not proof that code does not exist.

## Plan and equation analysis

Binding: experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md §§4.5,
7.1–7.4 and 7.7 (written against v40; rebound to v54 on 2026-11-04 with no change of behaviour: v54 keeps its §7.1 row and the §§4.5, 7.2–7.4, 7.7 rules this record cites). This is NOT reproduction of the image benchmarks. Source
encoder, common attention and common head freeze in phase 2. Only target
EEGNet and auxiliary centers/potentials train. No DANN/MMI addition: the plan
selects the cluster-transport mechanism, not the paper's default host algorithm.

- `losses.py`: squared Euclidean costs `(N,d),(K,d)->(N,K)`; Eq. 7 mean source
  cross-entropy; Eq. 8 nearest-center squared distances summed over bags;
  Eq. 11 `mean(lambda)+mean_v min_u(cost-lambda)`; Eq. 12 weighted objective.
- `model.py`: CommonMILModel subclass, identical inference path; two source
  prototypes are actual frozen head rows `(2,256)`; target centers `(4,256)`;
  potential vector `(4,)`. Windows `(m,C,512)` use existing EEGNet and common
  attention to become bags `(B,256)`. C is 14 or 29, never intersected.
- `training.py`: reuse common phase-1 checkpoint, shared bridge, batch helpers,
  evaluate/BA/label-free metrics, search registration and model_store. Separate
  dual ascent on detached costs then descent on centers/target encoder with
  detached potentials; no labels of target training bags read. Checkpoints
  for primary AND secondary epoch selections retain auxiliary state.

PDF cross-check: `pdftotext -layout` on the CVF PDF beside the Markdown, pp.
19999–20001: Eqs. 7, 8, 10, 11, 12 and Algorithm 1 checked against extracted
PDF text. The PDF exposes embedded LaTeX for these equations. Eq. 9 prints
inconsistent marginals (row 1/K, column 1/|Y|); for K=2|Y| their total masses
disagree. We implement explicit Eq. 11, not an infeasible primal/Sinkhorn.
Algorithm 1's line 6 uses k/(b*10), while its following prose says initial
10% iterations each epoch. We follow that prose and document integer rounding.
No change to the original paper is claimed.

Lineage gap: [16] Gangbo and McCann, The geometry of optimal transportation;
[13] Deng et al., ArcFace; [47] Wang et al., CosFace. Their identities were read
from the PDF bibliography (the Markdown's reference pages 9–10 have no text).
Filename search under guidance/ for arcface/cosface/gangbo and Markdown title
search returned no ingested lineage papers. No lineage theorem is re-proved;
only the explicit finite Eq. 11 is implemented and tested against closed-form/LP
cases. These missing lineage readings bound the audit, not an invented theorem.

Source hashes (SHA256): Markdown
`d6d5c4cd2a3afeed8d7e4c1e9cf9ced5c93679ae473c901af3c250f57482f917`;
PDF `d2044d4b20a6f321505c0be4bbe4e1a8d1b9c969e3f50b8bca20ece021434214`.

## Provenance and choices

| Choice | Value / implementation | Source / verdict |
|---|---|---|
| Encoder | Shared EEGNetTrunk/EEGNetHead, EEGNet-8,2, d=256 | Plan §7.3; controlled deviation from image ResNets. |
| MIL | CommonMILModel tanh attention; bag labels only | Plan §7.2; controlled deviation from image instances. |
| Trainable inference parts | Target encoder only; frozen source, attention, head including BN | Plan §7.3; controlled deviation from jointly learned source classifier. |
| Source prototypes | Actual frozen linear-head rows, not empirical means | Paper §4.1 and plan §7.4. |
| Q, K | Q=2, K=4 for binary labels | Paper §5.3 Table 6; plan §7.4. |
| Distance | Squared Euclidean, unnormalized features and centers | Paper §4.1 Eqs. 8–11. |
| Cluster loss reduction | Sum over target bags, no division by feature dimension | Paper Eq. 8. |
| CE | Mean cross-entropy on source bags, constant during phase 2 | Paper Eq. 7 / plan §7.3. |
| OT objective | Eq. 11 finite min dual, uniform means over centers/classes | Paper Eqs. 10–11; Eq. 9 inconsistency noted above. |
| Combined objective | CE + alphaCluster*cluster + alphaOT*OT | Paper Eq. 12. |
| Alpha defaults | 0.2, 0.2 independently searchable | Paper §5.3 Table 5; plan §7.7 allows separate weights. |
| Primal optimizer | SGD momentum 0.9 | Paper §5.1. |
| Primal learning rate / decay defaults | lr2=0.001, wd2=0, same for centers and target trunk | unspecified; conservative CPU smoke defaults, production search uses shared LR2/WEIGHT_DECAY axes. |
| Phase-one training | Existing common-group source trainer, including its normalized source loss, selected checkpoint and schedule | Plan §7.3; controlled deviation from original image training. |
| Default host | No DANN or MMI | Plan §7.4 controlled mechanism; deviation from paper §5.1 host DANN+MMI. |
| Dual window | First ceil(0.1 * batches) each epoch | Paper Algorithm 1 following prose; rounding unspecified, ensure tiny epochs update. |
| Dual optimizer | SGD ascent, no momentum/decay, default lr 0.1 | unspecified; finite subgradient ascent, separate from primal. |
| Dual steps | 20 per active batch, multiplier {0.5,0.75,1,1.5} | unspecified; finite optimization budget, never claim exact optimum. |
| Potential initialization/gauge | Zero, subtract mean after each dual update | unspecified; Eq. 11 invariant to additive constants. |
| Target-center initialization | Repeat head rows twice plus seeded N(0,0.01²) noise | unspecified; retain source scale while breaking ties. |
| Ties in nearest centers | torch min, first index | unspecified; deterministic subgradient at nondifferentiable ties. |
| Epoch/step budget | Explicit shared schedule, no default epoch cap | Plan §7.3, §7.7; pilot owns budget. |
| Checkpoint | Last epoch of the run (no early stopping, no restore); the search's pooled (trial, epoch) selection breaks ties at the lowest trial and earliest epoch | Plan v54 §§4.5, 7.7 (row corrected 2026-11-04: it described a first-maximum validation checkpoint). |
| Primary/secondary selection | Shared BA and H(mean)-mean H on validation bags, pooled over the three inner splits per epoch by `search` | Plan v54 §4.5 (row corrected 2026-11-04: it named v40's four-rotation `search.robust`). |
| Source training CE batching | Cycle existing interleaved source batches | unspecified; diagnostic constant only, target labels never used. |
| Hyperparameter spaces | Shared lr2/wd2; generic LOSS_WEIGHT for weights/dualLR, KERNEL_MULTIPLIER for dual-step budget | Existing search generic rule; dual axes unspecified mapping. |
| Precision | float32 trunk, float64 representations/centers/potentials | unspecified by COT paper; reuse existing CommonMILModel precision contract. |
| Phase-2 storage | Shared model_store plus own source digest in hyper and load check | Plan §7.3; compensates for shared CODE_FILES excluding comparator code. |

There are **11 unspecified rows**, counted with
`grep '^|' REPLICATION.md | grep -c unspecified`.

## Verification (2026-11-02)

Target: `implementations/Domain_Adaptation-worktrees/cot`, detached at
`e4bf5325dbb1c3755efe4fed667ad428371b87d5`; clean before writes. Its excluded
`.venv` symlinks the target's own venv (torch 2.13.0). Always prefix PYTHONPATH:
the editable installation otherwise resolves to the other checkout. No existing
module, parent comparators/__init__.py, partition, or main checkout was edited.
No third-party code was adopted and no package was installed.

Commands run inside this worktree:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_balladeer_comparator_cot.py -q
# exit 0: 15 passed
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests -q -k balladeer
# exit 0: 265 passed, 2 skipped, 821 deselected (4 Optuna experimental warnings)
PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_balladeer_comparator_cot.py -q -s -k synthetic_two_phase_smoke
# exit 0: 1 passed, 14 deselected
```

The test-first run before implementing losses.py failed collection with
`ModuleNotFoundError: No module named 'MIL_CREDA_EEG.comparators.cot.losses'`.
Invariants cover closed-form costs/CE/gradients, summed clustering loss/scaling,
dual invariance to a constant potential, agreement with an independent scipy
linprog primal (consistent uniform marginals, <0.03 finite-budget dual gap),
detachment in both optimization directions, first-10% schedule, shared inference
and actual classifier-row prototypes, all finite target/center gradients,
bit-identical frozen parameters and buffers, no target-training-label access,
reproducibility, disjoint role guards, primary AND secondary state restoration,
explicit search registration, and shared model-store round-trip with auxiliaries
and own-code digest. Every declared invariant has a test.

Portable smoke test printed seed 710912115, CPU, seeded synthetic BALLADEER
windows (29 source channels to 14 target channels, 512 samples, ragged bags),
phase-one loss `0.03875390134467817`, phase-two losses
`[7.160633956410263, 7.29684490722917, 6.475836533215493]`. One source epoch and
three adaptation steps; loss moved, not monotonically. No real data or toy set
was available in this isolated worktree.

Additional scratch smoke (4 bags/domain, each 3 or 4 windows) ran:

```bash
PYTHONPATH=$PWD/src .venv/bin/python /var/folders/y9/9hfxsk810hv08cnh64bncjq80000gn/T/tmp.FtohdWtpjh/smoke_cot.py
```

Exit 0: CPU, seed 710912115 derived from unit id `COT:synthetic:rotation0`;
one phase-one epoch, three phase-two epochs (six adaptation steps). Phase-two
losses `[6.773286560393651, 5.194294686176067, 4.369120754620813]`, cluster losses
`[30.38093076841052, 22.486014523879682, 18.36019899569471]`, selected epoch 0,
secondary epoch 0, centers `[4,256]`, `frozen_source_head_attention: true`.
Synthetic smoke is execution evidence only, not replicated paper numbers.
Scratch retained at `/var/folders/y9/9hfxsk810hv08cnh64bncjq80000gn/T/tmp.FtohdWtpjh`.

Integration: explicitly call `MIL_CREDA_EEG.comparators.cot.training.register()`
before shared search.run_phase_two_study. Shared search pools the three inner splits
per epoch (v54 §4.5) under common trial budgets. Phase-one records come
from the common group. Shared search.save_phase_two stores the selected full
state; own load_phase_two verifies the extra source digest. The campaign refits the
label-free (trial, epoch) on its own and stores it under a distinct root.
History records already have the dashboard keys; use shared two_phase.write_history
under MIL-CREDA/Results/history*.json at campaign time. No Results tree is written
by these library hooks or synthetic tests.

The shared read-only proposal-implementation probe returned `nextStep: convert`
(inference calls the whole checkout mixed because data/audit files use NumPy).
Its stage reference was read. No conversion of existing modules was undertaken:
this unit is PyTorch already, and changing shared files is outside the edit grant.

### Later parallel run: assessment refused, not a launch decision

The later campaign has multiple transfer/rotation/seed units. Ran the loaded
remote-parallelism skill's `assess --axis fold` without guessed remote inputs
(exit 2), which answered exactly:

> measured inputs are required, never guessed; missing: --units or --units-file, --unit-minutes, --owners, --sessions-per-account, --session-limit-hours, --quota-hours, --setup-minutes

No split verdict can be derived from this CPU synthetic smoke. The handoff owes
a measured pilot/rehearsal, dated quota/session readings and unit list, followed
by assess again. Checklist read via the skill and its `checklist` command:

1. Split by declared units, never arms/comparisons; each campaign unit carries every arm.
2. Seed from `int(sha256(unit_id).hexdigest()[:8],16) % 2**31`; trainer receives this seed (the caller owns unit ids), never seed by account/order.
3. Pin identical published commit and requirements across workers.
4. Request identical accelerator model.
5. Stamp Python, torch, CUDA, cuDNN, accelerator, commit, requirements digest, data checksum/preparation; require identicalAcrossShards and verify returned stamps.
6. torch.manual_seed per unit, deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8, cudnn.benchmark=False; measure their cost.
7. Use identical officially prepared dataset versions.
8. Rehearse each account separately.
9. Never pool time/memory across hardware.
10. One account per person per service.
11. One service per campaign unless identical environment stamps are declared and verified.

No GPU, remote worker, real-data campaign, paper image benchmark, exact dual
convergence for arbitrary costs, or numerical fidelity to reported accuracies was
tested. Full-data hyperparameter selection, placement, and launch belong to later
stretches and the operator, not this implementation. Lineage reading gaps remain
listed above; all explicit operational equations used here were PDF-checked.
