# HASD — BALLADEER controlled variant (not published-result reproduction)

## Code search and route (2026-11-02 UTC)

- Ingested paper, abstract and code-availability statement: links to https://github.com/TumVink/HASD.
- `git ls-remote https://github.com/TumVink/HASD HEAD`: `128c5e156ff48dab962d2af4ebd45326c25bca67`.
- Cloned outside worktree; `git ls-tree -r HEAD`: **only** `README.md` and `docs/HASD_bridge.png`. No implementation, config, tests or license in this tree. No upstream bytes vendored because reuse permission is not stated. `third_party/hasd/PIN.json` records facts only. Operator authorized completing this placeholder by implementing missing method.
- Web searches: `"HASD" "Hierarchical" "github"` and `"Jingsong Liu" "HASD" code`: official placeholder and MICCAI page; no other working release returned.
- Author page https://tumvink.github.io/ and venue https://papers.miccai.org/miccai-2025/0402-Paper1963.html fetched: official code link, not another working release.
- https://arxiv.org/html/2506.23673v2 fetched and compared: confirms all five displays, the unmatched DAS bracket, and the prose saying gamma fixed at 0.001 while the equation uses epsilon for entropy. This ambiguity is not resolved by HTML.

Route: **reimplementation of missing method**, pinned upstream placeholder, no adopted executable code and no adopted license.

Primary source: `guidance/experimental-papers/hasd-2506.23673/hasd-2506.23673.md` (read whole).
Binding protocol: `experiments/experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md`, §§4.5, 7.1–7.4, 7.7. This implements its expressly controlled target-to-source variant, not original HASD inference or reported pathology experiments.
Lineage read: `guidance/introduction/ilse2018/ilse2018.md` whole (ABMIL §§2.4, 4). Needed but not found ingested by filename search and Markdown title search under `guidance/`: Cuturi (2013), Sinkhorn distances, and Peyré/Cuturi (2019), Computational Optimal Transport. KL relaxation conventions below remain explicit assumptions, not claimed lineage-verified details.

## Plan and equation analysis (before implementation)

Existing layout, no shared-file edits:
- `model.py`: native ungated ABMIL, shared EEGNetTrunk and EEGNetHead; source and target independent. Windows per bag `(m,C,512)`, C=14 or 29 -> embeddings `(m,256)` -> native softmax weights `(m,)` -> one bag vector `(256,)` -> common head `(2,)`. Four-phase-group architecture is HASD, not common wrapper.
- `prototypes.py`: label-free deterministic Lloyd clustering of initial target embeddings `(m,256)` -> fixed memberships `(m,)` -> centroids `(min(k,m),256)`. Same memberships before/after adaptation prevent centroid permutation masquerading as geometry change.
- `losses.py`: domain representation display §2.1 is implemented by model.embed (ragged list, no padding). DAS §2.2: current **bag** embeddings `(Bt,256)` and frozen source bag embeddings `(Bs,256)` -> cosine cost `(Bt,Bs)` -> log-domain Sinkhorn plan of same shape -> scalar transport + entropy + both marginal KL terms. SGIR §2.2: lists of initial/current prototype matrices -> sum of unsquared Frobenius norms of Gram differences; PACR §2.2: initial/current softmax attention vectors on corresponding prototypes -> sum of squared differences. Total §2.2: exactly DAS + lambda_sgir SGIR + lambda_pacr PACR.
- `training.py`: shared source-phase trainer, bridge/copy, BagTensor, batching, evaluate, balanced_accuracy, label_free_score, assignment, bag_data, model_store and search hooks reused. Reference target encoder deep-copied **after** source->target compatible-layer bridge (including heterogeneous montage initialization), eval/frozen. Phase 2 fits target trunk only; source/attention/head never change or accumulate BN statistics. Target training labels never read. Validation labels only select epoch; no test data supplied to trainer. Explicit registration avoids import side effects on sibling tests.
- `smoke.py`: CPU synthetic shapes because gitignored BALLADEER data absent from worktree; not evidence of pathology or EEG accuracy.

## Provenance and assumptions

One row per choice. `unspecified` means the paper does not fix it, even if protocol or lineage supplies a defensible choice. No public code to audit against equations.

| Choice | Implementation / value | Source / status |
|---|---|---|
| Domain representation | Ragged list of per-window feature matrices | HASD §2.1 domain display |
| Encoder | Existing EEGNet-8,2 (8,2,16; temporal64, separable16; pools4/8; BN/ELU; dropout .25; spatial maxnorm1) | Controlled deviation from UNI, plan §7.3; reuse `eegnet.py` |
| Head | Existing EEGNetHead linear+softmax once per bag, maxnorm .25, two classes | Controlled deviation from original MLP/pathology tasks, plan §7.3 |
| Native aggregator family | Ungated ABMIL tanh-softmax weighted sum, no MIL-CREDA consensus or l1 constraint | HASD §2.1; Ilse §2.4 Eqs.7–8 |
| Aggregator instantiation/capacity | No bias, width128, float64, Xavier V/w, fixed across trials | unspecified: HASD does not select gated vs ungated or width; Ilse §4 provides width128 and Glorot as a possible lineage choice |
| Direction / transformation | Target EEGNet to frozen source, no extra learned T | Controlled deviation, plan §7.4 (paper moves source to target) |
| DAS unit | Cosine cost between full native aggregated bag vectors | Controlled deviation, plan §7.3 says native losses on bags; paper DAS is on patches/prototypes |
| Cosine convention | Cost 1-cosine, norm epsilon1e-12; zero vector has cosine0 | unspecified: paper gives cosine similarity as example, not precise cost or zero convention |
| Entropic coefficient | Default epsilon=.001; searchable | unspecified: §2.2 prose says gamma=.001, presumed epsilon; HTML repeats ambiguity; no silent correction |
| Entropy definition | Sum gamma*(log gamma-1), minimized | unspecified: H sign/offset absent; choose negative entropy for convex entropic OT |
| Partial relaxation | tau*(KL(rowmass,uniform)+KL(colmass,uniform)); generalized KL incl -x+y | unspecified: paper says marginal KL but supplies no formula or reference measure |
| Partial activation | tau>0 default .1; tau=0 selects balanced solver, not unconstrained zero-penalty transport | unspecified: paper activates partial only under prevalence shift; plan §7.4 mandates partial, no target-label prevalence test |
| Solver | Detached float64 log-space generalized Sinkhorn, max10000, tolerance1e-9; refuse nonconvergence | unspecified: iterations/stopping/gradient convention not stated; envelope gradient of converged optimum |
| Prototype count | k=10 capped by available windows | HASD §2.3, Table2; small-bag cap unspecified, no padding/duplicate windows |
| Clustering convention | Deterministic farthest-first init + Lloyd20; fixed initial target memberships, empty groups omitted; differentiable centroid means | unspecified: clustering algorithm/init/schedule not stated; fixed correspondences make before/after comparison meaningful |
| Prototype scope | SGIR/PACR use prototypes, DAS uses full bag vectors, inference all windows | Controlled deviation for bag-level DAS per plan §7.3; HASD §2.3 clustering retained for regularizers |
| Gram convention | XX^T, no normalization, unsquared Frobenius norm, sum over bags | HASD §2.2 SGIR display |
| PACR convention | Sum squared differences of in-bag normalized attention over corresponding prototypes | HASD §2.2 PACR display; interpreting Atten as distribution from accompanying prose |
| Total | DAS + lambda_sgir*SGIR + lambda_pacr*PACR | HASD §2.2 total display |
| Regularizer weights | Defaults1,1; searched via existing loss-weight rule [1e-3,10] log | unspecified: no lambda values in paper; protocol §7.7 supplies search rule |
| OT search | epsilon,tau via existing loss-weight rule [1e-3,10] log | unspecified: protocol requests OT regularization, exact axes/convention not fixed; bounded numeric coefficients, no new distribution |
| Phase1 optimizer/schedule | Shared Adam/source-train early-stop/source-validation checkpoint restore; lr1/wd1 and epoch cap/patience supplied by caller | Controlled protocol §7.3; no paper optimizer/schedule supplied |
| Phase1 sigma metadata | Shared trainer measures/stores sigma; never read by HASD prediction/loss | unspecified: compatibility metadata from reuse, not an HASD bandwidth or searched axis |
| Phase2 optimizer | Adam target trunk only, standard beta(.9,.999)/eps1e-8, lr2/wd2 supplied | unspecified: paper gives no optimizer; reuse target-only protocol §§7.3–7.4 |
| Phase2 regularizer modes | Current target train BN/dropout; initial reference eval; validation eval | unspecified: paper uses pretrained features, no stochastic EEG semantics; stable frozen reference |
| Batches/schedule | Caller bagsPerStep, epochs2, seed; shared shuffled target / class queues source | unspecified: paper supplies no schedule; protocol pilot decides, no full-run defaults |
| Epoch/config selection | Primary target-validation BA; secondary shared H(mean)-mean(H); same fold/train seed constraints through shared search | Plan §4.5, §7.3; not original pathology evaluation |
| Persistence and assignment | Existing model_store save/load, frozen partition and assignment, no re-split | Plan §§4.2–4.3,7.3; checkpoint retains partition/code metadata |

## Verification record

Tests were written first; the initial focused pytest exited 2 because the package did not yet exist. Final outcomes, measured 2026-11-02 from the worktree root with `PYTHONPATH=$PWD/src`:

- `.venv/bin/python -m pytest tests/test_balladeer_comparator_hasd.py -q` -> 19 passed, exit 0.
- `.venv/bin/python -m pytest tests -q -k balladeer` -> 269 passed, 2 skipped, 821 deselected, exit 0.
- `.venv/bin/python -m MIL_CREDA_EEG.comparators.hasd.smoke` (run from a scratch directory outside the worktree) -> exit 0. Seeded synthetic data, not BALLADEER observations; CPU; seed 20261102; 29 -> 14 channels, 512 samples; 2 phase-1 and 3 phase-2 epochs. Phase-1 loss 0.0378, 0.0369. Phase-2 objective 0.3164, 0.3219, 0.3133 = DAS (0.1950, 0.1961, 0.1959) + 0.001 x SGIR (115.06, 117.37, 111.88) + 0.1 x PACR (0.0638, 0.0834, 0.0551). Predictions finite, shape [4, 2]. This is execution evidence only, not convergence or reproduced performance.
- Every module under `comparators/hasd/` declares `__provenance__`.

Observation for review: unnormalized SGIR (Frobenius norm of Gram differences) is about two orders of magnitude above DAS and PACR on these shapes, so the searched `lambda_sgir` range decides whether SGIR dominates the objective.

Unspecified rows: 14 (rows whose source column begins with `unspecified`).

Not tested: real BALLADEER data, the full search, GPU execution, any comparison with the paper's reported numbers.
