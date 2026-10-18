# Official comparator losses

## Objective
Bring the comparator losses chosen in `MIL-CREDA/embudo-mil-creda.html` into the repo, copied from each paper's own GitHub repository and cited, so they can be compared against CREDA first (non-MIL), later on bags.

## Scope
- New package `src/CREDA/comparators/` only. No edits to existing modules (`src/CREDA/*.py` carry uncommitted work from another session).
- Verbatim copy of the upstream core-loss code, with a header citing paper, repo, commit SHA and upstream path. A thin adapter per method exposes a common call on (f_s, f_t, g_s, g_t, y_s).
- PDFs downloaded and ingested only in the session scratchpad, deleted afterwards.

## Sources (verified 2026-10-12)
| Method | Repo | Upstream path |
|---|---|---|
| SHOT | tim-learn/SHOT | object/image_target.py (IM loss, obtain_label), object/loss.py |
| CDAN+E | thuml/CDAN | pytorch/loss.py (CDAN, Entropy), network.py (AdversarialNetwork, grl) |
| DSAN/LMMD | easezyc/deep-transfer-learning | UDA/pytorch1.0/DSAN/lmmd.py |
| CS-DA | ywzcode/CS-adv | CSutils.py (CondCSD, CS, GaussianMatrix) |
| PFAN | Xiewp/PFAN | model.py (protoloss, adloss), pseudo.py — TF1, must be translated |
| JUMBOT | kilianFatras/JUMBOT | digits/jumbot.py |

## Tasks
- [x] T1 Vendor the seven upstream loss sources with citation headers (route: delegated writer; trigger: 2+ non-trivial files).
- [x] T2 Adapters with a common signature + smoke tests (finite loss, gradient reaches f_t).
- [x] T3 Resolve ambiguities against the PDFs: PFAN L143 centroid update, DSAN class mask, CDAN+E weight normalisation, CS-DA extra marginal term.
- [x] T4 Delete scratchpad clones and PDFs (done 2026-10-12, scratchpad empty).

## Constraints
- Do not branch or commit: the checkout is shared with another session with uncommitted work; the operator decides commits.

## Progress
### 2026-10-12 — T1-T3 (route: delegated writer; trigger: 2+ non-trivial files)
Upstream clones (`git clone --depth 1`, scratchpad only) and HEAD SHAs:
SHOT f7d555a0d53b525b885e5ef2a887a267a5be3c36; CDAN c904cd3f2fe092bb5175be1c5b6e42fa2036dece;
deep-transfer-learning 2969ed6c740d04283aa54af4e2c267cb980c96fe; CS-adv 79d010b5e125bf4558d9318bdcd633937e0c9274;
PFAN 328c4c5b4e24a6d04b06136d9e83d8acb391a289; JUMBOT af9a22d800cf9b9c1243ab125314eefaf06fdda8;
MS-DA-MIL-CNN 9e6754525ac9a6e2aab2f6665fa700b57066dc61.

Package: `src/CREDA/comparators/` — `<method>/upstream_*.py` (verbatim, citation header, `# LOCAL EDIT:` markers),
`<method>/adapter.py`, `pfan/upstream/*.txt` (TF1 reference, never imported) + `pfan/pfan_torch.py` (translation),
`CITATIONS.md`, registry `REGISTRY`/`build()` in `__init__.py`. Inline-loop code (SHOT IM loss, JUMBOT cost, DANN-MIL
lambda schedule) is wrapped in a `def` with the upstream lines byte-identical, indentation included.

LOCAL EDITs: `.cuda()` -> device-agnostic (SHOT L197, L250; CDAN loss.py L31; lmmd.py L37-39, L43; jumbot.py L102);
`iter_test.next()` -> `next(iter_test)` (SHOT L247); `np.float` -> `float` (CDAN network.py L11, removed in NumPy 1.24);
package-relative imports (SHOT L10, model.py L4); dropped `from utils import model_eval` (jumbot.py L23); wrapper defs/returns.

Packages: `pot` 0.9.7.post1 installed into `.venv` (JUMBOT; upstream pinned POT 0.7.0).

Verification: RED observed first (collection ImportError, exit 2). GREEN: `pytest tests/test_comparators_official.py -q`
exit 0, 16 passed. Mutating pfan_torch.py back to the upstream L143 form fails the deviation lock (1 failed), restored ->
passes. `pytest -q -x --co tests` exit 0, 1262 collected.

#### Ambiguity verdicts (T3; PDFs from arXiv read in scratchpad only)
- **PFAN L143** — RESOLVED as instructed. Paper Sec. 3.3 Eq. 8 / Alg. 1 l.12 updates c^T from c^T_{k(I-1)}; model.py
  keeps `target_moving_centroid` (L14, assigned L195) but L143 reads the source one. Translation uses the commented L142
  form; locked by `test_pfan_target_centroid_uses_target_moving_centroid`. RECORDED, not resolved: paper Eq. 7-8 uses
  the running mean of local prototypes and a cosine-adaptive rho^2, the code a fixed decay 0.3 (kept); paper Eq. 9 sums
  over classes, code L5 means over CxD (kept); paper EHTS thresholds tau (Eq. 3-4), code keeps top-3 per class (kept).
  Also: train.py L215-218 builds `dic_t` from TRUE target labels (`batch_ty`); written to a file, not used by the loss.
- **DSAN class mask** — code DEVIATES from the paper. Paper Eq. 6/8 average over all C classes with factor 1/C and no
  mask. lmmd.py L70-86 zeroes every class not present in BOTH the source labels and the target argmax, then divides by
  the number of shared classes. Identical to the paper only when every class appears on both sides of the batch.
  Kept verbatim. Also: requires n_s == n_t (mask broadcast at L73); bandwidth uses the detached mean distance.
- **CDAN+E weight normalisation** — code DEVIATES. Paper Eq. 10: expectations of w(H)=1+e^{-H} times the log-loss,
  unnormalised. loss.py L33-43 normalises the weights to sum 1 WITHIN each domain, adds them, and divides by their total
  (=2), i.e. the mean of two self-normalised per-domain weighted averages; the entropy also receives a GRL-reversed
  gradient via `entropy.register_hook(grl_hook(coeff))` (L33). Kept verbatim.
- **CS-DA marginal term** — paper INCLUDES it. Eq. 15: L_cls + lambda D_CS(p_s(z),p_t(z)) + beta sum_n D_CCS, with
  lambda = beta = 1 (Sec. 4.2), matching the unweighted `CS` at train script L296. Adapter `include_marginal=True`
  by default, `False` gives the conditional term alone. Recorded: the script's Step C (L372-378) also adds `CS_loss`
  where paper Eq. 18 has D_CCS only; out of the adapter's scope (MCD two-classifier step).

#### Adapter decisions to review
- DANN-MIL needs MIL attention (1, n_s+n_t); the adapter raises without it (uniform attention gives a zero reversed
  gradient, upstream semantics). Non-MIL use needs an operator decision on the attention source.
- DANN-MIL `DomainPredictor` first layer resized from 25088 (VGG16) to `feature_dim`; widths 4096/1024 kept.
- Upstream bug kept verbatim: `ClassPredictor.forward` calls `self.attention` (defined as `attention_mechanism`).
- JUMBOT defaults are the class signature (eta1=0.001, eta2=0.0001); digits/train.py L84-87 uses eta1=eta2=0.1.
- CDAN+E and LMMD refuse n_s != n_t (ValueError), as upstream's batch split requires.
- Not zero on identical batches by construction (not asserted): JUMBOT, CDAN+E, DANN-MIL, SHOT.

Next: T4 (parent deletes scratchpad `upstream/`, `pdfs/`, `build_vendor.py`, logs). Commit is the operator's decision.

### 2026-10-12 — Scope correction (operator)
Scope is the non-MIL comparison against CREDA only. DANN-MIL (Hashimoto, MIL bags) was out of scope and was removed:
package `dann_mil/`, registry entry, `__all__`, CITATIONS row and BibTeX, test helper. Six comparators remain
(shot, cdan_e, dsan_lmmd, cs_da, pfan, jumbot). `pytest tests/test_comparators_official.py -q`: exit 0, 15 passed.
Nothing extends to MIL until the operator asks.
