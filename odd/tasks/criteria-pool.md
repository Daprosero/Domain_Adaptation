Objective: random-pool test of label-free phase-2 selection criteria against the oracle, per method, from frozen phase-1 bases.
Branch: Diego. Delivery: ask-on-risk. Test runner: `.venv/bin/python -m pytest tests/`.

## Specs

- S1. Phase 1 frozen: one base per unit (seed 1, nested-searched settings); seeds 2-3 removed ("las que sobran se van para que no exista posibilidad de usarlas").
- S2. Units: 12 dev (d 1.5/2.5 x shift 0.5/1.5/3 x rotation 0/1) + 6 boundary d = 1.0 (same shifts, rotations 0/1).
- S3. Phase 2 for all 8 methods: 100 fixed epochs, lr schedule lr2 * (1 + 10p)^-0.75 (T-GARNet / DANN), no lambda ramp ("la rampa no va pero lo demás si").
- S4. Spaces: every tunable hyperparameter searched. MIL-CREDA: lr2, wd2, lambda_global, lambda_local, tau_local [1e-3, 1] log, gamma [0, 1], tau_att [1, 10] log. Comparators: their own phase-2 space + tau_att [1, 10] log. gamma and tau_local MIL-CREDA only ("gamma y el local se suman a mil creda es propio de nuestro método").
- S5. RandomSampler, 100 trials per method, phase-2 seeds k = 1..5; seed k fixes the draw and the training init; trials sharded by range (4 x 25) with identical results to an unsharded run.
- S6. Per trial: params, BA/AUC on the 120-subject evaluation pool (oracle) and on the 12 target-test subjects, and target-train / source-test outputs (probabilities, representations) so every criterion is computed offline.
- S7. Launch only after smoke, account validation and the operator's explicit OK.

## Tasks

- T1 [S3] lr schedule hook, all 8 trainers — inline — done (5 tests; binds on all 8, CASUAL 2 optimizers)
- T2 [S4-S6] criteria_pool module + tests — inline — done (5 tests; local 100-epoch trial ~157 s across 8 methods)
- T3 [S2] six d = 1.0 bases — inline — done (predict BA .50 in 5 of 6)
- T4 [S5, S7] local smoke, Kaggle smoke, timing — inline — pending

## Log

- L1 (2026-11-15) "listo, me gusta, hacele al plan" — plan as approved in the session (criteria-selection test, Musgrave-style shared random pool).
- L2 Measured: tau_local [0.1, 10] gives a uniform Eq. 34 correspondence (entropy >= .99); range moved to [1e-3, 1].
- L3 Suite: 829 passed, 1 pre-existing failure (test_the_old_creda_comparators_are_gone, from 4a0e6393). Operator authorized commit + push; launch stays pending ("que quede solo el envío").
- L4 (2026-11-16) Kaggle smoke (1 process, os.cpu_count threads): 625 s per 8-method trial set, 4x the Mac. Local throughput (8-method sets/hour): Mac 1x4 23, 4x2 32.7; WSL i7-10750H 1x6 13.7, 3x2 14.0, 6x1 16.5. "probá en kaggle con 4 procesos de 1 hilo" -> `run(procs, threads)`: a spawn pool shares the shard's trials; rows equal the single-process ones (test).
