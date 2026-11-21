Objective: one notebook per paper stage that regenerates the paper's inputs and results from the public Kaggle data, plus a Results_Paper notebook that rebuilds every figure and table from the stored summaries; BALLADEER removed; history slimmed.
Branch: repro-notebooks (merged to Diego and main at the end). Delivery: ask-on-risk. Test runner: `.venv/bin/python -m pytest tests/`.

## Specs

- S1. Remove BALLADEER: "hay que quitarlo junto con esa base de datos porque no apica para nuestro paper, y reduciomos espacio, se va todo lo realcionado." This covers the notebook, the BALLADEER-only modules and tests, the code paths that only BALLADEER uses, the `balladeer-existing-local` Kaggle mount, and finally the local `MIL-CREDA-EEG/Data/` (73 GB, irreversible, done last).
- S2. EEG usage example: "debe existir un ejemplo de uso de milcreda en eeg así como esta en el de sinteticos de toyset".
- S3. Toyset: "se debe actulizar ese cuaderno a la ultima versión de milcreda y que sea funcional" (MIL-CREDA/Notebooks/Toyset.ipynb).
- S4. Stage notebooks: "cuadernos que producen los resultados y los deja en results, como la base de datos en kaggle es publica, deben los cuadernos conectar con esa db, y mostrar en ese cuaderno como se produjo los resultados". One notebook per stage: "no realices reparto deja un solo cuaderno por etapa o resultado o sección". Each takes `SPLIT = None` (everything in one account) or a portion `(k, n)`, plus a merge cell.
- S5. Reproducibility: same parameters as the paper runs; "no exactos, puede variar, en lo posible ojala sean los mismos". Each stage notebook ends with a cell that compares its numbers with the paper's.
- S6. Results_Paper: "ese ultimo lo llamaría results paper, porque genera con lo gaurdao los resultados de lo que hay hoy en el paper". It rebuilds every paper figure and table from tracked summaries, with no training.
- S7. GitHub without LFS: "solo se sube lo github permita sin usar ls", "garantizando que los cuaderno produzcna los insumos que necesiten los cuadenros para producir los results". No tracked file over 50 MB.
- S8. README: explains which notebook to run per result, that the paper ran over several Kaggle accounts, and how to split (`SPLIT`) or run everything in one account.
- S9. History rewrite to slim clones, approved by the user ("si"): filter-repo plus force-push to main; preserve the collaborator's uncommitted changes in the working copy.

## Tasks

- T1 [S4, S7] repro.py + IEEE manifest + find_ieee fallback — inline — done (f73f592 before rewrite; 11 tests)
- T2 [S3] Toyset notebook — delegated (parallel) — done (6.1 min headless; 15 tutorial tests)
- T3 [S2] 00_Tutorial_EEG — delegated (parallel) — done (6.5 min headless; ADHD pick reproduces BA 0.833 / AUC 0.972)
- T4 [S4, S5] 01_Phase1_Bases — delegated (parallel) — done (4/4 paper numbers)
- T5 [S4, S5] 02_Synthetic_Criteria_Pool — delegated (parallel) — done (32/32)
- T6 [S4, S5] 03_Synthetic_Mechanisms — delegated (parallel) — done (98/98; paper prints 0.008 where the stored Holm p is 0.00749)
- T7 [S4, S5] 04_ADHD_Pool — delegated (parallel) — done (95/95)
- T8 [S4, S5] 05_Attention_Influence — delegated (parallel) — done (32/32; pair IEEE-v236 -> v265)
- T9 [S4, S5] 06_ADHD_Graph_Analysis — delegated (parallel) — done (21/21; full run 5 min locally)
- T10 [S6] Results_Paper — inline — done (282/282 in 15 s)
- T11 [S8] README + requirements — inline — done (4bf2085)
- T12 [S1] BALLADEER removal — inline — done (f340693: 21 modules, 17 tests, tools, 1165 job folders, notebook); manifests restored to the paper's (00b9c0c)
- T13 [S9] history rewrite + force-push — inline — done (pack 709 MB -> 161 MB; main/Diego/tag trees identical; 2 empty commits dropped)
- T14 [S1] delete local Data/ — inline — done (73 GB)

## Log

- L1 (2026-11-21) User request, verbatim: "listo, entonces, hay que quitarlo junto con esa base de datos porque no apica para nuestro paper, y reduciomos espacio, se va todo lo realcionado. despues para mi debe existir un ejemplo de uso de milcreda en eeg así como esta en el de sinteticos de toyset(se debe actulizar ese cuaderno a la ultima versión de milcreda y que sea funcional), despues debe cuadernos que producen los resultados y los deja en results, como la base de datos en kaggle es publica, deben los cuadernos conectar con esa db, y mostrar en ese cuaderno como se produjo los resultados. primero enuemra los cuadernos que deben crearse para que realemnte sea reproducible el paper, guardadno en resuts las graficas y tablas, modelos y demaás, pero solo se sube lo github permita sin usar ls o como se llama lo de archivos grandes, pero garantizando que los cuaderno produzcna los insumos que necesiten los cuadenros para producir los results, dejanod los que ya están en el paper, pero que si alguien quiere reproducir, sepa segun el redme que debe correr un cuaderno, no realices reparto deja un solo cuaderno por etapa o resultado o sección dejando en readme que eso se corrio en varias cunetas y que se debe hacer un script de envios o divisiones y como hacerlos, o hacer los cuadernos con split=None (corre todo en un cuaderno/cuenta) o porciones, es unda idea pero primero necesito mirar tu propuesta"
- L2 (2026-11-21) "ese ultimo lo llamaría results paper, porque genera con lo gaurdao los resultados de lo que hay hoy en el paper. los demás desde que garanticen que se reproduzcan los resultados (no exactos, puede variar, en lo posible ojala sean los mismos ) esta bien, es decir, mismos parametros, etc."
- L3 (2026-11-21) History rewrite approved: "si".
- L4 Evidence: daprosero/mi-tdah-dataset is public (CC0, v5); IEEE ADHD files download one by one with kagglehub, about 33 MB in total. crit3.py is in Results/v77-pool24 and v74-criteria-pool; xgb84.py is in Results/v84-gsa.
- L5 (2026-11-21) Evidence: the tracked Manifests were an older set; restored from archive/ieee-pool-slim (34/34 bases byte-identical to the Kaggle clones). Full suite after the removal: the only failures were tests of deleted modules and tests reading an untracked job folder, fixed with tests/fixtures. History rewritten with git filter-repo --strip-blobs-with-ids (92 blobs, 339 MB, none in main or in the archive tags' tips) and force-pushed; local branches reset with `git reset --mixed`, the collaborator's three modified files unchanged (md5).
- L6 Open: main.tex:402 prints p_Holm 0.008 for the mean aggregation (stored 0.00749 -> 0.007); BALLADEER leftovers on disk that the collaborator's .gitignore keeps on purpose: Models/Balladeer (442 MB), Results/launch_v59 (7.8 GB), Results/hpo/*.db; three detached worktrees (band-row, criteria-pool, oracle-screen) keep the old history alive locally (pack 334 MB).
