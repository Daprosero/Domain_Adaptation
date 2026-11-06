# MIL-CREDA-EEG: replication index of the comparison methods

The eight comparators of the full BALLADEER campaign (experiments plan v58, Sec. 7.1),
one row each. Every row only points at the comparator's own `REPLICATION.md`: the route
is quoted from that file, the bound revision is the `__provenance__` revision of the
comparator's modules, and the `unspecified` count is the number of table rows in that
file marking an item `unspecified` (a cell that begins with the word, in any column).
Nothing here is a new claim; read each file for the correspondence table itself.

| Comparator | Record | Route, as its file states it | Bound revision | `unspecified` rows |
|---|---|---|---|---|
| AdaptMIL (AdaptMIL-LMMD) | [REPLICATION.md](../src/MIL_CREDA_EEG/comparators/adaptmil/REPLICATION.md) | "reimplementation of the ABMIL mechanism with the LMMD objective ... **controlled variant**, not a replication of published results. No upstream AdaptMIL repository was adopted" | `experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md` | 14 |
| CASUAL | [REPLICATION.md](../src/MIL_CREDA_EEG/comparators/casual/REPLICATION.md) | "reimplementation. No adopted repository, commit or license; no third-party code copied." | `experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md` | 11 |
| CCA-LSC | [REPLICATION.md](../src/MIL_CREDA_EEG/comparators/cca_lsc/REPLICATION.md) | "paper-based reimplementation, not adopted code." | `experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md` | 11 |
| CLUDA | [REPLICATION.md](../src/MIL_CREDA_EEG/comparators/cluda/REPLICATION.md) | "independent controlled-variant implementation; read-only upstream audit, no upstream source vendored, no upstream patches." | `experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md` | 6 |
| CLUST, source-free variant | [REPLICATION.md](../src/MIL_CREDA_EEG/comparators/clust/REPLICATION.md) | "reimplementation; no usable CLUST implementation located in the searches below. No adopted repository, commit, license or vendored source." | `experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md` | 15 |
| COT | [REPLICATION.md](../src/MIL_CREDA_EEG/comparators/cot/REPLICATION.md) | "Reimplementation; no upstream repository adopted, commit/license not applicable." | `experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md` | 8 |
| CS+CCS | [REPLICATION.md](../src/MIL_CREDA_EEG/comparators/cs_ccs/REPLICATION.md) | "reimplementation from the paper. The repository has no license, so it is not vendored and no line is copied" | `experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md` | 8 |
| UDA-DDA | [REPLICATION.md](../src/MIL_CREDA_EEG/comparators/uda_dda/REPLICATION.md) | "paper-based reimplementation. Official code exists but carries no license, so nothing was adopted, vendored or copied" | `experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md` | 11 |

None of the eight adopts upstream code: all are reimplementations or controlled variants
under the common bag-level protocol of Secs. 7.2-7.4.
