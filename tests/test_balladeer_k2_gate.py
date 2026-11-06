"""Sec. 6.2/6.3 (v56): the K2 mechanism diagnostics and the reading gate.

The probe tests are read only where the local term had room: median entropy
ratio < 0.9 OR median room ratio >= 1.0 (plan v56, "Condición de lectura").
Otherwise each K2 criterion row carries the verdict "no concluyente (el
mecanismo no tuvo margen)" and an `inconclusive_no_room` flag -- never "not
supported" -- and the four tests are still computed and reported with Holm
over four. Sec. 6.3's "both terms vs global-only" row carries the same flag
for a core pair only when NEITHER direction meets the gate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from MIL_CREDA_EEG import mechanisms as mech
from MIL_CREDA_Benchmark.poc_math import entropy_ratio


def test_entropy_ratio_is_one_uniform_and_zero_one_hot():
    assert entropy_ratio(torch.full((5,), 0.2)) == pytest.approx(1.0)
    one_hot = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert entropy_ratio(one_hot) == pytest.approx(0.0)
    assert np.isnan(entropy_ratio(torch.tensor([1.0])))


def _constructed(tau_loc=0.01):
    # 4 source bags: classes [0, 0, 1, 1]; 2 target bags of classes [0, 1].
    source_labels = np.array([0, 0, 1, 1])
    target_classes = np.array([0, 1])
    own_index = np.array([0, 2])  # target j's own source bag
    # Class 0 kernels equal -> uniform pi -> entropy ratio 1.
    # Class 1 one kernel dominates -> near one-hot at a small tau -> ratio ~0.
    K_st = np.array([[1.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 10.0, 0.0]])
    # Start distances: own is nearest in H; spread is the same-class source spread.
    D_start = np.array([[0.2, 0.9, 5.0, 5.0],
                        [5.0, 5.0, 0.4, 0.9]])
    D_start_ss = np.array([[0.0, 1.0, 9.0, 9.0],
                           [1.0, 0.0, 9.0, 9.0],
                           [9.0, 9.0, 0.0, 2.0],
                           [9.0, 9.0, 2.0, 0.0]])
    return dict(K_st=K_st, source_labels=source_labels, target_classes=target_classes,
                own_index=own_index, D_start=D_start, D_start_ss=D_start_ss, tau_loc=tau_loc)


def test_transfer_diagnostics_in_range_on_the_constructed_case():
    d = mech.k2_transfer_diagnostics(transfer="L1->L6", **_constructed())
    # uniform (class 0) and near one-hot (class 1): median of ~1 and ~0.
    assert d["median_entropy_ratio"] == pytest.approx(0.5, abs=0.05)
    # each target's own source bag is its kernel-nearest same-class source bag.
    assert d["fraction_own_nearest"] == pytest.approx(1.0)
    # own distance 0.2 over spread 1.0, and 0.4 over 2.0 -> median room 0.2.
    assert d["median_room_ratio"] == pytest.approx(0.2, abs=1e-9)
    assert d["probe_subjects"] == 2


def test_push_with_and_without_the_local_term():
    args = _constructed()
    D_end_full = args["D_start"] - 0.1          # local term pushed own closer
    D_end_no_local = args["D_start"] - 0.02
    d = mech.k2_transfer_diagnostics(transfer="L1->L6", D_end_full=D_end_full,
                                     D_end_no_local=D_end_no_local, **args)
    assert d["push_with_local"] == pytest.approx(0.1, abs=1e-9)
    assert d["push_without_local"] == pytest.approx(0.02, abs=1e-9)


def test_the_reading_gate():
    assert mech.k2_has_room(0.5, 0.5) is True       # low entropy -> read
    assert mech.k2_has_room(0.95, 1.2) is True       # enough room -> read
    assert mech.k2_has_room(0.95, 0.5) is False      # neither -> inconclusive


def _separation_rows(transfer):
    return [{"model": "MIL-CREDA", "transfer": transfer, "subject": f"{transfer}-S{i}",
             "separation": 0.1 * (i + 1)} for i in range(8)]


def test_gate_met_leaves_no_flag_and_gate_failed_marks_inconclusive():
    rows = _separation_rows("L1->L6") + _separation_rows("L6->Robots")
    weaker = [{**r, "model": mech.K2_NO_LOCAL, "separation": r["separation"] / 2} for r in rows]
    diagnostics = {"L1->L6": {"transfer": "L1->L6", "median_entropy_ratio": 0.5,
                              "median_room_ratio": 0.5},          # read
                   "L6->Robots": {"transfer": "L6->Robots", "median_entropy_ratio": 0.95,
                                  "median_room_ratio": 0.5}}      # inconclusive
    out = mech.k2_tests(rows, weaker, diagnostics=diagnostics)
    # all four still computed and Holm'd over four
    assert len(out) == 4 and all(r["p_value"] is not None for r in out)
    assert all("m=4" in r["correction"] for r in out)
    read = [r for r in out if r["transfer"] == "L1->L6"]
    inconclusive = [r for r in out if r["transfer"] == "L6->Robots"]
    assert all(r["inconclusive_no_room"] is False and "verdict" not in r for r in read)
    assert all(r["inconclusive_no_room"] is True for r in inconclusive)
    assert all(r["verdict"] == mech.K2_INCONCLUSIVE for r in inconclusive)
    assert all(r["verdict"] != "not supported" for r in inconclusive)


def test_diagnostics_table_carries_the_declared_columns():
    diagnostics = [{"transfer": "L1->L6", "probe_subjects": 7, "median_entropy_ratio": 0.4,
                    "fraction_own_nearest": 0.7, "median_room_ratio": 1.3,
                    "push_with_local": 0.12, "push_without_local": 0.05},
                   {"transfer": "L6->Robots", "probe_subjects": 5, "median_entropy_ratio": 0.95,
                    "fraction_own_nearest": 0.4, "median_room_ratio": 0.5,
                    "push_with_local": 0.01, "push_without_local": 0.02}]
    table = mech.k2_diagnostics_table(diagnostics)
    assert all(list(r) == list(mech.K2_DIAGNOSTICS_COLUMNS) for r in table)
    read, inconc = table
    assert read["Condición de lectura"] != mech.K2_INCONCLUSIVE   # low entropy + room
    assert inconc["Condición de lectura"] == mech.K2_INCONCLUSIVE


def test_section_6_3_flag_only_when_both_directions_fail():
    both_fail = [{"transfer": "a->b", "median_entropy_ratio": 0.95, "median_room_ratio": 0.5},
                 {"transfer": "b->a", "median_entropy_ratio": 0.92, "median_room_ratio": 0.3}]
    one_reads = [both_fail[0],
                 {"transfer": "b->a", "median_entropy_ratio": 0.5, "median_room_ratio": 0.5}]
    assert mech.both_vs_global_only_inconclusive(both_fail) is True
    assert mech.both_vs_global_only_inconclusive(one_reads) is False
