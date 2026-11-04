"""Experiments plan v50, Sec. 4.6: the expected precision of the core-pair contrasts."""

from __future__ import annotations

import pytest

from MIL_CREDA_EEG import statistics as st


def _points(values):
    return [round(100 * v, 1) for v in values]


@pytest.mark.parametrize("controls, cases, half_widths, detectable", [
    (45, 82, [8.9, 8.3, 7.3], [8.2, 11.6, 14.2]),       # level L1<->L11 and device L6<->Robots
    (33, 33, [11.8, 11.1, 9.7], [10.9, 15.4, 18.9]),    # task CogniFit<->Robots
])
def test_expected_precision_reproduces_the_plan_numbers(controls, cases, half_widths,
                                                        detectable):
    out = st.expected_precision(controls, cases)
    assert out["ba_levels"] == [0.6, 0.7, 0.8]
    assert out["discordances"] == [0.1, 0.2, 0.3]
    assert _points(out["ci_half_width"]) == half_widths
    assert _points(out["minimum_detectable_difference"]) == detectable
    assert out["alpha"] == 0.05 and out["power"] == 0.8 and out["holm"] is False
    assert out["n_controls"] == controls and out["n_cases"] == cases


def test_expected_precision_refuses_an_empty_class():
    with pytest.raises(st.StatisticsError, match="class"):
        st.expected_precision(0, 10)
