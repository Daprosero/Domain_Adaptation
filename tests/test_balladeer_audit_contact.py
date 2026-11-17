"""Experiments plan v50, Sec. 2.5: the contact-scale measures only read their inputs."""

from __future__ import annotations

import numpy as np

from MIL_CREDA_EEG import audit_contact

RATE = 128.0


def test_contact_scale_measures_never_alter_the_input():
    rng = np.random.default_rng(7)
    n = int(RATE * 4) * 6
    signal = rng.normal(0.0, 10.0, size=(n, 3))
    signal[5:40, 1] = np.nan
    cq = np.repeat(rng.choice([0.0, 1.0, 2.0, 4.0], size=(6, 3)), int(RATE * 4), axis=0)
    cq[100:200, 2] = np.nan
    cq[600:700, 0] = 2.0
    overall = rng.choice([0.0, 50.0, 100.0], size=n)
    overall[:10] = np.nan
    before = {"signal": signal.copy(), "cq": cq.copy(), "overall": overall.copy()}

    counts = audit_contact.value_counts(cq[:, 0])
    proxies = audit_contact.window_proxies(signal, RATE)
    levels, pure = audit_contact.window_levels(cq, RATE)
    proxies_before = {k: v.copy() for k, v in proxies.items()}
    levels_before, pure_before = levels.copy(), pure.copy()
    rows = audit_contact.calibration_rows(proxies, levels, pure)
    table = audit_contact.overall_vs_channels(cq, overall)

    assert sum(counts.values()) == n
    assert proxies["line_ratio"].shape == (6, 3) and levels.shape == (6, 3)
    assert rows and table
    for name, array in (("signal", signal), ("cq", cq), ("overall", overall)):
        assert array.dtype == before[name].dtype
        np.testing.assert_array_equal(array, before[name])
    for name, values in proxies.items():
        np.testing.assert_array_equal(values, proxies_before[name])
    np.testing.assert_array_equal(levels, levels_before)
    np.testing.assert_array_equal(pure, pure_before)
