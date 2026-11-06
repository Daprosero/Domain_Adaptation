"""Sampling-rate audit, per record and never per subject.

Experiments plan v55, Sec. 2.2. The effective rate is derived from the
timestamps and from the sample counter and compared with what the manufacturer
declares. Two hard defects are declared:

* an effective rate outside the manufacturer's MENU -- membership in a discrete
  set, not in a range: Emotiv declares 128 and 256 samples per second;
* an effective rate different from the record's nominal one, which signals
  packet loss; the counter tells the two apart (jumps mean loss, a contiguous
  counter means a real configuration).

**No tolerance is invented here.** A measured rate is never exactly 128: clock
drift alone moves it by hundredths. So the comparison is categorical. The
effective rate is assigned to the NEAREST menu entry in log-space, and its ratio
to the nominal to the nearest of 1/2, 1 and 2 -- the plan names half-rate loss
as the plausible mechanism. The continuous deviation is reported beside it, so
a reader sees the distribution the category was read from.

The span-derived rate counts holes as missing samples, so one long hole drags
it below the nominal. The median step between consecutive timestamps is
reported beside it: a hole leaves it at the configured rate, a slower
configuration does not. (CGX timestamps arrive in bursts, so its median step
is not a rate; it is reported and not read.)

A legitimate rate inside the menu that differs from the working rate (256 Hz
against the common 128 Hz) is configuration, not defect: Sec. 2.7 resolves it.

The CGX menu is not declared by the plan; the data paper's nominal 500 Hz is
used as a one-entry menu, and said so in every report.
"""

from __future__ import annotations

import numpy as np

from MIL_CREDA_EEG.audit_integrity import counter_modulus, lost_before_each_sample

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v55.md",
    "sections": ["2.2"],
    "experiments": ["2.2"],
    "invariants": [
        "nearest_menu_rate_is_taken_in_log_space",
        "a_rate_half_the_nominal_is_flagged_as_packet_loss",
        "a_nominal_rate_outside_the_menu_is_a_hard_defect",
        "a_legitimate_high_menu_rate_is_configuration_not_defect",
        "a_long_hole_lowers_the_span_rate_but_not_the_step_rate",
    ],
}

#: Emotiv EPOC X / EPOC+ manufacturer menu (plan Sec. 2.2, manuals cited there).
EMOTIV_MENU = (128.0, 256.0)

#: CGX: the plan declares no menu; the data paper's nominal rate stands in.
CGX_MENU = (500.0,)

_RATIO_CLASSES = (0.5, 1.0, 2.0)


def nearest_in_menu(rate: float, menu: tuple[float, ...]) -> float:
    """The menu entry closest to `rate` in log-space."""
    menu = np.asarray(menu, dtype=float)
    return float(menu[np.argmin(np.abs(np.log(menu) - np.log(rate)))])


def effective_rate(timestamps: np.ndarray) -> float:
    span = float(timestamps[-1] - timestamps[0])
    return (len(timestamps) - 1) / span if span > 0 else float("nan")


def audit_rate(timestamps: np.ndarray, counter: np.ndarray | None,
               nominal: float | None, menu: tuple[float, ...]) -> dict:
    """Effective rate, its menu entry, and the two declared hard defects."""
    timestamps = np.asarray(timestamps, dtype=float)
    effective = effective_rate(timestamps)
    span = float(timestamps[-1] - timestamps[0])
    steps = np.diff(timestamps)
    steps = steps[steps > 0]
    median_step_rate = float(1.0 / np.median(steps)) if len(steps) else None
    lost = None
    if counter is not None and np.isfinite(counter).any():
        lost = int(lost_before_each_sample(counter, counter_modulus(counter)).sum())
    corrected = ((len(timestamps) - 1 + lost) / span
                 if lost is not None and span > 0 else None)
    nearest = nearest_in_menu(effective, menu) if np.isfinite(effective) else None
    nominal_in_menu = nominal is not None and float(nominal) in {float(m) for m in menu}
    ratio = effective / nominal if nominal else None
    ratio_class = (nearest_in_menu(ratio, _RATIO_CLASSES)
                   if ratio is not None and np.isfinite(ratio) and ratio > 0 else None)
    nearest_equals_nominal = nearest is not None and nominal is not None \
        and nearest == float(nominal)
    reasons = []
    if not nominal_in_menu:
        reasons.append("nominal_rate_outside_manufacturer_menu")
    if nearest is None or ratio_class != 1.0 or not nearest_equals_nominal:
        reasons.append("effective_rate_differs_from_nominal")
    return {
        "nominal_rate": float(nominal) if nominal is not None else None,
        "menu": [float(m) for m in menu],
        "effective_rate": float(effective),
        "counter_corrected_rate": float(corrected) if corrected is not None else None,
        "median_step_rate": median_step_rate,
        "relative_deviation": (float(effective / nominal - 1.0) if nominal else None),
        "nearest_menu_rate": nearest,
        "nominal_in_menu": bool(nominal_in_menu),
        "nearest_equals_nominal": bool(nearest_equals_nominal),
        "ratio_class": ratio_class,
        "half_of_nominal": ratio_class == 0.5,
        "counter_contiguous": (lost == 0) if lost is not None else None,
        "hard_defect": bool(reasons),
        "hard_defect_reasons": reasons,
    }
