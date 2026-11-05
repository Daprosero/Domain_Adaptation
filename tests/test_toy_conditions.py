"""The toy draw's run-3 conditions: a declared rotation angle and a smooth
subject field along the roll's parameter (`toy_bags.construir`).

Both are off by default, and off is the draw exactly as it was: the golden
hashes are the ones `tests/test_toy_subject_shift.py` pins, taken before any
option existed.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from CREDA.toy_experiment import DURAS, perturbar
from MIL_CREDA_Benchmark import toy_bags, toy_run

GOLDEN = {
    "rollo": "179de3633815a4c5add08eca1618a9344a12f8a9cd98f54a9f7df1187210a376",
    "toroide": "5a6a4452a8df7e982656ca5586ff120044a581dec3acfdb92d60ce4fc32e07d1",
    "lunas": "bf9707fe85fd9616601e2fc2c002875ca95c71daf313083eb3e864e7c94c864b",
}
FIELD = 0.45


def _digest(f) -> str:
    h = hashlib.sha256()
    for a in (f.Xs, f.ys, f.bs, f.Xt, f.yt, f.bt, f.Xt_crudo):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("family", sorted(GOLDEN))
def test_the_new_options_at_their_off_values_leave_the_draw_unchanged(family):
    f = toy_run.build_family(family, 10, toy_run.FULL, rotation_degrees=None,
                             subject_field=0.0)
    assert _digest(f) == GOLDEN[family]
    assert f.campo is None


def test_a_declared_rotation_is_perturbar_at_that_angle():
    f = toy_run.build_family("rollo", 10, toy_run.FULL, rotation_degrees=25.0)
    base = toy_run.build_family("rollo", 10, toy_run.FULL)
    for name in ("Xs", "ys", "bs", "yt", "bt", "Xt_crudo"):
        assert np.array_equal(getattr(f, name), getattr(base, name))
    rng = np.random.default_rng(10)
    fn = dict(DURAS)["rollo"]
    fn(toy_run.FULL.points_per_domain, rng)            # the source draw
    expected = perturbar(base.Xs, base.Xt_crudo, rng, grados=25.0)
    assert np.allclose(f.Xt, expected)
    assert not np.allclose(f.Xt, base.Xt)


@pytest.fixture(scope="module")
def raw():
    f = toy_run.build_family("rollo", 10, toy_run.FULL)
    return f


def test_the_subject_field_is_smooth_and_invertible_along_t():
    width = 0.08
    t = np.linspace(1.5 * np.pi, 4.5 * np.pi, 20001)
    for phase in (0.0, 1.3, 4.0):
        moved = toy_bags.t_bajo_campo(t, FIELD * width, toy_bags.PERIODOS_DEL_CAMPO, phase)
        assert np.all(np.diff(moved) > 0)                        # invertible
        shift = moved - t
        assert np.max(np.abs(shift)) <= FIELD * width + 1e-12    # bounded
        assert np.max(np.abs(np.diff(shift))) < 1e-3              # smooth on the grid


def test_the_subject_field_moves_points_along_the_roll_only(raw):
    X = raw.Xt_crudo
    moved = toy_bags.campo_de_sujeto(X, 0.04, toy_bags.PERIODOS_DEL_CAMPO, 0.7)
    t0, t1 = toy_bags.parametro("rollo", X), toy_bags.parametro("rollo", moved)
    assert np.allclose(moved[:, 1], X[:, 1])                    # height untouched
    assert np.allclose(moved[:, 0], t1 * np.cos(t1))            # still on the roll
    assert np.allclose(moved[:, 2], t1 * np.sin(t1))
    assert np.max(np.abs(t1 - t0)) <= 0.04 + 1e-12


def test_below_half_a_band_the_field_keeps_every_band_nearest_its_own_place():
    """The field's own guarantee, on ONE draw so the two domains' independent
    quantile jitter cannot enter: every band slid by the field is still
    nearest (in median `t`) to where it was, among its class's bands."""
    on = toy_run.build_family("rollo", 10, toy_run.FULL, subject_field=FIELD)
    record = on.campo
    assert record["fraccionDeBanda"] == FIELD
    assert record["amplitudT"] == pytest.approx(FIELD * record["espaciadoMinimoT"])
    shifts = []
    for X, y, b in ((on.Xs, on.ys, on.bs), (on.Xt_crudo, on.yt, on.bt)):
        t = toy_bags.parametro("rollo", X)
        moved = toy_bags.t_bajo_campo(t, record["amplitudT"], record["periodos"], record["fase"])
        for c in np.unique(y):
            bands = np.unique(b[y == c])
            centres = np.array([np.median(t[(y == c) & (b == k)]) for k in bands])
            for k in bands:
                m = (y == c) & (b == k)
                assert bands[np.argmin(np.abs(centres - np.median(moved[m])))] == k
                shifts.append(np.median(moved[m] - t[m]))
    # Bands move differently: shifts of both signs, spread over the amplitude.
    assert min(shifts) < 0 < max(shifts)
    assert np.std(shifts) > 0.3 * record["amplitudT"]


def test_the_target_record_carries_the_field_it_was_drawn_with():
    on = toy_run.build_family("rollo", 10, toy_run.FULL, subject_field=FIELD)
    off = toy_run.build_family("rollo", 10, toy_run.FULL)
    assert np.array_equal(on.Xt_crudo, off.Xt_crudo) and np.array_equal(on.bt, off.bt)
    assert not np.allclose(on.Xt, off.Xt)


def test_the_subject_field_is_refused_off_the_roll():
    with pytest.raises(ValueError):
        toy_run.build_family("toroide", 10, toy_run.FULL, subject_field=FIELD)
