"""The per-pair (subject-like) target shift of `toy_bags.construir`.

Off by default, and off must be the draw as it was: the golden hashes below
were taken from `toy_run.build_family(family, 10, toy_run.FULL)` before the
option existed. On, every (class, band) pair of the target gets its own
rotation and translation, while source bag k and target bag k keep their
band, so they stay partners.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from CREDA.toy_experiment import ANGULO_SIGMA
from MIL_CREDA_Benchmark import toy_bags, toy_run

GOLDEN = {
    "rollo": "179de3633815a4c5add08eca1618a9344a12f8a9cd98f54a9f7df1187210a376",
    "toroide": "5a6a4452a8df7e982656ca5586ff120044a581dec3acfdb92d60ce4fc32e07d1",
    "lunas": "bf9707fe85fd9616601e2fc2c002875ca95c71daf313083eb3e864e7c94c864b",
}


def _digest(f: toy_bags.Familia) -> str:
    h = hashlib.sha256()
    for a in (f.Xs, f.ys, f.bs, f.Xt, f.yt, f.bt, f.Xt_crudo):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("family", sorted(GOLDEN))
def test_the_option_off_is_byte_identical_to_the_draw_before_it(family):
    assert _digest(toy_run.build_family(family, 10, toy_run.FULL)) == GOLDEN[family]
    off = toy_run.build_family(family, 10, toy_run.FULL, per_pair_shift=0.0)
    assert _digest(off) == GOLDEN[family]
    assert off.desplazamientos is None


@pytest.fixture(scope="module")
def pair():
    off = toy_run.build_family("rollo", 10, toy_run.FULL)
    on = toy_run.build_family("rollo", 10, toy_run.FULL,
                              per_pair_shift=toy_bags.ESCALA_POR_PAR_DECLARADA)
    return off, on


def test_on_only_the_target_points_move(pair):
    off, on = pair
    for name in ("Xs", "ys", "bs", "yt", "bt", "Xt_crudo"):
        assert np.array_equal(getattr(off, name), getattr(on, name)), name
    assert not np.allclose(off.Xt, on.Xt)


def test_on_partners_keep_their_band_and_stay_near_their_region(pair):
    off, on = pair
    record = on.desplazamientos
    esc = float(off.Xs.std(0).mean())
    for c in np.unique(on.yt):
        for k in np.unique(on.bt):
            m = (on.yt == c) & (on.bt == k)
            moved = np.linalg.norm(on.Xt[m].mean(0) - off.Xt[m].mean(0))
            # The centroid moves by the pair's own translation only (the
            # rotation is about the centroid), bounded by a few declared scales.
            expected = np.linalg.norm(record[(int(c), int(k))]["traslacion"])
            assert moved == pytest.approx(expected, abs=1e-9)
            assert moved < 4 * toy_bags.ESCALA_POR_PAR_DECLARADA * esc


def test_on_every_pair_gets_its_own_rotation_and_translation(pair):
    _, on = pair
    record = on.desplazamientos
    assert len(record) == 3 * toy_run.FULL.bags_per_class
    translations = np.array([r["traslacion"] for r in record.values()])
    angles = np.array([r["angulo"] for r in record.values()])
    assert len({tuple(np.round(t, 12)) for t in translations}) == len(record)
    assert len(set(np.round(angles, 12))) == len(record)
    # Centered across pairs: the shared part stays the global shift's.
    assert np.allclose(translations.mean(0), 0.0, atol=1e-9)
    assert np.std(np.degrees(angles)) == pytest.approx(ANGULO_SIGMA, rel=0.35)


def test_on_the_draw_is_reproducible_from_the_seed(pair):
    _, on = pair
    again = toy_run.build_family("rollo", 10, toy_run.FULL,
                                 per_pair_shift=toy_bags.ESCALA_POR_PAR_DECLARADA)
    assert np.array_equal(on.Xt, again.Xt)
    other = toy_run.build_family("rollo", 11, toy_run.FULL,
                                 per_pair_shift=toy_bags.ESCALA_POR_PAR_DECLARADA)
    assert not np.allclose(on.desplazamientos[(0, 0)]["traslacion"],
                           other.desplazamientos[(0, 0)]["traslacion"])


def test_the_rotation_is_rigid_about_the_pair_centroid(pair):
    off, on = pair
    m = (on.yt == 1) & (on.bt == 4)
    a, b = off.Xt[m], on.Xt[m]
    da = np.linalg.norm(a[:, None] - a[None], axis=-1)
    db = np.linalg.norm(b[:, None] - b[None], axis=-1)
    assert np.allclose(da, db, atol=1e-9)


def test_a_negative_scale_is_refused():
    with pytest.raises(ValueError):
        toy_run.build_family("rollo", 10, toy_run.FULL, per_pair_shift=-0.1)
