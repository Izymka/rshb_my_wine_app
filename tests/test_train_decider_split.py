"""Разбиение живых кадров на калибровку (K) и порог (P) в scripts/train_decider.py."""

import numpy as np
import pytest

import scripts.train_decider as td


@pytest.fixture
def live_queries(monkeypatch):
    monkeypatch.setattr(td, "FAMILIES", {
        "fanagoriya-a": "Фанагория", "fanagoriya-b": "Фанагория",
        "abrau-a": "Абрау", "esse-a": "Esse", "leto-a": "LETO", "zaharin-a": "Захарьин",
    })
    wines, groups = [], []
    for wine, frames in (("fanagoriya-a", 3), ("fanagoriya-b", 2), ("abrau-a", 2),
                         ("esse-a", 2), ("leto-a", 2), ("zaharin-a", 1)):
        wines += [wine] * frames
        groups += ["live-organizer"] * frames
    for wine, group, frames in (
        ("unknown:unknown-fanagoriya-classic", "live-unknown-hard", 3),
        ("unknown:unknown-org-fanagoriya-alveus", "live-unknown-organizer", 2),
        ("unknown:unknown-massandra-kagor", "live-unknown-hard", 2),
        ("unknown:unknown-inkerman-merlo", "live-unknown-hard", 2),
        ("unknown:import-portia", "live-unknown-easy", 3),
        ("unknown:import-dalat", "live-unknown-easy", 3),
    ):
        wines += [wine] * frames
        groups += [group] * frames
    wines += ["synthetic-wine"] * 4
    groups += ["synthetic"] * 4
    known = np.array([not w.startswith(td.UNKNOWN_PREFIX) for w in wines])
    return wines, known, groups


def test_split_is_disjoint_by_winery_and_skips_synthetic(live_queries):
    wines, known, groups = live_queries
    roles = td.split_live(wines, known, groups, seed=7)
    assert set(roles[[g == "synthetic" for g in groups]]) == {""}
    sides: dict[str, set] = {}
    for wine, group, is_known, role in zip(wines, groups, known, roles, strict=True):
        if group.startswith("live-"):
            sides.setdefault(td.split_key(wine, bool(is_known), group), set()).add(role)
    # Одна винодельня внутри слоя — всегда по одну сторону: близнецы не разъезжаются.
    assert all(len(side) == 1 for side in sides.values())
    # Две бутылки Фанагории (разные вина) идут вместе.
    fanagoriya = {roles[i] for i, w in enumerate(wines) if w.startswith("fanagoriya-")}
    assert len(fanagoriya) == 1


def test_split_has_every_stratum_on_both_sides_and_is_deterministic(live_queries):
    wines, known, groups = live_queries
    roles = td.split_live(wines, known, groups, seed=3)
    assert (roles == td.split_live(wines, known, groups, seed=3)).all()
    for stratum in ("known", "unknown-hard", "unknown-easy"):
        in_stratum = [
            role for role, w, k, g in zip(roles, wines, known, groups, strict=True)
            if g.startswith("live-") and td.live_stratum(bool(k), g) == stratum
        ]
        assert {"K", "P"} <= set(in_stratum), stratum


def test_split_calibration_fits_on_k_and_thresholds_on_p():
    rng = np.random.default_rng(0)
    known = np.r_[np.ones(60, bool), np.zeros(40, bool)]
    correct = known & (rng.random(100) < 0.8)
    raw = np.clip(np.where(correct, 0.8, 0.1) + rng.normal(0, 0.1, 100), 0.01, 0.99)
    roles = np.array(["K", "P"] * 50, dtype=object)
    result = td.split_calibration(raw, correct, known, roles, budget=0.1)
    assert result["k_queries"] == result["p_queries"] == 50
    assert result["p_known"] + result["p_unknown"] == 50
    assert result["false_answer_rate"] <= 0.1
    assert result["calib_weight"] > 0
