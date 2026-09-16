"""Whitening: форма, нормировка, сохранение и то, ради чего он заведён, — разжатие пространства."""

import numpy as np
import pytest

from wine_scanner.embed import Whitening


def crowded(n: int = 400, d: int = 64, seed: int = 0) -> np.ndarray:
    """Векторы с одним громким общим направлением — модель «все бутылки на белом»."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=d)
    x = rng.normal(size=(n, d)) + 6.0 * common
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def nearest_neighbour_share(vectors: np.ndarray, above: float = 0.95) -> float:
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1.0)
    return float((sims.max(axis=1) > above).mean())


def test_output_is_unit_length_and_of_requested_dim():
    x = crowded()
    w = Whitening.fit(x, dim=16)
    y = w.apply(x)

    assert y.shape == (len(x), 16)
    assert y.dtype == np.float32
    assert np.allclose(np.linalg.norm(y, axis=1), 1.0, atol=1e-5)


def test_single_vector_is_accepted():
    w = Whitening.fit(crowded(), dim=8)

    assert w.apply(crowded()[0]).shape == (1, 8)


def test_whitening_spreads_a_crowded_space():
    x = crowded()
    before = nearest_neighbour_share(x)
    after = nearest_neighbour_share(Whitening.fit(x, dim=32).apply(x))

    assert before > 0.9
    assert after < 0.1


def test_roundtrip_through_disk(tmp_path):
    x = crowded()
    w = Whitening.fit(x, dim=8, power=0.25)
    w.save(tmp_path / "w.npz")
    loaded = Whitening.load(tmp_path / "w.npz")

    assert loaded.dim == 8 and loaded.power == 0.25
    assert np.allclose(loaded.apply(x), w.apply(x))


def test_too_few_vectors_is_an_error():
    with pytest.raises(ValueError):
        Whitening.fit(crowded(n=10), dim=16)
