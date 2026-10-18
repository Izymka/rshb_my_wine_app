"""Дописывание и пересчёт карточек собранного индекса (wine_scanner.index.patch)."""

import json

import faiss
import numpy as np
import pytest
import torch
from PIL import Image

import wine_scanner.index.patch as patch_module
from wine_scanner.index.patch import IndexPatch
from wine_scanner.rerank.store import TENSOR_FIELDS

DIM = 4


class ColorEmbedder:
    """Вектор — средний цвет вырезки плюс константа, нормированный: проверяемо и без модели."""

    def encode_image(self, image):
        mean = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255
        vector = torch.tensor([*mean, 1.0])
        return vector / vector.norm()


class FakeMatcher:
    def __init__(self, **kwargs):
        pass

    def describe(self, image):
        features = {name: torch.zeros(1, 2) for name in TENSOR_FIELDS}
        return {**features, "image_size": image.size}


class NoOCR:
    def __init__(self, **kwargs):
        pass

    def read(self, image, use_cache=False):
        return []


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(patch_module, "build_embedder", lambda **kwargs: ColorEmbedder())
    monkeypatch.setattr(patch_module, "XFeatMatcher", FakeMatcher)
    monkeypatch.setattr("wine_scanner.ocr.LabelOCR", NoOCR)
    source = tmp_path / "index"
    (source / "descriptors").mkdir(parents=True)
    vectors = np.eye(DIM, dtype=np.float32)[:2]
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    faiss.write_index(index, str(source / "vectors.faiss"))
    meta = {"dim": DIM, "item_ids": ["a", "b"], "payloads": [{"slug": "a"}, {"slug": "b"}]}
    (source / "meta.json").write_text(json.dumps(meta))
    (source / "config.json").write_text(json.dumps({"model": "m", "weights": "w", "items": 2}))
    (source / "descriptors" / "a.npz").write_bytes(b"old-a")
    return source


def test_add_appends_card_without_touching_source(index_dir, tmp_path):
    patch = IndexPatch(index_dir, device="cpu")
    patch.open_target(tmp_path / "out")
    red = Image.new("RGB", (32, 32), (255, 0, 0))
    vector = patch.add("c", {"slug": "c", "name": "Новое"}, red)
    patch.save("added_from_site", {"added": ["c"]})

    out = tmp_path / "out"
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["item_ids"] == ["a", "b", "c"] and meta["payloads"][2]["name"] == "Новое"
    saved = faiss.read_index(str(out / "vectors.faiss"))
    assert saved.ntotal == 3
    assert np.allclose(saved.reconstruct(2), vector)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["items"] == 3 and config["added_from_site"] == {"added": ["c"]}
    assert (out / "descriptors" / "c.npz").exists()
    # Исходный индекс не тронут.
    assert faiss.read_index(str(index_dir / "vectors.faiss")).ntotal == 2
    with pytest.raises(ValueError):
        patch.add("a", {}, red)


def test_replace_breaks_hard_link_before_writing(index_dir, tmp_path):
    patch = IndexPatch(index_dir, device="cpu")
    patch.open_target(tmp_path / "out")
    patch.replace("a", Image.new("RGB", (32, 32), (0, 255, 0)))
    patch.save("label_overrides", {})
    # Дескриптор исходного индекса (жёсткая ссылка) остался прежним, у копии — новый файл.
    assert (index_dir / "descriptors" / "a.npz").read_bytes() == b"old-a"
    assert (tmp_path / "out" / "descriptors" / "a.npz").read_bytes() != b"old-a"


def test_shared_patch_reuses_models(index_dir, tmp_path, monkeypatch):
    """Копия индекса для самопоиска и итоговая берут модели у первого патча, а не грузят свои."""
    built = []
    monkeypatch.setattr(
        patch_module, "build_embedder", lambda **kwargs: built.append(1) or ColorEmbedder()
    )
    first = IndexPatch(index_dir, device="cpu")
    first.open_target(tmp_path / "first")
    first.add("c", {"slug": "c"}, Image.new("RGB", (8, 8), "red"))
    second = IndexPatch(index_dir, device="cpu", share=first)
    assert len(built) == 1
    assert second.embedder is first.embedder
    assert second._matcher is first._matcher and second._ocr is first._ocr
    # Состояние индекса у копии своё: карточка первого патча в неё не попала.
    assert "c" not in second.position and len(second.vectors) == 2
