"""Обучить whitening на векторах готового индекса и положить рядом с ним.

    uv run python scripts/fit_whitening.py --index models/index_platform --dim 512

Индекс не пересобирается: сырые векторы остаются в vectors.faiss, а пайплайн при загрузке
применяет whitening.npz и к каталогу, и к запросу. Так преобразование можно подобрать и
переподобрать за секунды, не платя полтора часа за прогон каталога через модель.
Размерность и степень записываются в config.json — версия индекса от этого меняется, как и
должна: ответы на разных whitening в логах обязаны различаться.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from wine_scanner.embed import Whitening
from wine_scanner.index import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("models/index_platform"))
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument("--remove", action="store_true", help="снять whitening с индекса")
    args = parser.parse_args()

    target = args.index / "whitening.npz"
    config_path = args.index / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    if args.remove:
        target.unlink(missing_ok=True)
        config.pop("whiten", None)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print("whitening снят")
        return

    index = VectorIndex.load(args.index)
    raw = index.index.reconstruct_n(0, index.index.ntotal)
    whitening = Whitening.fit(raw, dim=args.dim, power=args.power)
    whitening.save(target)
    config["whiten"] = {"dim": args.dim, "power": args.power}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # Насколько разжалось пространство: доля карточек, чей ближайший сосед ближе 0.95.
    def crowding(vectors: np.ndarray) -> float:
        sims = vectors @ vectors.T
        np.fill_diagonal(sims, -1.0)
        return float((sims.max(axis=1) > 0.95).mean())

    print(
        f"whitening dim={args.dim} power={args.power} -> {target}; "
        f"соседей ближе 0.95: {crowding(raw):.1%} -> {crowding(whitening.apply(raw)):.1%}"
    )


if __name__ == "__main__":
    main()
