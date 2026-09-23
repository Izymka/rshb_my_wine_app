"""Build canonical 512px label crops and register their provenance in PostgreSQL."""

import argparse
import hashlib
from pathlib import Path

from wine_scanner.db import ImageDerivative, WineImage, database_url, session_factory
from wine_scanner.detect import build_cropper
from wine_scanner.embed import load_image, pick_device
from wine_scanner.image_preprocess import LABEL_PREPROCESS_VERSION, prepare_label_image


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("models/rtdetr_label"))
    parser.add_argument("--out", type=Path, default=Path("data/media/labels/v1"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device = pick_device()
    cropper = build_cropper("cascade", args.weights, device)
    factory = session_factory(database_url())
    with factory.begin() as session:
        images = session.query(WineImage).order_by(WineImage.id).all()
        for image in images:
            target = args.out / f"{image.sha256}.png"
            existing = (
                session.query(ImageDerivative)
                .filter_by(
                    source_image_id=image.id,
                    kind="label_square",
                    pipeline_version=LABEL_PREPROCESS_VERSION,
                )
                .one_or_none()
            )
            if existing and target.exists() and not args.force:
                continue
            try:
                # cropper includes the geometric square; apply photometry explicitly here to
                # retain the exact coefficients in the database.
                square, bottle_box, label_box = cropper.detector.crop_with_metadata(
                    load_image(image.storage_path)
                )
                prepared, photo = prepare_label_image(square)
                target.parent.mkdir(parents=True, exist_ok=True)
                prepared.save(target, format="PNG", compress_level=1)
                values = {
                    "storage_path": str(target),
                    "sha256": file_hash(target),
                    "status": "ready",
                    "photometric": {
                        "white_balance": photo.white_balance,
                        "luminance_low": photo.luminance_low,
                        "luminance_high": photo.luminance_high,
                    },
                    "bottle_box": list(bottle_box) if bottle_box else None,
                    "label_box": list(label_box) if label_box else None,
                }
            except Exception as error:  # keep a visible, queryable failure rather than hiding it
                values = {
                    "storage_path": str(target),
                    "sha256": "",
                    "status": f"failed:{type(error).__name__}",
                    "photometric": {},
                    "bottle_box": None,
                    "label_box": None,
                }
            if existing is None:
                existing = ImageDerivative(
                    source_image_id=image.id,
                    kind="label_square",
                    pipeline_version=LABEL_PREPROCESS_VERSION,
                    **values,
                )
                session.add(existing)
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
    print(f"processed {len(images)} catalogue images on {device}")


if __name__ == "__main__":
    main()
