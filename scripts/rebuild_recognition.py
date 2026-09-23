"""Rebuild measured recognition artifacts after RT-DETR training has completed.

Commands run in the current Python environment. Each stage has a separate log and errors
stop the chain. Run only after the label checkpoint is frozen; test is evaluated last.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STEPS = (
    "audit",
    "padding",
    "branches",
    "index",
    "whitening",
    "features",
    "catboost",
    "benchmark",
    "notebooks",
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--weights", default="models/rtdetr_label_recrop_bf16")
    parser.add_argument("--start", choices=STEPS, default="audit")
    args = parser.parse_args()
    weights = Path(args.weights)
    training = json.loads((weights / "training.json").read_text(encoding="utf-8"))
    if training["history"][-1]["mean_iou"] <= 0:
        raise ValueError("Refusing to build artifacts from a detector with zero validation IoU")
    with (weights / "model.safetensors").open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    log_root = Path("data/derived/rebuild_logs")
    log_root.mkdir(parents=True, exist_ok=True)
    state = {"checkpoint_sha256": checkpoint_hash, "weights": str(weights), "stages": {}}
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
        "WINE_OCR": "paddle",
        "WINE_VLM": "0",
        "OMP_NUM_THREADS": "4",
    }

    def run(stage, arguments):
        if STEPS.index(stage) < STEPS.index(args.start):
            return
        command = [sys.executable, *map(str, arguments)]
        state["stages"][stage] = {
            "status": "running",
            "command": command,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        status = log_root / "status.json"
        status.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(stage, "started", flush=True)
        with (log_root / f"{stage}.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        state["stages"][stage].update(
            status="complete" if result.returncode == 0 else "failed", returncode=result.returncode
        )
        status.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{stage} failed; inspect {log_root / (stage + '.log')}")
        print(stage, "complete", flush=True)

    crop_root = "data/derived/rtdetr_retrained_audit"
    run(
        "audit",
        [
            "scripts/audit_rtdetr_crops.py",
            "--weights",
            weights,
            "--out",
            crop_root,
            "--reuse-bottles-from",
            "data/derived/rtdetr_audit",
            "--batch-size",
            "12",
        ],
    )
    run(
        "padding",
        [
            "scripts/evaluate_padding.py",
            "--crop-root",
            crop_root,
            "--detector-meta",
            weights / "training.json",
            "--out",
            "data/derived/padding_retrained_validation.json",
        ],
    )
    run(
        "branches",
        [
            "scripts/evaluate_branches.py",
            "--crop-root",
            crop_root,
            "--detector-meta",
            weights / "training.json",
            "--tag",
            "retrained",
            "--device",
            "cuda",
        ],
    )
    padding = json.loads(
        Path("data/derived/padding_retrained_validation.json").read_text(encoding="utf-8")
    )
    branches = json.loads(Path("data/derived/branches_retrained.json").read_text(encoding="utf-8"))
    color = ",".join(map(str, padding["results"][padding["selection"]]["fill"]))
    index = "models/index_platform"
    features = "eval/results/features_platform_recrop.jsonl"
    decider = "models/decider_platform_paddle"
    run(
        "index",
        [
            "scripts/build_index.py",
            "--out",
            index,
            "--catalog",
            "platform",
            "--model",
            "models/siglip2",
            "--weights",
            weights,
            "--fit",
            "pad",
            "--precision",
            "fp16",
            "--batch-size",
            "8",
            "--pad-color",
            color,
            "--local-preprocess",
            branches["selected_local"],
            "--ocr-preprocess",
            branches["selected_ocr"],
        ],
    )
    run("whitening", ["scripts/fit_whitening.py", "--index", index, "--dim", "256"])
    run(
        "features",
        [
            "scripts/build_platform_features.py",
            "--index",
            index,
            "--sources",
            "synthetic,train",
            "--out",
            features,
        ],
    )
    run(
        "catboost",
        [
            "scripts/train_decider.py",
            "--features",
            features,
            "--out",
            decider,
            "--scenario",
            "real",
            "--family-map",
            "data/catalog/catalog.csv",
        ],
    )
    run(
        "benchmark",
        [
            "eval/platform_benchmark.py",
            "--index",
            index,
            "--decider",
            decider,
            "--tag",
            "rtdetr-recrop-catboost-v3",
            "--no-cache",
            "--out",
            "eval/results/platform_recrop_runs.jsonl",
            "--dump",
            "eval/results/platform_recrop_outcomes.jsonl",
        ],
    )
    for script, notebook in (
        ("scripts/build_crop_audit_notebook.py", "notebooks/02_rtdetr_crop_audit.ipynb"),
        (
            "scripts/build_pipeline_experiments_notebook.py",
            "notebooks/03_pipeline_rebuild_experiments.ipynb",
        ),
    ):
        run("notebooks", [script])
        run(
            "notebooks",
            [
                "-m",
                "jupyter",
                "nbconvert",
                "--execute",
                "--to",
                "notebook",
                "--inplace",
                notebook,
                "--ExecutePreprocessor.timeout=300",
            ],
        )
    print("Recognition rebuild completed", flush=True)


if __name__ == "__main__":
    main()
