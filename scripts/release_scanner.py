"""Package, install, and publish the scanner's fixed set of release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/release"
FOLDER_ID = "1xZ1ye2XNLb3Zd6witTz5ZjVWpEfFdTmQ"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
ARCHIVES = {
    "wine_scanner_artifacts.tar": (
        "models/index_platform_sq_v3",
        "models/decider_platform_sq_v3_v7",
        "models/rtdetr_label_recrop_bf16",
        "data/catalog/catalog.csv",
        "data/catalog/originals",
    ),
    "siglip2.tar": ("models/siglip2",),
}
FILES = (*ARCHIVES, "SHA256SUMS", "README_DEPLOY.txt")
ESSENTIAL = (
    "models/index_platform_sq_v3/vectors.faiss",
    "models/index_platform_sq_v3/meta.json",
    "models/decider_platform_sq_v3_v7/model.cbm",
    "models/decider_platform_sq_v3_v7/meta.json",
    "models/rtdetr_label_recrop_bf16",
    "models/siglip2/model.safetensors",
    "data/catalog/catalog.csv",
    "data/catalog/originals",
)


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sums(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name not in ARCHIVES or len(digest) != 64 or not all(
            character in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Недопустимая строка SHA256SUMS: {line}")
        result[name] = digest
    if set(result) != set(ARCHIVES):
        raise ValueError("SHA256SUMS должен содержать ровно два релизных архива")
    return result


def verify_archives(expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        path = RELEASE / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Ошибка SHA256: {path}")
        print(f"{name}: OK")


def safe_extract(archive: Path, allowed: tuple[str, ...]) -> None:
    with tarfile.open(archive, "r") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not any(member.name == path or member.name.startswith(path + "/")
                           for path in allowed)
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Недопустимый объект в архиве: {member.name}")
            target = ROOT.joinpath(*name.parts)
            if not target.resolve().is_relative_to(ROOT):
                raise ValueError(f"Путь выходит за пределы проекта: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"Не удалось прочитать {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def check_files() -> None:
    missing = [name for name in ESSENTIAL if not (ROOT / name).exists()]
    if missing:
        raise ValueError(f"После распаковки отсутствуют файлы: {', '.join(missing)}")


def gdown_listing() -> dict[str, str]:
    output = subprocess.check_output(["gdown", FOLDER_URL, "--json"], text=True)
    listing = {Path(item["path"]).name: item["url"] for item in json.loads(output)}
    if not set(FILES) <= listing.keys():
        raise ValueError(f"В папке Google Drive нет файлов: {set(FILES) - listing.keys()}")
    return listing


def download_from_drive() -> None:
    RELEASE.mkdir(parents=True, exist_ok=True)
    remote = os.environ.get("DRIVE_REMOTE", "").strip()
    if remote:
        if not remote.endswith(":"):
            raise ValueError("DRIVE_REMOTE должен иметь вид gdrive:")
        run("rclone", "copy", remote, str(RELEASE), "--drive-root-folder-id", FOLDER_ID,
            "--include", "SHA256SUMS", "--include", "README_DEPLOY.txt", "-v")
        expected = sums((RELEASE / "SHA256SUMS").read_text(encoding="utf-8"))
        installed = RELEASE / ".installed.json"
        if installed.exists() and json.loads(installed.read_text()) == expected:
            check_files()
            return
        run("rclone", "copy", remote, str(RELEASE), "--drive-root-folder-id", FOLDER_ID,
            "--include", "*.tar", "-v")
    else:
        listing = gdown_listing()
        run("gdown", listing["SHA256SUMS"], "-O", str(RELEASE / "SHA256SUMS"))
        expected = sums((RELEASE / "SHA256SUMS").read_text(encoding="utf-8"))
        installed = RELEASE / ".installed.json"
        if installed.exists() and json.loads(installed.read_text()) == expected:
            check_files()
            return
        for name in (*ARCHIVES, "README_DEPLOY.txt"):
            path = RELEASE / name
            if name in ARCHIVES and path.exists() and sha256(path) == expected[name]:
                continue
            if path.exists() and name in ARCHIVES:
                path.unlink()  # A stale archive must never be resumed into a new release.
            run("gdown", listing[name], "-O", str(path), "--continue", "--retries", "3")

    verify_archives(expected)
    for name, allowed in ARCHIVES.items():
        safe_extract(RELEASE / name, allowed)
    check_files()
    (RELEASE / ".installed.json").write_text(json.dumps(expected, indent=2) + "\n")


def digest(paths: list[Path]) -> str:
    hasher = hashlib.sha1()
    for path in paths:
        if path.exists():
            hasher.update(path.name.encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:12]


def check_health() -> None:
    index = ROOT / "models/index_platform_sq_v3"
    decider = ROOT / "models/decider_platform_sq_v3_v7"
    expected_version = {
        "index": digest([index / name for name in
                         ("vectors.faiss", "meta.json", "config.json", "whitening.npz")]
                        + [ROOT / "data/splits/catalog_excluded_slugs.csv"]),
        "decider": digest([decider / name for name in ("model.cbm", "ranker.cbm", "meta.json")]),
    }
    expected_threshold = json.loads((decider / "meta.json").read_text())["threshold"]
    for _ in range(60):
        try:
            with urllib.request.urlopen("http://localhost:8080/health", timeout=5) as response:
                health = json.load(response)
            if health.get("status") != "ok":
                raise ValueError(f"Сервис не готов: {health}")
            if health.get("version") != expected_version:
                raise ValueError(f"Версии артефактов не совпадают: {health.get('version')}")
            if health.get("threshold") != expected_threshold:
                raise ValueError(f"Порог не совпадает: {health.get('threshold')}")
            devices = health.get("devices", {})
            if (devices.get("ocr") != "gpu:0" or
                    not all(str(devices.get(key, "")).startswith("cuda")
                            for key in ("embed", "rerank"))):
                raise ValueError(f"Модели работают не на GPU: {devices}")
            print(json.dumps(health, ensure_ascii=False, indent=2))
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5)
    raise TimeoutError("Сервис не ответил на /health за 5 минут; см. docker compose logs scanner")


def deploy() -> None:
    for command in ("docker", "python3"):
        if not shutil.which(command):
            raise RuntimeError(f"Нужна команда {command}")
    if not shutil.which("rclone" if os.environ.get("DRIVE_REMOTE") else "gdown"):
        raise RuntimeError("Установите gdown >= 6.1 или задайте DRIVE_REMOTE и настройте rclone")
    run("docker", "compose", "version")
    run("docker", "info", "--format", "{{.ServerVersion}}")
    download_from_drive()
    env = ROOT / ".env"
    if not env.exists():
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        env.write_text(example.replace("\nWINE_VLM=1\n", "\nWINE_VLM=0\n", 1),
                       encoding="utf-8")
        print("Создан .env с WINE_VLM=0. Для облачного судьи добавьте ключ и включите WINE_VLM=1.")
    run("docker", "compose", "up", "-d", "--build", "scanner")
    check_health()


def prepare() -> None:
    check_files()
    RELEASE.mkdir(parents=True, exist_ok=True)
    temporary_archives = []
    for name, paths in ARCHIVES.items():
        target = RELEASE / name
        temporary = target.with_suffix(".tar.tmp")
        with tarfile.open(temporary, "w") as tar:
            for path in paths:
                source = ROOT / path
                if not source.exists():
                    raise FileNotFoundError(source)
                tar.add(source, arcname=path, recursive=True)
        temporary_archives.append((temporary, target))
    for temporary, target in temporary_archives:
        temporary.replace(target)
        print(f"Готов {target}")
    expected = {name: sha256(RELEASE / name) for name in ARCHIVES}
    (RELEASE / "SHA256SUMS").write_text(
        "".join(f"{digest} *{name}\n" for name, digest in expected.items()),
        encoding="utf-8",
    )
    shutil.copyfile(ROOT / "docs/DEPLOY_SCANNER.md", RELEASE / "README_DEPLOY.txt")
    verify_archives(expected)


def publish() -> None:
    remote = os.environ.get("DRIVE_REMOTE", "").strip()
    if not remote or not remote.endswith(":"):
        raise ValueError("Задайте DRIVE_REMOTE=gdrive: (настроенный Google Drive remote rclone)")
    expected = sums((RELEASE / "SHA256SUMS").read_text(encoding="utf-8"))
    verify_archives(expected)
    if (RELEASE / "README_DEPLOY.txt").read_bytes() != (
        ROOT / "docs/DEPLOY_SCANNER.md"
    ).read_bytes():
        raise ValueError("README_DEPLOY.txt не совпадает с docs/DEPLOY_SCANNER.md")
    print(f"Публикация {', '.join(FILES)} в {FOLDER_URL}")
    if input("Заменить файлы в Google Drive? Введите yes: ").strip() != "yes":
        raise RuntimeError("Публикация отменена")
    run("rclone", "copy", str(RELEASE), remote, "--drive-root-folder-id", FOLDER_ID,
        *(item for name in FILES for item in ("--include", name)), "--checksum", "-v")
    run("rclone", "check", str(RELEASE), remote, "--drive-root-folder-id", FOLDER_ID,
        *(item for name in FILES for item in ("--include", name)), "--one-way", "-v")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "prepare", "publish"))
    args = parser.parse_args()
    {"deploy": deploy, "prepare": prepare, "publish": publish}[args.action]()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
