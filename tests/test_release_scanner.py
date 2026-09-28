"""Small, offline checks for the release install boundary."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import release_scanner


class ReleaseScannerTests(unittest.TestCase):
    def test_matching_installed_release_skips_large_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            expected = {name: "a" * 64 for name in release_scanner.ARCHIVES}
            (release / ".installed.json").write_text(json.dumps(expected))
            manifest = "".join(f"{digest} *{name}\n" for name, digest in expected.items())
            calls = []

            def fake_run(*args: str) -> None:
                calls.append(args)
                (release / "SHA256SUMS").write_text(manifest)

            listing = {name: f"https://drive.google.com/file/d/{name}" for name in
                       release_scanner.FILES}
            with (patch.object(release_scanner, "RELEASE", release),
                  patch.object(release_scanner, "gdown_listing", return_value=listing),
                  patch.object(release_scanner, "run", side_effect=fake_run),
                  patch.object(release_scanner, "check_files"),
                  patch.dict("os.environ", {"DRIVE_REMOTE": ""})):
                release_scanner.download_from_drive()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], listing["SHA256SUMS"])

    def test_rejects_unexpected_checksum_entry(self) -> None:
        with self.assertRaises(ValueError):
            release_scanner.sums("a" * 64 + " *.env\n")

    def test_extracts_expected_file_and_rejects_outside_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "release.tar"
            with tarfile.open(archive, "w") as tar:
                body = b"artifact"
                member = tarfile.TarInfo("models/siglip2/config.json")
                member.size = len(body)
                tar.addfile(member, io.BytesIO(body))
            with patch.object(release_scanner, "ROOT", root):
                release_scanner.safe_extract(archive, ("models/siglip2",))
            self.assertEqual((root / "models/siglip2/config.json").read_bytes(), b"artifact")

            with tarfile.open(archive, "w") as tar:
                member = tarfile.TarInfo("models/siglip2/../../.env")
                member.size = 1
                tar.addfile(member, io.BytesIO(b"x"))
            with patch.object(release_scanner, "ROOT", root):
                with self.assertRaises(ValueError):
                    release_scanner.safe_extract(archive, ("models/siglip2",))
            self.assertFalse((root / ".env").exists())


if __name__ == "__main__":
    unittest.main()
