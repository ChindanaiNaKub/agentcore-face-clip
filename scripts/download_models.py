"""Download and verify the official InsightFace models.

The models are for non-commercial research only. The repository MIT license
does not grant commercial rights to them.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"
BUFFALO_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
SWAP_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx"
SWAP_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
BUFFALO_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)


@dataclass(frozen=True)
class ModelSpec:
    url: str
    sha256: str
    filename: str


MODELS = (
    ModelSpec(BUFFALO_URL, BUFFALO_SHA256, "buffalo_l.zip"),
    ModelSpec(SWAP_URL, SWAP_SHA256, "inswapper_128.onnx"),
)


def default_root() -> Path:
    configured = os.environ.get("INSIGHTFACE_ROOT")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "ncnx" / "insightface"


def verify_sha256(path: Path, expected: str) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def _download(spec: ModelSpec, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"Downloading {spec.url}")
    urllib.request.urlretrieve(spec.url, partial)
    if not verify_sha256(partial, spec.sha256):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {spec.filename}")
    partial.replace(destination)


def _verified_archive(spec: ModelSpec, model_root: Path) -> Path:
    archive = model_root / spec.filename
    if not archive.is_file() or not verify_sha256(archive, spec.sha256):
        _download(spec, archive)
    return archive


def extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"unsafe path in model archive: {member.filename}")
        bundle.extractall(destination)


def download_models(root: Path) -> Path:
    model_root = root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    buffalo_dir = model_root / "buffalo_l"
    if not all((buffalo_dir / name).is_file() for name in BUFFALO_FILES):
        buffalo_spec = MODELS[0]
        archive = _verified_archive(buffalo_spec, model_root)
        with tempfile.TemporaryDirectory(prefix=".buffalo_l-", dir=model_root) as temporary:
            staging = Path(temporary) / "buffalo_l"
            staging.mkdir()
            extract_zip(archive, staging)
            if not all((staging / name).is_file() for name in BUFFALO_FILES):
                raise RuntimeError("buffalo_l.zip does not contain the expected model files")
            if buffalo_dir.exists():
                shutil.rmtree(buffalo_dir)
            staging.replace(buffalo_dir)
        print(f"Verified buffalo_l in {buffalo_dir}")

    swap_spec = MODELS[1]
    swap_path = model_root / swap_spec.filename
    if not swap_path.is_file() or not verify_sha256(swap_path, swap_spec.sha256):
        _download(swap_spec, swap_path)
    print(f"Verified inswapper_128.onnx in {swap_path}")
    print("Model license: non-commercial research only.")
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    args = parser.parse_args(argv)
    try:
        download_models(args.root)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
