import hashlib
import zipfile

import pytest

from scripts.download_models import extract_zip, verify_sha256


def test_verify_sha256(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model bytes")
    assert verify_sha256(path, hashlib.sha256(b"model bytes").hexdigest())
    assert not verify_sha256(path, "0" * 64)


def test_extract_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")
    destination = tmp_path / "models"
    with pytest.raises(RuntimeError, match="unsafe path"):
        extract_zip(archive, destination)
    assert not (tmp_path / "escape.txt").exists()


def test_extract_zip_installs_expected_files(tmp_path):
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("det_10g.onnx", b"model")
    destination = tmp_path / "models"
    extract_zip(archive, destination)
    assert (destination / "det_10g.onnx").read_bytes() == b"model"
