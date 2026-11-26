import json
from pathlib import Path

import pytest

from scripts import face_swap
from scripts.face_swap import SwapResult, TargetFaces


class FakeEngine:
    def analyze(self, target: Path):
        return ({"face_id": "face_0001"},)

    def swap(self, source: Path, target: Path, selection: TargetFaces, output: Path, compare: bool = False):
        output.write_bytes(b"clip")
        return SwapResult(({"face_id": "face_0001"},), 1)


def test_cli_analyze_and_swap(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(face_swap, "InsightFaceEngine", FakeEngine)
    assert face_swap.main(["target.mp4", "--analyze"]) == 0
    assert json.loads(capsys.readouterr().out) == {"faces": [{"face_id": "face_0001"}]}

    output = tmp_path / "swapped.mp4"
    assert face_swap.main(
        ["source.jpg", "target.mp4", "--face-id", "face_0001", "--out", str(output)]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["swapped_faces"] == 1
    assert output.read_bytes() == b"clip"


def test_cli_requires_one_selection_mode_for_swap():
    with pytest.raises(SystemExit):
        face_swap.main(["source.jpg", "target.mp4"])
    with pytest.raises(SystemExit):
        face_swap.main(["source.jpg", "target.mp4", "--all", "--face-id", "face_0001"])
    with pytest.raises(SystemExit):
        face_swap.main(["target.mp4", "--analyze", "--all"])
