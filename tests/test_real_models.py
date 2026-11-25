import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.face_swap import InsightFaceEngine, TargetFaces

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("INSIGHTFACE_ROOT", "")) if os.environ.get("INSIGHTFACE_ROOT") else None
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL_TESTS") != "1"
    or MODEL_ROOT is None
    or not (MODEL_ROOT / "models" / "buffalo_l" / "det_10g.onnx").is_file()
    or not (MODEL_ROOT / "models" / "inswapper_128.onnx").is_file(),
    reason="set RUN_REAL_MODEL_TESTS=1 and INSIGHTFACE_ROOT to run large-model checks",
)


def make_two_face_video(path: Path) -> None:
    left = cv2.imread(str(ROOT / "testdata" / "rose.jpg"))
    right = cv2.imread(str(ROOT / "testdata" / "young_man.jpg"))
    assert left is not None and right is not None
    frame = np.full((512, 1024, 3), 255, dtype=np.uint8)
    frame[:, :512] = cv2.resize(left, (512, 512))
    frame[:, 512:] = cv2.resize(right, (512, 512))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (1024, 512))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(frame)
    writer.release()


def first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        assert ok
        return frame
    finally:
        capture.release()


def mean_change(original: np.ndarray, swapped: np.ndarray, left: bool) -> float:
    region = (slice(100, 400), slice(100, 400) if left else slice(612, 912))
    return float(np.abs(original[region].astype(float) - swapped[region].astype(float)).mean())


def test_real_models_analyze_all_and_selected_face(tmp_path):
    target = tmp_path / "two_faces.mp4"
    make_two_face_video(target)
    engine = InsightFaceEngine(MODEL_ROOT)
    faces = engine.analyze(target)
    assert [face["face_id"] for face in faces] == ["face_0001", "face_0002"]

    all_output = tmp_path / "all.mp4"
    all_result = engine.swap(
        ROOT / "testdata" / "franklin.jpg", target, TargetFaces(), all_output
    )
    one_output = tmp_path / "one.mp4"
    one_result = engine.swap(
        ROOT / "testdata" / "franklin.jpg",
        target,
        TargetFaces(False, "face_0001"),
        one_output,
    )
    assert all_result.swapped_faces == 6
    assert one_result.swapped_faces == 3

    original = first_frame(target)
    all_frame = first_frame(all_output)
    one_frame = first_frame(one_output)
    assert mean_change(original, all_frame, True) > 1
    assert mean_change(original, all_frame, False) > 1
    assert mean_change(original, one_frame, True) > 1
    assert mean_change(original, one_frame, False) < 4
