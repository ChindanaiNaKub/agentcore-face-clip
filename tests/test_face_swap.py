from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.face_swap import (
    FaceDetection,
    FaceSwapError,
    FaceTracker,
    TargetFaces,
    analyze_video,
    choose_largest_face,
    decode_base64,
    parse_target_faces,
    swap_video,
)


class TwoFaceDetector:
    def __call__(self, frame):
        del frame
        detections = [
            FaceDetection((20.0, 30.0, 100.0, 130.0), (1.0, 0.0)),
            FaceDetection((180.0, 30.0, 280.0, 130.0), (0.0, 1.0)),
        ]
        return detections, [face.bbox for face in detections]


class FillSwapper:
    def __call__(self, frame, bbox):
        left, top, right, bottom = (int(value) for value in bbox)
        frame[top:bottom, left:right] = (0, 255, 0)
        return frame


def make_video(path: Path, *, frames: int = 10, fps: float = 5, size=(320, 160)) -> None:
    width, height = size
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    frame[30:130, 20:100] = (0, 0, 255)
    frame[30:130, 180:280] = (255, 0, 0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for _ in range(frames):
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


def region_change(left: np.ndarray, right: np.ndarray, bounds: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bounds
    return float(np.abs(left[y1:y2, x1:x2].astype(float) - right[y1:y2, x1:x2].astype(float)).mean())


def test_parse_target_faces_accepts_only_documented_shapes():
    assert parse_target_faces("all") == TargetFaces()
    assert parse_target_faces({"face_id": "face_0001"}) == TargetFaces(False, "face_0001")


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        [],
        "ALL",
        "",
        {},
        {"face_id": ""},
        {"face_id": "   "},
        {"face_id": True},
        {"face_id": "face_0001", "extra": 1},
    ],
)
def test_parse_target_faces_rejects_invalid_shapes(value):
    with pytest.raises(FaceSwapError):
        parse_target_faces(value)


def test_base64_validation_enforces_encoding_and_byte_limit():
    assert decode_base64("YWJj", "target_video", 3) == b"abc"
    with pytest.raises(FaceSwapError, match="valid base64"):
        decode_base64("not base64!", "target_video", 10)
    with pytest.raises(FaceSwapError, match="exceeds 3 bytes"):
        decode_base64("YWJjZA==", "target_video", 3)


def test_tracker_keeps_moving_face_id_and_adds_new_face_in_left_order():
    tracker = FaceTracker()
    first = FaceDetection((100.0, 0.0, 200.0, 100.0), (1.0, 0.0))
    assignments, tracks = tracker.update([first], 0)
    assert assignments == ("face_0001",)

    moved = FaceDetection((130.0, 0.0, 230.0, 100.0), (1.0, 0.0))
    new = FaceDetection((0.0, 0.0, 100.0, 100.0), (0.0, 1.0))
    assignments, tracks = tracker.update([moved, new], 1)
    assert assignments == ("face_0001", "face_0002")
    assert [track.face_id for track in tracks] == ["face_0001", "face_0002"]
    assert tracks[0].first_frame == 0
    assert tracks[0].frames == 2
    assert tracks[1].first_frame == 1


def test_tracker_does_not_match_different_embedding_at_same_box():
    tracker = FaceTracker()
    tracker.update([FaceDetection((0.0, 0.0, 100.0, 100.0), (1.0, 0.0))], 0)
    assignments, tracks = tracker.update(
        [FaceDetection((0.0, 0.0, 100.0, 100.0), (0.0, 1.0))], 1
    )
    assert assignments == ("face_0002",)
    assert len(tracks) == 2


def test_choose_largest_source_face():
    detections = [
        FaceDetection((0.0, 0.0, 10.0, 10.0), (1.0,)),
        FaceDetection((20.0, 0.0, 120.0, 100.0), (0.0,)),
    ]
    assert choose_largest_face(detections) == 1
    with pytest.raises(FaceSwapError, match="no face found in source"):
        choose_largest_face([])


def test_analyze_and_swap_two_face_synthetic_video(tmp_path):
    target = tmp_path / "target.mp4"
    make_video(target)
    detector = TwoFaceDetector()
    swapper = FillSwapper()

    faces = analyze_video(target, detector)
    assert [face["face_id"] for face in faces] == ["face_0001", "face_0002"]
    assert all(face["frames"] == 10 for face in faces)

    all_output = tmp_path / "all.mp4"
    all_result = swap_video(target, all_output, TargetFaces(), detector, swapper)
    assert all_result.swapped_faces == 20

    one_output = tmp_path / "one.mp4"
    one_result = swap_video(
        target, one_output, TargetFaces(False, "face_0001"), detector, swapper
    )
    assert one_result.swapped_faces == 10

    original = first_frame(target)
    all_frame = first_frame(all_output)
    one_frame = first_frame(one_output)
    assert region_change(original, all_frame, (35, 45, 85, 115)) > 20
    assert region_change(original, all_frame, (195, 45, 265, 115)) > 20
    assert region_change(original, one_frame, (35, 45, 85, 115)) > 20
    assert region_change(original, one_frame, (195, 45, 265, 115)) < 5


def test_swap_preserves_target_audio_and_writes_h264(tmp_path):
    import subprocess

    silent = tmp_path / "silent.mp4"
    target = tmp_path / "target_audio.mp4"
    make_video(silent)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-i", str(silent), "-map", "0:a:0", "-map", "1:v:0",
            "-c:v", "copy", "-c:a", "aac", str(target),
        ],
        check=True,
    )
    output = tmp_path / "swapped.mp4"
    swap_video(target, output, TargetFaces(), TwoFaceDetector(), FillSwapper())
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,codec_name", "-of", "json", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    streams = json.loads(probe.stdout)["streams"]
    codecs = {stream["codec_type"]: stream["codec_name"] for stream in streams}
    assert codecs["video"] == "h264"
    assert codecs["audio"] == "aac"
    capture = cv2.VideoCapture(str(output))
    try:
        frames = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frames += 1
    finally:
        capture.release()
    assert frames == 10


def test_swap_rejects_unknown_face_and_removes_partial_output(tmp_path):
    target = tmp_path / "target.mp4"
    output = tmp_path / "swapped.mp4"
    make_video(target)
    with pytest.raises(FaceSwapError, match="was not found"):
        swap_video(
            target, output, TargetFaces(False, "face_9999"), TwoFaceDetector(), FillSwapper()
        )
    assert not output.exists()


def test_analyze_reports_no_target_face(tmp_path):
    target = tmp_path / "target.mp4"
    make_video(target)
    with pytest.raises(FaceSwapError, match="no face found in target"):
        analyze_video(target, lambda frame: ([], []))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frames": 21, "fps": 1}, "exceeds 20 seconds"),
        ({"frames": 151, "fps": 30}, "exceeds 150 frames"),
        ({"frames": 1, "size": (1282, 32)}, "exceeds 1280 pixels"),
    ],
)
def test_target_limits_are_enforced(tmp_path, kwargs, message):
    target = tmp_path / "target.mp4"
    make_video(target, **kwargs)
    with pytest.raises(FaceSwapError, match=message):
        analyze_video(target, TwoFaceDetector())
