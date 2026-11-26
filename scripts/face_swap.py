"""Face insertion with deterministic target-face tracks.

Usage:
    python3 scripts/face_swap.py TARGET.mp4 --analyze
    python3 scripts/face_swap.py SOURCE.jpg TARGET.mp4 --all --out out/swapped.mp4
    python3 scripts/face_swap.py SOURCE.jpg TARGET.mp4 --face-id face_0001 --out out/swapped.mp4
    python3 scripts/face_swap.py SOURCE.jpg TARGET.mp4 --all --compare --out out/compare.mp4
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

MAX_TARGET_BYTES = 50 * 1024 * 1024
MAX_TARGET_SECONDS = 20.0
MAX_TARGET_FRAMES = 20
MAX_TARGET_PIXELS = 1280
MIN_COSINE = 0.5
MIN_IOU = 0.2

BBox = tuple[float, float, float, float]


class FaceSwapError(ValueError):
    """An input or processing error that a client can correct."""


@dataclass(frozen=True)
class TargetFaces:
    all_faces: bool = True
    face_id: str | None = None


@dataclass(frozen=True)
class FaceDetection:
    bbox: BBox
    embedding: tuple[float, ...]


@dataclass
class FaceTrack:
    face_id: str
    embedding: tuple[float, ...]
    bbox: BBox
    first_frame: int
    last_frame: int
    frames: int

    def as_dict(self) -> dict:
        return {
            "face_id": self.face_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames": self.frames,
            "bbox": list(self.bbox),
        }


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class SwapResult:
    faces: tuple[dict, ...]
    swapped_faces: int


def parse_target_faces(value: object) -> TargetFaces:
    if isinstance(value, str) and value == "all":
        return TargetFaces()
    if isinstance(value, dict):
        if set(value) != {"face_id"}:
            raise FaceSwapError("target_faces must be 'all' or an object with only face_id")
        face_id = value["face_id"]
        if not isinstance(face_id, str) or not face_id.strip():
            raise FaceSwapError("target_faces.face_id must be a non-empty string")
        return TargetFaces(all_faces=False, face_id=face_id)
    raise FaceSwapError("target_faces must be 'all' or an object with only face_id")


def decode_base64(value: object, field: str, max_bytes: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise FaceSwapError(f"{field}.bytes_b64 must be a non-empty base64 string")
    if len(value) > ((max_bytes + 2) // 3) * 4:
        raise FaceSwapError(f"{field} exceeds {max_bytes} bytes")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FaceSwapError(f"{field}.bytes_b64 is not valid base64") from exc
    if len(data) > max_bytes:
        raise FaceSwapError(f"{field} exceeds {max_bytes} bytes")
    return data


def parse_upload(value: object, field: str, max_bytes: int) -> bytes:
    if not isinstance(value, dict):
        raise FaceSwapError(f"{field} must be an object with name and bytes_b64")
    if set(value) != {"name", "bytes_b64"}:
        raise FaceSwapError(f"{field} must contain only name and bytes_b64")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise FaceSwapError(f"{field}.name must be a non-empty string")
    return decode_base64(value["bytes_b64"], field, max_bytes)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


def _iou(left: BBox, right: BBox) -> float:
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _merge_embedding(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    if len(left) != len(right):
        return tuple(right)
    merged = tuple(a + b for a, b in zip(left, right))
    norm = math.sqrt(sum(x * x for x in merged))
    return tuple(x / norm for x in merged) if norm else tuple(right)


class FaceTracker:
    """Greedy identity matching without a tracking dependency."""

    def __init__(self) -> None:
        self.tracks: list[FaceTrack] = []

    def update(
        self, detections: Sequence[FaceDetection], frame_index: int
    ) -> tuple[tuple[str, ...], tuple[FaceTrack, ...]]:
        ordered = sorted(enumerate(detections), key=lambda item: (item[1].bbox[0], item[0]))
        candidates = []
        for track_index, track in enumerate(self.tracks):
            for detection_index, (_, detection) in enumerate(ordered):
                cosine = _cosine(track.embedding, detection.embedding)
                overlap = _iou(track.bbox, detection.bbox)
                if cosine >= MIN_COSINE and overlap >= MIN_IOU:
                    candidates.append(
                        (-(cosine + overlap), track_index, detection_index, detection)
                    )
        candidates.sort(key=lambda item: item[:3])

        assignments: list[str | None] = [None] * len(detections)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_index, detection_index, detection in candidates:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            track = self.tracks[track_index]
            track.embedding = _merge_embedding(track.embedding, detection.embedding)
            track.bbox = detection.bbox
            track.last_frame = frame_index
            track.frames += 1
            assignments[ordered[detection_index][0]] = track.face_id
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for detection_index, (original_index, detection) in enumerate(ordered):
            if detection_index in matched_detections:
                continue
            track = FaceTrack(
                face_id=f"face_{len(self.tracks) + 1:04d}",
                embedding=tuple(detection.embedding),
                bbox=detection.bbox,
                first_frame=frame_index,
                last_frame=frame_index,
                frames=1,
            )
            self.tracks.append(track)
            assignments[original_index] = track.face_id

        return tuple(face_id for face_id in assignments if face_id is not None), tuple(self.tracks)


def choose_largest_face(detections: Sequence[FaceDetection]) -> int:
    if not detections:
        raise FaceSwapError("no face found in source image")

    def rank(index: int) -> tuple[float, float, float]:
        box = detections[index].bbox
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        return area, -box[0], -box[1]

    return max(range(len(detections)), key=rank)


def _open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    if not path.is_file():
        raise FaceSwapError(f"target video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FaceSwapError("target video could not be read")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise FaceSwapError("target video has invalid dimensions")
    if max(width, height) > MAX_TARGET_PIXELS:
        capture.release()
        raise FaceSwapError(f"target video exceeds {MAX_TARGET_PIXELS} pixels on the longest side")
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise FaceSwapError("target video has an invalid frame rate")
    if frame_count > MAX_TARGET_FRAMES:
        capture.release()
        raise FaceSwapError(f"target video exceeds {MAX_TARGET_FRAMES} frames")
    if frame_count > 0 and frame_count / fps > MAX_TARGET_SECONDS + 1e-6:
        capture.release()
        raise FaceSwapError(f"target video exceeds {MAX_TARGET_SECONDS:g} seconds")
    return capture, VideoInfo(width, height, fps)


def _check_frame(frame: np.ndarray, info: VideoInfo, frame_index: int) -> None:
    height, width = frame.shape[:2]
    if (width, height) != (info.width, info.height):
        raise FaceSwapError("target video changes dimensions between frames")
    if frame_index >= MAX_TARGET_FRAMES:
        raise FaceSwapError(f"target video exceeds {MAX_TARGET_FRAMES} frames")
    if (frame_index + 1) / info.fps > MAX_TARGET_SECONDS + 1e-6:
        raise FaceSwapError(f"target video exceeds {MAX_TARGET_SECONDS:g} seconds")


Detector = Callable[[np.ndarray], tuple[list[FaceDetection], list[object]]]
Swapper = Callable[[np.ndarray, object], np.ndarray]

LABEL_FACE_ID_COLOR = (0, 0, 255)
LABEL_SWAPPED_COLOR = (0, 255, 0)


def _gender(model_face: object) -> str:
    sex = getattr(model_face, "sex", "")
    return {"M": "M", "F": "F"}.get(str(sex).upper(), "")


def _annotate(frame: np.ndarray, labels: Sequence[tuple[BBox, str, tuple]], shift: int) -> np.ndarray:
    """Draw one box plus label per face; shift moves the box to a stacked copy."""
    for (x1, y1, x2, y2), text, color in labels:
        left, top, right, bottom = (int(value) for value in (x1 + shift, y1, x2 + shift, y2))
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        if text:
            cv2.rectangle(frame, (left, top - 18), (left + 10 * len(text), top), color, -1)
            cv2.putText(frame, text, (left + 2, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _frame_labels(
    assignments: Sequence[str], detections: Sequence[FaceDetection], model_faces: Sequence[object]
) -> list[tuple[BBox, str, tuple]]:
    labels = []
    for face_id, detection, model_face in zip(assignments, detections, model_faces):
        gender = _gender(model_face)
        text = f"{face_id} {gender}".strip()
        labels.append((detection.bbox, text, LABEL_FACE_ID_COLOR))
    return labels


def _marked_labels(labels: list[tuple[BBox, str, tuple]], selection: TargetFaces) -> list[tuple[BBox, str, tuple]]:
    """Labels for the swapped side: swapped faces get a green 'swapped' tag."""
    return [
        (bbox, f"{text} swapped", LABEL_SWAPPED_COLOR)
        if selection.all_faces or text.startswith(selection.face_id or "\0")
        else item
        for item in labels
        for bbox, text in [(item[0], item[1])]
    ]


def analyze_video(target: Path, detector: Detector) -> tuple[dict, ...]:
    capture, info = _open_video(target)
    tracker = FaceTracker()
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            _check_frame(frame, info, frame_index)
            detections, _ = detector(frame)
            tracker.update(detections, frame_index)
            frame_index += 1
    finally:
        capture.release()
    if frame_index == 0:
        raise FaceSwapError("target video contains no readable frames")
    if not tracker.tracks:
        raise FaceSwapError("no face found in target video")
    return tuple(track.as_dict() for track in tracker.tracks)


def _start_ffmpeg(target: Path, output: Path, info: VideoInfo) -> subprocess.Popen:
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{info.width}x{info.height}",
        "-framerate", f"{info.fps:.9g}", "-i", "pipe:0",
        "-i", str(target),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-af", "apad", "-shortest", "-movflags", "+faststart", str(output),
    ]
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise FaceSwapError("ffmpeg is required but could not be started") from exc


def swap_video(
    target: Path,
    output: Path,
    selection: TargetFaces,
    detector: Detector,
    swapper: Swapper,
    compare: bool = False,
) -> SwapResult:
    capture, info = _open_video(target)
    tracker = FaceTracker()
    swapped_faces = 0
    process: subprocess.Popen | None = None
    write_info = VideoInfo(info.width * 2, info.height, info.fps) if compare else info
    try:
        process = _start_ffmpeg(target, output, write_info)
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            _check_frame(frame, info, frame_index)
            detections, model_faces = detector(frame)
            assignments, _ = tracker.update(detections, frame_index)
            labels = _frame_labels(assignments, detections, model_faces)
            if compare:
                original = frame.copy()
            for face_id, model_face in zip(assignments, model_faces):
                if selection.all_faces or face_id == selection.face_id:
                    frame = swapper(frame, model_face)
                    swapped_faces += 1
            if compare:
                frame = np.hstack([original, frame])
                _annotate(frame, labels, 0)
                _annotate(frame, _marked_labels(labels, selection), info.width)
            try:
                process.stdin.write(frame.tobytes())
            except OSError as exc:
                raise FaceSwapError("ffmpeg stopped before all frames were written") from exc
            frame_index += 1
        if frame_index == 0:
            raise FaceSwapError("target video contains no readable frames")
        if not tracker.tracks:
            raise FaceSwapError("no face found in target video")
        if not selection.all_faces and not any(
            track.face_id == selection.face_id for track in tracker.tracks
        ):
            raise FaceSwapError(f"target face {selection.face_id!r} was not found")
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code:
            raise FaceSwapError(f"ffmpeg failed: {stderr or f'exit code {return_code}'}")
        return SwapResult(
            faces=tuple(track.as_dict() for track in tracker.tracks),
            swapped_faces=swapped_faces,
        )
    except Exception:
        if process is not None:
            process.kill()
            process.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        if process is not None and process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass


def _default_model_root() -> Path:
    configured = os.environ.get("INSIGHTFACE_ROOT")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "ncnx" / "insightface"


def _providers() -> list[str]:
    """NCNX_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider for GPU runtimes."""
    configured = os.environ.get("NCNX_ONNX_PROVIDERS")
    return configured.split(",") if configured else ["CPUExecutionProvider"]


class InsightFaceEngine:
    """Lazy CPU model adapter shared by all requests."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else _default_model_root()
        self._detector = None
        self._swapper = None
        self._lock = threading.RLock()

    def _load_detector(self):
        if self._detector is not None:
            return self._detector
        model_dir = self.root / "models" / "buffalo_l"
        if not (model_dir / "det_10g.onnx").is_file():
            raise FaceSwapError(
                "buffalo_l models are missing; run python3 scripts/download_models.py "
                f"--root {self.root}"
            )
        try:
            from insightface.app import FaceAnalysis

            self._detector = FaceAnalysis(
                name="buffalo_l", root=str(self.root), providers=_providers()
            )
            self._detector.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as exc:
            raise FaceSwapError(f"could not load buffalo_l models: {exc}") from exc
        return self._detector

    def _load_swapper(self):
        if self._swapper is not None:
            return self._swapper
        model_path = self.root / "models" / "inswapper_128.onnx"
        if not model_path.is_file():
            raise FaceSwapError(
                "inswapper_128.onnx is missing; run python3 scripts/download_models.py "
                f"--root {self.root}"
            )
        try:
            import insightface

            self._swapper = insightface.model_zoo.get_model(
                str(model_path), providers=_providers()
            )
        except Exception as exc:
            raise FaceSwapError(f"could not load inswapper_128.onnx: {exc}") from exc
        return self._swapper

    def _detect(self, frame: np.ndarray) -> tuple[list[FaceDetection], list[object]]:
        faces = sorted(self._load_detector().get(frame), key=lambda face: float(face.bbox[0]))
        detections = []
        for face in faces:
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = face.embedding
            detections.append(
                FaceDetection(
                    bbox=tuple(float(value) for value in face.bbox),
                    embedding=tuple(float(value) for value in embedding),
                )
            )
        return detections, faces

    def _load_source(self, source: Path) -> object:
        try:
            data = np.fromfile(source, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        except (cv2.error, OSError) as exc:
            raise FaceSwapError("source image could not be read") from exc
        if image is None:
            raise FaceSwapError("source image could not be read")
        detections, faces = self._detect(image)
        return faces[choose_largest_face(detections)]

    def analyze(self, target: Path) -> tuple[dict, ...]:
        with self._lock:
            return analyze_video(target, self._detect)

    def swap(
        self, source: Path, target: Path, selection: TargetFaces, output: Path, compare: bool = False
    ) -> SwapResult:
        if output.resolve() in {source.resolve(), target.resolve()}:
            raise FaceSwapError("output path must differ from the source and target paths")
        with self._lock:
            source_face = self._load_source(source)
            swap_model = self._load_swapper()
            return swap_video(
                target,
                output,
                selection,
                self._detect,
                lambda frame, face: swap_model.get(frame, face, source_face, paste_back=True),
                compare=compare,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--analyze", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--all", action="store_true")
    modes.add_argument("--face-id")
    parser.add_argument("--out", type=Path, default=Path("out/swapped.mp4"))
    parser.add_argument("--compare", action="store_true", help="side-by-side original and swap with face boxes")
    args = parser.parse_args(argv)

    try:
        engine = InsightFaceEngine()
        if args.analyze:
            if len(args.files) != 1 or args.all or args.face_id:
                parser.error("--analyze requires one TARGET.mp4 and no face selection")
            result = {"faces": list(engine.analyze(args.files[0]))}
        else:
            if len(args.files) != 2 or not (args.all or args.face_id):
                parser.error("a swap requires SOURCE.jpg TARGET.mp4 and --all or --face-id")
            selection = TargetFaces() if args.all else parse_target_faces({"face_id": args.face_id})
            args.out.parent.mkdir(parents=True, exist_ok=True)
            swapped = engine.swap(args.files[0], args.files[1], selection, args.out, compare=args.compare)
            result = {
                "faces": list(swapped.faces),
                "swapped_faces": swapped.swapped_faces,
                "output": str(args.out),
            }
    except (FaceSwapError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
