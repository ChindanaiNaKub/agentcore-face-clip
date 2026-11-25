"""Smoke test a running face-insertion API with the real models.

Usage:
    PYTHONPATH=scripts:agent python3 -m uvicorn main:app --app-dir agent --port 8080
    python3 agent/smoke_test.py [TARGET.mp4]

Without TARGET.mp4, this script creates a temporary two-face video from testdata.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def upload(path: Path) -> dict:
    return {"name": path.name, "bytes_b64": base64.b64encode(path.read_bytes()).decode("ascii")}


def make_two_face_video(path: Path) -> None:
    left = cv2.imread(str(ROOT / "testdata" / "rose.jpg"))
    right = cv2.imread(str(ROOT / "testdata" / "young_man.jpg"))
    if left is None or right is None:
        raise RuntimeError("testdata face images could not be read")
    frame = np.full((512, 1024, 3), 255, dtype=np.uint8)
    frame[:, :512] = cv2.resize(left, (512, 512))
    frame[:, 512:] = cv2.resize(right, (512, 512))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (1024, 512))
    if not writer.isOpened():
        raise RuntimeError("could not create the smoke-test video")
    for _ in range(10):
        writer.write(frame)
    writer.release()


def check_clip(data: bytes, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    capture = cv2.VideoCapture(str(path))
    try:
        frames = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frames += 1
    finally:
        capture.release()
    if frames == 0:
        raise RuntimeError(f"returned clip has no readable frames: {path}")
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "testdata" / "franklin.jpg")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out")
    args = parser.parse_args(argv)

    try:
        with tempfile.TemporaryDirectory() as temporary:
            target = args.target
            if target is None:
                target = Path(temporary) / "two_faces.mp4"
                make_two_face_video(target)
            target_upload = upload(target)
            analyzed = post(f"{args.url}/analyze", {"target_video": target_upload})
            faces = analyzed.get("faces", [])
            face_ids = [face["face_id"] for face in faces]
            if not face_ids:
                raise RuntimeError("analyze returned no face IDs")
            if args.target is None and face_ids != ["face_0001", "face_0002"]:
                raise RuntimeError(f"expected two deterministic face IDs, got {face_ids}")

            source_upload = upload(args.source)
            all_result = post(
                f"{args.url}/invocations",
                {
                    "source_face": source_upload,
                    "target_video": target_upload,
                    "target_faces": "all",
                },
            )
            one_result = post(
                f"{args.url}/invocations",
                {
                    "source_face": source_upload,
                    "target_video": target_upload,
                    "target_faces": {"face_id": face_ids[0]},
                },
            )
            all_frames = check_clip(
                base64.b64decode(all_result["clip_b64"], validate=True),
                args.out_dir / "smoke_all.mp4",
            )
            one_frames = check_clip(
                base64.b64decode(one_result["clip_b64"], validate=True),
                args.out_dir / "smoke_one.mp4",
            )
            expected_all = sum(face["frames"] for face in faces)
            selected = next(face for face in faces if face["face_id"] == face_ids[0])
            if all_result["swapped_faces"] != expected_all:
                raise RuntimeError("all mode did not swap every tracked face")
            if one_result["swapped_faces"] != selected["frames"]:
                raise RuntimeError("selected mode did not swap every occurrence of its track")
            if all_frames != one_frames:
                raise RuntimeError("all and selected clips have different frame counts")
    except (KeyError, OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    print(
        "PASS",
        json.dumps(
            {
                "face_ids": face_ids,
                "all_swapped_faces": all_result["swapped_faces"],
                "selected_face": face_ids[0],
                "selected_swapped_faces": one_result["swapped_faces"],
                "frames": all_frames,
                "outputs": [str(args.out_dir / "smoke_all.mp4"), str(args.out_dir / "smoke_one.mp4")],
            }
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
