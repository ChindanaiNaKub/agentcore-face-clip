"""Detect faces, classify gender, crop each face, render a video clip.

Usage: python3 scripts/face_clip.py IMG [IMG ...] --out out/faces.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path

import boto3
from PIL import Image, ImageDraw, ImageFont, ImageOps

FRAME = 512  # square output frame size
SECONDS = 2  # display time per face
MARGIN = 0.3  # crop margin as a fraction of face size
FONT = ImageFont.load_default(32)


def detect_faces(client, path: Path) -> list:
    """Return the Rekognition FaceDetails list for one image."""
    # Rekognition uses PascalCase; Converse uses camelCase. Each API keeps
    # its native casing in boto3 1.43+.
    return client.detect_faces(
        Image={"Bytes": path.read_bytes()}, Attributes=["ALL"]
    )["FaceDetails"]


def crop(img: Image.Image, box: dict) -> Image.Image:
    """Crop the face box plus margin, clamped to the image, fit to a square."""
    w = box["Width"] * img.width
    h = box["Height"] * img.height
    left = max(0, box["Left"] * img.width - MARGIN * w)
    top = max(0, box["Top"] * img.height - MARGIN * h)
    right = min(img.width, left + w * (1 + 2 * MARGIN))
    bottom = min(img.height, top + h * (1 + 2 * MARGIN))
    return ImageOps.fit(img.crop((left, top, right, bottom)), (FRAME, FRAME))


def label(face_img: Image.Image, gender: str, confidence: float) -> None:
    """Write the gender prediction onto the crop, in place."""
    text = f" {gender} {confidence:.0f}% "
    bar = FONT.getbbox(text)[3] + 12
    d = ImageDraw.Draw(face_img)
    d.rectangle([0, 0, FRAME, bar], fill=(0, 0, 0))
    d.text((8, 6), text, font=FONT, fill=(255, 255, 255))


def render(frames_dir: Path, out: Path) -> None:
    """Encode the numbered PNG frames into the output clip."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", f"1/{SECONDS}",
            "-i", str(frames_dir / "%03d.png"),
            "-vf", "fps=25,format=yuv420p",
            "-c:v", "libx264", "-movflags", "+faststart",
            str(out),
        ],
        check=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("out/faces.mp4"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out.parent / "frames"
    frames_dir.mkdir(exist_ok=True)
    # shortcut: stale frames from an earlier run leak into the clip
    for old in frames_dir.glob("*.png"):
        old.unlink()

    client = boto3.client("rekognition")
    counts = {}
    n = 0
    for path in args.images:
        faces = detect_faces(client, path)
        print(f"{path.name}: {len(faces)} face(s)")
        for face in faces:
            gender = face["Gender"]
            counts[gender["Value"]] = counts.get(gender["Value"], 0) + 1
            img = crop(Image.open(path).convert("RGB"), face["BoundingBox"])
            label(img, gender["Value"], gender["Confidence"])
            img.save(frames_dir / f"{n:03d}.png")
            n += 1

    if n == 0:
        print("no faces found in any image", file=sys.stderr)
        return 1
    render(frames_dir, args.out)
    print(f"{n} faces {counts} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
