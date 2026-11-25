"""AgentCore HTTP agent: faces -> gender -> video clip.

POST /invocations with {"images": [{"name": "a.jpg", "bytes_b64": "..."}]}.
Returns {"faces": [...], "counts": {...}, "clip_b64": "<mp4>"}.
"""
import base64
import binascii
import tempfile
from pathlib import Path

import boto3
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from face_clip import FRAME, crop, detect_faces, label, render

MAX_IMAGES = 10
MAX_BYTES = 8 * 1024 * 1024  # per image, keep under AgentCore body limits

app = FastAPI()


@app.get("/ping")
async def ping():
    return {"status": "Healthy"}


@app.get("/health")
async def health():
    return {"status": "Healthy"}


@app.post("/invocations")
async def invoke(req: dict):
    imgs = req.get("images") if isinstance(req, dict) else None
    if not isinstance(imgs, list) or not 1 <= len(imgs) <= MAX_IMAGES:
        return JSONResponse({"error": f"images must be a list of 1-{MAX_IMAGES} items"}, status_code=400)

    rekognition = boto3.client("rekognition")
    faces, counts = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames = tmp / "frames"
        frames.mkdir()
        n = 0
        for i, item in enumerate(imgs):
            if not isinstance(item, dict) or not item.get("bytes_b64"):
                return JSONResponse({"error": f"images[{i}] needs bytes_b64"}, status_code=400)
            try:
                data = base64.b64decode(item["bytes_b64"], validate=True)
            except (binascii.Error, ValueError):
                return JSONResponse({"error": f"images[{i}] bytes_b64 is not valid base64"}, status_code=400)
            if len(data) > MAX_BYTES:
                return JSONResponse({"error": f"images[{i}] exceeds {MAX_BYTES} bytes"}, status_code=400)
            name = item.get("name", f"image{i}")
            (tmp / name).write_bytes(data)
            detected = detect_faces(rekognition, tmp / name)
            for face in detected:
                g = face["Gender"]
                counts[g["Value"]] = counts.get(g["Value"], 0) + 1
                img = crop(decode_image(data), face["BoundingBox"])
                label(img, g["Value"], g["Confidence"])
                img.save(frames / f"{n:03d}.png")
                faces.append({"name": name, "gender": g["Value"], "confidence": g["Confidence"]})
                n += 1
        if n == 0:
            return JSONResponse({"faces": [], "counts": {}, "clip_b64": None})
        render(frames, tmp / "faces.mp4")
        clip = base64.b64encode((tmp / "faces.mp4").read_bytes()).decode()
    return {"faces": faces, "counts": counts, "clip_b64": clip, "frames": FRAME}


def decode_image(data: bytes):
    from PIL import Image
    import io
    return Image.open(io.BytesIO(data)).convert("RGB")
