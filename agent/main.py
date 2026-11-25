"""HTTP agent for face insertion into a supplied target video."""
from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from scripts.face_swap import (  # noqa: E402
        MAX_TARGET_BYTES,
        FaceSwapError,
        InsightFaceEngine,
        TargetFaces,
        parse_target_faces,
        parse_upload,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts.face_swap"}:
        raise
    from face_swap import (  # noqa: E402
        MAX_TARGET_BYTES,
        FaceSwapError,
        InsightFaceEngine,
        TargetFaces,
        parse_target_faces,
        parse_upload,
    )

MAX_SOURCE_BYTES = 8 * 1024 * 1024


def create_app(engine: InsightFaceEngine | None = None) -> FastAPI:
    app = FastAPI()
    app.state.engine = engine if engine is not None else InsightFaceEngine()

    def error(message: str) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=400)

    @app.get("/ping")
    def ping():
        return {"status": "Healthy"}

    @app.get("/health")
    def health():
        return {"status": "Healthy"}

    @app.post("/analyze")
    def analyze(req: dict):
        if set(req) != {"target_video"}:
            return error("analyze accepts only target_video")
        try:
            target_bytes = parse_upload(req["target_video"], "target_video", MAX_TARGET_BYTES)
            with tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "target.mp4"
                target.write_bytes(target_bytes)
                faces = app.state.engine.analyze(target)
            return {"faces": [dict(face) for face in faces]}
        except FaceSwapError as exc:
            return error(str(exc))

    @app.post("/invocations")
    def invoke(req: dict):
        if set(req) != {"source_face", "target_video", "target_faces"}:
            return error("invocations accepts only source_face, target_video, and target_faces")
        try:
            selection = parse_target_faces(req["target_faces"])
            source_bytes = parse_upload(req["source_face"], "source_face", MAX_SOURCE_BYTES)
            target_bytes = parse_upload(req["target_video"], "target_video", MAX_TARGET_BYTES)
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source = directory / "source.img"
                target = directory / "target.mp4"
                output = directory / "swapped.mp4"
                source.write_bytes(source_bytes)
                target.write_bytes(target_bytes)
                result = app.state.engine.swap(source, target, selection, output)
                clip = base64.b64encode(output.read_bytes()).decode("ascii")
            return {
                "faces": [dict(face) for face in result.faces],
                "swapped_faces": result.swapped_faces,
                "clip_b64": clip,
            }
        except FaceSwapError as exc:
            return error(str(exc))

    return app


app = create_app()
