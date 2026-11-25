import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import agent.main as main_module
from agent.main import MAX_SOURCE_BYTES, create_app
from scripts.face_swap import FaceSwapError, SwapResult, TargetFaces


class FakeEngine:
    def __init__(self) -> None:
        self.analyze_calls = 0
        self.swap_calls = 0
        self.source_name = None
        self.target_name = None
        self.selection = None

    def analyze(self, target: Path):
        self.analyze_calls += 1
        self.target_name = target.name
        return (
            {"face_id": "face_0001", "first_frame": 0, "last_frame": 1, "frames": 2, "bbox": [0, 0, 1, 1]},
            {"face_id": "face_0002", "first_frame": 0, "last_frame": 1, "frames": 2, "bbox": [2, 0, 3, 1]},
        )

    def swap(self, source: Path, target: Path, selection: TargetFaces, output: Path):
        self.swap_calls += 1
        self.source_name = source.name
        self.target_name = target.name
        self.selection = selection
        output.write_bytes(b"fake mp4")
        return SwapResult(
            faces=({"face_id": "face_0001"}, {"face_id": "face_0002"}),
            swapped_faces=4 if selection.all_faces else 2,
        )


def upload(data: bytes, name="input.bin") -> dict:
    return {"name": name, "bytes_b64": base64.b64encode(data).decode("ascii")}


def payload(target_faces="all") -> dict:
    return {
        "source_face": upload(b"source", "../../source.jpg"),
        "target_video": upload(b"target", "../../target.mp4"),
        "target_faces": target_faces,
    }


def test_health_and_ping_do_not_need_models():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        assert client.get("/ping").json() == {"status": "Healthy"}
        assert client.get("/health").json() == {"status": "Healthy"}
    assert engine.analyze_calls == engine.swap_calls == 0


def test_analyze_accepts_only_target_video_and_returns_stable_ids():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post("/analyze", json={"target_video": upload(b"target")})
        assert response.status_code == 200
        assert [face["face_id"] for face in response.json()["faces"]] == [
            "face_0001", "face_0002"
        ]
        assert engine.analyze_calls == 1
        assert engine.target_name == "target.mp4"
        assert client.post("/analyze", json={"unexpected": []}).status_code == 400
        assert client.post(
            "/analyze", json={"target_video": upload(b"target"), "source_face": upload(b"source")}
        ).status_code == 400


@pytest.mark.parametrize("selection", ["all", {"face_id": "face_0001"}])
def test_invocations_returns_faces_count_and_clip(selection):
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post("/invocations", json=payload(selection))
    assert response.status_code == 200
    result = response.json()
    assert [face["face_id"] for face in result["faces"]] == ["face_0001", "face_0002"]
    assert result["swapped_faces"] == (4 if selection == "all" else 2)
    assert base64.b64decode(result["clip_b64"], validate=True) == b"fake mp4"
    assert engine.source_name == "source.img"
    assert engine.target_name == "target.mp4"
    assert engine.selection == (
        TargetFaces() if selection == "all" else TargetFaces(False, "face_0001")
    )


@pytest.mark.parametrize(
    "selection",
    [
        True,
        False,
        None,
        [],
        "ALL",
        {},
        {"face_id": ""},
        {"face_id": "  "},
        {"face_id": True},
        {"face_id": "face_0001", "extra": 1},
    ],
)
def test_invocations_rejects_invalid_target_selection_without_running_engine(selection):
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post("/invocations", json=payload(selection))
    assert response.status_code == 400
    assert "target_faces" in response.json()["error"]
    assert engine.swap_calls == 0


@pytest.mark.parametrize("field", ["source_face", "target_video"])
def test_invocations_rejects_invalid_base64_without_running_engine(field):
    engine = FakeEngine()
    request = payload()
    request[field]["bytes_b64"] = "not base64!"
    with TestClient(create_app(engine)) as client:
        response = client.post("/invocations", json=request)
    assert response.status_code == 400
    assert "valid base64" in response.json()["error"]
    assert engine.swap_calls == 0


def test_source_byte_limit_is_enforced_at_api_boundary():
    engine = FakeEngine()
    request = payload()
    request["source_face"] = upload(b"x" * (MAX_SOURCE_BYTES + 1))
    with TestClient(create_app(engine)) as client:
        response = client.post("/invocations", json=request)
    assert response.status_code == 400
    assert f"exceeds {MAX_SOURCE_BYTES} bytes" in response.json()["error"]
    assert engine.swap_calls == 0


def test_target_byte_limit_is_enforced_at_api_boundary(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(main_module, "MAX_TARGET_BYTES", 3)
    request = payload()
    request["target_video"] = upload(b"four")
    with TestClient(create_app(engine)) as client:
        response = client.post("/invocations", json=request)
    assert response.status_code == 400
    assert "exceeds 3 bytes" in response.json()["error"]
    assert engine.swap_calls == 0


def test_extra_keys_are_rejected():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        assert client.post("/invocations", json={"unexpected": []}).status_code == 400
        request = payload()
        request["extra"] = True
        assert client.post("/invocations", json=request).status_code == 400
    assert engine.swap_calls == 0


def test_engine_errors_are_clear_client_errors():
    class NoFaceEngine(FakeEngine):
        def analyze(self, target: Path):
            raise FaceSwapError("no face found in target video")

    with TestClient(create_app(NoFaceEngine())) as client:
        response = client.post("/analyze", json={"target_video": upload(b"target")})
    assert response.status_code == 400
    assert response.json() == {"error": "no face found in target video"}
