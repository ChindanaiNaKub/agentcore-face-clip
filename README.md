# Face insertion agent

Insert a source face into a supplied target video. The agent runs as an HTTP service on Amazon Bedrock AgentCore Runtime.

Choose one of two target modes:

- `all` swaps every detected face.
- `{"face_id": "face_0001"}` swaps one tracked face.

`POST /analyze` detects faces without swapping them. It returns stable track IDs for the selected mode.

## Architecture

1. InsightFace `buffalo_l` detects faces and embeddings.
2. Greedy cosine-similarity and bounding-box IoU matching assigns track IDs.
3. InsightFace `inswapper_128.onnx` inserts the largest source face.
4. Frames pass to ffmpeg through a raw-video pipe.
5. ffmpeg writes H.264 MP4 and keeps target audio when present.
6. FastAPI serves the pipeline. One lock serializes access to the shared CPU models.

The swap model loads on the first swap request. No language model or AWS model access is required.

## Model license

The code is under the MIT license. The InsightFace `buffalo_l` and `inswapper_128.onnx` models are **for non-commercial research only**. The code license does not grant commercial rights to these models. Obtain the required permission before commercial use.

See the [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) for model details.

## Requirements

- Python 3.12 or later
- ffmpeg on `PATH`
- Docker with buildx for the ARM64 container build

## Install and download models

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_models.py --root ~/.cache/ncnx/insightface
export INSIGHTFACE_ROOT=~/.cache/ncnx/insightface
```

The downloader verifies these SHA-256 values before installing the models:

- `buffalo_l.zip`: `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`
- `inswapper_128.onnx`: `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af`

Model binaries stay outside git. The default root is `~/.cache/ncnx/insightface`.

## Command line

List target face IDs:

```bash
python3 scripts/face_swap.py TARGET.mp4 --analyze
```

Swap all target faces:

```bash
python3 scripts/face_swap.py SOURCE.jpg TARGET.mp4 --all --out out/swapped.mp4
```

Swap one tracked face:

```bash
python3 scripts/face_swap.py SOURCE.jpg TARGET.mp4 --face-id face_0001 --out out/swapped.mp4
```

`--all` and `--face-id` are mutually exclusive. A swap requires one of them.

## HTTP API

Start the service:

```bash
PYTHONPATH=scripts:agent python3 -m uvicorn main:app --app-dir agent --host 127.0.0.1 --port 8080
```

Check `GET /ping` or `GET /health`. Both return `{"status": "Healthy"}` without loading models.

Analyze a target video:

```json
POST /analyze
{"target_video": {"name": "target.mp4", "bytes_b64": "..."}}
```

The response contains `faces`. Each face has `face_id`, `first_frame`, `last_frame`, `frames`, and its latest `bbox` as `[left, top, right, bottom]`.

Insert a face:

```json
POST /invocations
{
  "source_face": {"name": "source.jpg", "bytes_b64": "..."},
  "target_video": {"name": "target.mp4", "bytes_b64": "..."},
  "target_faces": "all"
}
```

Set `target_faces` to `{"face_id": "face_0001"}` to swap one track. The response contains `faces`, the `swapped_faces` count, and `clip_b64` as a base64 MP4.

The API ignores client filenames and uses fixed temporary filenames. Source images are limited to 8 MiB. Target videos are limited to 50 MiB, 20 seconds, 150 frames, and 1280 pixels on the longest side.

Track IDs are deterministic. Detections sort by left edge. New IDs follow first-appearance order. Later detections match tracks by embedding cosine similarity plus bounding-box IoU.

## Verify locally

Run the focused tests:

```bash
python3 -m pytest -q
```

Run the optional real-model test:

```bash
RUN_REAL_MODEL_TESTS=1 INSIGHTFACE_ROOT=~/.cache/ncnx/insightface python3 -m pytest -q tests/test_real_models.py
```

Start the API, then run the smoke test in another terminal:

```bash
python3 agent/smoke_test.py
```

The smoke test creates a temporary two-face video. It checks analyze, all-face swap, and selected-face swap. It saves `out/smoke_all.mp4` and `out/smoke_one.mp4`. Pass a target path to test another video.

## Build the container

AgentCore Runtime requires ARM64.

```bash
docker buildx build --platform linux/arm64 --load -f agent/Dockerfile -t face-insert-agent:test .
```

Run the health check:

```bash
docker run --rm --platform linux/arm64 -p 127.0.0.1:8080:8080 face-insert-agent:test
```

Call `http://127.0.0.1:8080/ping` from another terminal.

The image downloads and verifies the models at build time. Models are stored at `/opt/insightface/models` and are readable by the non-root runtime user.

## Deploy to AgentCore Runtime

1. Push the ARM64 image to a private ECR repository.
2. Create an execution role with ECR pull and CloudWatch Logs permissions.
3. Create an HTTP Runtime on port 8080 with `/invocations` as the request path.
4. Create a Runtime endpoint and wait for `READY`.
5. Invoke the endpoint with the script below.

Use IAM SigV4 for inbound authentication. Use VPC mode before production use. Confirm model license rights before any commercial deployment.

## Invoke a deployed Runtime

Swap all faces:

```bash
python3 scripts/invoke_deployed.py <runtime-arn> testdata/franklin.jpg TARGET.mp4 --all --region ap-southeast-2 --endpoint production
```

Swap one face:

```bash
python3 scripts/invoke_deployed.py <runtime-arn> testdata/franklin.jpg TARGET.mp4 --face-id face_0001 --region ap-southeast-2 --endpoint production
```

Add `--profile <profile>` for a named AWS profile. The script saves `out/deployed_swapped.mp4` by default. Use `--out` to change the path.

## Demo results

Tested on 2026-11-26 with a CC BY 3.0 *Tears of Steel* excerpt and public-domain Grace Hopper and Alan Turing portraits.

| Run | Target | Tracks | Face swaps per source | Result |
| --- | --- | --- | --- | --- |
| Local | 6 seconds, 960×400, 144 frames | 2 | 288 | H.264/AAC MP4 |
| AgentCore | 2 seconds, 640×266, 48 frames | 3 | 96 | H.264/AAC MP4 |

AgentCore Runtime V2 used HTTP, ARM64, and `ap-southeast-2`. The endpoint reached `READY`. The first invocation took 571.75 seconds. The second took 332.16 seconds. The three-track AgentCore result shows one target identity split during motion.

Validation finished with `48 passed, 1 skipped`. The real-model test also passed.

## Limitations

- The InsightFace models are for non-commercial research only. The MIT code license does not grant commercial model rights.
- Processing uses CPU models and one lock. Requests are serialized. Long clips can exceed the AgentCore 15-minute synchronous timeout.
- Tracking uses cosine similarity and bounding-box overlap. Occlusion, fast motion, and large pose changes can split or merge identities. Track IDs apply only to one video.
- Only the largest face in the source image is used. Output quality varies with resolution, lighting, pose, and occlusion.
- The API loads full base64 requests and responses into memory. Input caps limit size, but large clips remain slow.
- The demo uses AgentCore `PUBLIC` network mode. The image also has known critical and high dependency findings. Patch, rescan, and use VPC mode before production.
- Face insertion can create misleading media. Obtain consent and follow applicable law. Source and target media licenses still apply.
- AgentCore, ECR, KMS, and CloudWatch Logs can incur charges while resources remain active.

## Test data

See [testdata/README.md](testdata/README.md) for image sources and licenses. Face insertion can create misleading media. Use it only with consent and follow applicable law.
