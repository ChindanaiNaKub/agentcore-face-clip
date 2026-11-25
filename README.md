# AgentCore face clip

Detect faces, predict gender, crop each face, and render the crops into an MP4 clip. The agent runs as an HTTP service on Amazon Bedrock AgentCore Runtime.

## Demo

Open [out/deployed_faces.mp4](out/deployed_faces.mp4) to see the deployed result.

The demo uses three photos. It finds three faces, two Male and one Female. The clip is H.264, 512x512, 25 fps, and 6 seconds long.

Gender is a model prediction. It can be wrong. Do not use it for sensitive decisions.

## Architecture

1. Amazon Rekognition `DetectFaces` returns face boxes and gender predictions.
2. Pillow crops each face and adds a label.
3. ffmpeg renders the crops into an MP4 clip.
4. FastAPI serves the pipeline over HTTP.
5. AgentCore Runtime runs the ARM64 container.

The pipeline does not call a language model. OpenRouter and Amazon Bedrock model access are not required.

## Requirements

- Python 3.12 or later
- AWS credentials with `rekognition:DetectFaces`
- ffmpeg on `PATH`
- Docker with buildx for the container build

## Run locally

Install the dependencies.

```bash
python3 -m pip install -r requirements.txt
```

Set your AWS region and profile if needed.

```bash
export AWS_REGION=ap-southeast-2
export AWS_PROFILE=personal
```

Run the pipeline.

```bash
python3 scripts/face_clip.py testdata/*.jpg --out out/faces.mp4
```

Run the HTTP agent.

```bash
PYTHONPATH=scripts:agent python3 -m uvicorn main:app --app-dir agent --host 127.0.0.1 --port 8080
```

Run the smoke test in another terminal.

```bash
python3 agent/smoke_test.py
```

The request shape is a JSON object with an `images` list. Each item has a `name` and a base64 encoded `bytes_b64` value. The response contains `faces`, `counts`, and `clip_b64`.

## Build the container

AgentCore Runtime requires ARM64.

```bash
docker buildx build --platform linux/arm64 --load -f agent/Dockerfile -t face-clip-agent:test .
```

Test the health endpoint.

```bash
docker run --rm --platform linux/arm64 -p 127.0.0.1:8080:8080 face-clip-agent:test
```

Call `http://127.0.0.1:8080/ping` from another terminal.

## Deploy to AgentCore Runtime

1. Push the ARM64 image to a private ECR repository.
2. Create an execution role with ECR pull, CloudWatch Logs, and Rekognition permissions.
3. Create an HTTP Runtime on port 8080 with `/invocations` as the request path.
4. Create a Runtime endpoint and wait for `READY`.
5. Invoke the endpoint with the script below.

Use IAM SigV4 for inbound authentication. Use VPC mode before production use.

## Invoke a deployed Runtime

```bash
python3 scripts/invoke_deployed.py <runtime-arn> --region ap-southeast-2 --endpoint production
```

Add `--profile <profile>` if the credentials use a named AWS profile. The script sends the three test photos, checks the face counts, and saves `out/deployed_faces.mp4`.

## Verification

The deployed Runtime returned HTTP 200. CloudWatch logs showed ffmpeg running on ARM64. CloudWatch metrics in `AWS/Bedrock-AgentCore` recorded invocations, sessions, and zero errors.

The console can lag behind a new call. Check the Runtime logs and metrics if the Agents card still shows no activity.

## Test data

See [testdata/README.md](testdata/README.md) for sources and licenses.

## License

The code is under the MIT license. See [LICENSE](LICENSE). The test images keep their own licenses.
