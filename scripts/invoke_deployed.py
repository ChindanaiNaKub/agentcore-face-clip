"""Invoke the deployed face clip agent and save its MP4 response."""

import argparse
import base64
import json
import sys
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHOTOS = [ROOT / "testdata" / name for name in ("rose.jpg", "young_man.jpg", "franklin.jpg")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_arn")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="ap-southeast-2")
    parser.add_argument("--endpoint", default="production")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "deployed_faces.mp4")
    parser.add_argument("photos", nargs="*", type=Path, default=DEFAULT_PHOTOS)
    args = parser.parse_args()

    payload = {
        "images": [
            {
                "name": path.name,
                "bytes_b64": base64.b64encode(path.read_bytes()).decode(),
            }
            for path in args.photos
        ]
    }
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = session.client(
        "bedrock-agentcore",
        config=Config(retries={"total_max_attempts": 5, "mode": "adaptive"}, read_timeout=300),
    )
    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=args.runtime_arn,
            qualifier=args.endpoint,
            runtimeSessionId=str(uuid.uuid4()),
            contentType="application/json",
            accept="application/json",
            payload=json.dumps(payload).encode(),
        )
        if response["statusCode"] != 200:
            raise RuntimeError(f"runtime returned HTTP {response['statusCode']}")
        result = json.loads(response["response"].read())
        if not result.get("faces") or not result.get("clip_b64"):
            raise RuntimeError("runtime returned no faces or no clip")
        if sum(result["counts"].values()) != len(result["faces"]):
            raise RuntimeError("face counts do not match the face list")
        clip = base64.b64decode(result["clip_b64"], validate=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(clip)
    except ClientError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    print("PASS", result["counts"], "clip bytes", len(clip), "output", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
