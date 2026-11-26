"""Invoke a deployed face-insertion Runtime and save its MP4 response.

Usage:
    python3 scripts/invoke_deployed.py RUNTIME_ARN SOURCE.jpg TARGET.mp4 --all
    python3 scripts/invoke_deployed.py RUNTIME_ARN SOURCE.jpg TARGET.mp4 --face-id face_0001
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_arn")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--all", action="store_true")
    modes.add_argument("--face-id")
    parser.add_argument("--compare", action="store_true", help="request the side-by-side review clip")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="ap-southeast-2")
    parser.add_argument("--endpoint", default="production")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "deployed_swapped.mp4")
    args = parser.parse_args(argv)

    try:
        payload = {
            "source_face": {
                "name": args.source.name,
                "bytes_b64": base64.b64encode(args.source.read_bytes()).decode("ascii"),
            },
            "target_video": {
                "name": args.target.name,
                "bytes_b64": base64.b64encode(args.target.read_bytes()).decode("ascii"),
            },
            "target_faces": "all" if args.all else {"face_id": args.face_id},
        }
        if args.compare:
            payload["compare"] = True
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        client = session.client(
            "bedrock-agentcore",
            config=Config(retries={"total_max_attempts": 5, "mode": "adaptive"}, read_timeout=900),
        )
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
        if not all(
            isinstance(face, dict) and isinstance(face.get("face_id"), str)
            for face in result["faces"]
        ):
            raise RuntimeError("runtime returned an invalid face list")
        if not isinstance(result.get("swapped_faces"), int) or result["swapped_faces"] <= 0:
            raise RuntimeError("runtime returned an invalid swapped_faces count")
        clip = base64.b64decode(result["clip_b64"], validate=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(clip)
    except (BotoCoreError, OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    print(
        "PASS",
        "faces", [face["face_id"] for face in result["faces"]],
        "swapped", result["swapped_faces"],
        "clip bytes", len(clip),
        "output", args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
