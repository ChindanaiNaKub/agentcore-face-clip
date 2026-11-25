"""Check the AWS setup for the face-to-video assignment.

Usage: python3 scripts/check_aws.py [photo.jpg]

Without a photo, checks AWS services and local tools. With a photo, also
runs Rekognition DetectFaces and prints each face's gender and box.
"""
import importlib.metadata
import shutil
import sys

import boto3
from botocore.exceptions import ClientError


def check(name, fn):
    try:
        print(f"PASS {name}: {fn()}")
    except Exception as exc:
        print(f"FAIL {name}: {exc}")


def identity():
    return boto3.client("sts").get_caller_identity()["Arn"]


def rekognition():
    count = len(boto3.client("rekognition").list_collections()["CollectionIds"])
    return f"service reachable, {count} collections"


def bedrock_catalog():
    count = len(boto3.client("bedrock").list_foundation_models()["modelSummaries"])
    return f"{count} models listed"


def bedrock_converse():
    # shortcut: plain amazon.nova-2-lite-v1:0 is rejected, on-demand needs a profile id
    client = boto3.client("bedrock-runtime")
    try:
        r = client.converse(
            modelId="global.amazon.nova-2-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": "Reply with the single word OK"}]}],
        )
        return f"nova-2-lite replied {r['output']['message']['content'][0]['text']!r}"
    except ClientError as exc:
        msg = exc.response.get("Error", {}).get("Message", "")
        if "Too many tokens" in msg:
            print("WARN bedrock converse: account daily token cap reached, retry tomorrow")
            return "throttled today (account daily cap)"
        raise


def agentcore_runtimes():
    runtimes = boto3.client("bedrock-agentcore-control").list_agent_runtimes()["agentRuntimes"]
    return ", ".join(f"{r['agentRuntimeName']}={r['status']}" for r in runtimes) or "none"


def package(name):
    return f"{name} {importlib.metadata.version(name)}"


def ffmpeg():
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("not on PATH")
    return path


def detect_faces(path):
    with open(path, "rb") as f:
        r = boto3.client("rekognition").detect_faces(
            Image={"Bytes": f.read()}, Attributes=["ALL"]
        )
    lines = []
    for i, face in enumerate(r["FaceDetails"], 1):
        g = face["Gender"]
        b = face["BoundingBox"]
        lines.append(
            f"face {i}: {g['Value']} ({g['Confidence']:.0f}%) "
            f"box=({b['Left']:.2f},{b['Top']:.2f},{b['Width']:.2f},{b['Height']:.2f})"
        )
    return "\n      ".join(lines) if lines else "no faces found"


def main() -> int:
    check("identity", identity)
    check("rekognition", rekognition)
    check("bedrock catalog", bedrock_catalog)
    check("bedrock converse", bedrock_converse)
    check("agentcore runtimes", agentcore_runtimes)
    check("agentcore sdk", lambda: package("bedrock-agentcore"))
    check("pillow", lambda: package("pillow"))
    check("ffmpeg", ffmpeg)
    if len(sys.argv) > 1:
        check("detect_faces", lambda: detect_faces(sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
