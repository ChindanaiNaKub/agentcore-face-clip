"""Smoke test for the agent: start it, send the test photos, check the clip.

Usage: PYTHONPATH=scripts:agent python3 -m uvicorn main:app --port 8080 &
       python3 agent/smoke_test.py [photo ...]
"""
import base64
import json
import sys
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080/invocations"
DEFAULTS = ["rose.jpg", "young_man.jpg", "franklin.jpg"]


def main() -> int:
    names = sys.argv[1:] or DEFAULTS
    payload = {"images": [
        {"name": n, "bytes_b64": base64.b64encode((Path("testdata") / n).read_bytes()).decode()}
        for n in names
    ]}
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    r = json.load(urllib.request.urlopen(req, timeout=300))
    assert r["faces"], "no faces returned"
    assert r["clip_b64"], "no clip returned"
    assert sum(r["counts"].values()) == len(r["faces"])
    print("PASS", r["counts"], "clip bytes", len(base64.b64decode(r["clip_b64"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
