"""Script to test real Cover Test JSON sample against the RemiCare Screening API.

Reads:
    data/raw/sample.json
Sends to:
    POST /api/v1/screening/analyze

Supports both live running server (http://127.0.0.1:8000) and direct in-memory
FastAPI TestClient execution if the external server is not currently running.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_sample_json(sample_path: str) -> Dict[str, Any]:
    """Load and validate JSON from file."""
    if not os.path.exists(sample_path):
        print(f"Error: Sample file not found at {sample_path}")
        sys.exit(1)

    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


__test__ = False


def send_http_request(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send payload to live HTTP server via httpx."""
    import httpx

    print(f"Connecting to live server: {url} ...")
    with httpx.Client(timeout=10.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def send_testclient_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send payload via in-memory FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from app.main import app

    print("Running in-memory FastAPI TestClient ...")
    client = TestClient(app)
    response = client.post("/api/v1/screening/analyze", json=payload)
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Test Cover Test JSON with Screening API")
    parser.add_argument(
        "--file",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "sample.json"),
        help="Path to Cover Test JSON sample",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Custom API URL (e.g. http://127.0.0.1:8000/api/v1/screening/analyze)",
    )
    args = parser.parse_args()

    sample_path = os.path.abspath(args.file)
    print("=" * 60)
    print("RemiCare Strabismus AI - Real Sample Screening Test")
    print("=" * 60)
    print(f"Sample file: {sample_path}")

    payload = load_sample_json(sample_path)
    sample_id = payload.get("sampleId", "unknown")
    cycles = payload.get("cycles", [])
    total_samples = sum(len(c.get("samples", [])) for c in cycles)

    print(f"Payload sampleId : {sample_id}")
    print(f"Cycles in payload: {len(cycles)}")
    print(f"Total raw samples: {total_samples}")
    print("-" * 60)

    # Determine execution mode: live server or TestClient
    target_url = args.url or "http://127.0.0.1:8000/api/v1/screening/analyze"
    response_data = None

    if args.url:
        try:
            response_data = send_http_request(args.url, payload)
        except Exception as e:
            print(f"HTTP request failed: {e}")
            sys.exit(1)
    else:
        # Check if local server is listening
        try:
            import httpx

            r = httpx.get("http://127.0.0.1:8000/health", timeout=1.0)
            if r.status_code == 200:
                response_data = send_http_request(target_url, payload)
        except Exception:
            # Server not running -> use TestClient directly
            response_data = send_testclient_request(payload)

    print("-" * 60)
    print("SCREENING API RESPONSE:")
    print("-" * 60)
    print(json.dumps(response_data, indent=2))
    print("=" * 60)

    # Verification of acceptance criteria
    status = response_data.get("status")
    gate_status = response_data.get("quality", {}).get("status")
    cycles_analyzed = response_data.get("analysis", {}).get("cyclesAnalyzed", 0)

    print(f"Screening Status : {status}")
    print(f"Quality Gate     : {gate_status}")
    print(f"Cycles Analyzed  : {cycles_analyzed}")
    print(f"Notice           : {response_data.get('notice')}")

    if gate_status == "PASS" and cycles_analyzed == len(cycles):
        print("\nSUCCESS: Sample processed successfully through all pipeline gates.")
    else:
        print(f"\nNOTE: Pipeline returned {status} (Reason: {response_data.get('reason')})")


if __name__ == "__main__":
    main()
