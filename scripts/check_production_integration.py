"""Verify deployed contracts with synthetic inputs; storage writes are opt-in."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re
import time
import uuid

import httpx
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="https://remicare-strabismus-ai.onrender.com")
    parser.add_argument("--frontend", default="https://aidetecteye.vercel.app")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--write-synthetic-cover-session", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new report path")
    checks = {}
    with httpx.Client(timeout=90) as client:
        health = client.get(args.backend + "/health").json()
        assert health.get("deploymentCommit", "").startswith(args.expected_commit), health
        assert health["models"]["hirschberg"]["loaded"] is True, health
        checks["health"] = health
        for endpoint in ("/api/v1/research/measurements", "/api/v1/cover-test/sessions"):
            cors = client.options(args.backend + endpoint, headers={"Origin": args.frontend,
                "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
            assert cors.status_code == 200
            assert cors.headers["access-control-allow-origin"] == args.frontend
        checks["cors"] = "PASS"

        image = Image.new("RGB", (448, 224), (150, 130, 120))
        draw = ImageDraw.Draw(image)
        for x in (110, 338):
            draw.ellipse((x - 35, 60, x + 35, 150), fill=(30, 30, 30))
            draw.ellipse((x - 2, 98, x + 2, 102), fill="white")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        payload = {"schemaVersion": "remicare-research-quality-v0.1", "featureVersion": "research-geometry-v0.1",
            "testType": "HIRSCHBERG", "sessionId": "synthetic-deployment-probe", "requestId": "synthetic-deployment-probe",
            "distance_bucket": "30_50_CM", "eligibility": {"consent": True, "redFlag": False},
            "imageDataUrl": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as executor:
            analysis = executor.submit(client.post, args.backend + "/api/v1/research/measurements", json=payload)
            time.sleep(0.1)
            ping_start = time.monotonic()
            ping = client.get(args.backend + "/ping")
            checks["ping_during_hirschberg_ms"] = round((time.monotonic() - ping_start) * 1000, 1)
            assert ping.status_code == 200
            result = analysis.result()
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["aiPrediction"]["status"] in ("PREDICTED", "INCONCLUSIVE"), body
        checks["hirschberg"] = {"http_status": result.status_code,
            "round_trip_ms": round((time.monotonic() - start) * 1000, 1),
            "model_id": body["aiPrediction"].get("modelId"), "status": body["aiPrediction"]["status"]}

        if args.write_synthetic_cover_session:
            session_id = str(uuid.uuid4())
            metadata = {"sessionId": session_id, "cycleCount": 3, "samplingRateHz": 60,
                "clientMetadata": {"synthetic": True, "purpose": "deployment_smoke_test"}}
            files = {}
            for cycle in (1, 2, 3):
                samples = [{"t": round(i * 1000 / 60, 4), "phase": "BASELINE" if i < 4 else "TRACKING",
                    "leftX": .4, "leftY": .5, "rightX": .6, "rightY": .5,
                    "leftValid": True, "rightValid": True, "trackingQuality": 1.0} for i in range(12)]
                raw = {"samples": samples, "coveredEye": "LEFT" if cycle % 2 else "RIGHT"}
                files[f"cycle_{cycle}_raw"] = (f"synthetic_cycle_{cycle}.json", json.dumps(raw), "application/json")
            start = time.monotonic()
            response = client.post(args.backend + "/api/v1/cover-test/sessions",
                data={"session_metadata": json.dumps(metadata)}, files=files)
            assert response.status_code == 201, response.text
            saved = response.json()
            assert saved["saved"] is True and saved["cyclesSaved"] == 3, saved
            assert saved["aiResult"]["reason"] == "MODEL_NOT_LOADED", saved
            checks["cover_test"] = {"http_status": 201, "saved": True, "cycles": 3,
                "synthetic_session_id": session_id, "reason": "MODEL_NOT_LOADED",
                "round_trip_ms": round((time.monotonic() - start) * 1000, 1)}

        page = client.get(args.frontend)
        assert page.status_code == 200
        js_path = re.search(r'src="(/assets/index-[^"]+\.js)"', page.text).group(1)
        javascript = client.get(args.frontend + js_path).text
        assert args.backend in javascript
        assert "timeoutMs:25000" not in javascript and "timeoutMs:25e3" not in javascript
        assert "MODEL_NOT_LOADED" in javascript
        checks["frontend"] = {"status": "PASS", "asset": js_path, "backend_url_present": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
