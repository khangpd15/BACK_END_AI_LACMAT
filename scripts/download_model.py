"""Model download script for Render build/startup phase.

Verifies that app/models/best_model.onnx is present.
If absent, downloads the model from MODEL_DOWNLOAD_URL.
"""

import os
from pathlib import Path
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DEST = PROJECT_ROOT / "app" / "models" / "best_model.onnx"

def ensure_model():
    if MODEL_DEST.is_file() and MODEL_DEST.stat().st_size > 1_000_000:
        print(f"[ModelDownload] Model already present: {MODEL_DEST} ({MODEL_DEST.stat().st_size} bytes)")
        return True

    MODEL_DEST.parent.mkdir(parents=True, exist_ok=True)
    download_url = os.getenv("MODEL_DOWNLOAD_URL", "").strip()

    if not download_url:
        print("[ModelDownload Warning] best_model.onnx is not present and MODEL_DOWNLOAD_URL is not set.")
        print("[ModelDownload] If the file is tracked in Git, ensure it was checked out properly.")
        return False

    print(f"[ModelDownload] Downloading model from {download_url} to {MODEL_DEST}...")
    try:
        urllib.request.urlretrieve(download_url, str(MODEL_DEST))
        print(f"[ModelDownload] Successfully downloaded model: {MODEL_DEST.stat().st_size} bytes")
        return True
    except Exception as e:
        print(f"[ModelDownload Error] Failed to download model: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    success = ensure_model()
    sys.exit(0 if success else 1)
