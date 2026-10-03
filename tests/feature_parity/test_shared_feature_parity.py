import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services.shared_feature_contract import (
    ALL_SHARED_FEATURES,
    extract_shared_features_vector,
)


FIXTURE = Path(__file__).with_name("fixture_cover_test_samples.json")
JS_EXTRACTOR = Path(__file__).with_name("js_shared_feature_extractor.mjs")


def _run_js_extractor() -> dict:
    node = os.getenv("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for JS/Python feature parity test")
    completed = subprocess.run(
        [node, str(JS_EXTRACTOR), str(FIXTURE)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_js_and_python_shared_features_match():
    samples = json.loads(FIXTURE.read_text(encoding="utf-8"))
    py_features = extract_shared_features_vector(samples)
    js_features = _run_js_extractor()

    assert list(py_features.keys()) == ALL_SHARED_FEATURES
    assert list(js_features.keys()) == ALL_SHARED_FEATURES

    for name in ALL_SHARED_FEATURES:
        py_val = py_features[name]
        js_val = js_features[name]
        assert (py_val is None) == (js_val is None), name
        if py_val is not None:
            assert math.isclose(float(py_val), float(js_val), abs_tol=1e-6), name
