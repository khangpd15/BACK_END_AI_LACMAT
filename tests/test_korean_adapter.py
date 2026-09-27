"""Unit tests for Korean Strabismus Dataset Adapter.

Verifies:
1. test_normal mapping strictly to 'NORMAL'
2. test_strabismus mapping strictly to 'STRABISMUS'
3. CSV column parsing (TIME -> t, LPCX -> leftX, etc.)
4. LPV / RPV validity mapping (1 -> True, 0 -> False)
5. Timestamp parsing and sampling statistics (estimatedHz ~60Hz, mean/median intervals)
6. Missing coordinate handling (retains None, NO 0.0 or 0.5 sentinel, phase='UNKNOWN')
7. Conversion report schema and integrity
8. Binary label mapping (NORMAL -> 0, STRABISMUS -> 1)
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from app.services.korean_adapter import (
    calculate_sampling_stats,
    convert_korean_test_folders,
    extract_features_from_korean_record,
    parse_korean_csv,
    resolve_label_from_folder,
)


# =====================================================================
# Test 1: test_normal mapping
# =====================================================================
def test_normal_mapping():
    """Verify that test_normal folders map strictly to NORMAL."""
    assert resolve_label_from_folder("data/test_normal/sample.csv") == "NORMAL"
    assert resolve_label_from_folder("data\\test_normal\\sub\\sample.csv") == "NORMAL"
    assert resolve_label_from_folder("data/normal/1-_all_gaze.csv") == "NORMAL"
    assert resolve_label_from_folder("C:/path/to/test_normal/file.csv") == "NORMAL"


# =====================================================================
# Test 2: test_strabismus mapping (strictly binary, no sub-types)
# =====================================================================
def test_strabismus_mapping():
    """Verify that test_strabismus maps to STRABISMUS and never sub-classes."""
    assert resolve_label_from_folder("data/test_strabismus/sample.csv") == "STRABISMUS"
    assert resolve_label_from_folder("data\\test_strabismus\\1-15_all_gaze.csv") == "STRABISMUS"
    # Even if sub-type folders exist in training data, they must map to STRABISMUS
    assert resolve_label_from_folder("data/esotropia/sample.csv") == "STRABISMUS"
    assert resolve_label_from_folder("data/exotropia/sample.csv") == "STRABISMUS"
    assert resolve_label_from_folder("data/hypertropia/sample.csv") == "STRABISMUS"


# =====================================================================
# Test 3: CSV parsing
# =====================================================================
def test_csv_parsing():
    """Verify CSV reading and column mapping into NormalizedKoreanRecord."""
    csv_content = """MEDIA_ID,TIME(2020/05/27 14:46:42.702),LPCX,LPCY,LPV,RPCX,RPCY,RPV
0,0.00000,0.31118,0.46009,1,0.58613,0.47523,1
0,0.01636,0.31120,0.46010,1,0.58615,0.47525,1
0,0.03198,0.31122,0.46012,1,0.58617,0.47526,1
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
        tf.write(csv_content)
        temp_path = tf.name

    try:
        record, errors = parse_korean_csv(temp_path, label="NORMAL")
        assert errors == []
        assert record is not None
        assert record.source == "KOREAN_EYE_TRACKER"
        assert record.deviceType == "INFRARED_EYE_TRACKER"
        assert record.label == "NORMAL"
        assert len(record.samples) == 3

        s0 = record.samples[0]
        assert s0["t"] == pytest.approx(0.0)
        assert s0["leftX"] == pytest.approx(0.31118)
        assert s0["leftY"] == pytest.approx(0.46009)
        assert s0["leftValid"] is True
        assert s0["rightX"] == pytest.approx(0.58613)
        assert s0["rightY"] == pytest.approx(0.47523)
        assert s0["rightValid"] is True
        assert s0["phase"] == "UNKNOWN"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================================
# Test 4: LPV / RPV mapping
# =====================================================================
def test_lpv_rpv_mapping():
    """Verify that LPV/RPV == 1 maps to True and 0 maps to False."""
    csv_content = """TIME,LPCX,LPCY,LPV,RPCX,RPCY,RPV
0.00000,0.30,0.45,1,0.58,0.47,0
0.01667,0.30,0.45,0,0.58,0.47,1
0.03333,0.30,0.45,0,0.58,0.47,0
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
        tf.write(csv_content)
        temp_path = tf.name

    try:
        record, _ = parse_korean_csv(temp_path, label="STRABISMUS")
        assert record is not None
        assert record.samples[0]["leftValid"] is True
        assert record.samples[0]["rightValid"] is False

        assert record.samples[1]["leftValid"] is False
        assert record.samples[1]["rightValid"] is True

        assert record.samples[2]["leftValid"] is False
        assert record.samples[2]["rightValid"] is False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================================
# Test 5: Timestamp parsing and sampling statistics (~60Hz)
# =====================================================================
def test_timestamp_parsing_and_sampling_stats():
    """Verify that ~16.6ms intervals calculate to approximately 60Hz."""
    # 60Hz timestamps in seconds
    ts = np.array([0.0, 0.016667, 0.033333, 0.050000, 0.066667, 0.083333])
    stats = calculate_sampling_stats(ts)

    assert stats.estimatedHz == pytest.approx(60.0, abs=1.0)
    assert stats.medianIntervalMs == pytest.approx(16.67, abs=0.1)
    assert stats.meanIntervalMs == pytest.approx(16.67, abs=0.1)
    assert stats.durationMs == pytest.approx(83.33, abs=0.5)


# =====================================================================
# Test 6: Missing coordinates (NO sentinel 0 or 0.5 substitution)
# =====================================================================
def test_missing_coordinates_no_fake_zeros():
    """Verify missing/NaN coordinates are kept as None, not replaced with 0 or 0.5."""
    csv_content = """TIME,LPCX,LPCY,LPV,RPCX,RPCY,RPV
0.00000,,0.45,0,0.58,,0
0.01667,0.31,,0,,0.47,0
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
        tf.write(csv_content)
        temp_path = tf.name

    try:
        record, _ = parse_korean_csv(temp_path, label="NORMAL")
        assert record is not None

        s0 = record.samples[0]
        assert s0["leftX"] is None  # NOT 0.0, NOT 0.5
        assert s0["leftY"] == pytest.approx(0.45)
        assert s0["rightX"] == pytest.approx(0.58)
        assert s0["rightY"] is None  # NOT 0.0, NOT 0.5

        s1 = record.samples[1]
        assert s1["leftX"] == pytest.approx(0.31)
        assert s1["leftY"] is None
        assert s1["rightX"] is None
        assert s1["rightY"] == pytest.approx(0.47)

        # Confirm phase is UNKNOWN
        assert s0["phase"] == "UNKNOWN"
        assert s1["phase"] == "UNKNOWN"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================================
# Test 7: Conversion report schema
# =====================================================================
def test_conversion_report_schema():
    """Verify batch conversion produces complete and valid report schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        norm_dir = os.path.join(tmp_dir, "test_normal")
        strab_dir = os.path.join(tmp_dir, "test_strabismus")
        out_dir = os.path.join(tmp_dir, "normalized")
        feats_csv = os.path.join(tmp_dir, "features.csv")
        rep_json = os.path.join(tmp_dir, "report.json")

        os.makedirs(norm_dir, exist_ok=True)
        os.makedirs(strab_dir, exist_ok=True)

        sample_csv = "TIME,LPCX,LPCY,LPV,RPCX,RPCY,RPV\n0.0,0.3,0.4,1,0.5,0.4,1\n0.016,0.3,0.4,1,0.5,0.4,1\n"
        with open(os.path.join(norm_dir, "norm_01.csv"), "w") as f:
            f.write(sample_csv)
        with open(os.path.join(strab_dir, "strab_01.csv"), "w") as f:
            f.write(sample_csv)

        report = convert_korean_test_folders(
            normal_dir=norm_dir,
            strabismus_dir=strab_dir,
            normalized_out_dir=out_dir,
            features_out_csv=feats_csv,
            report_out_json=rep_json,
        )

        assert report["totalFiles"] == 2
        assert report["successfulFiles"] == 2
        assert report["failedFiles"] == 0
        assert report["normalFiles"] == 1
        assert report["strabismusFiles"] == 1
        assert report["totalSamples"] == 4
        assert report["missingColumns"] == 0
        assert report["errors"] == []
        assert os.path.exists(rep_json)
        assert os.path.exists(feats_csv)


# =====================================================================
# Test 8: Binary label mapping (NORMAL = 0, STRABISMUS = 1)
# =====================================================================
def test_binary_label_mapping():
    """Verify feature extraction maps NORMAL to 0 and STRABISMUS to 1."""
    csv_content = """TIME,LPCX,LPCY,LPV,RPCX,RPCY,RPV
0.0,0.31,0.46,1,0.58,0.47,1
0.016,0.31,0.46,1,0.58,0.47,1
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
        tf.write(csv_content)
        temp_path = tf.name

    try:
        norm_record, _ = parse_korean_csv(temp_path, label="NORMAL")
        strab_record, _ = parse_korean_csv(temp_path, label="STRABISMUS")

        norm_feat = extract_features_from_korean_record(norm_record)
        strab_feat = extract_features_from_korean_record(strab_record)

        assert norm_feat["label_name"] == "NORMAL"
        assert norm_feat["label"] == 0

        assert strab_feat["label_name"] == "STRABISMUS"
        assert strab_feat["label"] == 1
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
