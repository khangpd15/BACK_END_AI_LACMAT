import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from hirschberg_folder_manifest import LABEL_SOURCE, summarize  # noqa: E402


def test_folder_manifest_summary_keeps_generated_groups_in_one_split():
    rows = [
        {
            "class_label": "esotropia",
            "original_split": "train",
            "generated_split": "train",
            "group_id": "esotropia:001",
            "sha256": "a",
        },
        {
            "class_label": "esotropia",
            "original_split": "val",
            "generated_split": "train",
            "group_id": "esotropia:001",
            "sha256": "b",
        },
        {
            "class_label": "normal",
            "original_split": "test",
            "generated_split": "test",
            "group_id": "normal:001",
            "sha256": "c",
        },
    ]

    summary = summarize(rows)

    assert summary["labelSource"] == LABEL_SOURCE
    assert summary["originalCrossSplitGroupCount"] == 1
    assert summary["generatedCrossSplitGroupCount"] == 0
    assert summary["labelCounts"] == {"esotropia": 2, "normal": 1}
