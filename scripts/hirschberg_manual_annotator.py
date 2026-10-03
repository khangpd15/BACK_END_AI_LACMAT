"""Manual Hirschberg landmark annotation tool.

This is a small OpenCV GUI for research use. It never edits source images.

Click order per image:
1. OD pupil/iris center (patient right eye, viewer-left half)
2. OD corneal reflex center
3. OS pupil/iris center (patient left eye, viewer-right half)
4. OS corneal reflex center

Keys:
- u: undo last point
- s: save current image once 4 points are present
- n: skip current image
- q: quit

Default output: processed/hirschberg_manual_annotations.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "harschberg_data_detect"
DEFAULT_OUTPUT = PROJECT_ROOT / "processed" / "hirschberg_manual_annotations.csv"
CLASS_DIRS = {
    "esotropia": "esotropia_harschberg",
    "exotropia": "exotropia_harschberg",
    "normal": "normal_harschberg",
}
POINT_ORDER = [
    ("od_pupil_x", "od_pupil_y", "OD pupil/iris center"),
    ("od_reflex_x", "od_reflex_y", "OD reflex center"),
    ("os_pupil_x", "os_pupil_y", "OS pupil/iris center"),
    ("os_reflex_x", "os_reflex_y", "OS reflex center"),
]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def iter_images(dataset_root: Path) -> Iterable[ImageRecord]:
    for label, folder_name in CLASS_DIRS.items():
        folder = dataset_root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                yield ImageRecord(path=path, label=label)


def load_existing(output_csv: Path) -> Dict[str, Dict[str, str]]:
    if not output_csv.exists():
        return {}
    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["relative_path"]: row for row in csv.DictReader(handle)}


def write_rows(output_csv: Path, rows: Dict[str, Dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "relative_path",
        "class_label",
        "od_pupil_x",
        "od_pupil_y",
        "od_reflex_x",
        "od_reflex_y",
        "os_pupil_x",
        "os_pupil_y",
        "os_reflex_x",
        "os_reflex_y",
        "notes",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])


def draw_overlay(image, points: List[Tuple[int, int]], prompt: str):
    display = image.copy()
    h, w = display.shape[:2]
    cv2.line(display, (w // 2, 0), (w // 2, h), (128, 128, 128), 1)
    colors = [(255, 128, 0), (255, 255, 255), (0, 192, 255), (255, 255, 0)]
    for idx, (x, y) in enumerate(points):
        color = colors[idx % len(colors)]
        cv2.circle(display, (x, y), 5, color, -1)
        cv2.putText(display, str(idx + 1), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.rectangle(display, (0, 0), (w, 42), (0, 0, 0), -1)
    cv2.putText(display, prompt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(display, "u=undo s=save n=skip q=quit", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    return display


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate Hirschberg pupil/reflex centers on accepted crops.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-existing", action="store_true", help="Review images that already have annotations.")
    args = parser.parse_args()

    rows = load_existing(args.output_csv)
    records = list(iter_images(args.dataset_root))
    if not args.include_existing:
        records = [
            rec for rec in records
            if str(rec.path.relative_to(PROJECT_ROOT)).replace("\\", "/") not in rows
        ]

    window = "Hirschberg manual annotation"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    current_points: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 4:
            current_points.append((int(x), int(y)))

    cv2.setMouseCallback(window, on_mouse)

    for idx, rec in enumerate(records, 1):
        image = cv2.imread(str(rec.path))
        if image is None:
            continue
        current_points.clear()
        rel = str(rec.path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        while True:
            next_label = POINT_ORDER[len(current_points)][2] if len(current_points) < 4 else "ready to save"
            prompt = f"{idx}/{len(records)} {rec.label} {rec.path.name} | click: {next_label}"
            cv2.imshow(window, draw_overlay(image, current_points, prompt))
            key = cv2.waitKey(50) & 0xFF
            if key == ord("u") and current_points:
                current_points.pop()
            elif key == ord("n"):
                break
            elif key == ord("q"):
                write_rows(args.output_csv, rows)
                cv2.destroyAllWindows()
                return 0
            elif key == ord("s") and len(current_points) == 4:
                row = {
                    "relative_path": rel,
                    "class_label": rec.label,
                    "notes": "",
                }
                for (x_key, y_key, _), (x, y) in zip(POINT_ORDER, current_points):
                    row[x_key] = str(x)
                    row[y_key] = str(y)
                rows[rel] = row
                write_rows(args.output_csv, rows)
                break

    write_rows(args.output_csv, rows)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
