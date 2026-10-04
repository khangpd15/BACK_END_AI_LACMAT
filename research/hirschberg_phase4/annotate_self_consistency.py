"""Blind local OpenCV re-annotation UI for four pupil/reflex landmarks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2


TARGETS = ("OD pupil", "OD reflex", "OS pupil", "OS reflex")
TARGET_KEYS = {ord(str(index)): target for index, target in enumerate(TARGETS, start=1)}
WINDOW = "Blind repeat annotation"
MAX_WIDTH, MAX_HEIGHT = 1400, 850
HEADER_HEIGHT = 58


def _atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def annotate(manifest_path: Path, image_root: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("samples"), list):
        raise ValueError("Unsupported sample manifest")
    payload = {"format_version": 1, "source_manifest": str(manifest_path), "annotations": []}
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or not isinstance(payload.get("annotations"), list):
            raise ValueError("Existing output is not a compatible annotation file")
    saved = {row["sample_id"]: row for row in payload["annotations"]}

    for sample in manifest["samples"]:
        if sample["sample_id"] in saved:
            continue
        image_path = (image_root / sample["relative_path"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {sample['relative_path']}")
        height, width = image.shape[:2]
        scale = min(MAX_WIDTH / width, MAX_HEIGHT / height, 1.0)
        display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        display = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
        points: dict[str, tuple[float, float]] = {}
        selected_target = TARGETS[0]

        def on_mouse(event, x, y, flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN and y >= HEADER_HEIGHT:
                native_x = min(width - 1, max(0, x / scale))
                native_y = min(height - 1, max(0, (y - HEADER_HEIGHT) / scale))
                points[selected_target] = (float(native_x), float(native_y))

        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, on_mouse)
        while True:
            canvas = cv2.copyMakeBorder(display, HEADER_HEIGHT, 0, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(24, 24, 24))
            for target, (px, py) in points.items():
                cv2.drawMarker(canvas, (round(px * scale), round(py * scale) + HEADER_HEIGHT), (0, 255, 0),
                               cv2.MARKER_CROSS, 16, 2)
                cv2.putText(canvas, target, (round(px * scale) + 8, round(py * scale) + HEADER_HEIGHT - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            status = " | ".join(f"{i}:{t}={'set' if t in points else 'missing'}"
                                for i, t in enumerate(TARGETS, start=1))
            help_line = f"{sample['sample_id']}  active: {selected_target} | click=set; 1-4=target; S=save; N=skip; Q=quit"
            cv2.putText(canvas, help_line, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, status, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(25) & 0xFF
            if key in TARGET_KEYS:
                selected_target = TARGET_KEYS[key]
            elif key in (ord("s"), ord("S")) and len(points) == 4:
                row = {
                    "sample_id": sample["sample_id"],
                    "relative_path": sample["relative_path"],
                    "width": width,
                    "height": height,
                    "points": {target: list(points[target]) for target in TARGETS},
                }
                payload["annotations"] = [*payload["annotations"], row]
                saved[sample["sample_id"]] = row
                _atomic_save(output_path, payload)
                break
            elif key in (ord("n"), ord("N")):
                break
            elif key in (ord("q"), ord("Q"), 27):
                cv2.destroyAllWindows()
                return
        cv2.destroyWindow(WINDOW)
    cv2.destroyAllWindows()
    print(f"Saved {len(payload['annotations'])}/{len(manifest['samples'])} annotations to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    annotate(Path(args.manifest), Path(args.root), Path(args.output))


if __name__ == "__main__":
    main()
