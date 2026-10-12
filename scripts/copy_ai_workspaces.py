"""Copy research assets into task folders without changing runtime or source data."""

from pathlib import Path
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "ai"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    if DEST.exists():
        raise SystemExit("ai/ already exists; refusing to overwrite copies.")
    pairs = []

    def add(source, target):
        source = ROOT / source
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    pairs.append((path, DEST / target / path.relative_to(source)))
        else:
            pairs.append((source, DEST / target))

    add("app/models/research", Path("hirschberg/models/candidates"))
    add("app/models/best_model.onnx", Path("hirschberg/models/representation/best_model.onnx"))
    add("app/models/best_model.pt", Path("hirschberg/models/representation/best_model.pt"))
    add("harschberg_data_detect", Path("hirschberg/datasets/harschberg_data_detect"))
    add("dataset_by_pedseye_manifest", Path("hirschberg/datasets/dataset_by_pedseye_manifest"))
    add("datasets/pedseye_hirschberg_manifest_v0.1.jsonl", Path("hirschberg/datasets/manifests/pedseye_hirschberg_manifest_v0.1.jsonl"))
    for path in sorted((ROOT / "processed").glob("hirschberg*.csv")):
        add(path.relative_to(ROOT), Path("hirschberg/processed") / path.name)
    add("research/hirschberg_phase4", Path("hirschberg/source_snapshot/research/hirschberg_phase4"))
    add("docs/research/hirschberg", Path("hirschberg/docs"))
    for path in sorted((ROOT / "scripts").glob("*.py")):
        if "hirschberg" in path.name or path.name == "prepare_pedseye_phase_a.py":
            add(path.relative_to(ROOT), Path("hirschberg/source_snapshot/scripts") / path.name)
    add("filter_harschberg.py", Path("hirschberg/source_snapshot/filter_harschberg.py"))
    for path in sorted((ROOT / "reports").iterdir()):
        if "hirschberg" in path.name or "pedseye" in path.name or path.name in {
            "audit_samples", "detector_failures", "oracle_plots",
            "normal_high_oracle_asymmetry_grid.jpg", "strabismus_low_oracle_asymmetry_grid.jpg",
        }:
            add(path.relative_to(ROOT), Path("hirschberg/reports") / path.name)

    add("data/VIDEO_COVER_TEST", Path("cover_test/datasets/VIDEO_COVER_TEST"))
    add("data/normalized/korean", Path("cover_test/datasets/normalized/korean"))
    add("research/cover_test_timestamp", Path("cover_test/source_snapshot/research/cover_test_timestamp"))
    add("reports/cover_timestamp_30_60", Path("cover_test/reports/cover_timestamp_30_60"))
    add("docs/research/cover_test_model_audit.md", Path("cover_test/docs/cover_test_model_audit.md"))
    # Source-checkpoint inventory only; never manufacture a Cover Test artifact.
    for path in sorted((ROOT / "models").rglob("*")):
        if path.is_file() and path.suffix in {".joblib", ".pkl", ".onnx", ".pt", ".pth", ".h5"}:
            add(path.relative_to(ROOT), Path("cover_test/models") / path.relative_to(ROOT / "models"))

    destinations = [target for _, target in pairs]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Duplicate destination in copy plan")
    inventory = []
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(source)
        shutil.copy2(source, target)
        if digest(target) != before or digest(source) != before:
            raise RuntimeError(f"Copy verification failed: {source}")
        inventory.append({"source": source.relative_to(ROOT).as_posix(),
                          "copy": target.relative_to(ROOT).as_posix(),
                          "bytes": target.stat().st_size, "sha256": before})
    (DEST / "cover_test/models").mkdir(parents=True, exist_ok=True)
    (DEST / "cover_test/models/.gitkeep").touch()
    (DEST / "copy_inventory.json").write_text(
        json.dumps({"operation": "copy_only", "source_files_unchanged": True,
                    "files": inventory}, indent=2), encoding="utf-8")
    for branch in ("hirschberg", "cover_test"):
        rows = [r for r in inventory if r["copy"].startswith(f"ai/{branch}/")]
        print(f"{branch}: {len(rows)} verified copies, {sum(r['bytes'] for r in rows)/1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
