# Hirschberg Geometry Research Report

Status: research candidate analysis only. No CNN training. No production API integration.

## Environment

- Python env: `.venv312`
- PyTorch added: no
- MediaPipe: 0.10.14
- OpenCV: 4.14.0
- scikit-learn: 1.9.1
- Repeated CV config: repeats=5, splits=5
- Permutations: 20

## Dataset

- Accepted images: 136
- Quality usable by current detector: 73
- Class counts: `{'esotropia': 55, 'exotropia': 40, 'normal': 41}`
- Quality counts: `{'missing_reflex_od=large_or_elongated_glare_os=large_or_elongated_glare': 16, 'missing_reflex_od=large_or_elongated_glare_os=clustered_reflex_candidates': 3, 'missing_reflex_od=large_or_elongated_glare_os=detected_with_secondary_candidates': 8, 'missing_reflex_od=detected_os=large_or_elongated_glare': 8, 'ok': 73, 'missing_reflex_od=detected_with_secondary_candidates_os=large_or_elongated_glare': 4, 'missing_reflex_od=large_or_elongated_glare_os=detected': 12, 'missing_reflex_od=detected_with_secondary_candidates_os=clustered_reflex_candidates': 3, 'missing_reflex_od=clustered_reflex_candidates_os=detected_with_secondary_candidates': 2, 'missing_reflex_od=detected_os=reflex_candidate_too_far': 2, 'missing_reflex_od=large_or_elongated_glare_os=low_peak_brightness': 2, 'missing_reflex_od=clustered_reflex_candidates_os=detected': 3}`
- OD reflex status counts: `{'large_or_elongated_glare': 41, 'detected': 44, 'detected_with_secondary_candidates': 46, 'clustered_reflex_candidates': 5}`
- OS reflex status counts: `{'large_or_elongated_glare': 28, 'clustered_reflex_candidates': 6, 'detected_with_secondary_candidates': 48, 'detected': 50, 'reflex_candidate_too_far': 2, 'low_peak_brightness': 2}`
- Near-duplicate group count: 135
- Near-duplicate pairs at aHash <= 8: 1

## Detector Rules

- Pupil/iris center: dark connected component in the central eye crop, with Hough fallback.
- Reflex search: broad iris-centered region, then reject candidates outside the stricter pupil/iris radius.
- Hard rejects: large/elongated bright components, clustered competing glints, low peak brightness, and far candidates.
- Candidate choice: compact, bright, iris-proximal component; secondary candidates are flagged in status.

## Feature Contract

Primary diagnostic feature is binocular asymmetry:

`signed_asymmetry_lr = os_offset_norm - od_offset_norm`

where each offset is `(reflex_x - iris_center_x) / iris_radius`.

## Binary Strabismus vs Normal

- Model: Logistic Regression, repeated StratifiedGroupKFold
- Balanced accuracy: 0.5616
- Macro-F1: 0.5611
- ROC-AUC: 0.5383255813953488
- Balanced accuracy 95% bootstrap CI: {'low': 0.4605588430851064, 'high': 0.6588801076213914}
- Permutation p-value: 0.2857142857142857
- Permutation balanced accuracy p95: 0.6090697674418605
- Confusion matrix: `[[75, 75], [81, 134]]`

## Eso vs Exo Direction

Skipped by stop rule because the binary geometry signal did not exceed the permutation baseline.

## Manual Annotation / Detector Error

- Manual annotation CSV: `processed\hirschberg_manual_annotations.csv`
- Manual annotated images loaded: 62
- Detector reflex error px: `{'count': 94, 'mean': 10.527627470591591, 'median': 2.23606797749979, 'p90': 31.609424406122734}`

Run `python scripts/hirschberg_manual_annotator.py` to create annotations, then rerun this script with `--manual-csv processed/hirschberg_manual_annotations.csv`.

## Stop Rule

The binary result is not convincingly above the permutation baseline. Stop here and inspect label quality, detector error, and geometry sign conventions before trying any other model.

Likely causes to inspect:

- Detector: Tightened detector reduces large reflex outliers but still excludes many eyes with glare, clustered glints, low brightness, or far candidates.
- Labels: Folder labels are not backed by an auditable clinician manifest in this repo.
- Geometry rule: The assumed sign convention signed_asymmetry_lr = os - od may not match all crops/gaze directions.
- Domain: Accepted images are legacy 224x224 crops, not production phone full-face captures.
