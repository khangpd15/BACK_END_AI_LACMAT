# Phase 4 A: first increment

Date: 2026-10-04. Branch: `codex/hirschberg-phase4-a`.
Noncommercial screening research only; not diagnosis or a substitute for care.

## Implemented And Verified

- Native-coordinate manifest validation, separate pupil/limbus/glint targets,
  explicit permission/consent/clinical-label review references for human records.
- Patient/image/hash overlap rejection, one preselected image per patient.
- A0 dimensionless geometry + StandardScaler/logistic regression, fitted only on
  training records. Evaluation is a separate CLI and never fits/tunes.
- Five-column confusion including abstain, all-sample class recall and referral
  sensitivity/specificity/rate. Missing geometry never becomes a fake glint.
- Physical-scale helper requires individual measured WTW; no default mm or PD.
- Synthetic-only smoke run: 48 train records, 16 validation records. This merely
  tests software, not patient generalization. Normal and pseudo intentionally share
  a generating distribution. No clinical improvement percentage is claimed.
- Legacy CSV import: 62 records, all HOLD, no patient IDs inferred, coordinates
  remain legacy_crop. Original CSV and images untouched. No image files were read.
- Tests: 21 new tests plus 1 existing manifest regression test, all passed with
  Python 3.12.14, numpy 2.5.3, sklearn 1.9.1, pytest 9.1.1.

Generated run artifacts are local/ignored under `runs/`; no existing model or
report was overwritten. Output writers refuse replacement. Hashes/environment
versions are recorded, but a fully pinned environment is still outstanding.

## Not Completed, No Claim Of Full Phase 4 Completion

M0 permission/consent/patient linking and M1 native gold clinical/landmark labels
are not established. A1 image ROI/ellipse/glint detector, pixel-error benchmark,
epicanthal/canthal automation, OOF/nested patient-CV, bootstrap CIs and the clinical
private test have not been implemented or run in this increment. No actual patient
training, clinical validation, runtime change, push or deployment occurred.

Next: review actual source/license/consent scope and clinician-confirmed records;
otherwise continue nonpatient infrastructure/synthetic detector fixtures only.
Changing manifest flags without genuine documentation does not satisfy M0.

## Verification Commands Used

```powershell
.\.venv312\Scripts\python.exe -B -m pytest research/hirschberg_phase4/tests tests/research/test_hirschberg_folder_manifest.py -q -p no:cacheprovider
.\.venv312\Scripts\python.exe -B -m research.hirschberg_phase4.synthetic --output research/hirschberg_phase4/runs/smoke_manifest.json
.\.venv312\Scripts\python.exe -B -m research.hirschberg_phase4.train --manifest research/hirschberg_phase4/runs/smoke_manifest.json --output-dir research/hirschberg_phase4/runs/smoke_a0
.\.venv312\Scripts\python.exe -B -m research.hirschberg_phase4.evaluate --manifest research/hirschberg_phase4/runs/smoke_manifest.json --model research/hirschberg_phase4/runs/smoke_a0/model.joblib --split val --output research/hirschberg_phase4/runs/smoke_a0/validation.json
.\.venv312\Scripts\python.exe -B -m research.hirschberg_phase4.import_legacy --csv processed/hirschberg_manual_annotations.csv --output research/hirschberg_phase4/runs/legacy_quarantine.json
```

Use new paths for subsequent runs; rerunning these exact writer commands must
refuse existing outputs rather than overwrite them.
