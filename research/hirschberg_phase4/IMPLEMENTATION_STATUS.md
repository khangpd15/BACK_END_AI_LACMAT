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

The project owner has now stated that all supplied data were confirmed by a
physician. This is recorded in `annotation/user_attestation_20261004.json` as
owner-reported clinical confirmation, not an independent medical review. Earlier
audit findings describe the metadata available at that time; absence of a stored
confirmation field does not mean that a physician did not review the data.

The owner clarified that physician confirmation covers disease labels only and
that the supplied images have permission and consent for research use. Research
image-use permission is accepted on that explicit statement, not reclassified as
a public dataset license or permission to redistribute images. The new attested
import records these facts without overwriting the historical HOLD import.
Verification of this increment: 62 attested rows, all with research-use permission
and owner-reported physician-confirmed disease labels; zero genuine patient IDs.
The extended test suite passes 23 tests including the existing manifest regression.

The owner has no patient-to-image mapping and delegated this decision. Per
`GROUPING_DECISION.md`, filename/hash groups are provisional similarity clusters,
not patient IDs. A 5-fold exploratory OOF classifier was run on the 62 legacy
annotated crops using these provisional groups. At hash threshold 4: BA 0.716,
macro-F1 0.694, with 59 groups; threshold 8 gives BA 0.828 but one validation fold
contains just one group. Neither is patient-independent validation or clinical
evidence. Features use legacy pupil/reflex clicks, not the runtime detector, and
the cohort has no pseudo cases. Full result and sensitivity analysis are in the
ignored local `runs/exploratory_primary4_20261004/report.json`.

Real patient linkage and native landmark review remain outstanding. Legacy
coordinates are not physician-confirmed. A1 image ROI/ellipse/glint detector,
pixel-error benchmark, epicanthal/canthal automation, nested patient-CV, bootstrap
CIs and the clinical private test have not been implemented. No final model,
clinical validation, runtime change, push or deployment occurred.

Next: use authorized supplied images only for exploratory development; obtain
genuine patient grouping before reporting generalization. Do not
manufacture patient IDs from image IDs or duplicate clusters.
User permission applies to supplied data, not external datasets or public release.

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
