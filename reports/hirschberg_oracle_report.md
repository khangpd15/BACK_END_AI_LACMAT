# Hirschberg Oracle Manual Geometry Analysis

Status: research-only; no CNN; no production API integration; no deployable model artifact.

## Detector Failure Analysis

- Manual images: 62
- Reflex point comparisons: 94
- P90 threshold: 31.61px
- P90+ failures exported: 10
- Failure folder: `reports/detector_failures/`
- Cause counts: `{'double_or_multiple_reflex_candidates': 7, 'large_or_elongated_glare_environment_or_eyelid': 2, 'eyelid_or_skin_bright_spot': 1}`

Recommended exclusions before using detector features:

- Reject images when detected reflex is on a large/elongated bright component.
- Reject or flag images with multiple bright candidates near the iris.
- Require detector/manual-like consistency checks: compact area, aspect ratio, and distance from pupil/iris center.
- Keep manual/oracle analysis separate until detector P90 error is much lower.

## Oracle Geometry

Offsets use manual pupil and reflex points. The horizontal axis is the line connecting the two pupils. Values are normalized by interpupillary distance. OD and OS are mirrored so positive is nasal for both eyes.

- Feature evaluated: `abs_asymmetry_nasal_ipd`
- Balanced accuracy: 0.6650
- Macro-F1: 0.6111
- Bootstrap CI: `{'balanced_accuracy': {'low': 0.5017810713627143, 'high': 0.8280500000000001}, 'macro_f1': {'low': 0.47662824831421857, 'high': 0.7312340312349447}}`
- Permutation p-value: 0.030969030969030968
- Permutation p95: 0.6508
- Confusion matrix: `[[29, 21], [65, 195]]`

## Detector vs Oracle

- Detector balanced accuracy from previous report: 0.5616
- Detector permutation p-value: 0.2857142857142857
- Oracle balanced accuracy: 0.6650

## Label / Image Hypothesis Checks

- Normal high-asymmetry grid: `reports/normal_high_oracle_asymmetry_grid.jpg`
- Strabismus low-asymmetry grid: `reports/strabismus_low_oracle_asymmetry_grid.jpg`
- Ambient-reflection automatic count: 2
- Note: Count is based on P90 detector-failure overlays classified as large/elongated glare; human review is still required to separate ambient reflection from eyelid/skin glare.

## Conclusion

Oracle geometry is above random while detector geometry was near random. This would indicate detector error is the primary bottleneck before any modeling.
