# hirschberg-candidate-v0.4 Model Card

Status: `research_candidate`

This model is for offline research comparison only. It is not deployed to
`/api/v1/strabismus/predict` and must not be presented as a clinical diagnosis
or clinical probability.

## Dataset

- Manifest: `D:\REMICARE-STRABISMUS-AI\datasets\pedseye_hirschberg_manifest_v0.1.jsonl`
- Dataset version: `pedseye_hirschberg_manifest_v0.1`
- Total samples used: `67`
- Class counts: `{'esotropia': 20, 'exotropia': 14, 'normal': 33}`
- Participant IDs: inferred near-duplicate clusters, not true patient IDs.
- Missing classes: `['pseudostrabismus', 'poor_quality']`

## Feature Contract

- Feature count: `73`
- Feature source: same 73-feature contract as `hirschberg_candidate_v0.1`.
- Crop aggregation: mean of service-compatible crop feature vectors.

## Selected Baseline

- Model family: `random_forest_balanced`
- CV: `{'name': 'StratifiedGroupKFold', 'splits': 5, 'group_source': 'manifest participant_id'}`
- Balanced accuracy: `0.4718`
- Macro-F1: `0.4666`
- Binary sensitivity: `0.5294`
- Binary specificity: `0.3939`
- PPV: `0.4737`
- NPV: `0.4483`
- ROC-AUC: `0.451`

## Recommendation

Do not deploy. The dataset is too small, lacks pseudostrabismus and poor-quality
examples, and does not contain true participant IDs. Use this candidate only as a
research baseline for error review and future data collection.
