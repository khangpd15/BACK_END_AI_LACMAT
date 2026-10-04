# hirschberg-candidate-v0.3-pedseye Model Card

Status: `research_candidate`

This artifact is for offline research only. It is not deployed to `/api/v1/strabismus/predict`
and must not be presented as a clinical diagnosis or clinical probability.

## Dataset

- Dataset: `dataset_by_pedseye_manifest-v0.1`
- Image count: `67`
- Class counts: `{'esotropia': 20, 'exotropia': 14, 'normal': 33}`
- Label source: folder labels from the user-provided Pedseye image set.
- Participant IDs: unavailable, so patient-level leakage cannot be ruled out.
- Pseudostrabismus class: absent.

## Feature Contract

- Feature count: `73`
- Extractor: `app.services.hirschberg_ai_service.extract_features_from_crop`
- Aggregation: mean of left/right heuristic crops for full-face images.

## Selected Model

- Model family: `random_forest_balanced`
- CV splits: `5`
- Balanced accuracy: `0.4617`
- Macro-F1: `0.4428`
- Binary strabismus sensitivity: `0.5294`
- Binary strabismus specificity: `0.3636`

## Recommendation

Do not deploy this artifact as production. Use it to compare candidate behavior and to
decide which Pedseye cases need manual review, especially false negatives and normal
false positives.
