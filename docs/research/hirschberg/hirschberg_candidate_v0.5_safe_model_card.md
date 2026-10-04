# hirschberg-candidate-v0.5-safe Model Card

Status: `research_candidate`

This candidate is wired for Hirschberg research screening only. It must not be
presented as diagnosis, clinical probability, or clearance.

## Why v0.5 Exists

`hirschberg_candidate_v0.3_pedseye` produced unsafe label flips on the local
Pedseye set. v0.5 keeps the same 73-feature service contract but adds a
confidence/margin abstention policy so weak cases return `INCONCLUSIVE`.

## Dataset

- Manifest: `D:\REMICARE-STRABISMUS-AI\datasets\pedseye_hirschberg_manifest_v0.1.jsonl`
- Dataset version: `pedseye_hirschberg_manifest_v0.1`
- Total trainable samples: `67`
- Class counts: `{'esotropia': 20, 'exotropia': 14, 'normal': 33}`
- Participant IDs: inferred near-duplicate groups, not real patient IDs.
- Missing expected classes: `['pseudostrabismus', 'poor_quality']`

## Feature Contract

- Feature count: `365`
- Extractor: `app.services.hirschberg_ai_service.extract_features_from_crop`
- Crop aggregation: mean feature vector across service-compatible eye crops.
- Classes: `['esotropia', 'exotropia', 'normal']`

## Selected Model

- Model family: `random_forest_pair_balanced`
- CV: `{'name': 'StratifiedGroupKFold', 'splits': 5, 'group_source': 'manifest participant_id inferred from near-duplicate clusters'}`
- Forced 3-class balanced accuracy: `0.5789`
- Forced 3-class macro-F1: `0.5669`
- Forced binary sensitivity: `0.6471`
- Forced binary specificity: `0.5152`

## Conservative Decision Policy

- Policy: `confidence_margin_abstention`
- Confidence threshold: `0.34`
- Margin threshold: `0.5`
- OOF coverage: `0.0448`
- OOF covered accuracy: `1.0`
- OOF confident wrong predictions: `0`
- OOF abstained count: `64`

## Research References Considered

- Corneal light-reflection/Hirschberg systems emphasize reflex/iris/pupil geometry and participant-level validation.
- Recent mobile-photo screening work supports feature-based Random Forest style classifiers over small interpretable geometry features.
- Public Roboflow 5-class strabismus data was identified as a future import candidate, but was not imported in this run.

## Deployment Recommendation

Deploy only as `research_candidate` with `INCONCLUSIVE` behavior enabled. Do not
use the output as a medical diagnosis. More clinician-confirmed images,
pseudostrabismus examples, poor-quality examples, and real participant IDs are
required before removing the abstention guard.
