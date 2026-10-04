# Pedseye Hirschberg Phase A Data QA Report

Created: 2026-10-04T07:03:11.424655+00:00

Status: `PASS_WITH_WARNINGS`

## Outputs

- Manifest: `datasets/pedseye_hirschberg_manifest_v0.1.jsonl`
- Validation summary: `reports/pedseye_hirschberg_manifest_v0.1_summary.json`
- Detector overlays: `reports/pedseye_detector_overlays` (67 images)

## Dataset Counts

| Class | Current images | Research minimum | Gap |
|---|---:|---:|---:|
| normal | 33 | 300 | 267 |
| esotropia | 20 | 300 | 280 |
| exotropia | 14 | 300 | 286 |
| pseudostrabismus | 0 | 300 | 300 |
| poor_quality | 0 | 200 | 200 |

## Split Counts

`{'train': 49, 'val': 9, 'test': 9}`

## Participant IDs

Participant IDs were inferred from near-duplicate image clusters:
`inferred_near_duplicate_cluster_not_real_patient_id`.

These IDs are not real patient identifiers and cannot prove patient-level independence.

## Duplicate Findings

- Exact SHA-256 duplicate count: `0`
- Near-duplicate threshold: `4`
- Near-duplicate pair count: `16`

## Warnings

- Class normal below research minimum: 33 < 300
- Class esotropia below research minimum: 20 < 300
- Class exotropia below research minimum: 14 < 300
- Missing class: pseudostrabismus
- Missing class: poor_quality
- participant_id values are inferred clusters, not true patient IDs; patient-level leakage risk remains.
- Near-duplicate image pairs at threshold <= 4: 16

## Research-Only Recommendation

Do not deploy a Hirschberg model from this dataset. Use this manifest and overlays
for manual review, label correction, pseudostrabismus/poor-quality collection, and
future participant-level data governance.
