# Hirschberg runtime optimization — 2026-10-16

Removed the mirror preview unused by the API, skipped visualization-only heatmaps/mask images, and removed the unused seed instance. The seed checkpoint remains required to initialize the iris architecture. Updated the importer so regeneration retains these optimizations; repeated import produced the identical manifest.

All 8 checkpoint SHA-256 values are unchanged. Crop resolution, decoding math, QC thresholds, class scores, and capture guards are unchanged. Optional visualization helpers keep their previous default for compatibility; production explicitly disables image generation.

## Regression validation

- 15/15 full API result dictionaries are exactly equal before and after, excluding latency.
- Includes 1 HUMAN_REVIEW with scores and 14 INCONCLUSIVE results. No preview score bypasses a failed QC gate.
- Synthetic real fusion logits are exactly equal.
- 27 model/API tests pass, including landmark/metadata equality for both eye sides and reflex QC equality without visualization images. One existing Starlette/httpx deprecation warning.
- Runtime integrity: 40 files verified.

## Measured Windows CPU results

| Metric | Before | After |
| --- | ---: | ---: |
| Model-loaded resident memory | 448.1 MiB | 447.9 MiB |
| Resident memory after image set | 615.1 MiB | 602.3 MiB |
| Process peak memory | 694.7 MiB | 685.4 MiB |
| Total time, 15 images | 16.150 s | 9.114 s |
| Median image time | 1.027 s | 0.482 s |
| Fusion classifier forwards | 13 | 1 |

Observed total time decreased about 43.6%. These are single sequential runs in fresh Windows CPU processes; cache, warmup, and system load can affect timing. This is not an independent clinical accuracy assessment, a repeated controlled benchmark, or a Render/Linux memory test. No worst-case 25MP/concurrent request validation was performed. Peak memory still exceeds 512 MiB. Provision memory with headroom (currently at least 2 GB suggested), then measure the deployed service. No deployment was performed.

The JSON report contains anonymous image hashes and per-case timings; it contains no original images or local source image paths.
