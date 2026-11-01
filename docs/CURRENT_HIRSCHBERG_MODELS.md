# Current bundle — 2026-11-01

Active model: matched_bundle_20261101_v1. Classifier retrained30 epochs on strict122 using matching pupil boundary fine-tune200, iris refined200, eyelid200. Reflex unchanged. All artifacts checked against model_bundle.json; legacy weights retained.

Geometry heuristics (0.4R vector/search ROI, 15% iris diameter difference, MediaPipe axis ratio0.85) are review warnings, not independently validated image rejection rules. Missing landmarks, invalid coordinates, reflex outside visible eye, missing boundary, unknown orientation/fixation remain blocking. Every class output remains research HUMAN_REVIEW; no clinical probability.

Clients must send the actual capture metadata: orientation=unmirrored or mirrored and fixation=confirmed only after confirmation. Defaults auto/unknown do not infer orientation from appearance. Newly confirmed patient_2–patient_5 metadata in research tests does not establish a default for future uploads.

39 backend tests pass. strict122:43/122 class proposals,37 agree folder labels among43;6 disagree. image_for_train:15/22 class proposals. These datasets have development/training exposure; no independent clinical accuracy. Same new weights with historical geometry gates give27/122 and11/22, so coverage increases mainly reflect policy changes. Existing backend processes must restart to replace the cached model.
