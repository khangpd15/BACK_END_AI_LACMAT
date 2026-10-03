# Backend Feature Contract

Contract version: `shared-v1.0.0`  
Runtime owner: backend Python in `app/services/shared_feature_contract.py`  
Frontend folder `D:\AI_Check_Lac` was referenced read-only. No frontend code was changed.

## Coordinate System

- Input coordinates are normalized image coordinates.
- Origin: top-left of image.
- X increases screen-right.
- Y increases screen-down.
- Time field `t` may be milliseconds or seconds; backend converts to seconds when elapsed span is greater than 200.
- Velocity is always computed from actual timestamp delta, not frame index or assumed FPS.

## Missing Values

- Extractor returns `None` for features that cannot be computed.
- Model services must reject missing/non-finite required features.
- No padding, truncation, zero-fill, or reshape is allowed to make inference run.

## Backend Shared Feature Order

| Index | Feature | Formula | Input | Unit | Missing |
|---:|---|---|---|---|---|
| 0 | `leftValidRatio` | count valid left / total | `leftValid` | ratio | never if total > 0 |
| 1 | `rightValidRatio` | count valid right / total | `rightValid` | ratio | never if total > 0 |
| 2 | `bothValidRatio` | count both valid / total | both validity flags | ratio | never if total > 0 |
| 3 | `meanDeltaX` | mean(`rightX-leftX`) | both-valid frames | normalized coord | `None` |
| 4 | `medianDeltaX` | median(`rightX-leftX`) | both-valid frames | normalized coord | `None` |
| 5 | `stdDeltaX` | population std(`rightX-leftX`) | both-valid frames | normalized coord | `None` |
| 6 | `minDeltaX` | min(`rightX-leftX`) | both-valid frames | normalized coord | `None` |
| 7 | `maxDeltaX` | max(`rightX-leftX`) | both-valid frames | normalized coord | `None` |
| 8 | `rangeDeltaX` | max-min delta X | both-valid frames | normalized coord | `None` |
| 9 | `meanAbsDeltaX` | mean(abs(`rightX-leftX`)) | both-valid frames | normalized coord | `None` |
| 10 | `meanDeltaY` | mean(`rightY-leftY`) | both-valid frames | normalized coord | `None` |
| 11 | `medianDeltaY` | median(`rightY-leftY`) | both-valid frames | normalized coord | `None` |
| 12 | `stdDeltaY` | population std(`rightY-leftY`) | both-valid frames | normalized coord | `None` |
| 13 | `minDeltaY` | min(`rightY-leftY`) | both-valid frames | normalized coord | `None` |
| 14 | `maxDeltaY` | max(`rightY-leftY`) | both-valid frames | normalized coord | `None` |
| 15 | `rangeDeltaY` | max-min delta Y | both-valid frames | normalized coord | `None` |
| 16 | `meanAbsDeltaY` | mean(abs(`rightY-leftY`)) | both-valid frames | normalized coord | `None` |
| 17 | `meanLeftX` | mean(`leftX`) | left-valid frames | normalized coord | `None` |
| 18 | `stdLeftX` | population std(`leftX`) | left-valid frames | normalized coord | `None` |
| 19 | `meanLeftY` | mean(`leftY`) | left-valid frames | normalized coord | `None` |
| 20 | `stdLeftY` | population std(`leftY`) | left-valid frames | normalized coord | `None` |
| 21 | `meanRightX` | mean(`rightX`) | right-valid frames | normalized coord | `None` |
| 22 | `stdRightX` | population std(`rightX`) | right-valid frames | normalized coord | `None` |
| 23 | `meanRightY` | mean(`rightY`) | right-valid frames | normalized coord | `None` |
| 24 | `stdRightY` | population std(`rightY`) | right-valid frames | normalized coord | `None` |
| 25 | `meanLeftVelocity` | mean(hypot(dX,dY)/dt) | consecutive left-valid frames | coord/s | `None` |
| 26 | `peakLeftVelocity` | max(hypot(dX,dY)/dt) | consecutive left-valid frames | coord/s | `None` |
| 27 | `meanRightVelocity` | mean(hypot(dX,dY)/dt) | consecutive right-valid frames | coord/s | `None` |
| 28 | `peakRightVelocity` | max(hypot(dX,dY)/dt) | consecutive right-valid frames | coord/s | `None` |
| 29 | `velocityDisparity` | abs(meanRightVelocity-meanLeftVelocity) | velocity features | coord/s | `None` |

## 10-15 FPS Candidate Feature Subset

Default declared subset in `fps_model_service.py`:

1. `leftValidRatio`
2. `rightValidRatio`
3. `bothValidRatio`
4. `meanDeltaY`
5. `medianDeltaY`
6. `stdDeltaY`
7. `minDeltaY`
8. `maxDeltaY`
9. `rangeDeltaY`
10. `meanAbsDeltaY`
11. `stdLeftX`
12. `stdLeftY`
13. `stdRightX`
14. `stdRightY`

If the artifact declares `feature_names`, the artifact list takes precedence and must match model `n_features_in_`.

## Separate Frontend Static Contract

`D:\AI_Check_Lac\FEATURE_CONTRACT_V1.md` documents a separate 10-feature client-side ONNX/static morphology contract. It is not the backend Cover Test model contract. Do not combine these feature orders without an explicit v2 migration and retraining plan.
