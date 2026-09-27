# Korean Model → RemiCare Feature Compatibility Report

This document reports the technical compatibility analysis between the **Korean Infrared Eye-Tracker Dataset Space (37 features)** and the **RemiCare Webcam + MediaPipe Screening Space**.

---

## 1. Summary of Comparative Findings

1. **Shared Feature Subspace:**
   - Out of the 34 numerical features in the Korean feature space, **26 core features** are mathematically identical in formula and can be computed on both domains (Inter-Ocular Disparity $\Delta X$ & $\Delta Y$, coordinate distributions, movement velocities, and detection validity ratios).
2. **Incompatible Features (Domain / Hardware Shifts):**
   - **4 features** (`estimatedHz`, `meanIntervalMs`, `sampleCount`, `durationMs`) are strictly tied to hardware (60 Hz infrared camera vs 30 Hz RGB webcam) or protocol duration (25-second continuous gaze vs 3-cycle Cover Test). Feeding these into a model trained on Korean data causes domain shift errors.
3. **RemiCare Cover-Test Specific Features (REMICARE_ONLY):**
   - RemiCare contains phase-specific refixation kinematics (`timeToPeak`, `movementDuration`, `baselineStdX`, `cycleConsistency`, `peakAbsDx relative to baseline`). The Korean dataset lacks phase tags (`phase = UNKNOWN`) and cannot generate these.
4. **Conclusion on Model Direct Transfer:**
   - The Korean 37-feature model **CANNOT** be fed RemiCare inputs directly without feature imputation or domain mismatch.
   - Per safety guidelines, **NO synthetic imputation** (0 filling, median filling, sentinel values) is permitted.
   - Solution: Build and evaluate a **Shared Feature Model** trained strictly on the **26 SHARED features**.

---

## 2. Master Feature Compatibility Table

| # | Feature Name | Korean Source | RemiCare Source | Same Formula | Classification | Note / Technical Assessment |
|---|---|---|---|---|---|---|
| 1 | `sampleId` | Filename | `sampleId` | Metadata | `SHARED` | Unique string identifier. |
| 2 | `label_name` | Folder name | N/A | Metadata | `KOREAN_ONLY` | Clinical reference label (NORMAL / STRABISMUS). RemiCare produces screening status. |
| 3 | `label` | `0` or `1` | N/A | Target | `KOREAN_ONLY` | Binary ML training target (0 = Normal, 1 = Strabismus). |
| 4 | `sampleCount` | Row count | Row count | YES | `INCOMPATIBLE` | Korean records ~1500 frames; RemiCare Cover Test records ~33-100 frames. |
| 5 | `leftValidRatio` | `count(LPV=1)/N` | `count(leftValid)/N` | YES | `SHARED` | Identical ratio of left eye valid frames. |
| 6 | `rightValidRatio`| `count(RPV=1)/N` | `count(rightValid)/N`| YES | `SHARED` | Identical ratio of right eye valid frames. |
| 7 | `bothValidRatio` | `count(both=1)/N` | `count(bothValid)/N` | YES | `SHARED` | Identical ratio of mutually tracked frames. |
| 8 | `estimatedHz` | `1000/med(dt)` | `1000/med(dt)` | YES | `INCOMPATIBLE` | Hardware discrepancy: Korean ~61.5 Hz vs Webcam ~30.0 Hz. |
| 9 | `durationMs` | `(t_end - t_0)*1000`| `(t_end - t_0)*1000`| YES | `INCOMPATIBLE` | Protocol discrepancy: ~16,000 ms vs ~1,000 - 3,000 ms. |
| 10| `meanIntervalMs`| `mean(dt)*1000` | `mean(dt)*1000` | YES | `INCOMPATIBLE` | Direct function of FPS (~16.6 ms vs ~33.3 ms). |
| 11| `meanDeltaX` | `mean(RPCX - LPCX)` | `mean(rightX - leftX)` | YES | `SHARED` | Core disparity signal: average horizontal inter-ocular distance. |
| 12| `medianDeltaX` | `median(RPCX - LPCX)`| `median(rightX - leftX)`| YES | `SHARED` | Median horizontal disparity (robust against blink artifacts). |
| 13| `stdDeltaX` | `std(RPCX - LPCX)` | `std(rightX - leftX)` | YES | `SHARED` | Horizontal disparity jitter / variation. |
| 14| `minDeltaX` | `min(RPCX - LPCX)` | `min(rightX - leftX)` | YES | `SHARED` | Minimum horizontal distance. |
| 15| `maxDeltaX` | `max(RPCX - LPCX)` | `max(rightX - leftX)` | YES | `SHARED` | Maximum horizontal distance. |
| 16| `rangeDeltaX` | `max - min` | `max - min` | YES | `SHARED` | Peak-to-peak horizontal disparity excursion. |
| 17| `meanAbsDeltaX` | `mean(abs(dx))` | `mean(abs(dx))` | YES | `SHARED` | Mean magnitude of horizontal ocular distance. |
| 18| `meanDeltaY` | `mean(RPCY - LPCY)` | `mean(rightY - leftY)` | YES | `SHARED` | Vertical disparity (hypertropia indicator / head tilt). |
| 19| `medianDeltaY` | `median(RPCY - LPCY)`| `median(rightY - leftY)`| YES | `SHARED` | Robust median vertical disparity. |
| 20| `stdDeltaY` | `std(RPCY - LPCY)` | `std(rightY - leftY)` | YES | `SHARED` | Vertical disparity fluctuation. |
| 21| `minDeltaY` | `min(RPCY - LPCY)` | `min(rightY - leftY)` | YES | `SHARED` | Minimum vertical distance. |
| 22| `maxDeltaY` | `max(RPCY - LPCY)` | `max(rightY - leftY)` | YES | `SHARED` | Maximum vertical distance. |
| 23| `rangeDeltaY` | `max - min` | `max - min` | YES | `SHARED` | Vertical excursion range. |
| 24| `meanAbsDeltaY` | `mean(abs(dy))` | `mean(abs(dy))` | YES | `SHARED` | Mean absolute vertical offset. |
| 25| `meanLeftX` | `mean(LPCX)` | `mean(leftX)` | YES | `SHARED` | Mean horizontal left eye position. |
| 26| `stdLeftX` | `std(LPCX)` | `std(leftX)` | YES | `SHARED` | Dispersion of left eye horizontal coordinate. |
| 27| `meanLeftY` | `mean(LPCY)` | `mean(leftY)` | YES | `SHARED` | Mean vertical left eye position. |
| 28| `stdLeftY` | `std(LPCY)` | `std(leftY)` | YES | `SHARED` | Dispersion of left eye vertical coordinate. |
| 29| `meanRightX` | `mean(RPCX)` | `mean(rightX)` | YES | `SHARED` | Mean horizontal right eye position. |
| 30| `stdRightX` | `std(RPCX)` | `std(rightX)` | YES | `SHARED` | Dispersion of right eye horizontal coordinate. |
| 31| `meanRightY` | `mean(RPCY)` | `mean(rightY)` | YES | `SHARED` | Mean vertical right eye position. |
| 32| `stdRightY` | `std(RPCY)` | `std(rightY)` | YES | `SHARED` | Dispersion of right eye vertical coordinate. |
| 33| `meanLeftVelocity`| `mean(dist/dt)` | `mean(dist/dt)` | YES | `SHARED` | Left eye average velocity ($coord/s$). |
| 34| `peakLeftVelocity`| `max(dist/dt)` | `max(dist/dt)` | YES | `SHARED` | Left eye peak saccadic speed ($coord/s$). |
| 35| `meanRightVelocity`| `mean(dist/dt)`| `mean(dist/dt)` | YES | `SHARED` | Right eye average velocity ($coord/s$). |
| 36| `peakRightVelocity`| `max(dist/dt)`| `max(dist/dt)` | YES | `SHARED` | Right eye peak saccadic speed ($coord/s$). |
| 37| `velocityDisparity`| `abs(vR - vL)` | `abs(vR - vL)` | YES | `SHARED` | Inter-ocular velocity discrepancy. |

---

## 3. RemiCare-Only Native Features (Not present in Korean Dataset)

The following Cover-Test specific features are computed in RemiCare but are **UNAVAILABLE** in the Korean dataset because the Korean dataset has no phase tagging (`phase = UNKNOWN`):

1. `cyclesAnalyzed`: Number of cover-uncover cycles completed (Korean has 0 cycles).
2. `cycleConsistency`: Cross-cycle peak displacement repeatability (Korean has no cycle structure).
3. `timeToPeak`: Time from uncover event to maximum refixation movement.
4. `movementDuration`: Elapsed duration of refixation tracking phase.
5. `baselineTrackedX`, `baselineTrackedY`: Median coordinate during BASELINE phase (Korean has no designated baseline phase).
6. `baselineStdX`, `baselineStdY`: Stability of eye position during baseline fixation.
7. `signedDx = currentX - baselineX`: Signed deviation of tracked eye from its baseline position.
8. `signedDy = currentY - baselineY`: Signed vertical deviation from baseline.
9. `peakAbsDx`, `peakAbsDy`: Peak magnitude of refixation deviation from baseline.

---

## 4. Transfer Strategy: Shared Feature Model

To enable cross-domain transfer without data fabrication or illegal imputation:
1. **Model Architecture:** Train a specialized `SharedFeatureModel` on the Korean training set using **only the 26 SHARED features**.
2. **Domain Evaluation:** Test the model on the Korean independent test set to obtain a verified baseline.
3. **Transfer Inference:** RemiCare extracts the identical 26 features from `sample.json` and evaluates inference safely through `SharedFeatureModel`.
