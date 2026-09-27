# Korean Strabismus Feature Dictionary

This document details the complete feature space of **37 attributes** extracted from the **Korean Strabismus Eye-Tracker Dataset** ([hyunwoongko/strabismus-recognition](https://github.com/hyunwoongko/strabismus-recognition)), evaluating formulas, sources, units, and compatibility with the **RemiCare Webcam + MediaPipe** system.

---

## 1. Overview of Korean Feature Space

The Korean Eye Tracker dataset time-series consists of infrared pupil tracking recording at approximately **60 Hz**. The dataset contains 3 metadata/target fields and **34 numerical technical features**:

1. **Identifiers & Targets (3):** `sampleId`, `label_name`, `label`
2. **Sampling & Validity (7):** `sampleCount`, `leftValidRatio`, `rightValidRatio`, `bothValidRatio`, `estimatedHz`, `durationMs`, `meanIntervalMs`
3. **Inter-Ocular Disparity - Horizontal (7):** `meanDeltaX`, `medianDeltaX`, `stdDeltaX`, `minDeltaX`, `maxDeltaX`, `rangeDeltaX`, `meanAbsDeltaX`
4. **Inter-Ocular Disparity - Vertical (7):** `meanDeltaY`, `medianDeltaY`, `stdDeltaY`, `minDeltaY`, `maxDeltaY`, `rangeDeltaY`, `meanAbsDeltaY`
5. **Single-Eye Coordinate Statistics (8):** `meanLeftX`, `stdLeftX`, `meanLeftY`, `stdLeftY`, `meanRightX`, `stdRightX`, `meanRightY`, `stdRightY`
6. **Kinematics & Velocities (5):** `meanLeftVelocity`, `peakLeftVelocity`, `meanRightVelocity`, `peakRightVelocity`, `velocityDisparity`

---

## 2. Complete Technical Feature Specifications

### 2.1 Identifiers & Ground Truth Labels

#### 1. `sampleId`
- **Formula:** Cleaned filename identifier (e.g. `korean_normal_1-_all_gaze`)
- **Source Columns:** Filename
- **Meaning:** Unique sample identifier
- **Unit:** String
- **Aggregation:** N/A (Metadata)
- **Available in Korean:** YES
- **Available in RemiCare:** YES (`sampleId` UUID)
- **Compatible:** YES (Identifier)
- **Limitation:** Formatting conventions differ; identifier only.

#### 2. `label_name`
- **Formula:** Extracted from parent directory name (`NORMAL` or `STRABISMUS`)
- **Source Columns:** Directory path
- **Meaning:** Categorical ground truth class
- **Unit:** Category string (`NORMAL` / `STRABISMUS`)
- **Aggregation:** N/A
- **Available in Korean:** YES
- **Available in RemiCare:** NO (RemiCare is an unlabelled screening client in production)
- **Compatible:** Target label only
- **Limitation:** Training/evaluation ground truth.

#### 3. `label`
- **Formula:** `0 if label_name == "NORMAL" else 1`
- **Source Columns:** Mapped from `label_name`
- **Meaning:** Binary classification target (0 = Normal, 1 = Strabismus)
- **Unit:** Integer {0, 1}
- **Aggregation:** N/A
- **Available in Korean:** YES
- **Available in RemiCare:** NO (RemiCare produces screening status, not clinical label)
- **Compatible:** Target label only
- **Limitation:** Strictly binary.

---

### 2.2 Sampling & Validity Features

#### 4. `sampleCount`
- **Formula:** $N = \text{len}(\text{rows})$
- **Source Columns:** Row index / `CNT`
- **Meaning:** Total number of recorded frames in time series
- **Unit:** Count (frames)
- **Aggregation:** Direct count
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** RemiCare cover test duration is fixed per protocol (~33 samples/3 cycles), while Korean records ~1500 frames (~25 sec).

#### 5. `leftValidRatio`
- **Formula:** $\frac{\sum [LPV == 1]}{N}$
- **Source Columns:** `LPV` (Korean) / `leftValid` (RemiCare)
- **Meaning:** Proportion of frames where left eye was detected validly
- **Unit:** Ratio $[0.0, 1.0]$
- **Aggregation:** Mean of boolean valid flags
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** CONDITIONALLY COMPATIBLE
- **Limitation:** In RemiCare Cover Test, left eye is intentionally occluded during LEFT cover phase, reducing this ratio by protocol design.

#### 6. `rightValidRatio`
- **Formula:** $\frac{\sum [RPV == 1]}{N}$
- **Source Columns:** `RPV` (Korean) / `rightValid` (RemiCare)
- **Meaning:** Proportion of frames where right eye was detected validly
- **Unit:** Ratio $[0.0, 1.0]$
- **Aggregation:** Mean of boolean valid flags
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** CONDITIONALLY COMPATIBLE
- **Limitation:** Reduced intentionally during RIGHT cover phase in RemiCare.

#### 7. `bothValidRatio`
- **Formula:** $\frac{\sum [LPV == 1 \land RPV == 1]}{N}$
- **Source Columns:** `LPV, RPV` (Korean) / `leftValid, rightValid` (RemiCare)
- **Meaning:** Proportion of frames where both eyes were tracked simultaneously
- **Unit:** Ratio $[0.0, 1.0]$
- **Aggregation:** Mean of conjunction of validity flags
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** CONDITIONALLY COMPATIBLE
- **Limitation:** Cover Test has lower both-valid ratio by clinical design due to alternating occlusion.

#### 8. `estimatedHz`
- **Formula:** $\frac{1000.0}{\text{median}(\Delta t_{\text{pos}})}$ where $\Delta t_{\text{pos}}$ is in milliseconds
- **Source Columns:** `TIME` (Korean) / `t` (RemiCare)
- **Meaning:** Estimated sampling frequency of tracking hardware
- **Unit:** Hertz (Hz)
- **Aggregation:** Inverse of median positive interval
- **Available in Korean:** YES (~61.5 Hz)
- **Available in RemiCare:** YES (~30.0 Hz)
- **Compatible:** INCOMPATIBLE (Domain Shift)
- **Limitation:** Hardware eye tracker operates at 60 Hz; standard webcam operates at 30 Hz.

#### 9. `durationMs`
- **Formula:** $(t_{\text{end}} - t_{\text{start}}) \times 1000.0$
- **Source Columns:** `TIME` (Korean) / `t` (RemiCare)
- **Meaning:** Total elapsed recording duration
- **Unit:** Milliseconds (ms)
- **Aggregation:** Max timestamp minus min timestamp
- **Available in Korean:** YES (~16,000 ms)
- **Available in RemiCare:** YES (~1,000 - 3,000 ms)
- **Compatible:** INCOMPATIBLE (Protocol length shift)
- **Limitation:** Protocol-dependent duration.

#### 10. `meanIntervalMs`
- **Formula:** $\text{mean}(\Delta t_{\text{pos}})$
- **Source Columns:** `TIME` (Korean) / `t` (RemiCare)
- **Meaning:** Average interval between consecutive recorded frames
- **Unit:** Milliseconds (ms)
- **Aggregation:** Arithmetic mean of consecutive positive time differences
- **Available in Korean:** YES (~16.6 ms)
- **Available in RemiCare:** YES (~33.3 ms)
- **Compatible:** INCOMPATIBLE (Hardware frame rate shift)
- **Limitation:** Directly reflects camera/sensor FPS.

---

### 2.3 Inter-Ocular Disparity Features (Horizontal: $\Delta X = \text{RightX} - \text{LeftX}$)

All $\Delta X$ features are computed strictly on frames where both eyes are valid simultaneously ($LPV=1 \land RPV=1$).

#### 11. `meanDeltaX`
- **Formula:** $\frac{1}{M} \sum_{i \in \text{BothValid}} (RPCX_i - LPCX_i)$
- **Source Columns:** `RPCX, LPCX` (Korean) / `rightX, leftX` (RemiCare)
- **Meaning:** Average horizontal distance between pupils (inter-ocular disparity)
- **Unit:** Normalized image width coordinate $[0.0, 1.0]$
- **Aggregation:** Mean
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES (Shared core signal)
- **Limitation:** Affected by participant distance to camera if not face-scale normalized.

#### 12. `medianDeltaX`
- **Formula:** $\text{median}(\{RPCX_i - LPCX_i \mid i \in \text{BothValid}\})$
- **Source Columns:** `RPCX, LPCX` (Korean) / `rightX, leftX` (RemiCare)
- **Meaning:** Robust median horizontal inter-ocular distance
- **Unit:** Normalized coordinate $[0.0, 1.0]$
- **Aggregation:** Median
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES (Shared core signal)
- **Limitation:** Highly resistant to single-frame tracker jitter.

#### 13. `stdDeltaX`
- **Formula:** $\sqrt{\frac{1}{M} \sum ((\Delta X_i) - \overline{\Delta X})^2}$
- **Source Columns:** `RPCX, LPCX`
- **Meaning:** Variation of horizontal inter-ocular distance during recording
- **Unit:** Normalized coordinate
- **Aggregation:** Standard deviation ($ddof=0$)
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** In strabismus, refixation movements cause higher disparity fluctuation.

#### 14. `minDeltaX`
- **Formula:** $\min_{i \in \text{BothValid}} (RPCX_i - LPCX_i)$
- **Source Columns:** `RPCX, LPCX`
- **Meaning:** Minimum observed horizontal distance between pupils
- **Unit:** Normalized coordinate
- **Aggregation:** Minimum
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** Sensitive to extreme blink-release artifacts.

#### 15. `maxDeltaX`
- **Formula:** $\max_{i \in \text{BothValid}} (RPCX_i - LPCX_i)$
- **Source Columns:** `RPCX, LPCX`
- **Meaning:** Maximum observed horizontal distance between pupils
- **Unit:** Normalized coordinate
- **Aggregation:** Maximum
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** Sensitive to momentary eye drift.

#### 16. `rangeDeltaX`
- **Formula:** $maxDeltaX - minDeltaX$
- **Source Columns:** `RPCX, LPCX`
- **Meaning:** Peak-to-peak horizontal disparity range
- **Unit:** Normalized coordinate
- **Aggregation:** Peak-to-peak amplitude
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** Captures dynamic refixation span.

#### 17. `meanAbsDeltaX`
- **Formula:** $\frac{1}{M} \sum |RPCX_i - LPCX_i|$
- **Source Columns:** `RPCX, LPCX`
- **Meaning:** Mean absolute horizontal distance between eyes
- **Unit:** Normalized coordinate
- **Aggregation:** Mean of absolute values
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** Equals `meanDeltaX` when right eye is consistently to the right of left eye.

---

### 2.4 Inter-Ocular Disparity Features (Vertical: $\Delta Y = \text{RightY} - \text{LeftY}$)

All $\Delta Y$ features are computed on mutually valid frames ($LPV=1 \land RPV=1$).

#### 18. `meanDeltaY`
- **Formula:** $\frac{1}{M} \sum_{i \in \text{BothValid}} (RPCY_i - LPCY_i)$
- **Source Columns:** `RPCY, LPCY` (Korean) / `rightY, leftY` (RemiCare)
- **Meaning:** Average vertical alignment disparity between pupils
- **Unit:** Normalized coordinate
- **Aggregation:** Mean
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** Reflects vertical strabismus (hypertropia/hypotropia) or head tilt.

#### 19. `medianDeltaY`
- **Formula:** $\text{median}(\{RPCY_i - LPCY_i \mid i \in \text{BothValid}\})$
- **Source Columns:** `RPCY, LPCY`
- **Meaning:** Median vertical inter-ocular disparity
- **Unit:** Normalized coordinate
- **Aggregation:** Median
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** Robust against vertical eyelid occlusions.

#### 20. `stdDeltaY`
- **Formula:** $\sqrt{\frac{1}{M} \sum ((\Delta Y_i) - \overline{\Delta Y})^2}$
- **Source Columns:** `RPCY, LPCY`
- **Meaning:** Standard deviation of vertical disparity
- **Unit:** Normalized coordinate
- **Aggregation:** Standard deviation
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** Indicates vertical stability.

#### 21. `minDeltaY`
- **Formula:** $\min_{i \in \text{BothValid}} (RPCY_i - LPCY_i)$
- **Source Columns:** `RPCY, LPCY`
- **Meaning:** Minimum vertical distance between eyes
- **Unit:** Normalized coordinate
- **Aggregation:** Minimum
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES

#### 22. `maxDeltaY`
- **Formula:** $\max_{i \in \text{BothValid}} (RPCY_i - LPCY_i)$
- **Source Columns:** `RPCY, LPCY`
- **Meaning:** Maximum vertical distance between eyes
- **Unit:** Normalized coordinate
- **Aggregation:** Maximum
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES

#### 23. `rangeDeltaY`
- **Formula:** $maxDeltaY - minDeltaY$
- **Source Columns:** `RPCY, LPCY`
- **Meaning:** Vertical disparity excursion range
- **Unit:** Normalized coordinate
- **Aggregation:** Peak-to-peak amplitude
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES

#### 24. `meanAbsDeltaY`
- **Formula:** $\frac{1}{M} \sum |RPCY_i - LPCY_i|$
- **Source Columns:** `RPCY, LPCY`
- **Meaning:** Mean absolute vertical offset between pupils
- **Unit:** Normalized coordinate
- **Aggregation:** Mean of absolute values
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES

---

### 2.5 Single-Eye Coordinate Statistics

#### 25. `meanLeftX` / 27. `meanLeftY`
- **Formula:** $\text{mean}(LPCX)$ / $\text{mean}(LPCY)$ on $LPV==1$
- **Source Columns:** `LPCX, LPCY`
- **Meaning:** Mean position of left pupil in normalized viewport
- **Unit:** Normalized image coordinates $[0.0, 1.0]$
- **Aggregation:** Mean
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** CONDITIONALLY COMPATIBLE
- **Limitation:** Highly dependent on head position in camera frame.

#### 26. `stdLeftX` / 28. `stdLeftY`
- **Formula:** $\text{std}(LPCX)$ / $\text{std}(LPCY)$ on $LPV==1$
- **Source Columns:** `LPCX, LPCY`
- **Meaning:** Jitter / movement dispersion of left eye
- **Unit:** Normalized coordinate
- **Aggregation:** Standard deviation
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** Uncontrolled head motion adds noise to std.

#### 29. `meanRightX` / 31. `meanRightY`
- **Formula:** $\text{mean}(RPCX)$ / $\text{mean}(RPCY)$ on $RPV==1$
- **Source Columns:** `RPCX, RPCY`
- **Meaning:** Mean position of right pupil in viewport
- **Unit:** Normalized coordinate
- **Aggregation:** Mean
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** CONDITIONALLY COMPATIBLE
- **Limitation:** Depends on participant centering.

#### 30. `stdRightX` / 32. `stdRightY`
- **Formula:** $\text{std}(RPCX)$ / $\text{std}(RPCY)$ on $RPV==1$
- **Source Columns:** `RPCX, RPCY`
- **Meaning:** Movement dispersion of right eye
- **Unit:** Normalized coordinate
- **Aggregation:** Standard deviation
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES

---

### 2.6 Kinematics & Velocities

#### 33. `meanLeftVelocity` / 35. `meanRightVelocity`
- **Formula:** $\text{mean}\left(\frac{\sqrt{(X_{i+1} - X_i)^2 + (Y_{i+1} - Y_i)^2}}{\Delta t_i}\right)$ for valid consecutive frames where $\Delta t > 0.001\text{ s}$
- **Source Columns:** `LPCX, LPCY, TIME` / `RPCX, RPCY, TIME`
- **Meaning:** Average velocity of pupil movement
- **Unit:** Normalized coordinate units per second ($coord/s$)
- **Aggregation:** Mean frame-to-frame velocity
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES (Shared formula)
- **Limitation:** Sensor noise at higher framerates can elevate mean velocity.

#### 34. `peakLeftVelocity` / 36. `peakRightVelocity`
- **Formula:** $\max\left(\frac{\sqrt{(X_{i+1} - X_i)^2 + (Y_{i+1} - Y_i)^2}}{\Delta t_i}\right)$
- **Source Columns:** `LPCX, LPCY, TIME` / `RPCX, RPCY, TIME`
- **Meaning:** Maximum saccadic or refixation speed reached by eye
- **Unit:** $coord/s$
- **Aggregation:** Maximum velocity
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** Outlier detection glitches can spike peak velocity.

#### 37. `velocityDisparity`
- **Formula:** $|meanRightVelocity - meanLeftVelocity|$
- **Source Columns:** Computed from left and right mean velocities
- **Meaning:** Disparity in movement dynamics between left and right eyes
- **Unit:** $coord/s$
- **Aggregation:** Absolute difference
- **Available in Korean:** YES
- **Available in RemiCare:** YES
- **Compatible:** YES
- **Limitation:** In unilateral strabismus, deviating eye may display different kinetic patterns.
