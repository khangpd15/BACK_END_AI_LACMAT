# Participant-Level Data Leakage Audit Report

**Audit Status:** `PARTICIPANT_LEVEL_LEAKAGE`

> **WARNING:** This audit is mandatory before claiming model generalization. If overlap > 0, the test set cannot be claimed as an independent participant-level evaluation.

## 1. Summary Statistics

- **Total Training Files:** 297
- **Unique Training Participants:** 112
- **Total Testing Files:** 41
- **Unique Testing Participants:** 15
- **Overlapping Participants:** **2**

## 2. Leakage Findings & Specific Overlaps

| Participant ID (Extracted) | Train File Count | Test File Count | Leakage Risk |
|---|---|---|---|
| `김민경` | 5 | 1 | HIGH (Same subject present in both sets) |
| `장진우` | 1 | 3 | HIGH (Same subject present in both sets) |

### Root Cause Analysis:
In the original Korean repository, certain subjects (e.g. `장진우`, `김민경`) had multiple recording sessions. One session was filed under `data/normal/` (training) while subsequent sessions were filed under `data/test_normal/` (testing). This demonstrates that the test set provided in the source repository is **session-split**, NOT strictly **participant-split**.

## 3. Scientific and Regulatory Implications

1. **Data Leakage Warning:** Performance figures on this test set include subject identity leakage. The test accuracy cannot be considered an unbiased estimate of generalization to unseen subjects.
2. **RemiCare Implication:** Because RemiCare tests completely novel subjects, performance on RemiCare cannot be predicted from this test set.
