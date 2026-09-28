# Korean 15 FPS Transfer Candidate

## Training outcome

- Source files: `338` Korean recordings
- Target training rate: `15 FPS`
- Phase-offset views per recording: `4`
- Augmented rows after conflict exclusion: `1324`
- Unique sessions: `331`
- Unique participants: `126`
- Selected feature configuration: `15FPS_STABILITY_14` (14 features)
- Logistic `C`: `0.03`
- Class weight: `None`
- Participant overlap in every validation fold: `0`
- Calibration: `NONE`
- Threshold tuning: `NONE`
- RemiCare labels used: `0`

## Participant-group validation

- Brier score: `0.20807069`
- Log loss: `0.71424499`
- ROC AUC: `0.67693057`
- Balanced accuracy at 0.5: `0.56921091`
- F1 at 0.5: `0.78159757`
- Mean probability std across four 15 FPS phase offsets: `0.01250537`

Legacy B2 configuration retrained/evaluated on the same 15 FPS participant folds:

- Brier score: `0.23774120`
- Log loss: `0.77277114`
- ROC AUC: `0.67708860`
- Balanced accuracy at 0.5: `0.63765803`

The selected 15 FPS candidate changes Brier by `-0.02967051` (negative is better) on these source-domain folds.

## Safety conclusion

This candidate improves sampling-rate alignment and is suitable for side-by-side technical comparison. It cannot be trained to guarantee `NORMAL > 80%` for a known user, because doing so would use the desired answer as a target and invalidate the experiment. It remains a Korean-source model under webcam domain shift, not a clinical diagnostic model.
