# Korean 10--15 FPS Robust Transfer Candidate

## Training outcome

- Source files: `338` Korean recordings
- Sampling augmentation: fixed `[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]` FPS (four phases each) plus `3` variable/drop-frame profiles
- Augmented rows after conflict exclusion: `8937`
- Unique sessions: `331`
- Unique participants: `126`
- Selected feature configuration: `15FPS_STABILITY_14` (14 features)
- Logistic `C`: `0.01`
- Class weight: `{0: 1.4, 1: 1.0}`
- Participant overlap in every validation fold: `0`
- Calibration: `NONE`
- Threshold tuning: `NONE`
- RemiCare labels used: `0`

## Participant-group validation on identical 10--15 FPS views

| Metric | New 10--15 FPS candidate | Previous 15 FPS configuration | Delta |
|---|---:|---:|---:|
| Brier score (lower better) | 0.20710619 | 0.19863040 | +0.00847579 |
| Log loss (lower better) | 0.73671995 | 0.59113044 | +0.14558951 |
| ROC AUC | 0.67367290 | 0.67769772 | -0.00402482 |
| Balanced accuracy at 0.5 | 0.57266158 | 0.55404239 | +0.01861919 |
| Mean probability std across rate profiles (lower better) | 0.01560107 | 0.00807619 | +0.00752489 |

## Safety conclusion

The model is selected only from participant-separated Korean validation. The saved RemiCare session is unlabelled and was not used as a training target. This remains a research transfer output under Korean eye-tracker to webcam domain shift, not a diagnosis or clinically validated probability.
