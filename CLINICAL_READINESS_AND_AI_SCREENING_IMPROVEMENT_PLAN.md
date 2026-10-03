# RemiCare Clinical Readiness & AI Screening Improvement Plan

Ngày kiểm tra: 2026-10-03  
Phạm vi:

- Frontend: `D:\AI_Check_Lac` (read-only, không sửa code)
- Backend: `D:\REMICARE-STRABISMUS-AI`

## Kết Luận Ngắn

**Chưa sẵn sàng tung ra lâm sàng như một công cụ hỗ trợ quyết định y khoa hoặc chẩn đoán.**

Hệ thống hiện phù hợp hơn cho:

- demo kỹ thuật có kiểm soát,
- nghiên cứu thu thập dữ liệu,
- pilot nội bộ với disclaimer rõ,
- workflow sàng lọc không chẩn đoán, có bác sĩ/chuyên viên nhãn khoa giám sát.

Không nên dùng độc lập để trả lời bệnh nhân theo nghĩa: “bình thường / bị lác / nguy cơ bệnh” cho tới khi có clinical validation độc lập, model registry đầy đủ, quản trị dữ liệu, quy trình regulatory, và kiểm thử an toàn/hiệu năng hoàn chỉnh.

## Chuẩn Đối Chiếu Bên Ngoài

Các nguồn chính thống được dùng để đặt tiêu chí readiness:

- WHO AI for health ethics: yêu cầu AI y tế bảo vệ an toàn con người, minh bạch, trách nhiệm, công bằng và quản trị xuyên suốt vòng đời.
  - https://iris.who.int/bitstream/handle/10665/341996/9789240029200-eng.pdf
- WHO regulatory considerations for AI in health: nhấn mạnh documentation/transparency, risk management, intended use, analytical/clinical validation, data quality, privacy/data protection.
  - https://www.who.int/publications/i/item/9789240078871
- FDA/IMDRF SaMD clinical evaluation: SaMD cần chứng minh safety, effectiveness, performance theo intended use.
  - https://www.fda.gov/medical-devices/software-medical-device-samd/global-approach-software-medical-device

## Tình Trạng Frontend `AI_Check_Lac`

### Điểm Mạnh

- Có flow sàng lọc hoàn chỉnh: position check, gaze 4 directions, Cover Test, Brock String, final summary.
- Có ngôn ngữ non-diagnostic trong nhiều module.
- Có quality gate cho Cover Test: kiểm tra valid cycles, NaN/Infinity, jitter, displacement mismatch, baseline/eye width.
- Time-series frontend dùng `performance.now()` và downsample theo timestamp, không dùng frame index làm thời gian.
- API client có timeout, abort, performance logging, và backend URL qua `VITE_AI_BACKEND_URL`.
- Test hiện có pass: `scripts/runAllTests.js` kiểm tra camera coordinate transform 5/5.

### Rủi Ro Chưa Sẵn Sàng Lâm Sàng

1. **Client-side ONNX vẫn zero-fill feature thiếu.**
   - `src/services/aiInferenceService.js` đưa missing/non-finite feature thành `0.0`.
   - Với lâm sàng, input contract thiếu phải trả `INCONCLUSIVE`, không cho model chạy.

2. **Có nhiều nguồn AI/threshold chưa thống nhất.**
   - Frontend ONNX static model: 10 features, threshold `0.5`.
   - Backend bilateral ROI ONNX: threshold `0.20`.
   - Backend Cover Test 10-15 FPS model: tabular/time-series consensus.
   - Cần model registry và policy rõ: nguồn nào là primary, source nào chỉ là supporting/research.

3. **AI image signal có thể đẩy final summary thành `SCREENING_ATTENTION`.**
   - Rule engine cẩn trọng về disclaimer, nhưng nếu AI image `SUSPICIOUS`, final status attention dù Cover Test không suspicious.
   - Trước clinical validation, AI image nên là “supporting signal / repeat test / clinician review”, không nên tự đẩy final action mạnh.

4. **Manual snap có thể bypass quality stability.**
   - `Gaze4DirectionsStep.jsx` có click-to-snap với `qualityScore: 0.95`.
   - Cần chỉ cho manual snap khi quality gate đạt điều kiện tối thiểu, hoặc đánh dấu `manual_capture_unverified`.

5. **Test coverage frontend còn mỏng.**
   - Hiện test script chủ yếu cover camera coordinate transform.
   - Chưa có automated tests cho AI feature contract, low-quality gating, Cover Test invalid series, final summary safety rules, backend failure fallback.

6. **ONNX Runtime Web tải WASM từ CDN.**
   - `aiInferenceService.js` cấu hình wasmPaths qua jsdelivr.
   - Với triển khai lâm sàng, nên pin/serve local asset, kiểm soát version, CSP, offline/failure behavior.

## Tình Trạng Backend `REMICARE-STRABISMUS-AI`

### Điểm Mạnh

- FastAPI có route health, CORS configurable, GZip, timing header.
- Model backend được load singleton, không load mỗi request.
- Có Supabase/PostgreSQL schema cho session/cycle/result metadata.
- Raw biometric image không được lưu ở image endpoint; Cover Test primary path lưu numeric trajectory và metadata.
- Có data integrity gate cho transfer endpoint.
- Đã có feature contract backend `shared-v1.0.0`.
- Đã siết các lỗi quan trọng ở pass trước:
  - 10-15 FPS model không zero-fill feature thiếu.
  - Missing model trả `INCONCLUSIVE`.
  - Model artifact/input dimension được kiểm tra.
  - Duplicate timestamp bị reject ở current và legacy session path.
- Test mới pass: `3 passed, 1 skipped`.

### Rủi Ro Chưa Sẵn Sàng Lâm Sàng

1. **Chưa có clinical validation độc lập trên RemiCare webcam data.**
   - Tài liệu hiện tại cũng ghi rõ Korean dataset/domain shift và no RemiCare clinical labels.
   - Không thể claim sensitivity/specificity/PPV/NPV/AUC lâm sàng.

2. **Dataset/training có rủi ro leakage/domain shift.**
   - Tài liệu backend ghi Korean source là session-split, không hoàn toàn patient-level split.
   - Model transfer từ infrared eye tracker sang webcam MediaPipe có domain shift lớn.

3. **Threshold chưa được clinical calibration.**
   - Bilateral ROI threshold `0.20`.
   - Frontend static ONNX threshold `0.5`.
   - Cover Test thresholds là engineering parameters.
   - Cần calibration protocol với ground truth nhãn khoa.

4. **Model registry mới ở dạng tài liệu/file, chưa enforced bằng DB/runtime đầy đủ.**
   - Cần registry table hoặc registry JSON signed/versioned với artifact hash, feature version, preprocessing version, metrics, intended use.

5. **Error envelope chưa đồng nhất.**
   - Một số endpoint trả `HTTPException.detail` dạng string/object khác nhau.
   - Clinical product cần error code chuẩn: `LOW_QUALITY_INPUT`, `FEATURE_CONTRACT_MISMATCH`, `MODEL_NOT_READY`, `TIMESERIES_INVALID`, etc.

6. **Performance report chưa có benchmark thực tế.**
   - Có total timing header, nhưng thiếu stage timing cho validation/storage/feature/inference/db.
   - Chưa có load/concurrency test cho production.

7. **Database migration/startup DDL cần hardening.**
   - Có auto-migration trong startup.
   - Với production clinical, migration phải có review, rollback, không drop table ngầm.

8. **Privacy governance chưa enforce bằng product flow.**
   - Có tài liệu data governance, nhưng cần consent, retention, deletion, audit trail, role-based access thực thi.

## Clinical Readiness Verdict

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Intended use rõ ràng | Một phần | Có disclaimer, nhưng cần khóa wording/product claims. |
| Analytical validation | Chưa đủ | Có feature parity/test mới, nhưng chưa đủ coverage. |
| Clinical validation | Chưa đạt | Chưa có RemiCare patient-level clinical ground truth validation. |
| Model versioning | Một phần | Có docs/guards, cần runtime registry chính thức. |
| Data privacy/governance | Một phần | Có code tránh lưu ảnh, nhưng cần consent/retention/deletion enforcement. |
| Safety/error handling | Một phần | Đã cải thiện backend, frontend còn zero-fill static ONNX. |
| Performance/reliability | Chưa đủ | Chưa có load test, monitoring, SLO, rollback. |
| Regulatory readiness | Chưa đạt | Cần regulatory classification + QMS + risk management. |

**Kết luận:** Không tung ra lâm sàng độc lập. Chỉ nên dùng trong nghiên cứu/pilot có giám sát, với output là “screening signal / cần kiểm tra thêm”, không phải diagnosis.

## Danh Sách Chức Năng Cần Cải Tiến — AI Sàng Lọc Là Trọng Tâm

### P0 — Chặn Sai Kết Quả Trước Khi Pilot

1. **Frontend AI contract guard**
   - Không zero-fill missing feature trong `aiInferenceService.js`.
   - Nếu feature thiếu/non-finite: trả `INCONCLUSIVE_FEATURE_CONTRACT`.
   - Validate `model_metadata.features.length === AI_FEATURE_ORDER.length`.

2. **Unify model decision policy**
   - Tạo một `screening_decision_policy`:
     - Cover Test dynamic = primary signal.
     - Image AI = supporting signal trước khi có validation.
     - Static client ONNX = dev/research only hoặc tắt trong production.
   - Không để AI image một mình đẩy final status mạnh nếu Cover Test tốt, trừ khi intended use/validation chứng minh.

3. **Model registry runtime**
   - Registry phải chứa: model id, version, artifact hash, feature version, preprocessing version, input dimension, threshold, trained data, validation metrics, status.
   - Backend `/health` trả model loaded + registry metadata.
   - Frontend hiển thị model source/version trong technical details.

4. **Low-quality handling**
   - Low quality, invalid ROI, missing landmarks, duplicate timestamps, insufficient cycles đều phải ra `INCONCLUSIVE`.
   - Không fallback thành normal/clear.

5. **Clinical wording lock**
   - Toàn UI/API dùng: “tín hiệu sàng lọc”, “cần kiểm tra thêm”, “chưa ghi nhận tín hiệu”.
   - Không dùng “chẩn đoán”, “xác suất mắc bệnh”, “bị lác” trong output user-facing.

### P1 — Clinical Validation Foundation

6. **Ground truth capture workflow**
   - Thêm schema nhập ground truth từ ophthalmologist/orthoptist/PACT.
   - Ground truth phải tách khỏi inference path, không gửi từ frontend trong request inference.

7. **Patient-level dataset split**
   - Dataset builder bắt buộc split theo patient/session group.
   - Không random split frame/cycle cùng người vào train/test.

8. **Clinical metrics dashboard**
   - Sensitivity, specificity, ROC-AUC, PR-AUC, PPV, NPV, confusion matrix.
   - Stratify theo tuổi, thiết bị, ánh sáng, kính, camera, distance, skin tone nếu có dữ liệu phù hợp.

9. **Threshold calibration**
   - Calibrate threshold theo intended use:
     - high sensitivity screening,
     - acceptable false positive rate,
     - age/device subgroup analysis.
   - Threshold thay đổi phải là model registry migration, không hard-code.

10. **Explainability/traceability**
   - Mỗi result lưu: request id, session id, model id/version, feature version, quality summary, reason codes, latency.
   - Không log raw face images.

### P2 — Nâng Cấp AI Sàng Lọc

11. **Temporal Cover Test feature v2**
   - Add dynamic features:
     - max/mean displacement,
     - peak/mean velocity,
     - acceleration,
     - movement latency,
     - movement duration,
     - settling time,
     - stability,
     - direction consistency.
   - Chỉ collect/analyze trước, không đưa vào production model nếu chưa retrain/validate.

12. **Fusion engine có quality-aware gating**
   - Inputs: Cover Test model, image model, static model, quality.
   - Output: `VALID`, `INDETERMINATE`, `SCREENING_SIGNAL`.
   - Không force fusion khi source quality thấp.

13. **Image AI pipeline hardening**
   - ROI detector contract rõ: both-eye ROI dimensions/aspect/content.
   - Store no image by default; optional research storage only with consent.
   - Add calibration test set for ROI model.

14. **Frontend/backed parity tests**
   - JS/Python parity cho:
     - Cover Test raw sample schema,
     - 30 shared backend features,
     - static 10-feature ONNX contract,
     - missing-value behavior.

15. **Model monitoring**
   - Track input drift, quality drift, model confidence distribution, INCONCLUSIVE rate.
   - Alert khi production distribution lệch training validation.

### P3 — Production/Clinical Operations

16. **Regulatory/QMS package**
   - Intended use statement.
   - Risk management file.
   - Software lifecycle SOP.
   - Verification/validation report.
   - Cybersecurity/privacy documentation.
   - Change control + model update process.

17. **Consent + deletion**
   - Consent UI for data capture.
   - Retention policy.
   - Delete/export session data.
   - Admin audit log.

18. **Reliability**
   - Load test backend.
   - Offline/timeout behavior.
   - Retry policy.
   - SLO for inference and storage.
   - Deployment rollback.

19. **Security**
   - CSP for frontend.
   - Pin/serve ONNX WASM locally instead of CDN for clinical mode.
   - Secret scanning.
   - Supabase RLS/access control review.

20. **Clinical pilot mode**
   - Disable diagnostic language.
   - Require clinician review flag.
   - Export clinician-facing evidence report.
   - Keep model output hidden/secondary if study protocol requires blinded comparison.

## Recommended Release Stages

### Stage A — Internal Engineering Demo

Allowed:

- Show camera flow.
- Save test sessions.
- Return non-diagnostic demo signals.

Not allowed:

- Clinical claims.
- Patient-facing medical decision.

### Stage B — Supervised Research Pilot

Requirements:

- Consent.
- Data retention/deletion.
- Ground truth workflow.
- Clinician oversight.
- All AI outputs labeled research/screening only.

### Stage C — Clinical Screening Support

Requirements:

- Patient-level validation.
- Locked model registry.
- Calibrated thresholds.
- Risk management + regulatory review.
- Monitoring and rollback.

## Immediate Next Actions

1. Fix frontend `aiInferenceService.js` missing feature behavior.
2. Decide whether to disable frontend static ONNX in production.
3. Create runtime model registry and model health contract.
4. Add clinical validation dataset schema and patient-level split tooling.
5. Add final summary safety test cases: low quality, AI suspicious but Cover Test normal, backend unavailable, missing ROI, duplicate timestamps.
