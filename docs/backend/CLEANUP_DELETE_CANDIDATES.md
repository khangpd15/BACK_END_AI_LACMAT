# Cleanup Delete Candidates

## Mục tiêu

Project chỉ giữ chức năng:

> **Binocular Screening / Sàng lọc hai mắt** đang chạy hiện tại.

Quy trình thực tế của chức năng cần bảo tồn:
```text
Webcam
  ↓
Theo dõi hai mắt (MediaPipe Face Mesh, Eye Tracking, One-Euro Filter, Eye ROI)
  ↓
Sampling khoảng 15 FPS (CoverTestTimeSeriesService)
  ↓
Thu thập trajectory / features (Baseline, Uncover Trajectory, 14 canonical features)
  ↓
Cover / Uncover test (CoverTestStep, ProtocolService, PositionCheck)
  ↓
AI inference (POST /api/v1/cover-test/sessions -> session_service -> fps_model_service)
  ↓
Model: remicare_15fps_candidate.joblib (Korean 10-15 FPS robust transfer candidate, v1.1.0, 14 features)
  ↓
Prediction: NORMAL / STRABISMUS / INCONCLUSIVE
  ↓
Hiển thị kết quả sàng lọc hai mắt (CoverTestStep result card, FinalScreeningResult)
```

> [!NOTE]
> **TRẠNG THÁI: ĐÃ THỰC HIỆN DỌN DẸP THÀNH CÔNG (CLEANUP COMPLETED)**
> Toàn bộ các file trong danh sách DELETE CANDIDATES đã được loại bỏ an toàn.
> Hệ thống đã được kiểm tra và xác minh:
> - Frontend build sạch 100% (`npm run build` thành công, 0 lỗi).
> - Backend FastAPI app import và nạp model thành công (`remicare_15fps_candidate.joblib` 14 features hoạt động).
> - Chỉ giữ lại duy nhất chức năng **Binocular Screening** đang chạy.

---

## 1. Files đề xuất xóa

Bảng tổng hợp các file mã nguồn, component, service, model và script dư thừa có bằng chứng xác thực không thuộc execution path của Binocular Screening:

| # | File | Loại | Lý do xóa | Evidence | Confidence |
|---|---|---|---|---|---|
| 1 | `src/pages/StaticEyeTest.jsx` | Old Frontend Page | Trang đo tỉ lệ tĩnh Phase 1 cũ; không nằm trong navigation bar, không tham gia sàng lọc hai mắt. | `App.jsx` tab không hiển thị trên UI; không có component nào gọi. | HIGH |
| 2 | `src/pages/StrabismusScreening.jsx` | Old Frontend Page | Trang Cover Test Phase 2 cũ; đã được thay thế hoàn toàn bởi quy trình hợp nhất trong `CoverTestStep.jsx`. | Không nằm trong navigation bar; 0 caller trong flow sàng lọc. | HIGH |
| 3 | `src/components/AIStatus.jsx` | Old Component | Thanh hiển thị telemetry của `StrabismusScreening.jsx` cũ. | Chỉ được import bởi `StrabismusScreening.jsx`. | HIGH |
| 4 | `src/components/Countdown.jsx` | Old Component | Component đếm ngược cũ; `CoverTestStep.jsx` đã tự render đếm ngược độc lập. | Chỉ được import bởi `StrabismusScreening.jsx`. | HIGH |
| 5 | `src/components/TestInstruction.jsx` | Old Component | Hướng dẫn text của màn hình cũ. | Chỉ được import bởi `StrabismusScreening.jsx`. | HIGH |
| 6 | `src/components/TestProgress.jsx` | Old Component | Tiến trình cũ; flow hiện tại dùng `ScreeningProgress.jsx`. | Chỉ được import bởi `StrabismusScreening.jsx`. | HIGH |
| 7 | `src/components/FixationTarget.jsx` | Duplicate Component | Điểm nhìn cố định cũ (554 B); flow hiện tại dùng `components/binocular/FixationTarget.jsx` (2.4 KB). | Chỉ được import bởi `StrabismusScreening.jsx`. | HIGH |
| 8 | `src/components/MedicalDisclaimer.jsx` | Old Component | Disclaimer cũ; màn hình kết quả hiện tại dùng banner disclaimer chuyên dụng. | Chỉ dùng ở `StrabismusScreening.jsx` và `StaticEyeTest.jsx`. | HIGH |
| 9 | `src/components/clinical/ClinicalCalibrationResult.jsx` | Old Component | Hiển thị độ lác lăng kính (Prism Diopters) cũ. | Chỉ được import bởi `StrabismusScreening.jsx`. | HIGH |
| 10 | `src/components/research/ClinicalCalibrationPanel.jsx` | Research Panel | Màn hình nghiên cứu thu thập dữ liệu PACT độc lập; không phục vụ sàng lọc. | Route phụ `/research/clinical-calibration`, độc lập hoàn toàn. | HIGH |
| 11 | `src/components/research/CalibrationAnalysis.jsx` | Research Component | Phân tích hồi quy PACT cho bảng nghiên cứu hiệu chuẩn. | Chỉ được import bởi `ClinicalCalibrationPanel.jsx`. | HIGH |
| 12 | `src/components/BrockStringTest.jsx` | Standalone Page | Trang thử nghiệm Brock String độc lập (không thuộc flow sàng lọc Cover Test). | Tab không hiển thị trong navbar chính. | HIGH |
| 13 | `src/components/audio/AudioDebugPanel.jsx` | Debug Component | Bảng debug âm thanh gắn dưới chân trang `App.jsx`. | Panel test nội bộ, không phục vụ người dùng sàng lọc. | HIGH |
| 14 | `src/components/debug/ScreeningDatasetDebugPanel.jsx` | Debug Component | Bảng debug kiểm tra dataset local gắn dưới chân trang `App.jsx`. | Panel test nội bộ, không phục vụ người dùng sàng lọc. | HIGH |
| 15 | `src/services/calibrationDatasetService.js` | Research Service | Tạo và export dataset PACT cho `ClinicalCalibrationPanel.jsx`. | Chỉ import bởi `ClinicalCalibrationPanel.jsx`. | HIGH |
| 16 | `src/services/calibrationModelService.js` | Dead Service | Service nạp model hiệu chuẩn lăng kính cũ. | 0 file import trong toàn bộ repository (Dead code). | HIGH |
| 17 | `src/services/calibrationService.js` | Old Service | Đổi chuyển vị sang lăng kính PACT. | Chỉ import bởi `StrabismusScreening.jsx`. | HIGH |
| 18 | `src/services/clinicalCalibrationGuard.js` | Old Guard | Kiểm soát an toàn chuyển đổi lăng kính PACT. | Chỉ import bởi `clinicalCalibrationService.js` và test cases. | HIGH |
| 19 | `src/services/clinicalCalibrationService.js` | Old Service | Dự đoán độ lệch lâm sàng PACT. | Chỉ import bởi `calibrationService.js`. | HIGH |
| 20 | `src/services/fusionService.js` | Old Service | Tổng hợp tín hiệu Cover Test + AI cũ của Phase 2. | Chỉ import bởi `StrabismusScreening.jsx`. | HIGH |
| 21 | `src/services/measurementResultService.js` | Old Service | Tổng hợp kết quả kiểu cũ cho BrockStringTest standalone. | Chỉ import bởi `BrockStringTest.jsx` và test cases. | HIGH |
| 22 | `src/services/cv/frontendQualityGate.js` | Unused CV Service | Cổng chất lượng phía frontend nhưng không được runtime gọi. | Chỉ import bởi `cvModulesTestCases.js`. | HIGH |
| 23 | `src/constants/clinicalCalibrationConfig.js` | Config | Cấu hình cho nghiên cứu hiệu chuẩn PACT. | Chỉ dùng bởi các module hiệu chuẩn PACT bị đề xuất xóa. | HIGH |
| 24–33 | `src/services/*TestCases.js` (10 files) | Test Files in src | 10 bộ test script đặt lẫn trong thư mục mã nguồn runtime (`src/services/`). | Chỉ được import bởi `scripts/runAllTests.js`. | HIGH |
| 34 | `scripts/runAllTests.js` | Test Runner | Script Node.js chạy 10 bộ test trong `src/services/`. | Không tham gia build hay runtime của ứng dụng. | HIGH |
| 35–42 | `training/*.py` (8 files) | Training Scripts | Các script huấn luyện mô hình ONNX cũ trong repo frontend (`train.py`, `export_onnx.py`, ...). | Không có runtime reference; chỉ dùng offline. | HIGH |
| 43–46 | `reports/*` (4 files) | Training Reports | File ảnh ma trận nhầm lẫn (`confusion_matrix.png`, `roc_curve.png`) và metric JSON cũ. | Artifact tĩnh xuất ra từ `training/`. | HIGH |
| 47–51 | `models/*` (frontend root, 5 files) | Model Duplicates | Thư mục `models/` ở root frontend (`trained_model.joblib`, `strabismus_model.onnx`, ...). | Browser runtime chỉ fetch từ `public/models/`. | HIGH |
| 52 | `public/models/clinical-calibration.metadata.json` | Public Asset | Metadata cho hiệu chuẩn PACT. | Không còn được tham chiếu. | HIGH |
| 53 | `demo.html` | Demo File | File HTML demo giao diện thử nghiệm độc lập. | Không nằm trong luồng Vite React app. | HIGH |
| 54 | `update_main.py` | Scratch Script | Script Python phụ để ở root frontend. | Không có liên kết runtime. | HIGH |
| 55 | `app/api/screening.py` | Old Backend API | Endpoint cũ `POST /api/v1/screening/analyze`. | Frontend không bao giờ gọi endpoint này. | HIGH |
| 56 | `app/services/screening.py` | Old Backend Service | Bộ điều phối phân tích sàng lọc Phase 1 cũ. | Chỉ được gọi bởi `app/api/screening.py`. | HIGH |
| 57 | `app/services/inference.py` | Old Inference Service | Nạp model cũ không tồn tại (`strabismus_model.joblib`). | Chỉ được gọi bởi `app/services/screening.py`. | HIGH |
| 58 | `app/services/preprocessing.py` | Old Backend Service | Tiền xử lý dữ liệu của pipeline cũ. | Chỉ được gọi bởi `app/services/screening.py`. | HIGH |
| 59 | `app/services/validation.py` | Old Backend Service | Validation của pipeline cũ. | Chỉ được gọi bởi `app/services/screening.py`. | HIGH |
| 60 | `app/services/feature_extraction.py` | Old Feature Service | Trích xuất đặc trưng cũ của Phase 1. | Chỉ được gọi bởi `app/services/screening.py`. | HIGH |
| 61 | `app/services/feature_contract.py` | Old Feature Contract | Hợp đồng đặc trưng cũ (30 features V1) đã bị thay thế bởi `shared_feature_contract.py`. | Chỉ được import bởi `training/shared_feature_model.py`. | HIGH |
| 62 | `app/services/korean_adapter.py` | Dataset Adapter | Parser đọc CSV dữ liệu nghiên cứu Hàn Quốc phục vụ huấn luyện offline. | Chỉ được import bởi script huấn luyện; không dùng lúc inference. | HIGH |
| 63–64 | `app/training/` (2 files) | Training inside app | Script huấn luyện mô hình đặt trong backend app (`shared_feature_model.py`). | 0 runtime import trong backend. | HIGH |
| 65–70 | `app/services/cv/*` (6 files: `canthal_normalizer.py`, `eye_roi.py`, `gaze_tracker.py`, `mediapipe_eye_extractor.py`, `one_euro_filter.py`, `__init__.py`) | Unused CV Modules | Các module Computer Vision viết bằng Python trên backend; toàn bộ CV thực tế chạy 100% trên trình duyệt (WebAssembly / MediaPipe). | Không tham gia `fps_model_service.py` hay `session_service.py`. | HIGH |
| 71–72 | `app/services/kinematics/*` (2 files: `refixation_detector.py`, `__init__.py`) | Unused Kinematics | Module phát hiện refixation saccade trên backend. | Chỉ được gọi bởi `feature_extraction.py` cũ. | HIGH |
| 73 | `app/models/shared_strabismus_model.joblib` | Old Model (407 KB) | Model Phase 4 cũ; 0 runtime reference. | Không được load bởi `fps_model_service` hay `korean_transfer`. | HIGH |
| 74 | `app/models/remicare_b1_invariant_model.joblib` | Exp Model (3.6 KB) | Model thử nghiệm B1 không dùng. | Không được load trong đường dẫn chính thức. | HIGH |
| 75 | `app/models/remicare_b2_rescaled_model.joblib` | Exp Model (3.8 KB) | Bản sao của model B2; backend dùng trực tiếp `remicare_transfer_model.joblib`. | File dư thừa. | HIGH |
| 76–80 | `models/*` (backend root, 5 files) | Model Duplicates | Thư mục `models/` ở root backend (`remicare_b1_invariant_model.joblib`, `korean_groupcv_transfer_candidate.joblib`, ...). | Backend nạp từ `app/models/`. | HIGH |
| 81 | `migrations/drop_cover_test_storage.sql` | Teardown Migration | Script xoá bảng cơ sở dữ liệu `cover_test_*`. | Không được dùng trong vận hành thông thường. | HIGH |
| 82–96 | `scripts/*` (backend, 15 files) | Offline Scripts | Các script huấn luyện candidate, audit xác suất, convert dataset (`train_10_15fps_transfer_candidate.py`, `audit_transfer_probability.py`, ...). | Toàn bộ là script offline, không liên quan runtime. | HIGH |
| 97–114 | `outputs/*` (backend, 18 files) | Training Logs & Output | Các file JSON kết quả Nested Cross-Validation, biểu đồ xác suất, báo cáo huấn luyện offline. | Artifact thử nghiệm nghiên cứu. | HIGH |
| 115–123 | `tests/*` (backend, 9 files) | Unit & Integration Tests | Bộ test pytest cho các module cũ và nghiên cứu (`test_cv_pipeline.py`, `test_korean_adapter.py`, ...). | Không phải runtime code. | HIGH |
| 124 | `data/` (2,633 files CSV/JSON) | Offline Research Data | Hơn 2,600 file CSV dữ liệu chuyển động mắt thô (`test_normal/`, `test_strabismus/`) và dataset xử lý (`processed/`). | Chỉ dùng cho script huấn luyện offline; không có API/service nào đọc khi người dùng sàng lọc. | HIGH |

---

## 2. Chi tiết từng file

Dưới đây là phân tích chi tiết các file dư thừa đại diện theo từng nhóm chức năng:

### DELETE CANDIDATE #1
**File:** `d:\AI_Check_Lac\src\pages\StaticEyeTest.jsx`  
**Loại:** Old Frontend Page  
**Lý do:** Trang đo góc mắt tĩnh thời kỳ đầu (Phase 1). Không phục vụ bài kiểm tra che mắt chuyển động (Cover Test) và không nạp mô hình 10–15 FPS.  
**References:**
- `src/App.jsx` khai báo lazy import nhưng tab `static` đã bị ẩn hoàn toàn khỏi navigation bar.
- Không có nút bấm hoặc liên kết nào trên UI dẫn tới trang này.  
**Runtime usage:** Không phát hiện lượt thực thi trong luồng người dùng.  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #2
**File:** `d:\AI_Check_Lac\src\pages\StrabismusScreening.jsx`  
**Loại:** Old Frontend Page  
**Lý do:** Màn hình Cover Test cũ của Phase 2. Toàn bộ logic sàng lọc che mắt hiện tại đã được nâng cấp và chạy trong `CoverTestStep.jsx` thuộc hợp phần `BinocularVisionScreening.jsx`.  
**References:**
- `src/App.jsx` khai báo lazy import nhưng tab `cover` đã bị ẩn khỏi thanh điều hướng chính.  
**Runtime usage:** Không tham gia vào quy trình sàng lọc hai mắt thực tế.  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #3
**File:** `d:\AI_Check_Lac\src\components\FixationTarget.jsx`  
**Loại:** Redundant / Duplicate Component  
**Lý do:** File điểm nhìn cố định phiên bản cũ (kích thước 554 bytes). Trong khi đó, luồng sàng lọc hai mắt hiện tại đang sử dụng `src/components/binocular/FixationTarget.jsx` (2,417 bytes) với tính năng khoá vị trí tâm mặt và bù trừ góc camera lật gương.  
**References:**
- Chỉ duy nhất trang cũ `StrabismusScreening.jsx` import file này.  
**Runtime usage:** Không được sử dụng trong `CoverTestStep.jsx`.  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #4
**File:** `d:\AI_Check_Lac\src\services\calibrationModelService.js`  
**Loại:** Dead Service  
**Lý do:** Mã nguồn cũ phục vụ nạp mô hình hiệu chuẩn lăng kính lâm sàng.  
**References:**
- Tìm kiếm toàn bộ mã nguồn không phát hiện bất kỳ file nào import hoặc gọi hàm từ file này.  
**Runtime usage:** Hoàn toàn không hoạt động (Dead code).  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #5
**File:** `d:\AI_Check_Lac\src\services\cv\frontendQualityGate.js`  
**Loại:** Unused CV Service  
**Lý do:** Bộ lọc chất lượng khung hình camera viết cho frontend nhưng không được gắn vào luồng xử lý video thực tế. Luồng thực tế dùng `screeningQualityGate.js` và `positionCalibrationService.js`.  
**References:**
- Chỉ duy nhất file test `cvModulesTestCases.js` import class `FrontendQualityGate`.  
**Runtime usage:** Không hoạt động lúc runtime.  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #6
**File:** `d:\AI_Check_Lac\src\services\aiBackendServiceTestCases.js` (và 9 file `*TestCases.js` khác trong `src/services/`)  
**Loại:** Test Suite đặt nhầm vị trí runtime  
**Lý do:** Các file chứa kịch bản kiểm thử tĩnh/giả lập được đặt trực tiếp bên trong thư mục sản phẩm `src/services/`.  
**References:**
- Chỉ được gọi bởi file chạy script `scripts/runAllTests.js`.  
**Runtime usage:** Không tham gia đóng gói ứng dụng web hay xử lý của người dùng.  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #7
**File:** `D:\REMICARE-STRABISMUS-AI\app\api\screening.py`  
**Loại:** Old Backend API Route  
**Lý do:** Router cung cấp endpoint cũ `POST /api/v1/screening/analyze`. Frontend hiện tại chỉ giao tiếp với backend qua endpoint lưu trữ và suy luận phiên `POST /api/v1/cover-test/sessions`.  
**References:**
- Frontend không hề gửi request đến `/api/v1/screening/analyze`.  
**Runtime usage:** Không có lưu lượng truy cập từ frontend.  
**Dependency risk:** Thấp (chỉ cần bỏ `app.include_router(screening_router)` trong `app/main.py` khi dọn dẹp).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #8
**File:** `D:\REMICARE-STRABISMUS-AI\app\services\screening.py`  
**Loại:** Old Backend Pipeline Orchestrator  
**Lý do:** Pipeline xử lý dữ liệu cũ từ Phase 1, gọi tuần tự `preprocessing.py`, `validation.py`, `feature_extraction.py`, và `inference.py`. Hiện tại toàn bộ quá trình phân tích và inference được thực hiện bởi `session_service.py` kết hợp `fps_model_service.py`.  
**References:**
- Chỉ được gọi bởi `app/api/screening.py`.  
**Runtime usage:** Không có.  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #9
**File:** `D:\REMICARE-STRABISMUS-AI\app\services\inference.py`  
**Loại:** Old Inference Abstraction  
**Lý do:** Service suy luận cũ tìm nạp file `models/strabismus_model.joblib` (file này không hề tồn tại trên hệ thống và luôn trả về INCONCLUSIVE). Mô hình sàng lọc thực sự đang chạy là `fps_model_service.py` với file `remicare_15fps_candidate.joblib`.  
**References:**
- Chỉ được gọi bởi `app/services/screening.py`.  
**Runtime usage:** Không có.  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #10
**File:** `D:\REMICARE-STRABISMUS-AI\app\services\cv\canthal_normalizer.py` (và các file `eye_roi.py`, `gaze_tracker.py`, `mediapipe_eye_extractor.py`, `one_euro_filter.py`)  
**Loại:** Unused Backend CV Implementation  
**Lý do:** Các module thị giác máy tính viết bằng Python/OpenCV trên backend. Trong kiến trúc hiện tại của RemiCare, toàn bộ quá trình xử lý camera, trích xuất điểm mốc MediaPipe Iris, lọc One-Euro và cắt vùng mắt diễn ra 100% cục bộ trên trình duyệt người dùng để bảo vệ quyền riêng tư và tối ưu độ trễ.  
**References:**
- Không được import bởi `session_service.py` hay `fps_model_service.py`.  
**Runtime usage:** Backend chỉ nhận chuỗi thời gian số toạ độ đã được trích xuất sẵn từ frontend.  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #11
**File:** `D:\REMICARE-STRABISMUS-AI\app\models\shared_strabismus_model.joblib`  
**Loại:** Old Model Artifact (407 KB)  
**Lý do:** Mô hình cũ huấn luyện từ Phase 4. Không tương thích với bộ 14 đặc trưng của pipeline 10–15 FPS hiện tại.  
**References:**
- Chỉ được nhắc đến trong script huấn luyện cũ `app/training/shared_feature_model.py`.  
**Runtime usage:** Không có service nào nạp file này khi chạy.  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

### DELETE CANDIDATE #12
**File:** `D:\REMICARE-STRABISMUS-AI\data\test_normal\` và `data\test_strabismus\` (Hơn 2,600 files CSV)  
**Loại:** Offline Raw Research Datasets  
**Lý do:** Dữ liệu theo dõi mắt 60 Hz của nghiên cứu Hàn Quốc được dùng để huấn luyện và chuyển giao mô hình offline. Không phục vụ vận hành thời gian thực.  
**References:**
- Không có endpoint hay service runtime nào đọc các thư mục này.  
**Runtime usage:** 0 MB đọc khi người dùng thực hiện sàng lọc.  
**Dependency risk:** Không có (Low).  
**Confidence:** HIGH  

---

## 3. Model có thể xóa

| Model | Path | Used by | Purpose | Status |
|---|---|---|---|---|
| `shared_strabismus_model.joblib` (407 KB) | `D:\REMICARE-STRABISMUS-AI\app\models\` | `app/training/shared_feature_model.py` (chỉ khi train) | Thử nghiệm Phase 4 cũ | **DELETE CANDIDATE** |
| `remicare_b1_invariant_model.joblib` (3.6 KB) | `D:\REMICARE-STRABISMUS-AI\app\models\` | Không có runtime reference | Mô hình thử nghiệm bất biến B1 | **DELETE CANDIDATE** |
| `remicare_b2_rescaled_model.joblib` (3.8 KB) | `D:\REMICARE-STRABISMUS-AI\app\models\` | Không có runtime reference | Bản sao trùng với `remicare_transfer_model.joblib` | **DELETE CANDIDATE** |
| `korean_groupcv_transfer_candidate.joblib` (4.4 KB) | `D:\REMICARE-STRABISMUS-AI\models\candidates\` | Không có runtime reference | Ứng viên thử nghiệm GroupCV | **DELETE CANDIDATE** |
| `korean_shared_model.joblib` (414 KB) | Root `models/` (ngoài `app/`) | Không có runtime reference | Bản sao trùng lặp ở ngoài thư mục app | **DELETE CANDIDATE** |
| `remicare_transfer_model.joblib` (3.8 KB) | Root `models/` (ngoài `app/`) | Không có runtime reference | Bản sao trùng lặp ở ngoài thư mục app | **DELETE CANDIDATE** |
| `trained_model.joblib` (30 KB) | `d:\AI_Check_Lac\models\` | Không có | Model frontend cũ trong thư mục root | **DELETE CANDIDATE** |
| `scaler.joblib` (823 B) | `d:\AI_Check_Lac\models\` | Không có | Bộ scale frontend cũ trong thư mục root | **DELETE CANDIDATE** |

> [!CAUTION]
> **TUYỆT ĐỐI KHÔNG ĐƯỢC XÓA MODEL SAU:**
> - `app/models/remicare_15fps_candidate.joblib` (Korean 10–15 FPS robust transfer candidate v1.1.0, 14 features). Đây là mô hình lõi đang phục vụ trực tiếp cho chức năng sàng lọc hai mắt hiện tại.

---

## 4. Frontend files có thể xóa

| File | Reason | References | Confidence |
|---|---|---|---|
| `src/pages/StaticEyeTest.jsx` | Trang đo tĩnh Phase 1 không còn sử dụng | Khai báo lazy trong `App.jsx`, không có nút gọi | HIGH |
| `src/pages/StrabismusScreening.jsx` | Màn hình Cover Test cũ được thay bằng `CoverTestStep.jsx` | Khai báo lazy trong `App.jsx`, không có nút gọi | HIGH |
| `src/components/AIStatus.jsx` | Thanh trạng thái chỉ dùng cho màn hình cũ | `StrabismusScreening.jsx` | HIGH |
| `src/components/Countdown.jsx` | Bộ đếm thời gian chỉ dùng cho màn hình cũ | `StrabismusScreening.jsx` | HIGH |
| `src/components/TestInstruction.jsx` | Hướng dẫn text chỉ dùng cho màn hình cũ | `StrabismusScreening.jsx` | HIGH |
| `src/components/TestProgress.jsx` | Thanh tiến trình chỉ dùng cho màn hình cũ | `StrabismusScreening.jsx` | HIGH |
| `src/components/FixationTarget.jsx` | Component điểm nhìn cũ 554 B; bản mới ở `binocular/` | `StrabismusScreening.jsx` | HIGH |
| `src/components/MedicalDisclaimer.jsx` | Disclaimer chung cũ không dùng cho flow mới | `StrabismusScreening.jsx`, `StaticEyeTest.jsx` | HIGH |
| `src/components/clinical/ClinicalCalibrationResult.jsx` | Hiển thị kết quả lăng kính cũ | `StrabismusScreening.jsx` | HIGH |
| `src/components/research/ClinicalCalibrationPanel.jsx` | Bảng nghiên cứu PACT độc lập | Route `/research/clinical-calibration` | HIGH |
| `src/components/research/CalibrationAnalysis.jsx` | Hợp phần con của bảng nghiên cứu PACT | `ClinicalCalibrationPanel.jsx` | HIGH |
| `src/components/BrockStringTest.jsx` | Trang Brock String độc lập (ngoài luồng sàng lọc Cover Test) | Khai báo lazy trong `App.jsx` | HIGH |
| `src/components/audio/AudioDebugPanel.jsx` | Bảng điều khiển debug âm thanh nội bộ | `App.jsx` | HIGH |
| `src/components/debug/ScreeningDatasetDebugPanel.jsx` | Bảng điều khiển debug dataset nội bộ | `App.jsx` | HIGH |
| `src/services/calibrationDatasetService.js` | Service sinh dataset cho nghiên cứu PACT | `ClinicalCalibrationPanel.jsx` | HIGH |
| `src/services/calibrationModelService.js` | Service chết không có file nào gọi | 0 references | HIGH |
| `src/services/calibrationService.js` | Chuyển đổi lăng kính PACT cũ | `StrabismusScreening.jsx` | HIGH |
| `src/services/clinicalCalibrationGuard.js` | Guard kiểm tra lăng kính cũ | `clinicalCalibrationService.js` | HIGH |
| `src/services/clinicalCalibrationService.js` | Dự đoán lăng kính cũ | `calibrationService.js` | HIGH |
| `src/services/fusionService.js` | Thuật toán hợp nhất tín hiệu cũ | `StrabismusScreening.jsx` | HIGH |
| `src/services/measurementResultService.js` | Xử lý kết quả kiểu cũ | `BrockStringTest.jsx` | HIGH |
| `src/services/cv/frontendQualityGate.js` | Gate kiểm tra phía client không dùng runtime | `cvModulesTestCases.js` | HIGH |
| `src/constants/clinicalCalibrationConfig.js` | Hằng số cấu hình nghiên cứu PACT | Các file hiệu chuẩn PACT | HIGH |
| `src/services/aiBackendServiceTestCases.js` | File test tĩnh đặt nhầm trong `src/` | `scripts/runAllTests.js` | HIGH |
| `src/services/audioServiceTestCases.js` | File test tĩnh đặt nhầm trong `src/` | `scripts/runAllTests.js` | HIGH |
| `src/services/clinicalCalibrationTestCases.js` | File test tĩnh đặt nhầm trong `src/` | `scripts/runAllTests.js` | HIGH |
| `src/services/coverTestRepairTestCases.js` | File test tĩnh đặt nhầm trong `src/` | `scripts/runAllTests.js` | HIGH |
| `src/services/coverTestSessionIdTestCases.js` | File test tĩnh đặt nhầm trong `src/` | `scripts/runAllTests.js` | HIGH |
| `src/services/coverTestTimeSeriesTestCases.js` | File test tĩnh đặt nhầm trong `src/` | `scripts/runAllTests.js` | HIGH |
| `src/services/cv/cvModulesTestCases.js` | File test tĩnh đặt nhầm trong `src/` | `scripts/runAllTests.js` | HIGH |
| `src/services/finalScreeningTestCases.js` | File test tĩnh đặt nhầm trong `src/` | `scripts/runAllTests.js` | HIGH |
| `src/services/screeningDatasetTestCases.js` | File test tĩnh đặt nhầm trong `src/` | `scripts/runAllTests.js` | HIGH |
| `src/services/screeningTestCases.js` | File test tĩnh đặt nhầm trong `src/` | `scripts/runAllTests.js` | HIGH |
| `scripts/runAllTests.js` | Runner chạy 10 file test trong `src/` | Không tham gia runtime | HIGH |
| `public/models/clinical-calibration.metadata.json` | Metadata hiệu chuẩn lăng kính | Không tham chiếu | HIGH |
| `demo.html` | Bản demo HTML đơn lẻ | Không nằm trong app React | HIGH |
| `update_main.py` | Script scratch sửa code cũ | Không dùng | HIGH |

---

## 5. Backend files có thể xóa

| File | Reason | References | Confidence |
|---|---|---|---|
| `app/api/screening.py` | Router `POST /api/v1/screening/analyze` cũ, frontend không gọi | `app/main.py` | HIGH |
| `app/services/screening.py` | Pipeline phân tích cũ của Phase 1 | `app/api/screening.py` | HIGH |
| `app/services/inference.py` | Nạp model cũ không tồn tại `strabismus_model.joblib` | `app/services/screening.py` | HIGH |
| `app/services/preprocessing.py` | Tiền xử lý dữ liệu cũ Phase 1 | `app/services/screening.py` | HIGH |
| `app/services/validation.py` | Kiểm tra dữ liệu cũ Phase 1 | `app/services/screening.py` | HIGH |
| `app/services/feature_extraction.py` | Trích xuất đặc trưng cũ Phase 1 | `app/services/screening.py` | HIGH |
| `app/services/feature_contract.py` | Hợp đồng 30 đặc trưng V1 cũ, đã thay bằng `shared_feature_contract.py` | `app/training/shared_feature_model.py` | HIGH |
| `app/services/korean_adapter.py` | Bộ chuyển đổi dữ liệu CSV Hàn Quốc offline | `app/training/shared_feature_model.py` | HIGH |
| `app/training/shared_feature_model.py` | Script huấn luyện đặt lẫn trong thư mục app | 0 runtime callers | HIGH |
| `app/training/__init__.py` | Init gói training | Không có | HIGH |
| `app/services/cv/canthal_normalizer.py` | Module CV trên backend; CV thực tế chạy ở client | `app/services/cv/__init__.py` | HIGH |
| `app/services/cv/eye_roi.py` | Module CV trên backend; CV thực tế chạy ở client | `app/services/cv/__init__.py` | HIGH |
| `app/services/cv/gaze_tracker.py` | Module CV trên backend; CV thực tế chạy ở client | `app/services/cv/__init__.py` | HIGH |
| `app/services/cv/mediapipe_eye_extractor.py` | Module CV trên backend; CV thực tế chạy ở client | `app/services/cv/__init__.py` | HIGH |
| `app/services/cv/one_euro_filter.py` | Module CV trên backend; CV thực tế chạy ở client | `app/services/cv/__init__.py` | HIGH |
| `app/services/cv/__init__.py` | Barrel export cho các module CV không dùng | Không tham chiếu | HIGH |
| `app/services/kinematics/refixation_detector.py` | Module phát hiện refixation saccade cũ | `app/services/kinematics/__init__.py` | HIGH |
| `app/services/kinematics/__init__.py` | Barrel export cho module kinematics cũ | Không tham chiếu | HIGH |
| `app/models/shared_strabismus_model.joblib` | Model 407 KB Phase 4 cũ, 0 runtime reference | `training/shared_feature_model.py` | HIGH |
| `app/models/remicare_b1_invariant_model.joblib` | Model thử nghiệm B1 không dùng | Không có | HIGH |
| `app/models/remicare_b2_rescaled_model.joblib` | Model thử nghiệm B2 trùng lặp | Không có | HIGH |
| `models/` (root ngoài app, 5 files) | Toàn bộ bản sao model trùng lặp ngoài thư mục `app/models/` | Không nạp lúc runtime | HIGH |
| `migrations/drop_cover_test_storage.sql` | Script xoá bảng DB | Không dùng trong vận hành | HIGH |

---

## 6. Training / Experiment files có thể xóa

| File | Reason | Runtime required? | Confidence |
|---|---|---|---|
| `training/evaluate.py` (frontend) | Đánh giá mô hình ONNX offline | NO | HIGH |
| `training/export_onnx.py` (frontend) | Xuất file `.onnx` offline | NO | HIGH |
| `training/extract_features.py` (frontend) | Trích xuất feature từ video tập huấn | NO | HIGH |
| `training/prepare_dataset.py` (frontend) | Chuẩn bị dataset offline | NO | HIGH |
| `training/run_pipeline.py` (frontend) | Điều phối pipeline huấn luyện client-side | NO | HIGH |
| `training/split_dataset.py` (frontend) | Chia tập train/val offline | NO | HIGH |
| `training/train.py` (frontend) | Huấn luyện mô hình ONNX offline | NO | HIGH |
| `training/validate_dataset.py` (frontend) | Kiểm tra tính toàn vẹn dataset offline | NO | HIGH |
| `reports/confusion_matrix.png` | Hình ảnh báo cáo huấn luyện | NO | HIGH |
| `reports/dataset_validation.json` | Báo cáo kiểm tra dataset | NO | HIGH |
| `reports/metrics.json` | Thống kê độ chính xác mô hình cũ | NO | HIGH |
| `reports/roc_curve.png` | Biểu đồ ROC curve mô hình cũ | NO | HIGH |
| `scripts/audit_transfer_probability.py` (backend) | Script audit phân bố xác suất transfer | NO | HIGH |
| `scripts/build_korean_shared_datasets.py` (backend) | Tạo tập dữ liệu liên kết Hàn Quốc | NO | HIGH |
| `scripts/convert_korean_dataset.py` (backend) | Chuyển đổi định dạng CSV dữ liệu | NO | HIGH |
| `scripts/create_ai_service.py` (backend) | Script tạo template service | NO | HIGH |
| `scripts/create_frontend_tests.py` (backend) | Script tạo test case frontend | NO | HIGH |
| `scripts/extract_remicare_features.py` (backend) | Trích xuất feature offline từ mẫu RemiCare | NO | HIGH |
| `scripts/run_frontend_tests.mjs` (backend) | Script chạy test integration frontend | NO | HIGH |
| `scripts/run_remicare_transfer.py` (backend) | Chạy thử nghiệm transfer offline | NO | HIGH |
| `scripts/test_live_frontend_integration.mjs` (backend) | Test kết nối live | NO | HIGH |
| `scripts/test_real_sample.py` (backend) | Kiểm thử mẫu đơn lẻ | NO | HIGH |
| `scripts/train_10_15fps_transfer_candidate.py` (backend) | Huấn luyện candidate 10–15 FPS (đã ra artifact) | NO | HIGH |
| `scripts/train_15fps_transfer_candidate.py` (backend) | Huấn luyện candidate 15 FPS (đã ra artifact) | NO | HIGH |
| `scripts/train_domain_adapted_models.py` (backend) | Huấn luyện mô hình thích ứng miền | NO | HIGH |
| `scripts/train_groupcv_transfer_candidate.py` (backend) | Huấn luyện ứng viên GroupCV | NO | HIGH |
| `scripts/update_cover_test_step.py` (backend) | Script hỗ trợ chỉnh sửa code cũ | NO | HIGH |
| `outputs/debug/transfer_probability/*` (9 files) | Log và ma trận phân tích xác suất | NO | HIGH |
| `outputs/retraining/10_15fps_candidate/*` (2 files) | Báo cáo huấn luyện candidate 10–15 FPS | NO | HIGH |
| `outputs/retraining/15fps_candidate/*` (3 files) | Báo cáo tính ổn định pha chuyển động | NO | HIGH |
| `outputs/retraining/groupcv_candidate/*` (4 files) | Báo cáo kiểm định chéo Nested OOF | NO | HIGH |
| `data/test_normal/*` (20 files CSV) | Dữ liệu chuyển động mắt thô người bình thường | NO | HIGH |
| `data/test_strabismus/*` (21 files CSV) | Dữ liệu chuyển động mắt thô người lác | NO | HIGH |
| `data/processed/*` (6 files CSV/JSON) | Tập đặc trưng đã xử lý phục vụ huấn luyện | NO | HIGH |
| `tests/*` (backend, 9 files) | Toàn bộ bộ test pytest backend | NO | HIGH |

---

## 7. Files KHÔNG ĐƯỢC XÓA

Các file cốt lõi đảm bảo quy trình **Binocular Screening (Sàng lọc hai mắt)** hoạt động bình thường từ camera đến hiển thị kết quả:

### Frontend (`d:\AI_Check_Lac`)
1. `src/main.jsx` — Điểm khởi động ứng dụng React
2. `src/App.jsx` — Cấu trúc trang chính, nạp `BinocularVisionScreening`
3. `src/App.css` — Định dạng layout ứng dụng
4. `src/index.css` — Toàn bộ thiết kế hệ thống giao diện, card hiển thị kết quả AI
5. `src/components/ErrorBoundary.jsx` — Bắt lỗi giao diện đảm bảo camera không crash trắng màn hình
6. `src/components/CameraView.jsx` — Khung phát video camera chống lật gương và đồng bộ layer
7. `src/components/EyeOverlay.jsx` — Lớp phủ trực quan hoá mắt và mống mắt thời gian thực
8. `src/components/binocular/BinocularVisionScreening.jsx` — Hợp phần điều phối quy trình sàng lọc hai mắt tích hợp
9. `src/components/binocular/CoverTestStep.jsx` — **Hợp phần kiểm tra Cover/Uncover test, ghi nhận 15 FPS và hiển thị card kết quả AI**
10. `src/components/binocular/PositionCheck.jsx` — Kiểm tra khoảng cách đầu và độ ổn định trước khi test
11. `src/components/binocular/FixationTarget.jsx` — Điểm nhìn trung tâm chuẩn hoá toạ độ
12. `src/components/binocular/ScreeningProgress.jsx` — Thanh hiển thị các bước của quy trình
13. `src/components/binocular/FinalScreeningResult.jsx` — Màn hình tổng hợp kết quả sàng lọc
14. `src/components/binocular/ScreeningSummary.jsx` — Wrapper hiển thị kết quả
15. `src/components/audio/AudioButton.jsx` — Nút bật/tắt hướng dẫn giọng nói
16. `src/hooks/useCamera.js` — Quản lý stream webcam liên tục
17. `src/hooks/useFaceMesh.js` — Vòng lặp nhận diện khuôn mặt MediaPipe WebAssembly
18. `src/hooks/useEyeTracking.js` — Tính toán toạ độ mống mắt và chất lượng theo dõi
19. `src/hooks/useSpeech.js` — Trợ lý giọng nói hướng dẫn từng pha che/mở mắt
20. `src/services/cameraService.js` — Tiện ích khởi tạo camera
21. `src/services/audioService.js` — Dịch vụ phát âm thanh và giọng nói
22. `src/services/binocularScreeningService.js` — Quản lý session và trạng thái sàng lọc
23. `src/services/coverTestMeasurementService.js` — Phân tích quỹ đạo mở mắt và tính baseline vững
24. `src/services/coverTestProtocolService.js` — Định dạng chu kỳ test, UUID v4, kiểm tra tính hợp lệ
25. `src/services/coverTestTimeSeriesService.js` — Bộ ghi chuỗi thời gian xấp xỉ 15 Hz không gây rò rỉ bộ nhớ
26. `src/services/eyeFeatureService.js` — Trích xuất góc canthal và toạ độ mống mắt
27. `src/services/faceMeshService.js` — Chỉ mục điểm mốc giải phẫu mắt MediaPipe
28. `src/services/positionCalibrationService.js` — Ước tính khoảng cách thực qua khoảng cách hai mống mắt
29. `src/services/screeningDatasetService.js` — Đóng gói dữ liệu phiên kiểm tra
30. `src/services/screeningImageCaptureService.js` — Cắt ảnh vùng mắt phục vụ lưu trữ
31. `src/services/screeningQualityGate.js` — Cổng kiểm soát chất lượng dữ liệu sàng lọc
32. `src/services/coverTest/coverTestPersistenceService.js` — Gửi dữ liệu phiên lên Cloud API
33. `src/services/cv/eyeRoiService.js` — Căn chỉnh và cắt ROI mắt xoay theo góc canthal
34. `src/services/cv/gazeTracker.js` — Chiếu vector cố định điểm nhìn
35. `src/services/cv/oneEuroFilter.js` — Bộ lọc One-Euro khử rung mống mắt
36. `src/utils/eyeCoordinateMapping.js` — Ánh xạ toạ độ mắt lật gương và câu hướng dẫn
37. `src/constants/audioConfig.js` — Cấu hình âm thanh
38. `src/constants/binocularScreeningConfig.js` — Cấu hình ngưỡng sàng lọc hai mắt
39. `src/constants/screeningConfig.js` — Cấu hình tần số lấy mẫu (15 Hz) và thời gian các pha
40. `src/api/client.js` — HTTP Client giao tiếp backend
41. `src/api/coverTestApi.js` — API call lưu trữ và suy luận `POST /api/v1/cover-test/sessions`
42. `src/api/healthApi.js` — API kiểm tra trạng thái backend
43. `src/api/index.js` — Barrel export API
44. `package.json`, `vite.config.js`, `index.html`, `.env.local` — Cấu hình hệ thống và môi trường

### Backend (`D:\REMICARE-STRABISMUS-AI`)
45. `app/main.py` — Khởi động FastAPI server, cấu hình CORS, lifespan preloading
46. `app/config.py` — Cấu hình ứng dụng và biến môi trường
47. `app/__init__.py` — Package init
48. `app/api/cover_test.py` — **Router chính `POST /api/v1/cover-test/sessions` nhận request từ frontend**
49. `app/api/__init__.py` — API package init
50. `app/db/database.py` — Quản lý kết nối PostgreSQL / Supabase connection pool
51. `app/db/__init__.py` — DB package init
52. `app/db/models/cover_test_session.py` — Model bảng `cover_test_sessions`
53. `app/db/models/cover_test_cycle.py` — Model bảng `cover_test_cycles`
54. `app/db/models/cover_test_result.py` — Model bảng `cover_test_results`
55. `app/db/models/__init__.py` — DB models package init
56. `app/db/repositories/cover_test_repository.py` — Thực hiện lưu dữ liệu phiên và kết quả AI vào PostgreSQL
57. `app/db/repositories/__init__.py` — Repositories package init
58. `app/schemas/cover_test.py` — Schema dữ liệu phiên kiểm tra che mắt
59. `app/schemas/screening.py` — Schema dữ liệu suy luận và request
60. `app/schemas/__init__.py` — Schemas package init
61. `app/services/fps_model_service.py` — **DỊCH VỤ SUY LUẬN LÕI: Nạp `remicare_15fps_candidate.joblib`, chạy 14 features và tổng hợp đa khung hình**
62. `app/services/shared_feature_contract.py` — **HỢP ĐỒNG ĐẶC TRƯNG CHÍNH: `extract_shared_features_vector()`, chuẩn hoá 14 đặc trưng cho model 10–15 FPS**
63. `app/services/cover_test/session_service.py` — **DỊCH VỤ ĐIỀU PHỐI LÕI: Xác thực dữ liệu 3 chu kỳ, đẩy Supabase Storage, gọi `fps_model_service` và lưu DB**
64. `app/services/cover_test/storage_service.py` — Đẩy dữ liệu quỹ đạo JSON lên Supabase Storage
65. `app/services/cover_test/__init__.py` — Storage services package init
66. `app/services/cv/temporal_aggregator.py` — Tổng hợp dự đoán đa khung hình qua cửa sổ thời gian
67. `app/services/cv/quality_gate.py` — Định nghĩa cấu trúc `QualityGateResult` dùng trong temporal consensus
68. `app/services/keep_alive.py` — Worker tự động ping giữ ấm server Render và connection pool DB
69. `app/services/__init__.py` — Services package init
70. `app/models/remicare_15fps_candidate.joblib` — **MÔ HÌNH LÕI DUY NHẤT ĐƯỢC CHỨC NĂNG SÀNG LỌC SỬ DỤNG**
71. `migrations/create_cover_test_storage.sql` — Định nghĩa cấu trúc các bảng lưu trữ trong PostgreSQL
72. `migrations/update_privacy_and_fps_fields.sql` — Bổ sung cột lưu trữ kết quả model 10–15 FPS
73. `requirements.txt`, `.env`, `.env.example`, `.gitignore` — Phụ thuộc và biến môi trường

---

## 8. UNCERTAIN

Các file cần kiểm tra thêm hoặc cần bước refactor trước khi có thể dọn dẹp an toàn:

| File | Why uncertain | What needs checking |
|---|---|---|
| `app/services/korean_transfer.py` | Hiện tại `session_service.py` và `app/main.py` vẫn import file này để khởi tạo lifespan và bổ sung danh sách `comparisonModels` (mặc dù frontend chỉ hiển thị thẻ kết quả của model 10–15 FPS). Nếu xoá ngay lúc này mà không refactor code backend, backend sẽ crash lúc khởi động. | Cần một bước refactor nhỏ trên backend để `session_service.py` và `main.py` lấy kết quả trực tiếp từ `fps_model_service.py` độc lập, sau đó mới xoá file này. |
| `app/models/korean_shared_model.joblib` | Là file model fallback của `korean_transfer.py`. Nếu xoá file này khi `korean_transfer.py` chưa được refactor, `korean_transfer.py` sẽ báo `FileNotFoundError`. | Giữ lại song hành cùng `korean_transfer.py` cho tới khi hoàn tất refactor tách rời so sánh model. |
| `app/models/remicare_transfer_model.joblib` | Model B2 được `korean_transfer.py` nạp làm model so sánh ban đầu. | Tương tự như trên, giữ lại song hành cho tới khi refactor backend. |
| `models/candidates/remicare_15fps_candidate.joblib` | Là đường dẫn fallback thứ 2 được `fps_model_service.py` kiểm tra nếu không tìm thấy ở `app/models/`. | Có thể xoá an toàn khi đảm bảo file chính `app/models/remicare_15fps_candidate.joblib` luôn tồn tại. |
| `app/api/transfer.py` | Endpoint `POST /api/v1/transfer/strabismus`. Được frontend gọi dự phòng trong trường hợp lưu cloud session bị lỗi. | Kiểm tra xem có muốn giữ endpoint dự phòng độc lập này không, hay chỉ duy trì duy nhất endpoint lưu trữ phiên `/api/v1/cover-test/sessions`. |
| `app/api/cover_test_session.py` | Endpoint legacy lưu trữ file trên ổ cứng máy chủ `/api/cover-test/sessions`. | Xác nhận không còn client bên ngoài nào gọi endpoint cũ này trước khi gỡ bỏ router. |
| `src/components/binocular/BrockStringStep.jsx` | Hiện đang được gọi ở Step 4 của `BinocularVisionScreening.jsx`. Mục tiêu yêu cầu chỉ giữ Cover Test và AI inference. Nếu xoá file này ngay, `BinocularVisionScreening.jsx` sẽ bị lỗi import và vỡ luồng 5 bước hiện tại. | Cần quyết định: Có rút gọn `BinocularVisionScreening.jsx` thành quy trình 3 bước (Vị trí -> Cover Test -> Kết quả) hay không. Nếu có, sau khi sửa code mới xoá file này. |
| `src/components/binocular/BrockStringVisual.jsx` | Visual canvas cho BrockStringStep. | Phụ thuộc vào quyết định đối với Brock String ở trên. |
| `src/services/brockStringMeasurementService.js` | Đo lường độ tụ/phân kỳ cho Brock String. | Phụ thuộc vào quyết định đối với Brock String ở trên. |
| `src/hooks/useStrabismusAI.js` | Hook chạy model ONNX trên trình duyệt. Hiện tại `BinocularVisionScreening.jsx` vẫn lấy tín hiệu phụ `smoothedPrediction` để ghi vào bản tóm tắt phiên. | Cần xác nhận: Có muốn loại bỏ hoàn toàn tín hiệu ONNX phụ trợ này để chỉ dùng 100% kết quả từ backend model 10–15 FPS hay không. |
| `src/services/aiInferenceService.js` | Service nạp `onnxruntime-web` cho `useStrabismusAI.js`. | Phụ thuộc vào quyết định với `useStrabismusAI.js`. |
| `public/models/strabismus_model.onnx` | Model ONNX 10 features chạy trong trình duyệt. | Phụ thuộc vào quyết định với `useStrabismusAI.js`. |
| `public/models/model_metadata.json` | Metadata cho model ONNX client. | Phụ thuộc vào quyết định với `useStrabismusAI.js`. |
| `src/services/aiBackendService.js` | Hàm `analyzeCoverTest` gọi API transfer dự phòng khi lưu trữ cloud session thất bại. | Giữ lại làm cơ chế fallback trừ khi quyết định loại bỏ hoàn toàn fallback API. |
| `src/api/transferApi.js` | File gọi HTTP cho API transfer dự phòng. | Đi kèm với `aiBackendService.js`. |
| `src/components/debug/CoverTestDebugPanel.jsx` | Panel hiển thị telemetry chi tiết trong `CoverTestStep.jsx`. | Có thể giữ lại phục vụ đội ngũ kỹ thuật theo dõi hoặc gỡ bỏ nếu muốn giao diện hoàn toàn tinh gọn. |

---

## 9. Summary

- **Total files scanned:** 2,886 files (bao gồm 253 file mã nguồn/cấu hình/mô hình và 2,633 file dữ liệu huấn luyện thô)
- **DELETE CANDIDATE:** 2,767 files (bao gồm 134 file mã nguồn, component, page, script, test, mô hình dư thừa + 2,633 file CSV/JSON dữ liệu nghiên cứu thô)
- **KEEP:** 76 files (43 file Frontend + 33 file Backend đảm bảo hoàn chỉnh luồng chạy thực tế)
- **UNCERTAIN:** 16 files (10 file Frontend + 6 file Backend đang liên kết phụ trợ, cần xác nhận thiết kế hoặc refactor trước khi xoá)
- **RESEARCH / OPTIONAL:** 27 files (23 file tài liệu đặc tả markdown + 4 file tài liệu tổng quan)
