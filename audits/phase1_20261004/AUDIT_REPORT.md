# PHASE 1: Audit Hirschberg hiện tại

Ngày kiểm tra: 2026-10-04. Workspace: `D:/REMICARE-STRABISMUS-AI`.
Commit đã đọc: `dcb87264e21f0932a9af4c7e18fd47a33ae1cbfd`.

**Giới hạn sử dụng: công cụ sàng lọc hỗ trợ nghiên cứu, không phải chẩn đoán y khoa và không thay thế khám chuyên khoa.** Chưa có bằng chứng xác nhận hiệu quả lâm sàng trên bệnh nhân mới.

Audit chỉ đọc code/dữ liệu/model hiện có. Chỉ tạo thư mục mới `audits/phase1_20261004/`; không train, sửa backend, đổi model, commit, push, deploy hay tải ảnh. Không đọc nội dung `.env`. Đã kiểm kê file toàn repo, loại môi trường ảo, Git và cache; đọc sâu các pipeline Hirschberg liên quan. Không có tuyên bố đã đánh giá từng ảnh bằng bác sĩ hoặc kiểm tra toàn bộ Cover Test.

## 1. Kết luận chính và mức nghiêm trọng

| Hạng | Mức độ | Phát hiện | Bằng chứng | Hệ quả |
|---|---|---|---|---|
| 1 | Nghiêm trọng | Mục tiêu 4 lớp chưa đạt: runtime v0.6 v2 chỉ có eso/exo/normal; không có ảnh thật pseudo trong hai tập hiện có | `evidence.json: image_summary, manifest`; `scripts/train_hirschberg_v06_ensemble_v2.py:54` | Không thể chứng minh hoặc trả đúng lớp pseudo bằng classifier hiện tại |
| 2 | Nghiêm trọng | Balanced accuracy 94,61% của v0.6 v2 tính trên tập chứa ảnh train, đồng thời loại ảnh UNCERTAIN | `scripts/train_hirschberg_v06_ensemble_v2.py:478, 498, 506`; `reports/hirschberg_candidate_v0.6_ensemble_v2_eval.json` | Không được hiểu là độ chính xác trên bệnh nhân mới |
| 3 | Nghiêm trọng | v0.2 đạt 100% trên bộ đặc trưng tổng hợp, không phải 450 ảnh bệnh nhân | `scripts/train_hirschberg_research_model.py:336, 473, 729`; `reports/hirschberg_candidate_v0.2_eval.json` | Con số 100% không chứng minh năng lực sàng lọc ảnh thật |
| 4 | Cao | Detector hiện tại sai điểm flash đáng kể: P90 37,22 px trên annotation hiện có | `evidence.json: current_eye_crop_detector_comparison` | Đặc trưng hình học có thể sai trước khi đến classifier |
| 5 | Cao | Thiếu danh tính nhóm bệnh nhân thật; chia v0.6 theo ảnh và có cặp gần giống qua train/val | `scripts/train_hirschberg_v06_ensemble_v2.py:188`; `evidence.json: duplicate_screen` | Nguy cơ leakage; chưa chứng minh đánh giá độc lập |
| 6 | Cao | Nhãn, nguồn, license và đồng thuận chưa có hồ sơ xác minh đầy đủ | Manifest Pedseye: `clinician_confirmed=false` 67/67; `scripts/hirschberg_folder_manifest.py:94` | Chưa đủ cơ sở coi nhãn thư mục là chuẩn lâm sàng hoặc dữ liệu có quyền sử dụng |
| 7 | Cao | Dependencies thiếu và EfficientNet có thể tự chuyển sang weights ngẫu nhiên | `requirements.txt`; `app/services/hirschberg_ai_service.py:60`; `scripts/train_hirschberg_v06_ensemble_v2.py:129` | Runtime có thể dùng nhánh khác model đã đánh giá; tái lập không bảo đảm |
| 8 | Trung bình | Dữ liệu ít, crop/độ phân giải khác miền ảnh production; tuning 125 cấu hình trên 27 ảnh | `evidence.json: image_summary`; `scripts/train_hirschberg_v06_ensemble_v2.py:317`; `reports/...v2_eval.json:tuning` | Dễ chọn cấu hình hợp tập nhỏ, khó biết hiệu quả thực |

Liên kết kiểm chứng: [evidence.json](D:/REMICARE-STRABISMUS-AI/audits/phase1_20261004/evidence.json), [script v0.6 v2](D:/REMICARE-STRABISMUS-AI/scripts/train_hirschberg_v06_ensemble_v2.py:478), [script v0.2](D:/REMICARE-STRABISMUS-AI/scripts/train_hirschberg_research_model.py:729), [detector](D:/REMICARE-STRABISMUS-AI/app/services/eye_crop_geometry_service.py:244).

## 2. Cấu trúc và công nghệ

| Khu vực | Nội dung đã kiểm kê |
|---|---|
| `app/` | 37 file Python; FastAPI/Pydantic, SQLAlchemy, API ảnh/measurement/Cover Test, services CV và inference |
| `scripts/` | 16 script Python; train v0.2/v0.3/v0.4/v0.5/v0.6, chuẩn bị manifest, đánh giá detector/oracle, governance |
| `cv/` | 2 file Python; quality gate |
| `tests/` | 13 file Python, 1 JS module, 1 fixture JSON; kiểm tra API/contract/model/research |
| `docs/` | 37 tài liệu Markdown; một số nội dung cũ mâu thuẫn với code/report hiện tại |
| `data/` | 42 JSON, 6 MP4, 1 CSV; bao gồm dữ liệu Cover Test/Korean, không tự coi là ảnh Hirschberg |
| `processed/` | 3 CSV: annotation thủ công, đặc trưng detector, đặc trưng oracle |
| `datasets/` | 1 manifest JSONL Pedseye |
| `reports/` | 10 JSON, 3 Markdown, 82 JPG, 6 PNG, 1 CSV; hình overlay/debug không tính là mẫu độc lập |
| `migrations/` | 3 SQL |
| Notebook/config | Không tìm thấy `.ipynb` hay YAML/YML trong phần repo được kiểm kê; có `app/config.py`, requirements, runtime, pytest, env example |

Runtime lựa chọn Hirschberg v0.6 v2 ở [hirschberg_ai_service.py](D:/REMICARE-STRABISMUS-AI/app/services/hirschberg_ai_service.py:24). Endpoint measurement có thể trả `INCONCLUSIVE/MEASUREMENT_ONLY` đồng thời kèm `aiPrediction`, vì trạng thái đo và trạng thái classifier là hai phần riêng ([research_measurement_service.py](D:/REMICARE-STRABISMUS-AI/app/services/research_measurement_service.py:684)).

README còn mô tả Korean transfer/đường dẫn cũ trong khi `app/main.py` đăng ký API hiện tại khác. Registry quảng bá 94,61% mà không nêu rõ đánh giá trên tập chứa ảnh train. Bằng chứng: [README](D:/REMICARE-STRABISMUS-AI/README.md), [main.py](D:/REMICARE-STRABISMUS-AI/app/main.py), [registry](D:/REMICARE-STRABISMUS-AI/docs/backend/MODEL_REGISTRY.md).

### Artifact hiện có

Đơn vị MiB = bytes / 1.048.576, tương ứng cách tính MB đã dùng trước đây.

| Artifact trong `app/models/` | MiB | Vai trò theo code/tài liệu hiện có |
|---|---:|---|
| `best_model.onnx` | 42,88 | Runtime ảnh nhị phân; cũng tham gia trích đặc trưng Hirschberg |
| `best_model.pt` | 42,97 | Checkpoint PyTorch; không tìm thấy runtime backend tải file này |
| `research/hirschberg_candidate_v0.1.joblib` | 1,24 | Classifier crop cũ; dữ liệu/manifest nguồn ngoài workspace chưa tái lập được |
| `research/hirschberg_candidate_v0.2.joblib` | 3,03 | Hybrid 35 đặc trưng; report 450 mẫu tổng hợp |
| `research/hirschberg_candidate_v0.3_pedseye.joblib` | 0,62 | Classifier Pedseye cũ |
| `research/hirschberg_candidate_v0.5_safe.joblib` | 1,42 | 365 đặc trưng cặp mắt, có cơ chế từ chối kết luận |
| `research/hirschberg_branch_v02_eyecrop.joblib` | 0,84 | Nhánh mới: 19 hình học + 16 EfficientNet |
| `research/hirschberg_candidate_v0.6_ensemble.joblib` | 0,0014 | Bundle v1; kích thước nhỏ không tự chứng minh thiếu/hỏng model |
| `research/hirschberg_candidate_v0.6_ensemble_v2.joblib` | 2,27 | Bundle runtime phối hợp hai classifier |

Không dùng dung lượng file để suy ra độ chính xác. Số bytes trong `evidence.json: artifacts`. Kiến trúc ONNX được service gọi là ResNet18; audit chưa kiểm tra graph/weights để xác nhận provenance kiến trúc hoặc cách train checkpoint gốc.

## 3. Kiểm kê dữ liệu và chất lượng

| Tập | Eso | Exo | Normal | Pseudo | Kích thước/định dạng |
|---|---:|---:|---:|---:|---|
| `harschberg_data_detect/` accepted | 55 | 40 | 41 | 0 | Toàn bộ JPEG 224×224 |
| `dataset_by_pedseye_manifest/` | 20 | 14 | 33 | 0 | Toàn bộ JPEG 855×235 |
| Annotation thủ công, tập con của accepted | 33 | 19 | 10 | 0 | CSV điểm đồng tử/flash |

Hai thư mục có 203 file accepted/Pedseye; không coi là 203 bệnh nhân độc lập. So sánh SHA-256 không tìm thấy cặp file giống byte trong 203 file này; kiểm tra dHash vẫn phát hiện ảnh gần giống. Bằng chứng: `evidence.json: image_summary, duplicate_screen`.

`report.csv` bộ lọc có 502 ảnh nguồn: 171 eso, 189 exo, 142 normal. PASS 136/502 = 27,09%; REJECT 366/502 = 72,91%. Thư mục debug thêm 20 ảnh, nên tổng 522 file JPG không phải 522 ảnh train. 500/502 ảnh dùng crop heuristic, chỉ 2 ảnh dùng FaceMesh. **PASS của bộ lọc không có nghĩa ảnh đạt chuẩn lâm sàng.** Bằng chứng: `evidence.json: filter_report`; [filter_harschberg.py](D:/REMICARE-STRABISMUS-AI/filter_harschberg.py:539).

Detector nghiên cứu cũ coi 73/136 ảnh usable = 53,68%, tức các detector có tiêu chí không thống nhất. Pedseye có 65/67 ảnh báo phản xạ cả hai mắt nhưng chỉ 15/67 tìm được đồng tử cả hai mắt; manifest gắn blur cho 56/67. Bằng chứng: [geometry report](D:/REMICARE-STRABISMUS-AI/reports/hirschberg_geometry_research_results.json), `evidence.json: manifest`.

Kiểm tra Laplacian mới chỉ là chỉ báo độ nét: ngưỡng 15 cho 1/136 crop và 16/67 Pedseye dưới ngưỡng. Nó khác ngưỡng/cách đọc ảnh của kiểm tra cũ; không dùng số này làm tỷ lệ ảnh đạt chuẩn hoặc quy lỗi cho nhãn. Audit không xác minh được mi che đồng tử, ảnh nghiêng, flash thật hay ánh sáng môi trường trên toàn bộ dataset. Manifest ghi eyes_closed/face_off_axis `null` cho 67/67; khoảng cách, thiết bị, flash và hướng ảnh đều chưa xác minh. Bằng chứng: `evidence.json: image_summary, manifest`; manifest JSONL.

Kiểm tra trực quan cặp `esotropia_055.jpg`/`esotropia_060.jpg` thấy rất giống nhau; SHA khác nhưng dHash bằng nhau. Grid normal đã lưu có các ảnh khác hướng nhìn, crop/màu sắc và ảnh trông lặp lại. Đây là dấu hiệu cần review quy trình chụp và nhãn, không phải kết luận chẩn đoán sai nhãn qua mắt thường. Bằng chứng: [grid](D:/REMICARE-STRABISMUS-AI/reports/normal_high_oracle_asymmetry_grid.jpg), `evidence.json: duplicate_candidate_examples`.

### Nguồn, quyền sử dụng và định danh

| Tập hiện có | Nguồn có thể xác minh trong repo | License/quyền truy cập/đồng thuận | Khả năng định danh |
|---|---|---|---|
| Accepted/rejected crop | File lọc giữ nhãn từ tên file/thư mục; tài liệu nhắc nguồn cũ `data_hirschberg/eye-classification` | Chưa xác minh; không thấy hồ sơ license/đồng thuận của ảnh cụ thể; nguồn gốc chưa có URL đã xác minh | Ảnh vùng mắt có thể góp phần nhận diện; không có patient ID thật |
| Pedseye | Nhãn `pedseye_folder`, ảnh người dùng đã có trong workspace | Chưa xác minh; tên thư mục không chứng minh dataset công khai được phép train; không có yêu cầu đăng ký/quyền truy cập được xác minh | Ảnh vùng mắt người thật; 52 ID suy ra từ độ giống ảnh, không phải ID ẩn danh bệnh nhân đã xác thực |
| Korean/Cover Test | JSON/time series và video hiện có; khác bài toán ảnh Hirschberg | Chưa xác minh trong audit này; không chuyển sang train Hirschberg | Tên file Korean chứa chuỗi có thể là tên người; không tái xuất chúng vào báo cáo |

Không truy cập hoặc tải nguồn ngoài trong PHASE 1. Không dùng những URL trong tài liệu cũ như nguồn đã xác minh. Research literature, quyền dataset và cơ sở quy đổi lâm sàng sẽ được kiểm chứng ở PHASE 2 nếu được xác nhận.

## 4. Annotation và leakage

CSV [hirschberg_manual_annotations.csv](D:/REMICARE-STRABISMUS-AI/processed/hirschberg_manual_annotations.csv) có 62 hàng, mỗi hàng bốn điểm: OD pupil, OD reflex, OS pupil, OS reflex, tọa độ pixel x/y. Cả 62 hàng có file ảnh, đủ 8 tọa độ hữu hạn trong biên ảnh, nhãn trùng thư mục, không có hàng trùng đường dẫn. Bao phủ 62/136 = 45,59%; 74 ảnh accepted chưa có annotation trong CSV này. Cột notes trống toàn bộ. Không có reviewer, ngày review, bác sĩ hoặc độ bất đồng giữa người gán. Đúng định dạng không bảo đảm đúng vị trí giải phẫu. Bằng chứng: `evidence.json: annotations`.

Chưa xác minh được nhãn giữa các lần chụp của cùng bệnh nhân, thứ tự mắt khi ảnh bị mirror, hoặc pseudo/normal có bác sĩ phân biệt. Pedseye ghi `clinician_confirmed=false` cho tất cả; script folder manifest ghi lời người dùng rằng nhãn từng được bác sĩ xác nhận nhưng cũng thừa nhận không có clinician manifest riêng. Không suy ra các nhãn đã được audit lâm sàng từ lời ghi trong script. Bằng chứng: [folder_manifest.py](D:/REMICARE-STRABISMUS-AI/scripts/hirschberg_folder_manifest.py:94).

Leakage là để bài kiểm tra chứa lại người/ảnh gần giống bài đã học. Kiểm tra dHash 64-bit, khoảng cách ≤4, trên 203 ảnh tìm 12 cặp cần review; đây là ứng viên gần giống, không phải chứng minh cùng bệnh nhân. Với split v0.6 v2 tái dựng theo seed 20261004 và thứ tự file như script, có 4 cặp qua train/val:

| Validation | Train | Khoảng cách dHash |
|---|---|---:|
| `esotropia_147.jpg` | `esotropia_083.jpg` | 3 |
| `esotropia_172.jpg` | `esotropia_126.jpg` | 3 |
| `exotropia_132.jpg` | `exotropia_105.jpg` | 3 |
| `normal_023.jpg` | `normal_015.jpg` | 3 |

Pedseye có 52 cluster suy ra, không cluster nào qua split trong manifest; điều này chỉ chứng minh nhất quán cluster, không chứng minh độc lập bệnh nhân. Script v0.3 dùng StratifiedKFold theo ảnh; v0.5 cải thiện bằng StratifiedGroupKFold nhưng train final trên toàn bộ 67 ảnh, kể cả hàng mang split test. Không còn test Pedseye giữ kín đối với model đó. Bằng chứng: [v0.3 training](D:/REMICARE-STRABISMUS-AI/scripts/evaluate_and_train_pedseye_hirschberg.py:233), [v0.5 training](D:/REMICARE-STRABISMUS-AI/scripts/train_hirschberg_v05_safe_screening.py:433).

Không thấy augmentation ngẫu nhiên trong train v0.6 v2: resize/trích đặc trưng rồi train cây. Lịch sử augmentation trước khi đưa ảnh vào thư mục hiện có chưa xác minh; tên file/cluster không đủ chứng minh không có augmentation trước split.

## 5. Pipeline và khả năng tái lập

| Pipeline | Thiết kế hiện có | Split/seed | Nhận xét |
|---|---|---|---|
| v0.2 | LightGBM multiclass, balanced weights, 160 cây, LR 0,06, depth 6; sigmoid calibration; 19+16 đặc trưng | Outer StratifiedGroupKFold 5, seed 20261004; dữ liệu tổng hợp | Inner calibration cv=3 không truyền groups; không phải đánh giá ảnh thật |
| v0.3 | So sánh logistic/RF/ExtraTrees/HistGradientBoosting trên Pedseye | StratifiedKFold theo ảnh | Không chặn cùng bệnh nhân trong CV |
| v0.5 | 365 đặc trưng cặp; so sánh model nhỏ; threshold/margin cho INCONCLUSIVE | Group CV trên cluster suy ra; seed cố định | Chọn model và threshold trên cùng OOF; thiếu outer evaluation/test riêng |
| v0.6 v2 | v0.5 + RF 120 cây, depth 5, min leaf 2, weights eso/exo/normal = 1/1,35/1; calibration cv=3; EfficientNet frozen | 109 train/27 val theo ảnh; seed 20261004 | 125 cấu hình fusion/threshold; chưa có test giữ kín; final metric chứa train |

ONNX/EfficientNet đóng vai trò tạo đặc trưng, không được fine-tune trong v0.6 v2. Vì classifier nhánh mới là RF nên không có LR/epoch/CNN training loss ở nhánh đó. Không tìm thấy nhật ký learning curves train/val theo epoch hoặc báo cáo train-vs-val đầy đủ của v0.6 v2; report có metric validation chọn tham số nhưng không phải kiểm tra độc lập. Bằng chứng: script v0.6 v2, report v2.

Tái lập full train chưa thực hiện: `.venv312` hiện thiếu `timm`, `torch`, `torchvision`, `lightgbm`; requirements cũng thiếu chúng và nhiều dependencies chỉ giới hạn tối thiểu, không khóa phiên bản. Không cài package hoặc tải pretrained weights trong phase chỉ đọc. EfficientNet fallback `pretrained=False` có thể tạo feature khác nhau giữa train và runtime vì weights ngẫu nhiên không được lưu kèm bundle hoặc gắn hash. Seed classifier không khắc phục seed/weights backbone. Đây là lỗi thiết kế tái lập, không chỉ thiếu dữ liệu. Bằng chứng: `evidence.json: environment_versions`, [EfficientNet runtime](D:/REMICARE-STRABISMUS-AI/app/services/hirschberg_ai_service.py:60), [requirements](D:/REMICARE-STRABISMUS-AI/requirements.txt).

Runtime fallback khi thiếu nhánh phụ trả dự đoán v0.5, nhưng payload vẫn lấy modelId và metrics từ bundle v0.6 v2, dù có `fallbackModelId/fallbackReason`. Vì vậy metrics kèm response không mô tả đúng nhánh đang thực thi. Không kết luận môi trường Render hiện giống máy local nếu chưa xem deployment/log. Bằng chứng: [fallback](D:/REMICARE-STRABISMUS-AI/app/services/hirschberg_ai_service.py:422), [payload metrics](D:/REMICARE-STRABISMUS-AI/app/services/hirschberg_ai_service.py:466).

Rủi ro code khác: `build_all_features()` append X5 trước khi xử lý X2, nhưng gặp lỗi thì không rollback/mask y; một lỗi ảnh có thể làm lệch số hàng/nhãn hoặc lỗi index. Scaler được fit trước inner calibration trong một số script, khiến inner fold không tách hoàn toàn bước chuẩn hóa. Chưa quan sát các lỗi này trong run cũ; ghi là rủi ro từ code, không quy chúng là nguyên nhân đã xảy ra. Bằng chứng: [build_all_features](D:/REMICARE-STRABISMUS-AI/scripts/train_hirschberg_v06_ensemble_v2.py:159), [v0.2 calibration](D:/REMICARE-STRABISMUS-AI/scripts/train_hirschberg_research_model.py:492).

## 6. Kết quả cũ: đọc report và tính lại metric

Precision: trong các ảnh model gọi là lớp X, có bao nhiêu đúng. Recall: trong các ảnh thật thuộc X, có bao nhiêu được nhận ra. F1 phối hợp hai chỉ số. Balanced accuracy là trung bình recall các lớp; coverage là tỷ lệ ảnh được trả nhãn thay vì UNCERTAIN. Các số sau tính lại từ confusion matrix đã lưu, **không phải train/inference lại toàn bộ model**.

| Kết quả lưu | Phạm vi | Balanced accuracy | Macro F1 | Coverage | Giá trị sử dụng |
|---|---|---:|---:|---:|---|
| v0.2 | 450 đặc trưng tổng hợp/150 nhóm giả lập | 1,0000 | 1,0000 | Không áp dụng | Chỉ kiểm tra trên dữ liệu sinh theo lớp |
| v0.4 selected | OOF 67 Pedseye | 0,4718 | 0,4666 | 100% nhãn | Thăm dò, cluster giả lập, chọn model trên OOF |
| v0.5 selected forced | OOF 67 Pedseye | 0,5789 | 0,5669 | 100% nhãn | Thăm dò, chưa test độc lập |
| v0.5 forced | 136 accepted crop | 0,3610 | 0,3267 | 100% | Nhận exo rất kém; độc lập bệnh nhân chưa xác minh |
| v0.5 abstained | 136 accepted crop | Không tính được | Không tính được | 0% | Từ chối tất cả theo policy đã lưu |
| v0.6 v1 | 45/136 ảnh được trả nhãn | 0,3576 | 0,3007 | 33,09% | Đã tuning; không phải test giữ kín |
| Nhánh v0.2 eye-crop mới | Toàn 136, có 109 ảnh train | 0,8151 | 0,8200 | 100% | Không dùng như kết quả test |
| v0.6 v2 headline | 105/136, có ảnh train | 0,9461 | 0,9484 | 77,21% | Bị lẫn train và loại UNCERTAIN |
| v0.6 v2 best validation | 13/27 ảnh val được trả nhãn | 0,7778 | 0,7643 | 48,15% | Tập dùng chọn trong 125 cấu hình; không phải test |

Nguồn: `reports/*eval.json`; số tính lại trong `evidence.json: saved_report_metrics`. Các tập khác nhau nên không lấy chênh lệch giữa dòng làm mức cải thiện đáng tin cậy. Report `pedseye_hirschberg_external_eval.json` cũ cũng có balanced accuracy 0,3737/macro F1 0,2328 của service lúc đó; không đồng nhất với phiên bản runtime hiện tại.

### Metric theo lớp

| Model/phạm vi | Lớp | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| v0.5 OOF Pedseye | Eso | 0,5909 | 0,6500 | 0,6190 |
| v0.5 OOF Pedseye | Exo | 0,5000 | 0,5714 | 0,5333 |
| v0.5 OOF Pedseye | Normal | 0,5862 | 0,5152 | 0,5484 |
| v0.5 forced 136 crop | Eso | 0,4884 | 0,7636 | 0,5957 |
| v0.5 forced 136 crop | Exo | 0,5714 | 0,1000 | 0,1702 |
| v0.5 forced 136 crop | Normal | 0,2093 | 0,2195 | 0,2143 |
| v0.6 v2 full, chỉ ảnh trả nhãn | Eso | 0,9583 | 0,9787 | 0,9684 |
| v0.6 v2 full, chỉ ảnh trả nhãn | Exo | 0,9615 | 0,8929 | 0,9259 |
| v0.6 v2 full, chỉ ảnh trả nhãn | Normal | 0,9355 | 0,9667 | 0,9508 |
| v0.6 v2 best val, chỉ ảnh trả nhãn | Eso | 0,7143 | 0,8333 | 0,7692 |
| v0.6 v2 best val, chỉ ảnh trả nhãn | Exo | 1,0000 | 0,5000 | 0,6667 |
| v0.6 v2 best val, chỉ ảnh trả nhãn | Normal | 0,7500 | 1,0000 | 0,8571 |

Pseudo: không có ảnh thật ở những đánh giá này, metric chưa tính được. v0.2 tổng hợp có precision/recall/F1 = 1 ở cả 5 lớp normal/eso/exo/pseudo/poor_quality; điều đó không thay thế mẫu pseudo thật.

Confusion matrix: hàng là nhãn thật, cột là dự đoán; thứ tự Eso, Exo, Normal.

```text
v0.5 OOF Pedseye       v0.5 forced 136 crop
13   0   7            42   2  11
 1   8   5            13   4  23
 8   8  17            31   1   9

v0.6 v2 full covered   v0.6 v2 best val covered
46   0   1             5   0   1
 2  25   1             2   2   0
 0   1  29             0   0   3
```

Validation: có 11 eso, 8 exo, 8 normal; UNCERTAIN lần lượt 5, 4, 5. Như vậy 14/27 ảnh không có nhãn; số phân loại đúng là 10/27 = 37,04% tổng ảnh val. Exo trả đúng lớp 2/8 = 25%, so với recall 50% tính trên 4 ảnh exo có nhãn. Đây là tỷ lệ trả đúng lớp trong audit, không gọi UNCERTAIN là NORMAL hoặc kết luận chuyển khám thất bại; đánh giá workflow sàng lọc cần định nghĩa xử lý UNCERTAIN riêng.

Headline full có 100 ảnh đúng trong 105 ảnh trả nhãn; trên toàn 136 là 73,53% trả đúng lớp, và vẫn bị lẫn train. Chỉ số `normal_specificity` trong script v0.6 thực chất là recall NORMAL (= specificity nhị phân bệnh/normal trên phần covered); không phải specificity one-vs-rest cho lớp NORMAL. Bằng chứng: [compute_metrics](D:/REMICARE-STRABISMUS-AI/scripts/train_hirschberg_v06_ensemble_v2.py:243).

### Baseline hình học và kiểm tra detector

Report cũ của detector: balanced accuracy 0,5616; macro F1 0,5611; permutation p=0,2857 trên 73 ảnh usable với repeated CV. Manual/oracle trên 62 ảnh: balanced accuracy 0,6650, macro F1 0,6111; CI balanced accuracy [0,5018; 0,8281]; permutation p=0,0310. Matrix oracle `[[29,21],[65,195]]` chứa 310 dự đoán do đánh giá lặp 5 lần, không phải 310 ảnh độc lập. Matrix detector chứa 365 dự đoán = 73×5. Oracle/detector khác tập con nên chênh lệch chỉ gợi ý detector là nút thắt, chưa phải so sánh nhân quả có kiểm soát. Bằng chứng: [oracle results](D:/REMICARE-STRABISMUS-AI/reports/hirschberg_oracle_results.json), [geometry results](D:/REMICARE-STRABISMUS-AI/reports/hirschberg_geometry_research_results.json).

Audit đã chạy lại **detector v2 hiện tại**, không train model, trên 62 ảnh/124 mắt:

| So sánh với annotation | Số điểm | Trung bình px | Median px | P90 px |
|---|---:|---:|---:|---:|
| Iris center detector với pupil center thủ công | 124 | 14,16 | 10,38 | 31,83 |
| Reflex detector với reflex thủ công | 124 | 13,54 | 3,12 | 37,22 |

Có 120 trạng thái DETECTED và 4 LOCAL_MAX_FALLBACK; không có điểm reflex bị từ chối trong 124 mắt này. Tâm iris và tâm pupil là hai mục tiêu khác nhau nên hàng đầu không phải sai số pupil detector thuần túy. Reflex cùng mục tiêu nhưng annotation chưa có xác nhận chuyên gia; số này đo độ không khớp annotation hiện có. Detector cũ P90 31,61 px chỉ tính 94 điểm tìm được, khác detector/tập điểm, không đủ để khẳng định v2 đã tệ hơn. Bằng chứng mới: `evidence.json: current_eye_crop_detector_comparison`.

## 7. Chẩn đoán giả thuyết và giới hạn

| Giả thuyết yêu cầu | Nhận định hiện tại | Bằng chứng/kiểm chứng còn thiếu |
|---|---|---|
| a. Dữ liệu ít cho end-to-end | Dữ liệu thật ít, nhưng v0.6 hiện không train CNN end-to-end; đây chưa phải lý do duy nhất | 136+67 file, thiếu pseudo; chưa có learning curve theo số bệnh nhân |
| b. Phân loại ảnh thay vì đo hình học | Repo đã có geometry + classifier nhỏ; nút thắt gồm detector và cách đánh giá, không chỉ chọn kiến trúc | Oracle 0,6650, detector 0,5616; chưa đối chiếu cùng cohort và cùng split |
| c. Flash nhỏ bị mất khi resize/augment | Nguy cơ có cơ sở trong code resize 224×224, chưa chứng minh mất flash ở bao nhiêu ảnh | Crop accepted đã 224×224; không có ảnh nguồn độ phân giải gốc để đối chiếu; lịch sử augment chưa rõ |
| d. Nhãn/annotation sai | Định dạng annotation tốt; nhãn chuẩn lâm sàng chưa có bằng chứng đầy đủ | 62 tọa độ hợp lệ; 0/67 clinician_confirmed Pedseye; cần reviewer và agreement |
| e. Pseudo cần vùng quanh mắt | Chưa thể đánh giá pseudo: không có mẫu thật và classifier runtime không có lớp | Đặc trưng pseudo đang là heuristic, không phải detector được xác nhận |
| f. Leakage/mất cân bằng | Có nguy cơ leakage theo ảnh; mất cân bằng vừa ở accepted, rõ hơn ở Pedseye, và thiếu lớp hoàn toàn | Accepted 55/40/41; Pedseye 20/14/33; 4 cặp dHash qua train/val; không patient ID thật |

Vấn đề geometry cụ thể từ code: thiếu reflex bị gán offset mặc định ±0,5 mm; intercanthal distance tính bằng khoảng cách hai tâm mắt ×0,45 chứ không đo hai khóe trong; pitch cố định 0; quality flag không yêu cầu reflex thật. Quy đổi dùng hằng iris 11,8 mm với bán kính ước lượng từ dark component, không có chuẩn kích thước/tuổi đã xác minh trong phase này. Những feature có hậu tố mm vì vậy là ước lượng dựa giả định, không phải phép đo mm đã hiệu chuẩn. Bằng chứng: [eye_crop_geometry_service.py](D:/REMICARE-STRABISMUS-AI/app/services/eye_crop_geometry_service.py:263). Không đưa ra hệ số mm sang prism diopter hoặc ngưỡng y khoa ở PHASE 1.

## 8. Đã đạt được, chưa đạt được và tái lập audit

Đã có backend hoạt động theo contract, artifact versioned, annotation có cấu trúc, manifest/QA, baseline classifier nhỏ, confusion matrix và oracle analysis. Đây là tài sản có thể tận dụng thay vì bắt đầu lại; bằng chứng nằm trong các script, CSV và report nêu trên.

Chưa có dataset 4 lớp được xác minh; test độc lập theo bệnh nhân; benchmark v0.6 đáng tin cậy trên ảnh mới; detector định vị phản xạ đủ chính xác được xác nhận; weights/dependencies có thể tái lập đầy đủ; hồ sơ label/license/đồng thuận. Vì vậy chưa thể kết luận mức cải thiện thực hoặc dùng các metric hiện có để xác nhận công cụ sàng lọc lâm sàng.

Chạy lại audit chỉ tạo/cập nhật `evidence.json` trong thư mục audit mới:

```powershell
.\.venv312\Scripts\python.exe -B audits/phase1_20261004/collect_evidence.py
```

Script kiểm kê file/size, đọc JPEG, SHA-256/dHash, kiểm tra CSV/manifest, tái dựng split, chạy detector v2 thuần CV, tính precision/recall/F1 từ report và ghi phiên bản môi trường. Không tải model/data, train, sửa ảnh nguồn hoặc ghi file vào thư mục backend. Không chạy lại train cũ vì thiếu dependencies/weights và script cũ mặc định ghi đè artifact/report. Không chạy unit tests backend trong audit này vì không thay đổi backend; unit tests cũng không chứng minh độ chính xác model.

### Tóm tắt 5 dòng

1. Dataset thật hiện có 136 crop + 67 Pedseye, thiếu hoàn toàn pseudo; chưa chứng minh độc lập bệnh nhân (`evidence.json`).
2. 62 annotation hợp lệ về định dạng, nhưng detector v2 có sai số flash P90 37,22 px (`evidence.json`).
3. v0.2 100% là kết quả tổng hợp; v0.6 v2 94,61% chứa ảnh train và loại UNCERTAIN (script/report đã dẫn).
4. Best validation chỉ trả nhãn 13/27 ảnh, nhận đúng lớp exo 2/8 tổng ảnh exo; chưa có test giữ kín (report v2).
5. Chưa sửa code/train/deploy; cần nghiên cứu nguồn, dữ liệu và quy trình đánh giá trước khi nâng model.

**Dừng PHASE 1. Bạn có xác nhận tiếp tục PHASE 2 để kiểm chứng tài liệu, thuật toán và dataset hợp pháp không?**
