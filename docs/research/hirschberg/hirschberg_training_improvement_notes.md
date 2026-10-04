# Nhận xét hiện trạng train AI Hirschberg và điều kiện để cải thiện độ chính xác

Ngày lập: 2026-10-04  
Trạng thái: research-only, không phải chẩn đoán lâm sàng

## 1. Kết luận ngắn

Model Hirschberg hiện tại chưa đủ tin cậy để deploy như một bộ phân loại chính thức. Kết quả kiểm chứng trên `dataset_by_pedseye_manifest` cho thấy model đang có xu hướng nhận diện nhiều ảnh bình thường thành lác và chưa phân biệt ổn định giữa `esotropia`, `exotropia`, và `normal`.

Candidate mới `hirschberg_candidate_v0.3_pedseye.joblib` có cải thiện một phần macro-F1 so với model service hiện tại, nhưng sensitivity trong cross-validation còn thấp. Vì mục tiêu sàng lọc là hạn chế bỏ sót ca lác, chưa nên đưa model này vào production hoặc endpoint đang chạy.

## 2. Hiện trạng dữ liệu

Dataset Pedseye mới:

| Class | Số ảnh |
|---|---:|
| esotropia | 20 |
| exotropia | 14 |
| normal | 33 |
| Tổng | 67 |

Hạn chế chính:

- Số ảnh quá ít cho bài toán computer vision y khoa.
- Chưa có `participant_id`, nên không thể chắc chắn train/test không bị rò rỉ cùng bệnh nhân.
- Chưa có class `pseudostrabismus`, trong khi đây là nhóm rất quan trọng để phân biệt giả lác với lác thật.
- Nhãn được lấy từ folder Pedseye do người dùng cung cấp; repo chưa có manifest xác nhận bởi bác sĩ/orthoptist độc lập.
- Ảnh có thể khác domain với ảnh người dùng upload thực tế: góc chụp, crop, ánh sáng, flash, khoảng cách, độ phân giải.

## 3. Kết quả kiểm chứng hiện tại

Model đang được service dùng thật: `hirschberg_candidate_v0.1.joblib`

Trên dataset Pedseye:

- Balanced accuracy: `0.3737`
- Macro-F1: `0.2328`
- Strabismus-vs-normal sensitivity: `0.9706`
- Strabismus-vs-normal specificity: `0.1212`
- False negative: `1` ảnh exotropia bị dự đoán normal

Candidate mới: `hirschberg_candidate_v0.3_pedseye.joblib`

- Model: `random_forest_balanced`
- Feature contract: `73` features
- CV balanced accuracy: `0.4617`
- CV macro-F1: `0.4428`
- Binary sensitivity: `0.5294`
- Binary specificity: `0.3636`

Nhận xét: v0.3 phân lớp tổng quát tốt hơn một chút, nhưng bỏ sót lác nhiều hơn trong CV. Do đó không nên thay model hiện tại bằng v0.3.

## 4. Vì sao accuracy hiện tại thấp

### 4.1. Dataset quá nhỏ

67 ảnh là chưa đủ để mô hình học ổn định. Với 3 class hiện tại, mỗi class nên có tối thiểu vài trăm ảnh đã kiểm duyệt. Nếu thêm `pseudostrabismus`, cần nhiều hơn nữa vì giả lác rất dễ nhầm với esotropia.

Mức khuyến nghị tối thiểu cho research:

| Class | Mức tối thiểu nên có |
|---|---:|
| normal | 300-500 ảnh |
| esotropia | 300-500 ảnh |
| exotropia | 300-500 ảnh |
| pseudostrabismus | 300-500 ảnh |
| poor_quality/rejected | 200+ ảnh |

Mức tốt hơn cho model ổn định:

- 1,000+ ảnh mỗi class.
- 300+ participant khác nhau.
- Có split theo participant, không split ngẫu nhiên theo ảnh.

### 4.2. Thiếu pseudostrabismus

Người dùng đã phát hiện có các trường hợp:

- lác trong thật
- lác ngoài thật
- giả lác
- bình thường

Nếu không train class `pseudostrabismus`, model sẽ buộc phải ép giả lác vào `normal`, `esotropia`, hoặc `exotropia`. Đây là nguyên nhân lớn gây sai.

Cần đưa `pseudostrabismus` thành class riêng trong manifest và report.

### 4.3. Chưa có manifest chuẩn

Cần có file manifest thay vì chỉ dựa vào tên folder.

Manifest nên có tối thiểu:

```json
{
  "sample_id": "uuid",
  "participant_id": "PT_0001",
  "image_path": "dataset/...",
  "label": "esotropia",
  "diagnosis_source": "pedseye_reference_or_ophthalmologist",
  "clinician_confirmed": false,
  "capture_protocol": "phone_flash_full_face",
  "age_group": "adult_18_plus",
  "split": "train"
}
```

Quan trọng nhất là `participant_id`. Nếu một bệnh nhân có nhiều ảnh, toàn bộ ảnh của bệnh nhân đó phải nằm trong cùng một split.

### 4.4. Protocol chụp chưa đồng nhất

Hirschberg phụ thuộc rất mạnh vào phản xạ ánh sáng giác mạc. Vì vậy ảnh train cần thống nhất:

- Nhìn thẳng camera.
- Có flash hoặc nguồn sáng cố định.
- Khoảng cách chụp được kiểm soát.
- Mặt gần chính diện, ít nghiêng đầu.
- Hai mắt nhìn rõ.
- Không dùng ảnh crop quá sát nếu production nhận ảnh mặt đầy đủ.
- Lưu thông tin ảnh bị mờ, tối, lệch mặt, thiếu phản xạ.

Nếu train bằng ảnh crop nhưng deploy bằng ảnh webcam/full-face thì model sẽ học sai domain.

### 4.5. Feature hiện tại chưa đủ robust

Feature v0.1/v0.3 đang dùng 73 features từ crop heuristic:

- iris/pupil estimate
- corneal reflex estimate
- horizontal intensity profile
- spatial grid intensity
- ONNX logits

Điểm yếu:

- Nếu reflex detector bắt nhầm highlight trên da/kính/mí mắt, vector Hirschberg sai.
- Nếu ảnh full-face không có landmarks, script phải crop trái/phải bằng heuristic nửa ảnh.
- Không có calibration theo camera/flash/khoảng cách.
- Chưa có rule mạnh để phân biệt pseudostrabismus dựa trên scleral exposure và reflex đối xứng.

## 5. Cần bổ sung gì để train chính xác hơn

### 5.1. Bổ sung dữ liệu theo class

Cần thu thập có chủ đích:

| Class | Cần bổ sung |
|---|---|
| normal | nhiều ảnh chính diện, đa tuổi, đa ánh sáng |
| esotropia | lác trong thật, nhiều mức độ |
| exotropia | lác ngoài thật, nhiều mức độ |
| pseudostrabismus | giả lác, đặc biệt trẻ em/nếp quạt/flat nasal bridge |
| poor_quality | mờ, tối, lệch mặt, thiếu mắt, thiếu reflex |

Không nên chỉ tăng ảnh bằng augmentation nếu dữ liệu gốc ít. Augmentation giúp regularization, không thay thế nhãn thật.

### 5.2. Chuẩn hóa manifest

Tạo manifest chính thức dạng JSONL/CSV/Parquet với:

- `sample_id`
- `participant_id`
- `image_path`
- `label`
- `diagnosis_source`
- `clinician_confirmed`
- `capture_condition`
- `head_pose_status`
- `split`
- `license`
- `consent_status`

Sau đó chạy governance:

- kiểm tra schema
- kiểm tra trùng SHA-256
- kiểm tra near-duplicate pHash
- kiểm tra participant không rò rỉ giữa train/val/test

### 5.3. Tách bộ kiểm chứng cố định

Không dùng tất cả ảnh để vừa train vừa báo kết quả.

Nên chia:

- `train`: 70%
- `val`: 15%
- `test`: 15%

Nếu dataset còn nhỏ, dùng cross-validation nhưng vẫn nên giữ một external test set không đụng vào khi chọn model.

### 5.4. Dùng metric đúng mục tiêu y khoa

Không chọn model chỉ theo accuracy.

Metric bắt buộc:

- confusion matrix 4-class hoặc 5-class
- macro-F1
- balanced accuracy
- per-class recall
- esotropia recall
- exotropia recall
- pseudostrabismus vs esotropia confusion rate
- binary strabismus-vs-normal sensitivity/specificity
- false negative review list

Ưu tiên:

1. Giảm false negative lác thật.
2. Giảm nhầm pseudostrabismus thành esotropia.
3. Sau đó mới tối ưu specificity.

### 5.5. Cải thiện detector trước khi cải thiện classifier

Classifier sẽ không tốt nếu feature đầu vào sai.

Cần audit riêng:

- reflex detector bắt đúng hay bắt nhầm.
- iris/pupil center đúng hay lệch.
- landmark có ổn định không.
- ảnh bị thiếu reflex nên trả `INELIGIBLE` hay vẫn dự đoán.

Nên tạo report overlay:

- vẽ iris center
- vẽ pupil center
- vẽ corneal reflex
- vẽ vector Hirschberg
- ghi status detector

Sau đó review bằng mắt các case sai.

## 6. Hướng train tiếp đề xuất

### Phase A - Data QA

1. Tạo manifest chuẩn cho `dataset_by_pedseye_manifest`.
2. Gán `participant_id` nếu biết ảnh nào cùng bệnh nhân.
3. Thêm class `pseudostrabismus`.
4. Chạy duplicate/near-duplicate check.
5. Tạo overlay review cho toàn bộ ảnh.

### Phase B - Baseline research

Train các model nhẹ, dễ kiểm soát:

- LogisticRegression `class_weight=balanced`
- RandomForest `class_weight=balanced`
- HistGradientBoosting
- calibrated model nếu dataset đủ lớn

Không nên vội dùng CNN thuần khi dataset nhỏ.

### Phase C - Feature cải tiến

Ưu tiên feature giải thích được:

- normalized Hirschberg vector mỗi mắt
- left/right vector asymmetry
- corneal reflex detected/missing
- reflex quality score
- scleral nasal/temporal ratio
- blur/brightness/head pose
- eye crop quality
- pseudostrabismus rule flag

Nếu có embedding deep learning, chỉ dùng như feature phụ, không dùng như bằng chứng y khoa.

### Phase D - Candidate versioning

Mỗi lần train phải tạo artifact mới:

- `hirschberg_candidate_v0.4.joblib`
- `hirschberg_candidate_v0.4_eval.json`
- `hirschberg_candidate_v0.4_model_card.md`

Không ghi đè:

- `hirschberg_candidate_v0.1.joblib`
- `hirschberg_candidate_v0.2.joblib`
- `hirschberg_candidate_v0.3_pedseye.joblib`

## 7. Điều kiện trước khi deploy model Hirschberg mới

Chỉ nên cân nhắc deploy research endpoint mới khi đạt tối thiểu:

- Có manifest chuẩn.
- Có `participant_id` và split không leakage.
- Có đủ 4 class: `normal`, `esotropia`, `exotropia`, `pseudostrabismus`.
- External test set chưa dùng trong training.
- Esotropia recall cao và ổn định.
- Exotropia recall cao và ổn định.
- Pseudostrabismus không bị nhầm nhiều thành esotropia.
- False negative list đã được review.
- Model card ghi rõ không phải chẩn đoán.

Không nên deploy vào production endpoint `/api/v1/strabismus/predict` nếu chỉ mới train trên dataset nhỏ.

## 8. Việc nên làm ngay tiếp theo

1. Tạo manifest chuẩn cho `dataset_by_pedseye_manifest`.
2. Bổ sung folder/class `pseudostrabismus`.
3. Thêm `participant_id` nếu có thể suy ra từ nguồn ảnh.
4. Tạo script overlay detector để review ảnh sai.
5. Giữ v0.3 làm baseline research, không deploy.
6. Train v0.4 chỉ sau khi dataset có thêm pseudostrabismus và manifest đạt governance.

## 9. Lệnh kiểm chứng hiện tại

```powershell
cd D:\REMICARE-STRABISMUS-AI
.\.venv312\Scripts\Activate.ps1
python scripts\evaluate_and_train_pedseye_hirschberg.py
python -m pip check
python -m pytest
```

Kết quả gần nhất:

- `pip check`: không có dependency lỗi.
- `pytest`: `38 passed, 1 skipped`.

