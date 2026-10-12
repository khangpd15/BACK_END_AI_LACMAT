# Hirschberg

## Cấu trúc

```text
models/candidates/             7 checkpoint Hirschberg v0.1–v0.6 và branch eyecrop
models/representation/         best_model.onnx, best_model.pt (bilateral ROI representation)
datasets/harschberg_data_detect/    Ảnh đã lọc, ảnh rejected, debug và report
datasets/dataset_by_pedseye_manifest/ Ảnh Pedseye theo thư mục nguồn
datasets/manifests/            Manifest Pedseye bản gốc
processed/                    Annotation thủ công và feature CSV
source_snapshot/scripts/      Code train, chuẩn bị dữ liệu và phân tích
source_snapshot/research/hirschberg_phase4/  Pipeline baseline/annotation và runs lịch sử
docs/                         Tài liệu nghiên cứu và model cards
reports/                      Kết quả đánh giá và ảnh phân tích
```

Checkpoint API hiện tham chiếu ở vị trí gốc: `app/models/research/hirschberg_candidate_v0.6_ensemble_v2.joblib`; bản copy tương ứng nằm trong `models/candidates/`. ONNX representation không phải detector reflex/limbus đã được xác nhận lâm sàng. Tổ chức thư mục không làm thay đổi mức độ xác nhận của model/dataset.

## Cách chạy train với dataset bản copy

Chạy tại `D:\REMICARE-STRABISMUS-AI`. Đây là hướng dẫn, chưa thực thi training. Chọn tên run mới để không ghi đè checkpoint và kết quả hiện có. Script vẫn dùng dependency và checkpoint v0.5 ở vị trí gốc theo mặc định.

```powershell
.\.venv312\Scripts\python.exe -B scripts/train_hirschberg_v06_ensemble_v2.py `
  --dataset-root ai/hirschberg/datasets/harschberg_data_detect `
  --model-output ai/hirschberg/models/runs/run_001/model.joblib `
  --branch-v02-output ai/hirschberg/models/runs/run_001/branch_v02.joblib `
  --eval-output ai/hirschberg/reports/run_001/eval.json `
  --card-output ai/hirschberg/reports/run_001/model_card.md
```

Annotation từ dataset bản copy, ghi file mới:

```powershell
.\.venv312\Scripts\python.exe -B scripts/hirschberg_manual_annotator.py `
  --dataset-root ai/hirschberg/datasets/harschberg_data_detect `
  --output-csv ai/hirschberg/processed/new_manual_annotations.csv
```

CSV và manifest lịch sử giữ nguyên đường dẫn nguồn. Không trộn annotation mới và cũ chỉ theo chuỗi đường dẫn; cần ánh xạ source/copy trong `../copy_inventory.json` và hash ảnh. Folder labels, inferred groups và annotation không tự trở thành diagnosis/patient ID đã được bác sĩ xác nhận.
