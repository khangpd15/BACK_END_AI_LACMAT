# Xác minh tích hợp Hirschberg — 22/10/2026

Nguồn: `D:\REMICARE_TRAIN_AI\HIRSCHBERG_METHOD`.
Đích: `app/hirschberg_runtime/` trong repository BE.

## Kết quả đồng bộ

Đã chạy lại `scripts/import_trained_hirschberg.py` từ nguồn trên. Bundle sau
import trùng với bundle đã được Git theo dõi: không có thay đổi checkpoint,
mã suy luận hoặc manifest. API đã nối vào runtime này qua
`app/services/trained_hirschberg_service.py` và `app/api/hirschberg.py`.

- Classifier: `screening_eyelid_branch_20261011_v1`.
- Pupil: `pupil_color_features_20261012_v1`, nhánh `control`.
- SHA-256 source runtime: `7bbb1f29efb3ff543780d4743713a84913ee65f8a38a7051d4f61365625baae8`.
- Manifest xác minh 40 file, tổng 101.075.063 byte.
- Cả 8 checkpoint `.pt` được Git theo dõi và nằm trong phần `app/` mà Docker copy.
- Không sao chép dataset, ảnh nguồn, virtualenv hoặc secret từ workspace train.

Run mới nhất `pupil_iris_continued_20261015_v3` không được kích hoạt:
`completion.json` của nguồn ghi `recommended_for_runtime=false`,
`runtime_promoted=false`, `independent_evaluation=false`. Cấu hình đang chạy
trong workspace train cũng giữ cặp model cũ. Đây là đồng bộ runtime được chọn,
không phải chuyển sang candidate mới nhất theo thời gian.

## Kiểm tra thực hiện

```powershell
.\.venv312\Scripts\python.exe -B scripts/import_trained_hirschberg.py
.\.venv312\Scripts\python.exe -B scripts/verify_runtime_bundle.py
.\.venv312\Scripts\python.exe -B -m pytest tests/model -q -p no:cacheprovider
```

- Import và xác minh hash: đạt.
- Bộ test BE/model: **27 passed**, 19,33 giây trên Windows CPU.
- Kiểm thử gồm nạp checkpoint thật, forward classifier, API ảnh trống/không hợp lệ,
  health/readiness, xử lý đồng thời, lưu session và hồi quy tối ưu landmark.
- Có 1 cảnh báo deprecation của Starlette TestClient với httpx; không có test lỗi.
- Lượt chạy trong sandbox bị kẹt khi asyncio tạo socket nội bộ. Đã dừng các lượt
  đó và chạy lại ngoài sandbox thành công; không thay đổi code để né lỗi môi trường.

Chưa build/chạy Docker hoặc deploy BE trong lần xác minh này. Sau khi deploy,
kiểm tra `/ready` trả 200 và `/api/v1/hirschberg/health` có `loaded=true`.
Không commit hoặc push thay người dùng.
