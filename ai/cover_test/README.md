# Cover Test

## Cấu trúc

```text
models/                         Chưa tìm thấy checkpoint Cover Test hiện có để copy
datasets/VIDEO_COVER_TEST/       6 video mẫu theo ESOTROPIA/EXOTROPIA/NORMAL
datasets/normalized/korean/     Chuỗi thời gian binocular eye-tracking và conversion report
source_snapshot/research/cover_test_timestamp/  Code nghiên cứu, permission example và license ledger
docs/                           Audit model Cover Test
reports/cover_timestamp_30_60/  Audit và trạng thái nghiên cứu
```

Không có checkpoint Cover Test trong các thư mục nguồn đã kiểm tra (`models/`, `research/cover_test_timestamp/`). `models/` được để trống có chủ ý, không tạo hoặc giả định model đã train.

Korean eye-tracking là dữ liệu nghiên cứu liên quan, không phải webcam Cover Test có nhãn chu kỳ cover/uncover. Video theo folder cũng không mặc định có nhãn lâm sàng đã xác minh, patient mapping hoặc timestamp/phase annotation. Tham khảo LICENSE_LEDGER và audit trước khi train.

## Audit dataset bản copy

Chạy từ `D:\REMICARE-STRABISMUS-AI`, dùng tên output mới. Lệnh dưới đây chưa được chạy khi tạo workspace:

```powershell
.\.venv312\Scripts\python.exe -B -m research.cover_test_timestamp.train `
  --data ai/cover_test/datasets/normalized/korean `
  --output ai/cover_test/reports/new_audit.json
```

## Train khi đủ quyền sử dụng và điều kiện dataset

`permission.example.json` là mẫu, không phải permission đã xác minh. Chỉ dùng file permission có bằng chứng thật; patient mapping phải từ nguồn xác nhận. Model từ dữ liệu Korean chưa được xác nhận cho webcam.

```powershell
.\.venv312\Scripts\python.exe -B -m research.cover_test_timestamp.train `
  --data ai/cover_test/datasets/normalized/korean `
  --train `
  --permission PATH_TO_VERIFIED_PERMISSION_JSON `
  --patient-mapping PATH_TO_VERIFIED_PATIENT_MAPPING_JSON `
  --output ai/cover_test/models/run_001
```

Không chạy trực tiếp source snapshot. Metadata và báo cáo lịch sử giữ nguyên nội dung nguồn. Dataset và model của runtime gốc không thay đổi.
