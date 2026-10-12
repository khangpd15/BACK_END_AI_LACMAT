# RemiCare Strabismus AI Backend

FastAPI backend cho Cover Test, sàng lọc ảnh hai mắt bằng ONNX và Hirschberg CNN.
Kết quả chỉ phục vụ nghiên cứu/sàng lọc sơ bộ, không phải chẩn đoán y khoa.

## Cấu trúc thư mục

```text
app/
  main.py                 # FastAPI, lifespan, middleware, health
  config.py               # Cấu hình môi trường
  api/                    # Cover Test, Hirschberg, ONNX, research measurement
  schemas/                # Pydantic request/response
  services/
    cover_test/           # Lưu session và dữ liệu
    cv/                   # Quality gates và temporal consensus
    ...                   # Inference, feature contract, keep-alive
  db/                     # Database, models, repositories
  models/                 # Model ONNX và artifact nghiên cứu
  hirschberg_runtime/     # Bundle CNN được kiểm tra SHA-256
migrations/               # SQL schema
scripts/                  # Chuẩn bị dữ liệu, training, import, verification
research/                 # Experiment source; archive/ lưu snapshot lịch sử
tests/                   # Kiểm thử contract, runtime và nghiên cứu
data/cover_test/sessions/ # Session local do API tạo, không commit nội dung
reports/production_integration/ # Bằng chứng kiểm tra deployment
docs/                    # Tài liệu kỹ thuật
audits/                  # Bằng chứng audit lịch sử
```

Chi tiết và lịch sử dọn dẹp: [Backend structure](docs/backend/BACKEND_STRUCTURE.md).
Dataset huấn luyện và output nghiên cứu đã được xóa khỏi BE. Training được thực
hiện tại `D:/REMICARE_TRAIN_AI/COVER_TEST_METHOD` và
`D:/REMICARE_TRAIN_AI/HIRSCHBERG_METHOD`. BE giữ model và code inference.

## Cài đặt và chạy

Python **3.12.14** theo `.python-version` và `runtime.txt`.

```powershell
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Điền cấu hình CORS, database và Supabase trong .env.
.\.venv312\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger: `http://127.0.0.1:8000/docs`.
Không ghi đè `.env` nếu đã có cấu hình local.

## API đang đăng ký

| API | Chức năng |
| --- | --- |
| `GET /ping` | Keep-alive |
| `GET /health` | Trạng thái backend và model |
| `GET /health/db` | Kiểm tra kết nối database |
| `POST /api/cover-test/sessions` | Lưu session local tương thích phiên bản cũ |
| `POST /api/v1/cover-test/sessions` | Lưu session vào PostgreSQL/Supabase và chạy inference tùy cấu hình |
| `GET /api/v1/strabismus/health` | Trạng thái ONNX |
| `POST /api/v1/strabismus/predict` | Sàng lọc ảnh ROI hai mắt bằng ONNX |
| `GET /api/v1/hirschberg/health` | Trạng thái CNN và checkpoint |
| `POST /api/v1/hirschberg/predict` | Hirschberg CNN với landmark/QC |
| `POST /api/v1/research/measurements` | Đo hình học phục vụ nghiên cứu |

Cover Test hiện trả `INCONCLUSIVE` khi chưa có artifact được nạp; các model
Korean transfer cũ đã được ngừng sử dụng. Không coi việc lưu session thành công
là bằng chứng model đang sẵn sàng.

## Cấu hình

Dùng [.env.example](.env.example) làm mẫu. Các nhóm cấu hình chính:

- `REMICARE_FRONTEND_ORIGINS`: danh sách origin frontend được phép truy cập.
- `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`: database/storage.
- `MODEL_PATH`: đường dẫn model ONNX của service ảnh hai mắt; mặc định
  `app/models/best_model.onnx` trong mẫu môi trường.
- `KEEP_ALIVE_ENABLED`, `KEEP_ALIVE_INTERVAL_SECONDS`, `RENDER_EXTERNAL_URL`:
  tự ping và giữ kết nối hoạt động.
- `ENABLE_AI_INFERENCE_ON_SAVE`, `ENABLE_RESEARCH_MEASUREMENT_API`: điều khiển
  inference khi lưu và API nghiên cứu.

Bundle Hirschberg nằm trong `app/hirschberg_runtime/`. Không đổi tên hoặc xóa
file trong bundle vì runtime kiểm tra manifest SHA-256 khi nạp.
Xem [Hirschberg runtime](docs/backend/HIRSCHBERG_RUNTIME.md).

## Kiểm thử

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
```

Các test parity cần Node.js; các test nghiên cứu dùng dataset đã xóa khỏi BE
cần chạy ở project train hoặc được cung cấp dữ liệu riêng.
Script kiểm tra deployment nằm trong `scripts/check_production_integration.py`;
kiểm tra ghi session cần bật tùy chọn riêng.

Chạy bộ lọc dataset từ repository root:

```powershell
.\.venv312\Scripts\python.exe scripts/filter_harschberg.py --help
```
