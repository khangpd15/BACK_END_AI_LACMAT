# RemiCare Hirschberg AI Backend

FastAPI backend chuyên biệt cho Hirschberg Test, định vị và sàng lọc bằng Hirschberg CNN.
Kết quả chỉ phục vụ nghiên cứu/sàng lọc sơ bộ, không phải chẩn đoán y khoa.

> Cấu hình deploy và phiên bản model: [DEPLOY_RENDER.md](DEPLOY_RENDER.md).

## Cấu trúc thư mục

```text
├── app/
│   ├── main.py                 # FastAPI, lifespan, middleware, health
│   ├── config.py               # Cấu hình môi trường Hirschberg
│   ├── api/                    # Hirschberg và research measurement
│   │   ├── hirschberg.py       # POST /api/v1/hirschberg/predict, GET sessions
│   │   └── research_measurement.py # POST /api/v1/research/measurements
│   ├── schemas/                # Pydantic request/response
│   │   ├── hirschberg.py
│   │   └── research_measurement.py
│   ├── services/
│   │   ├── hirschberg_session_service.py # Lưu session Hirschberg (local & DB)
│   │   ├── trained_hirschberg_service.py # Hirschberg CNN runtime
│   │   ├── hirschberg_ai_service.py      # Adapter cho Hirschberg CNN
│   │   ├── eye_crop_geometry_service.py  # Trích xuất hình học mắt
│   │   ├── cv/bilateral_quality_gate.py  # Quality gate ảnh hai mắt
│   │   └── keep_alive.py       # Tự ping giữ kết nối Render & DB
│   ├── db/
│   │   ├── database.py         # SQLAlchemy engine, session & auto migrations
│   │   ├── guid.py             # Type decorator UUID platform-independent
│   │   ├── models/             # hirschberg_session
│   │   └── repositories/       # hirschberg_repository
│   ├── models/                 # Artifact nghiên cứu lịch sử
│   └── hirschberg_runtime/     # Bundle CNN được kiểm tra SHA-256
├── data/hirschberg/sessions/   # Session local của Hirschberg (metadata.json, result.json)
├── migrations/                 # SQL schema
├── scripts/                    # Scripts chuẩn bị dữ liệu, import, verification
├── research/                   # Experiment source
├── tests/                      # Kiểm thử Hirschberg và nghiên cứu
└── render.yaml                 # Cấu hình deploy Render cho Hirschberg
```

## Cài đặt và chạy

Python **3.12.14** theo `.python-version` và `runtime.txt`.

```powershell
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# Điền cấu hình CORS, database và Supabase trong .env.
.\.venv312\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger API Docs: `http://127.0.0.1:8000/docs`.

## Deploy trên Render

Làm theo [DEPLOY_RENDER.md](DEPLOY_RENDER.md). Blueprint `render.yaml` đã cấu hình
Docker, PyTorch CPU, một worker, giới hạn thread và health check `/ready` chờ model nạp xong.
Kiểm tra artifact bằng `python scripts/verify_runtime_bundle.py` trước khi deploy.

## API đang đăng ký

| API | Chức năng |
| --- | --- |
| `GET /ready` | Readiness: 503 khi CNN chưa nạp, 200 khi sẵn sàng |
| `GET /ping` | Keep-alive fast ping |
| `GET /health` | Trạng thái backend và Hirschberg CNN |
| `GET /health/db` | Kiểm tra kết nối database |
| `GET /api/v1/hirschberg/health` | Trạng thái CNN và checkpoint |
| `POST /api/v1/hirschberg/predict` | Hirschberg CNN với landmark/QC + lưu session |
| `GET /api/v1/hirschberg/sessions/{session_id}` | Xem chi tiết session Hirschberg đã lưu |
| `GET /api/v1/hirschberg/sessions` | Danh sách session Hirschberg đã lưu |
| `POST /api/v1/research/measurements` | Đo hình học phục vụ nghiên cứu |

## Lưu trữ Session phân tách
- **Hirschberg:**
  - File local: `data/hirschberg/sessions/<session_id>/` (`metadata.json`, `result.json`).
  - Database: Bảng `hirschberg_sessions` trong PostgreSQL / Supabase.
  - Tuân thủ quyền riêng tư: Tuyệt đối không lưu ảnh sinh trắc học gốc của người dùng.
## Kiểm thử

```powershell
# Chạy bộ test cho Hirschberg Backend:
.\.venv312\Scripts\python.exe -m pytest tests/model/

```

## Pipeline Hirschberg duy nhất

Đã gỡ API Strabismus/ONNX và việc nạp ONNX khỏi backend. Frontend cần gọi
`POST /api/v1/hirschberg/predict`; các endpoint `/api/v1/strabismus/*` trả 404.
Trọng số và mã suy luận Hirschberg được giữ nguyên. Công cụ ONNX lịch sử ở
`research/legacy_onnx/`, được loại khỏi Docker image; dependency riêng nằm trong
`requirements-research.txt`. Bảng database cũ được giữ dữ liệu, không tự xóa.
