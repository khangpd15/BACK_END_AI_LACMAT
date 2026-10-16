# RemiCare Hirschberg AI Backend

FastAPI backend định vị đồng tử, mống mắt, phản quang, khe mi và sàng lọc bằng
Hirschberg CNN. Kết quả phục vụ nghiên cứu, cần chuyên gia xem xét.

Hướng dẫn deploy: [DEPLOY_RENDER.md](DEPLOY_RENDER.md).

## Cấu trúc đang chạy

- `app/main.py`: API, lifecycle và health/readiness.
- `app/api/hirschberg.py`: nhận ảnh, suy luận, lưu/đọc session local.
- `app/services/trained_hirschberg_service.py`: runtime CNN, kiểm tra SHA-256.
- `app/services/hirschberg_session_service.py`: metadata/result bằng JSON local.
- `app/hirschberg_runtime/`: checkpoint và mã suy luận tự chứa.
- `data/hirschberg/sessions/`: session JSON nếu bật `save_session`.
- `research/legacy_onnx/`, `research/legacy_database/`: lịch sử nghiên cứu, ngoài Docker image.
- `Dockerfile`, `render.yaml`: đóng gói CPU, một worker, health check `/ready`.

## Cài đặt và chạy

Python 3.12.14. Cài `requirements-dev.txt`, sao chép `.env.example` sang `.env`
và đặt `REMICARE_FRONTEND_ORIGINS` bằng origin frontend thật.
Không cần DATABASE_URL, Supabase hoặc PostgreSQL. Các biến database cũ không được sử dụng.

```powershell
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv312\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## API

| API | Chức năng |
| --- | --- |
| GET `/ping` | HTTP liveness |
| GET `/ready` | 200 khi CNN sẵn sàng, 503 khi chưa nạp |
| GET `/health` | Trạng thái model, HTTP keep-alive và sessionStorage=local |
| GET `/api/v1/hirschberg/health` | Phiên bản/checkpoint/error |
| POST `/api/v1/hirschberg/predict` | Multipart image, orientation, fixation, kind, save_session |
| GET `/api/v1/hirschberg/sessions` | Danh sách session local, limit/offset |
| GET `/api/v1/hirschberg/sessions/{session_id}` | Metadata/result local |
| POST `/api/v1/research/measurements` | Đo hình học nghiên cứu |

API `/api/v1/strabismus/*` và `/health/db` đã gỡ, trả 404.
Không có tác vụ ping, kết nối, đọc hoặc ghi database. HTTP self-ping là tùy chọn,
mặc định tắt với `KEEP_ALIVE_ENABLED=false`.

## Lưu session

`save_session=false` chỉ trả kết quả, không tạo session. Frontend hiện gửi tùy chọn này.
`save_session=true` lưu `metadata.json` và `result.json` trong
`data/hirschberg/sessions/<session_id>/`; không lưu ảnh upload.
Nếu ID đã tồn tại, server tạo ID mới và trả đúng ID để truy xuất lại.

Filesystem mặc định của Render là tạm: session local có thể mất sau redeploy/restart.
Nếu cần giữ lâu dài, gắn persistent disk phù hợp hoặc sao lưu thư mục session.
Dữ liệu đã tồn tại trong Supabase không bị thay đổi; backend hiện không truy xuất chúng.

## Kiểm thử và bộ nhớ

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/model/ -q
.\.venv312\Scripts\python.exe scripts/verify_runtime_bundle.py
```

27 test pass, gồm startup/health, session local, readiness, checkpoint thật và
đối chiếu tọa độ/QC khi tắt ảnh trực quan. Đã bỏ preview không dùng, ảnh heatmap
không trả qua API và instance seed thừa; giữ nguyên weights, preprocessing và QC.
15/15 kết quả trước/sau giống hoàn toàn (trừ latency). Trên Windows CPU, tổng thời
gian bộ ảnh giảm 16,15 xuống 9,11 giây; peak RAM 694,7 xuống 685,4 MiB.
Đây là phép đo một lượt, chưa kiểm chứng Linux/Render hay độ chính xác lâm sàng.
Chi tiết: `reports/production_integration/hirschberg_runtime_optimization_20261016.md`.
Chưa xác nhận chạy ổn trong 512 MB; nên chuẩn bị từ 2 GB RAM và đo trên Render thực tế.
