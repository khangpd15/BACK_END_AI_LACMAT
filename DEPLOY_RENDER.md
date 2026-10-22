# Deploy Hirschberg trên Render

Bundle đã tự chứa checkpoint và mã suy luận trong `app/hirschberg_runtime/`.
Server không cần truy cập `D:\REMICARE_TRAIN_AI` hoặc tải pretrained weights khi chạy.

Đã đồng bộ lại từ workspace train ngày 22/10/2026: bundle đang có trong Git trùng
với kết quả import, không phát sinh thay đổi checkpoint hoặc mã suy luận. Kiểm tra
40 file theo manifest đạt; bộ test `tests/model` đạt 27/27 trên Windows CPU.
Chi tiết: `reports/production_integration/hirschberg_sync_20261022.md`.

## Phiên bản được chọn

- Classifier: `screening_eyelid_branch_20261011_v1` (ResNet50 + geometry + khe mi).
- Pupil: `pupil_color_features_20261012_v1/control_final.pt`.
- Mã suy luận: đồng bộ workspace train ngày 16/10/2026, gồm input contract,
  kiểm tra visibility, EXIF, nhiều khuôn mặt, ánh xạ OD/OS và HUMAN_REVIEW.
- Run `pupil_iris_continued_20261015_v3` không được chọn: completion của run ghi
  `recommended_for_runtime=false`, `runtime_promoted=false`. Chưa có kiểm định độc lập.
- Hash checkpoint được giữ nguyên từ nơi train; manifest ghi hash mã nguồn và artifact.

## Trước khi deploy

1. Đưa code và toàn bộ file trong `app/hirschberg_runtime/` lên repository dùng để deploy.
   Không đưa `.env`, dataset, ảnh bệnh nhân hoặc virtualenv lên Git. Nếu dùng Git LFS,
   kiểm tra Render lấy được file nhị phân thật; bước build sẽ từ chối LFS pointer.
2. Chạy `python scripts/verify_runtime_bundle.py`.
3. Tạo Web Service từ repository; Root Directory là thư mục chứa `render.yaml`.
   Có thể dùng Blueprint để lấy cấu hình từ file này.
4. Chọn Docker runtime, Dockerfile Path `./Dockerfile`, Docker Context `.`.
   Để trống Docker Command để dùng CMD trong Dockerfile; không cần Build/Start Command riêng.
5. Dockerfile cài thư viện native cho OpenCV/MediaPipe, PyTorch CPU và kiểm tra import,
   dependency cùng hash artifact ngay lúc build. Container chạy dưới user không phải root.
6. Health Check Path: `/ready`; Python đã cố định `3.12.14` trong image.
   Các biến CPU nằm trong `render.yaml`.
7. Điền `REMICARE_FRONTEND_ORIGINS` bằng origin frontend thật, cách nhau bởi dấu phẩy.
   Backend không còn kết nối database/Supabase, không cần DATABASE_URL hoặc key Supabase.
   Các biến cũ nếu còn đặt trên Render không kích hoạt kết nối database.

## Bộ nhớ và vận hành

Chỉ dùng 1 worker: mỗi worker tạo thêm bản model trong RAM. Mỗi lần chỉ xử lý 1
request Hirschberg; request đồng thời nhận 503 với `Retry-After: 2`. Giới hạn CPU
thread là 1 mặc định; có thể đặt `HIRSCHBERG_CPU_THREADS` từ 1 đến 4 theo tài nguyên.
Cài PyTorch CPU để tránh tải các package CUDA. Chỉ cài một OpenCV distribution
(contrib, theo dependency MediaPipe); requirements-dev chứa pytest riêng.

Pipeline ONNX và tích hợp database/Supabase đã gỡ khỏi backend và Docker image.
Hirschberg giữ nguyên checkpoint, preprocessing và giới hạn ảnh.
Đã bỏ preview không được API sử dụng, ngừng tạo ảnh heatmap/mask trực quan và
bỏ instance seed thừa. Script import giữ các tối ưu khi tái tạo bundle; 8 checkpoint
không đổi SHA-256. 15/15 kết quả API giống hoàn toàn trước/sau (trừ latency).
Số đo Windows CPU trên 15 ảnh: sau nạp ~448 MiB; peak 694,7 → 685,4 MiB;
tổng thời gian 16,15 → 9,11 giây. Chỉ là một lượt đo; chưa kiểm chứng Linux/Render,
ảnh lớn nhất hay độ chính xác lâm sàng. 27 test pass và 40 file runtime được xác minh.
Chi tiết: `reports/production_integration/hirschberg_runtime_optimization_20261016.md`.
Nên chuẩn bị instance từ 2 GB RAM và đo trên Render thực tế; chưa xác nhận ổn định 512 MB.
Nếu gặp OOM, tăng RAM và kiểm tra ảnh kích thước lớn; không tăng worker để khắc phục.

Session chỉ lưu metadata/result local nếu `save_session=true`; frontend hiện gửi false.
Filesystem mặc định của Render là tạm, file có thể mất sau restart/redeploy. Muốn giữ
lâu dài cần persistent disk phù hợp hoặc cơ chế sao lưu. Ảnh upload không được lưu.
Backend không còn `/health/db`, không có ping hoặc tác vụ đọc/ghi database.

`/ping` xác nhận HTTP đang chạy. `/ready` trả 503 khi CNN chưa sẵn sàng và 200 sau
khi nạp thành công; Render chỉ chuyển traffic khi readiness thành công.
`/api/v1/hirschberg/health` cho biết checkpoint/version/error.

## Kiểm tra sau deploy

```bash
curl -f https://YOUR-SERVICE.onrender.com/ready
curl -f https://YOUR-SERVICE.onrender.com/api/v1/hirschberg/health
curl -f https://YOUR-SERVICE.onrender.com/api/v1/hirschberg/predict \
  -F 'image=@eye.jpg' -F 'orientation=unmirrored' \
  -F 'fixation=confirmed' -F 'save_session=false'
```

Chỉ xác nhận orientation/fixation khi có metadata thực tế. Mặc định API vẫn để
orientation auto/fixation unknown; ảnh thiếu điều kiện trả INCONCLUSIVE.
Kết quả có lớp luôn yêu cầu HUMAN_REVIEW, không phải chẩn đoán.

Tài liệu nền tảng: https://render.com/docs/python-version,
https://render.com/docs/health-checks, https://pytorch.org/get-started/previous-versions/.

## Giới hạn kiểm tra môi trường

Đã kiểm thử API/model trên Windows CPU và giải dependency wheel Linux bằng pip dry-run.
Docker daemon local chưa chạy nên chưa build/chạy container tại máy này; bước build
Docker trên Render sẽ kiểm tra import native và toàn bộ hash trước khi khởi động.
Chưa tạo service hoặc deploy trực tiếp lên tài khoản Render.
