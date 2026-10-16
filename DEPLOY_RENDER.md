# Deploy Hirschberg trên Render

Bundle đã tự chứa checkpoint và mã suy luận trong `app/hirschberg_runtime/`.
Server không cần truy cập `D:\REMICARE_TRAIN_AI` hoặc tải pretrained weights khi chạy.

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
   Điền `DATABASE_URL` nếu cần lưu session bền vững (PostgreSQL/Supabase pooler).
   Không dán secrets vào `render.yaml`. SQLite memory mặc định chỉ phù hợp thử nghiệm.

## Bộ nhớ và vận hành

Chỉ dùng 1 worker: mỗi worker tạo thêm bản model trong RAM. Mỗi lần chỉ xử lý 1
request Hirschberg; request đồng thời nhận 503 với `Retry-After: 2`. Giới hạn CPU
thread là 1 mặc định; có thể đặt `HIRSCHBERG_CPU_THREADS` từ 1 đến 4 theo tài nguyên.
Cài PyTorch CPU để tránh tải các package CUDA. Chỉ cài một OpenCV distribution
(contrib, theo dependency MediaPipe); requirements-dev chứa pytest riêng.

Pipeline ONNX đã gỡ khỏi backend và Docker image. Hirschberg giữ nguyên checkpoint,
preprocessing và giới hạn ảnh. Số đo 562 MB / peak 628 MB trước đây thuộc bản có cả
CNN + ONNX, không dùng làm số đo của bản hiện tại. Số đo mới trên Windows CPU: khoảng 469 MB sau khi nạp model; peak khoảng 711 MB
khi xử lý ba ảnh tham chiếu (không phải phép đo ảnh lớn nhất). Chi tiết trong
`reports/production_integration/hirschberg_only_20261016.json`.
Nên chuẩn bị instance từ 2 GB RAM và đo trên Linux thực tế; chưa xác nhận chạy ổn 512 MB.
Nếu gặp OOM, tăng RAM và kiểm tra ảnh kích thước lớn; không tăng worker để khắc phục.
File session trong `data/` dùng filesystem tạm của Render, có thể mất sau redeploy;
DATABASE_URL cần được cấu hình nếu muốn giữ dữ liệu. Ảnh upload không được lưu bởi runtime.

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
