# Phân tích pipeline xử lý ảnh Hirschberg

Đọc mã và đo trên ba ảnh tham chiếu có sẵn. Chỉ tạo báo cáo; không chỉnh mã suy luận,
checkpoint, preprocessing hoặc ngưỡng QC. Dữ liệu đo chi tiết trong file JSON cùng tên.

## 1. Luồng xử lý thực tế

1. API nhận JPEG/PNG/WebP dưới 10 MiB; kiểm tra ảnh tối thiểu 16 px mỗi chiều,
   tối đa 25 triệu pixel. PIL giải mã RGB theo tọa độ ảnh lưu.
2. MediaPipe FaceMesh tìm tối đa hai khuôn mặt. Ảnh có một mặt được lấy ROI hai mắt;
   nếu auto không thấy mặt, OpenCV eye cascade thử ghép cặp mắt. ROI ảnh mới được
   padding 10%; ảnh khớp SHA có ROI đóng băng để tái lập preprocessing trước đây.
3. Mỗi mắt chạy localizer pupil/reflex ở 256x256, iris bốn điểm ở 192x192
   (có thể thử lại với crop mở rộng), pupil direct-center ở 256x256 và U-Net khe mi 192x192.
   Kết quả pupil từ localizer chung được thay bằng pupil direct-center. Localizer chung
   vẫn cần để lấy reflex nên không thể bỏ cả mạng này chỉ vì pupil bị thay.
4. Tính vector pupil–reflex theo bán kính iris, đối chiếu hai mắt và hình học khe mi.
   Kiểm tra hướng ảnh, EXIF, nhiều mặt, ROI/visibility, reflex, tỷ lệ iris và fixation.
5. Kết quả chính chỉ gọi classifier sau gate. Classifier ghép hai crop mắt thành
   panel 224x224, chuẩn hóa ImageNet, dùng ResNet50 + 11 đặc trưng geometry +
   nhánh 10 đặc trưng khe mi để tạo điểm ba lớp normal/esotropia/exotropia.
6. Nếu gate không đạt, trả INCONCLUSIVE. Nếu có phân lớp, trạng thái HUMAN_REVIEW.
   Softmax chưa hiệu chuẩn; API đặt probabilities=null và trả uncalibratedModelScores.
   confidence là điểm lớp cao nhất, không phải số đo độ chính xác hay xác suất mắc bệnh.

## 2. Trọng số đang nằm trong RAM

| Đối tượng runtime | Tham số | State MiB (parameters + buffers) |
| --- | ---: | ---: |
| fusion: ResNet50 + geometry + khe mi | 23.775.622 | 90,900 |
| iris bốn điểm | 368.568 | 1,406 |
| pupil direct-center | 119.112 | 0,454 |
| pupil/reflex localizer | 118.968 | 0,454 |
| U-Net khe mi | 118.913 | 0,454 |
| seed | 118.968 | 0,454 |

Tổng state khoảng 94,1 MiB. Đây chỉ là tensor trọng số/buffer, không phải RAM cả
backend. Số đo trước đó sau gỡ database: khoảng 448 MB sau nạp, peak 689 MB
trong ba ảnh tham chiếu. PyTorch/MediaPipe/OpenCV, ảnh giải mã và tensor trung gian
chiếm phần còn lại; chưa đo riêng từng thư viện để gán chính xác số MB cho từng phần.
Code đã dùng eval() và torch.no_grad() trong các lượt suy luận.

## 3. Đo ba ảnh tham chiếu

| Ảnh (kích thước) | Thời gian local | Kết quả chính | Số forward fusion |
| --- | ---: | --- | ---: |
| 1280x1110 | 3.727 ms | INCONCLUSIVE: invalid_reflex_landmark | 0 |
| 615x248 | 759 ms | INCONCLUSIVE: invalid_reflex_landmark | 0 |
| 1920x2560 | 1.277 ms | INCONCLUSIVE: common_direction_ambiguous | 2 |

Mỗi ảnh trên chạy pr/iris/pupil/eyelid hai lần, tương ứng hai mắt. Seed không chạy.
Ảnh đầu tốn khoảng 1.913 ms trong fallback ghép cặp mắt OpenCV. Các mốc thời gian
được đo bằng wrapper/hook, có phần lồng nhau, không được cộng thẳng tất cả các cột.
Ba ảnh không đại diện cho toàn bộ dữ liệu; không dùng để suy độ chính xác hoặc throughput.

## 4. Phần xử lý dư đã xác minh

### Preview chạy thêm nhưng không có trong response API

Engine tạo model_preview riêng khi kind=auto. Nhánh này có kiểm tra reflex/input
nhưng không qua cùng capture_guard (fixation/common-direction) của kết quả chính.
mirror_preview chạy classifier trên ảnh gốc rồi ảnh lật ngang.

Trên ảnh thứ ba, kết quả chính bị capture_guard chặn, classifier_called=false;
forward hook vẫn ghi nhận fusion chạy hai lần cho preview, tổng nhánh preview khoảng
453 ms. Service chỉ lấy record.screening, không trả record.model_preview ra JSON.
Nếu ảnh vượt gate chính và cả preview chạy đủ, classifier có thể forward ba lần.

Ưu tiên xem xét loại/tắt preview trong production. Phải giữ nguyên kết quả screening
chính và đối chiếu regression; không dùng điểm preview để thay kết quả đã bị QC chặn.
Không thay đổi này trong lượt phân tích.

### Seed được nạp nhưng không được gọi

self.seed chỉ xuất hiện ở phần dựng/nạp/eval. Ba ảnh đều có zero forward cho seed.
Bỏ riêng instance này có thể giảm khoảng 0,45 MiB state, mức tiết kiệm nhỏ.
Không được xóa checkpoint INITIAL vì code dựng nhánh iris vẫn dùng nó.

### Heatmap phục vụ trực quan vẫn được tạo

decode() và predict_direct_pupil() tạo PIL heatmap ở cỡ crop gốc. Engine nhận maps,
pupil_maps và search_mask nhưng không xuất ảnh trực quan qua API. Có thể thêm chế độ
không tạo ảnh trực quan, giữ các phép tính cần cho tọa độ và metadata. Cần kiểm tra
kết quả trước–sau; chưa đo số MB tiết kiệm cụ thể.

### FaceMesh được dựng lại theo từng request

Mỗi lần process tạo FaceMesh mới trong context manager. Có thể thử tái sử dụng một
instance dưới khóa inference sẵn có; cần kiểm tra lifecycle và tính lặp lại đầu ra.
Chưa có phép đo riêng để cam kết phần RAM/thời gian có thể giảm.

## 5. Điều kiện ảnh hưởng độ tin cậy

Metadata khai báo independent_evaluation=false,
updated_pupil_classifier_pair_independently_evaluated=false và
calibrated_probabilities=false. Sai số pupil_train_mean_px ~1,87 là thống kê train,
không chứng minh sai số trên ảnh mới. Visibility của pupil có warning unreliable;
các ngưỡng QC/capture hiện được ghi là nghiên cứu, chưa kiểm định lâm sàng.

Ảnh có SHA khớp bộ ROI đóng băng và ảnh mới đi qua bước chọn ROI khác nhau. Cần đánh
giá riêng ảnh mới, tỷ lệ được chấp nhận/INCONCLUSIVE và lỗi localization trước khi
kết luận chất lượng triển khai. Không nới gate chỉ để tăng số ảnh có phần trăm.

## 6. Thứ tự tối ưu ít tác động đến kết quả

1. Tắt preview không được API sử dụng; đối chiếu screening chính trước–sau.
2. Không sinh heatmap/PIL trực quan trong chế độ API.
3. Bỏ instance seed không sử dụng, giữ checkpoint còn phục vụ iris.
4. Đo thử reuse FaceMesh, giữ static_image_mode và khóa xử lý.
5. Đo lại trên Linux/Render với ảnh được chấp nhận và ảnh độ phân giải cao.

Chưa đề xuất resize nhỏ hơn, đổi backbone hoặc quantize trong bước đầu vì các thay
đổi này cần đánh giá lại đầu ra/độ chính xác. Chưa thể cam kết đạt giới hạn 512 MB.
