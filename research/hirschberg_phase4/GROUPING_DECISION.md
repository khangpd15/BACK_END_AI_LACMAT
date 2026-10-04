# Quyết định nhóm dữ liệu khi thiếu mã bệnh nhân

Ngày: 2026-10-04. Cơ sở: người dùng xác nhận nhãn bệnh đã được bác sĩ kiểm tra,
ảnh được phép sử dụng và có đồng thuận; hiện không có bảng nối ảnh với mã bệnh
nhân. Chỉ dùng cho nghiên cứu phi thương mại. Không chẩn đoán.

## Quyết định

**Không suy mã bệnh nhân từ tên file hoặc độ giống ảnh. Không tạo patient ID giả.**
Một ảnh khác góc/ánh sáng vẫn có thể cùng bệnh nhân; ảnh gần trùng có thể là
hai frame/crop từ cùng lần chụp. Ngược lại, ảnh giống nhau không chứng minh cùng
người. Vì thế dữ liệu hiện tại chưa đủ để chia train/validation/test theo bệnh nhân.

Mã `group_id` trong feature CSV hiện có là nhóm near-duplicate tạo bằng hash ảnh,
không phải danh tính. Audit ghi nhận 136 accepted crop có 135 nhóm kiểu này:
134 nhóm một ảnh và một nhóm hai ảnh. Pedseye có nhóm near-duplicate suy ra từ
ảnh; audit trước đã thấy 4 cặp ứng viên qua split cũ. Các con số này giúp chặn
bản trùng, nhưng **không giải quyết được việc cùng bệnh nhân ở nhiều split**.

Đã chạy sensitivity OOF 5-fold trên 62 annotation, ghép nhóm hash bảo thủ ngưỡng
4. Có 59 nhóm cho 62 ảnh; fold validation có 11-13 ảnh. Hai ngưỡng rộng hơn cho
nhóm bị gộp mạnh: ở ngưỡng 8, có fold validation chỉ còn 1 nhóm; ngưỡng 12 còn
10 nhóm cho cả cohort và một fold chứa 52/62 ảnh. Vì vậy không chọn các kết quả
ngưỡng rộng làm số chính. Ngay ngưỡng 4 vẫn chỉ là group-held-out trên ảnh giống,
không chứng minh độc lập bệnh nhân. Nó được dùng như thăm dò feature, không phải
để quyết định chất lượng lâm sàng.

Ở ngưỡng 4, balanced accuracy ba lớp là 0,716, macro-F1 0,694; recall eso 0,758,
exo 0,789, normal 0,600; dummy majority BA 0,333. Confusion theo hàng/cột
eso, exo, normal: `[[25,3,5],[0,15,4],[2,2,6]]`. Một mô hình khác với ảnh/nhóm
ở ngưỡng 8 cho BA 0,828; đây là bằng chứng kết quả nhạy với clustering. Feature
lấy từ bốn tọa độ pupil/reflex cũ do người gán, chuẩn hóa bằng khoảng cách hai
pupil; không phải output detector, không dùng iris diameter, không phải mm.
Không có pseudo. Các giá trị này **không phải test accuracy hay hiệu quả lâm sàng**.

## Quy tắc dùng từ bây giờ

- Mọi ảnh cũ chỉ được dùng để phát triển/kiểm tra code; không có điểm số nào được
  gọi là patient-independent validation hoặc test.
- Hash chính xác và near-duplicate chỉ dùng để gắn cờ/review, không xác nhận bệnh
  nhân. Không đưa `group_id` này vào `patient_id`.
- Không train/evaluate A0 bốn lớp: dữ liệu hiện thiếu ảnh pseudostrabismus thật;
  annotation cũ thiếu limbus center/diameter cần cho feature A0. Hai điều này
  không được giải quyết bằng việc có nhãn bệnh đã được bác sĩ xác nhận.
- Có thể chạy thử detector trên ảnh crop cũ như exploratory development, nhưng
  sai số landmark đối chiếu với tọa độ cũ chỉ là agreement với annotator cũ,
  không phải ground truth bác sĩ.
- Khi có bảng nối pseudonymous image ID -> patient ID được giữ ở nơi riêng tư,
  nạp mã nhất quán, gộp mọi visit/ảnh/crop của cùng người rồi mới khóa patient
  split. Không đưa tên hoặc thông tin trực tiếp nhận dạng vào repo.

Kết quả/tham số và các cạnh hash ứng viên được ghi trong `runs/exploratory_primary4_20261004/`
(thư mục local bị ignore). Kết quả threshold 8/12 giữ để thấy sensitivity, không
chọn kết quả cao nhất rồi quảng bá.

Để chạy lại phân tích thăm dò với output mới, dùng từ repository root:

```powershell
.\.venv312\Scripts\python.exe -B -m research.hirschberg_phase4.exploratory --root . --output-dir research/hirschberg_phase4/runs/another_exploratory_run
```

## Mức thông tin đã xác nhận

Xác nhận của người dùng cho phép đánh dấu **nhãn bệnh** trong phạm vi dữ liệu họ
cung cấp và quyền/đồng thuận dùng ảnh cho nghiên cứu. Xác nhận đó không nói bác sĩ
đã kiểm tra tọa độ pupil/reflex, không tạo mã bệnh nhân, không xác nhận license
của dataset bên ngoài và không cho phép phát hành ảnh công khai.

Importer giữ các khác biệt này: nhãn bệnh owner-reported physician-confirmed;
landmark chưa xác nhận; ảnh cũ ở coordinate space `legacy_crop`; patient ID
unknown; `training_ready=false`. Đây là trạng thái trung thực, không phải lỗi cần
bỏ qua để chạy model.
