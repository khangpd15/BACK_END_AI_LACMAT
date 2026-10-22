# Hirschberg: điều kiện phân loại và thông báo kết quả

## Thay đổi ngày 22/10/2026

BE trước đây chặn classifier nếu tích vô hướng hai vector pupil–reflex dương
(`same_half_plane`). Điều kiện này chỉ mô tả hướng, không xét đầy đủ độ lớn
hay sự tương xứng của phản quang. Đã bỏ việc chặn riêng theo điều kiện này:
ảnh đạt input contract, QC landmark hai mắt và fixation được xác nhận sẽ đi
tiếp vào classifier. Không thêm quy tắc tự gán NORMAL từ hướng vector.

Capture policy được ghi là `research_capture_v2` trong manifest. Importer áp
dụng cùng thay đổi để không khôi phục cách chặn cũ khi đồng bộ lại nguồn train.
Đã tạo lại hash bundle; trọng số, preprocessing và ngưỡng QC không thay đổi.

Hirschberg đánh giá sự tương xứng của phản quang hai mắt; riêng cờ cùng hướng
không phải tiêu chuẩn xác định bình thường hay bất thường. Tham khảo:
https://morancore.utah.edu/basic-ophthalmology-review/alignment-assessment-hirschberg/

## Nội dung người dùng thấy

BE trả `message` ngắn, tách khỏi mã lý do và thông số kiểm tra chất lượng.
FE trong `D:\AI_Check_Lac` sử dụng thông báo kiểm soát, kể cả khi BE cũ vẫn gửi
đoạn mô tả kỹ thuật. Kết quả chưa đánh giá được chỉ có tiêu đề và hướng dẫn
chụp lại; không có badge lặp tiêu đề, phần trăm hoặc khung báo thiếu điểm.
Trang tổng kết cũng không hiện nguyên mã reason Hirschberg.

Kết quả có phân loại tiếp tục hiển thị gợi ý và điểm mô hình như trước. Cùng
hướng không bảo đảm kết quả NORMAL: classifier vẫn có thể đề xuất cả ba lớp.
Đây là sửa điều kiện chạy và cách hiển thị; chưa có đánh giá cohort độc lập
để tuyên bố độ chính xác cao hơn. Cần cập nhật cả BE và FE khi deploy.

## Xác minh

- Hash bundle đạt: 40 file.
- Test BE bao gồm cùng hướng với cả ba lớp, thiếu fixation/geometry, QC không
  đạt, thông báo đơn giản, nạp checkpoint thật và API.
- Test hợp đồng FE bao gồm ẩn điểm khi INCONCLUSIVE và không lộ đoạn kỹ thuật
  từ BE; frontend production build đạt.
- Không commit, push hoặc deploy trong lần chỉnh sửa này.
