# Hai workspace AI theo phương pháp

- `hirschberg/`: bản copy model, dataset ảnh, annotation, feature, tài liệu và source snapshot Hirschberg.
- `cover_test/`: bản copy video mẫu, chuỗi thời gian Korean, audit và source snapshot nghiên cứu Cover Test.
- `copy_inventory.json`: ánh xạ file gốc → bản copy, kích thước và SHA-256; mọi bản copy được kiểm tra bằng hash.

Đây là bản copy trong cùng repository, không phải hai repository độc lập. API, cấu hình runtime, dữ liệu gốc và checkpoint gốc không đổi. Không chạy training trong lúc tổ chức thư mục.

Đọc README của từng nhánh trước khi sử dụng. Các file `source_snapshot/` giữ nguyên code để tham khảo; chạy entrypoint hiện có từ repository root theo các lệnh đã ghi trong README, không chạy trực tiếp snapshot vì đường dẫn tương đối vẫn theo cấu trúc gốc.

Manifest, annotation CSV, báo cáo lịch sử và metadata checkpoint được copy nguyên byte để bảo toàn provenance. Các đường dẫn bên trong chúng có thể vẫn trỏ về vị trí gốc (vẫn tồn tại); không coi workspace này là bản export có thể di chuyển độc lập.

Các dataset, checkpoint và snapshot bản sao chỉ lưu local, được loại khỏi Git để tránh commit trùng dữ liệu lớn. Lệnh `scripts/copy_ai_workspaces.py` đã dùng để tạo bản copy; nó từ chối ghi đè khi `ai/` tồn tại.
