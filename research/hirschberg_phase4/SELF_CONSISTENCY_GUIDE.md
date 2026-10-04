# Gán lại tọa độ để đo độ nhất quán

Đây là phép đo **độ lặp lại của cùng một người gán**, không phải kiểm định detector
hay độ đúng so với bác sĩ. Nên đợi vài ngày sau lần gán cũ trước khi mở ảnh.

## Mẫu đã chọn

20/62 ảnh, lấy ngẫu nhiên có seed `20261004`, phân tầng theo nhãn: eso 11, exo 6,
normal 3. Manifest không chứa tọa độ cũ. UI chỉ đọc ảnh và manifest này.

## Cách gán

Từ repository root, chạy:

```powershell
.\.venv312\Scripts\python.exe -m research.hirschberg_phase4.annotate_self_consistency --manifest research/hirschberg_phase4/runs/self_consistency_20261004/sample.json --root . --output research/hirschberg_phase4/runs/self_consistency_20261004/reannotation.json
```

Trong cửa sổ ảnh, chọn điểm bằng `1` OD pupil, `2` OD reflex, `3` OS pupil,
`4` OS reflex; click để đặt hoặc thay điểm đang chọn. `S` lưu ảnh hiện tại khi đủ
bốn điểm, `N` bỏ qua ảnh này, `Q` thoát. Có thể chạy lại để tiếp tục ảnh chưa lưu.
Không mở CSV gốc trong lúc gán; giao diện không tải và không hiện tọa độ cũ.

## So sánh sau khi gán

```powershell
.\.venv312\Scripts\python.exe -m research.hirschberg_phase4.compare_self_consistency --csv processed/hirschberg_manual_annotations.csv --sample research/hirschberg_phase4/runs/self_consistency_20261004/sample.json --annotations research/hirschberg_phase4/runs/self_consistency_20261004/reannotation.json --output research/hirschberg_phase4/runs/self_consistency_20261004/comparison.json
```

Báo cáo tách pupil/reflex, cho mean/median/P90 theo pixel và theo tỷ lệ khoảng
cách hai pupil của tọa độ cũ; điểm thiếu được đếm là thiếu, không đổi thành 0.
Ảnh nguồn ở đây là crop cũ: pixel không phải tọa độ ảnh gốc độ phân giải đầy đủ.
Nếu độ lệch lặp lại vượt khoảng 3 px, bộ này không phù hợp làm chuẩn để tuyên bố
detector đạt mức 3 px. Kết luận chỉ nói về tính nhất quán với chính lần gán cũ.
