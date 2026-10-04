# Dự thảo protocol chụp và gán nhãn

Ngày: 2026-10-04. **Nghiên cứu sàng lọc, không chẩn đoán hoặc thay thế khám mắt.** Mọi con số pilot dưới đây là đề xuất kỹ thuật để phòng khám xem xét, không tiêu chuẩn lâm sàng đã được xác nhận. Chỉ bắt đầu sau approval, consent và thỏa thuận dữ liệu.

## 1. Trước thu thập

1. Chốt investigator, phòng khám, mục đích nghiên cứu/thương mại, đề cương và người giữ khóa định danh. Phòng khám duyệt an toàn flash, tuổi, tiêu chí loại trừ và quy trình dừng.
2. Có consent người lớn hoặc người giám hộ hợp lệ; assent trẻ khi phù hợp tuổi/năng lực theo quy định cơ sở. Không thu nếu trẻ phản đối/khó chịu; điều trị không phụ thuộc tham gia.
3. Tạo patient_id ngẫu nhiên, visit_id, image_id. Một người có nhiều lần khám vẫn cùng nhóm split. Khóa tra danh tính giữ ở phòng khám, không trong Git, không gửi lên Render.
4. Ghi consent scope/version, source_id, owner, permission record, withdrawal status. Không xác minh đủ -> quarantine, không train/test.

## 2. Setup ảnh native

Chốt một camera/lens/zoom/resolution trong pilot; hiệu chuẩn distortion riêng cấu hình đó. Tắt beauty filter, portrait mode, sửa red-eye tự động và digital zoom; ghi HDR/computational processing vì không phải máy nào cho tắt hoàn toàn. Không dùng ảnh qua chat đã nén. Giữ JPEG/HEIC gốc, nếu RAW khả dụng lưu thêm; không tuyên bố JPEG native là sensor RAW.

Đề xuất pilot: rear camera cố định, người ngồi thẳng, ánh sáng phòng ổn định; mục tiêu fixation gần trục camera/nguồn flash có vị trí ghi rõ. Bác sĩ chọn khoảng cách đo near fixation và giữ không đổi; có thể thử setup 40 cm nếu phù hợp khám, **không coi 40 cm là ngưỡng hiệu chuẩn phổ quát**. Ghi khoảng cách thực và offset flash-camera, không đoán bằng phần mềm.

Giữ nguyên mặt/hai mắt trong ảnh để lưu canthi/nếp da; thêm crop mắt native không rescale, liên kết ảnh gốc. Nếu scope consent chỉ cho vùng mắt, capture phù hợp và ghi thiếu full-face, không tự mở rộng consent. Mục tiêu kỹ thuật pilot iris >=150 px đường kính trên ảnh native để kiểm tra khả năng định vị; đây không phải bảo đảm <3 px. Nếu máy/khoảng cách không đạt, chỉnh setup đã được bác sĩ duyệt, không dí flash sát mắt.

Thu tối đa 3 ảnh frontal cho mỗi điều kiện đã duyệt, nghỉ/dừng theo phản ứng người tham gia. Không burst flash kéo dài để lấy số lượng. Bác sĩ quyết định chụp thêm không flash hoặc cover-condition/video có cần thiết và an toàn không. Không hướng dẫn bệnh nhân tự làm calibration hay cover test tại nhà.

Checklist từng ảnh: có focus rõ iris/limbus, hai mắt mở đủ, không motion blur, có glint hợp lệ, không bão hòa toàn vùng cần fit, gaze/pose đúng điều kiện. Nếu không đạt ghi lý do và `not_measurable`; không xóa im lặng, không gán điểm giả. Tránh thêm đèn tùy ý gây nhiều glint không biết nguồn.

Ghi metadata: thiết bị/OS/app, lens, native width/height, orientation, zoom, exposure/ISO nếu có, flash mode, distance, fixation target, head pose đo/quan sát, kính/tiếp xúc theo bác sĩ quyết định, điều kiện ánh sáng và số lần chụp. Tách EXIF định danh/GPS khỏi bản làm việc; giữ mapping orientation/crop để đổi tọa độ đúng.

## 3. Nhãn lâm sàng khác nhãn ảnh

Bác sĩ nhãn khoa xác nhận eso/exo/pseudo/normal từ khám thích hợp, ghi method, near/distance, fixation/correction, manifest/intermittent, laterality, magnitude PD nếu đã đo, thời điểm khám và mức chắc chắn. Cover-uncover/alternate prism cover khi khả thi do người được đào tạo thực hiện; các khám khác do bác sĩ quyết định. Không cưỡng ép phương pháp ở trẻ không hợp tác.

Ảnh frontal có thể không thể hiện lác intermittent tại thời điểm chụp. Vì vậy lưu riêng `clinical_diagnosis` và `image_manifest_alignment`; không ép chúng trùng nhau, cũng không biến ảnh có mắt thẳng thành normal khi hồ sơ bệnh có lác. Ngoài 4 lớp lưu `other_strabismus`, `mixed`, `indeterminate` để không ép bệnh ngoài phạm vi thành normal.

Pseudo đòi hỏi đánh giá lâm sàng không thấy lác thật ở điều kiện khám và ghi hình thái gây vẻ lác; không chỉ có epicanthal fold. Ghi giới hạn và follow-up do bác sĩ chỉ định. Normal không phải nhãn bảo đảm sức khỏe mắt toàn diện.

## 4. Hai người gán độc lập

- Huấn luyện rater bằng bộ 20-30 ảnh development có quyền, thống nhất OD/OS, canthus thật, pupil/limbus khác nhau, glint flash khác tear highlight. Ví dụ số lượng này là pilot, không tính vào test.
- Rater A và B gán trên ảnh native với zoom hiển thị; không xem nhãn người kia, model output hay nhãn thư mục. Khóa bản A/B trước hòa giải; giữ tọa độ float native, timestamp và version.
- Mỗi mắt: pupil boundary/ellipse + center; limbus boundary/ellipse + center/diameter; glint candidate, selected source, saturated/ambiguous; medial/lateral canthus; epicanthal fold boundary/extent nếu nhìn thấy. Trạng thái `visible`, `occluded`, `uncertain`, `absent`, không thay bằng (0,0).
- Tâm ellipse suy ra từ boundary khi che một phần cần đánh dấu inferred và confidence; không gọi tâm đó là điểm nhìn thấy trực tiếp. Glint bão hòa lớn phải có quy tắc xác định tâm/uncertainty, không chọn pixel trắng ngẫu nhiên.
- Nhãn lớp: hai bác sĩ độc lập xem hồ sơ khám theo cùng tiêu chí; kỹ thuật viên không thay người xác nhận chẩn đoán. Khi chỉ có một bác sĩ, ghi đúng giới hạn: chưa đo clinical inter-doctor agreement.
- Người thứ ba/bác sĩ adjudicate bất đồng, giữ cả A/B và nhãn cuối. Đồng thuận sau trao đổi không thay số liệu agreement **trước** trao đổi.

## 5. Đo đồng thuận

| Target | Báo cáo |
|---|---|
| Bốn lớp lâm sàng | Confusion A-vs-B, % agreement, **Cohen kappa không weighted** vì lớp không có thứ tự; CI95% bootstrap patient-level; mỗi lớp và số indeterminate |
| Landmark | Euclidean distance A-vs-B: mean, median, P90/P95, riêng OD/OS và target; native px + normalized iris diameter. Báo signed bias x/y; không trộn missing vào zero |
| Boundary | Dice/IoU và boundary distance cùng center error; Dice cao không chứng minh tâm đúng <3 px |
| WTW/ICD/PD đo liên tục | Bias + Bland-Altman limits of agreement, repeatability; CI theo người. ICC absolute agreement nếu tính phải nêu two-way random/fixed, single/average measure, không chỉ ghi “ICC” |
| Epicanthal visibility/mức độ | Agreement cho nominal/ordinal đúng định nghĩa; separate không đánh giá được, không bỏ khó |

Kappa được định nghĩa trong [scikit-learn docs](https://sklearn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html). Kappa bị ảnh hưởng tần suất lớp; báo raw agreement và ma trận, không dùng một số duy nhất. ICC trên tọa độ thô có thể cao do ROI lớn dù sai nhiều px, nên không dùng thay Euclidean error.

**Mốc pilot đề xuất, không tiêu chuẩn công nhận:** rater landmark P90 <=2 px trên ảnh đo được và clinical kappa khoảng >=0,8 kèm CI để xem xét tiếp; nếu không đạt, sửa định nghĩa/đào tạo và đo lại. Đạt mốc không chứng minh nhãn luôn đúng. Trường hợp hình ảnh không đủ xác định <3 px thì ghi bất định, không ép rater đồng ý.

## 6. Đánh giá detector và classifier sau này

Không dùng test trong pilot cải tiến. Split bệnh nhân trước augmentation; mắt trái/phải, mọi visit/burst và crop của người đó nằm cùng split. Gold test do rater độc lập và bác sĩ adjudicate, không dùng nhãn sinh tự động hoặc dự đoán của model làm chuẩn.

Detector: mục tiêu P90 <=3 px native, riêng glint/pupil/limbus; report CI95%, successful measurement coverage, candidate-ID error và tail failure. Kèm tỷ lệ <=3 px trên ảnh đủ nhãn, và thành công end-to-end trên toàn tập (missing/failure không được biến mất). Stratify theo blur/occlusion, iris size, camera, lớp và độ tuổi; không âm thầm loại frame khó.

Classifier: patient-level recall từng lớp, specificity/precision/F1, balanced accuracy/macroF1, confusion có cột abstain. Báo cả toàn mẫu và mẫu covered; indeterminate clinical không dùng làm label chắc chắn nhưng phải báo count/flow. Ngưỡng chọn trên development, private test giữ kín. Với vài trăm người, ưu tiên classifier nhỏ và ablation cùng cohort, chưa có cam kết accuracy.

## 7. Hồ sơ tối thiểu, không chứa tên

```text
patient_id, visit_id, image_id, source_id, consent_scope_id, rights_status
native_width, native_height, orientation_transform, crop_transform
camera_config_id, distance_cm, fixation_condition, capture_time
clinical_diagnosis_A, clinical_diagnosis_B, adjudicated_diagnosis
exam_method, clinical_certainty, image_manifest_alignment, reviewer_id
landmarks_A, landmarks_B, adjudicated_landmarks, visibility, uncertainty
WTW_OD_mm, WTW_OS_mm, measurement_device, repeatability
split_id, annotation_version, withdrawal_status, quality_flags
```

Bản raw/khóa nhận dạng chỉ nằm trong hệ thống được duyệt, mã hóa và phân quyền, không public cloud ngoài thỏa thuận. Consent không đặt chung dataset gửi kỹ sư. Ghi retention và quyền rút lui theo mẫu được cơ sở duyệt. Dataset/model version phải truy nguyên được các scope cho phép.
