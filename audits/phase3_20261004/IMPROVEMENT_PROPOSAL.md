# PHASE 3: Đề xuất cải tiến Hirschberg

Ngày: 2026-10-04. Chủ dự án xác nhận **nghiên cứu phi thương mại**, cho phép PHASE 3. Chưa chọn phương án PHASE 4.

**Công cụ hỗ trợ nghiên cứu sàng lọc, không chẩn đoán và không thay thế khám mắt.** Các mức dữ liệu, thời gian và tiêu chí dưới đây là đề xuất thiết kế, không phải kết quả thực nghiệm hoặc tiêu chuẩn công nhận lâm sàng. Chỉ tạo tài liệu mới; không tải ảnh/weights, train, sửa hệ thống, commit, push hay deploy.

Tài liệu nền: [audit PHASE 1](../phase1_20261004/AUDIT_REPORT.md), [nguồn PHASE 2](../phase2_20261004/LITERATURE_AND_DATASETS.md), [protocol](../phase2_20261004/CAPTURE_AND_ANNOTATION_PROTOCOL.md), [consent dự thảo](../phase2_20261004/CONSENT_TEMPLATE.md), [email dự thảo](../phase2_20261004/CLINIC_CONTACT_EMAIL.md).

## 1. Quyết định khuyến nghị

**Chọn A, bắt đầu A0: gold landmarks + classifier nhỏ, sau đó A1: detector cổ điển native-resolution.** Chỉ thêm keypoint model A2 nếu lỗi A1 trên development cho thấy cần. Thử C ở bước pretrain định vị khi quyền nguồn đã rõ. B chỉ là đối chứng tiếp theo, không phải thay toàn hệ thống ngay.

Lý do từ audit: 62 ảnh đã có tọa độ hợp lệ; glint detector lệch annotation P90 37,22 px, nhưng chưa xác minh gold; dữ liệu không có pseudo; con số v0.6 94,61% chứa train và bỏ abstain. Vì vậy chưa có căn cứ chọn CNN lớn hoặc hứa tăng accuracy. Repo đã có pipeline hình học và script oracle để tham khảo, không cần viết lại mọi thứ.

**Không thể tăng độ chính xác bằng cách đo đẹp hơn trên nhãn chưa chuẩn.** Thứ tự: quyền dữ liệu/patient ID -> gold lâm sàng và landmarks -> đối chứng cùng cohort -> sửa detector/capture -> thêm bệnh nhân đúng lớp -> cân nhắc model lớn hơn. Nếu gold geometry cũng phân biệt kém, phải sửa target/features/capture trước khi đổ công vào detector.

## 2. So sánh ba phương án

| Tiêu chí | A. Hình học hai tầng + classifier nhỏ | B. CNN pretrained + multi-task | C. Tổng hợp để pretrain/augment |
|---|---|---|---|
| Vai trò | Baseline chính, khuyến nghị | Đối chứng sau A, có thể khám phá feature hình ảnh | Bổ trợ A/B, không baseline lâm sàng độc lập |
| Cấu trúc | ROI -> pupil/limbus/glint/canthi -> feature -> logistic regression; boosted tree chỉ nếu cần | Encoder ảnh -> heatmap/boundary + class; hai crop mắt native và context quanh mắt | Gaussian glint hoặc render mắt có camera/đèn kiểm soát -> pretrain detector -> fine-tune real |
| Độ phức tạp | Thấp ở A0/A1; vừa khi thêm A2 | Cao: GPU, preprocessing, loss, weights, export | Thấp cho Gaussian; cao cho render quang học/epicanthal |
| Dữ liệu khởi động đề xuất | 40-60 bệnh nhân development có gold để kiểm tra detector; thử classifier khi đủ lớp và nhóm | Cùng cohort A để so công bằng; chỉ train class head khi có clinical labels đủ lớp | Synthetic tùy sinh; vẫn cần real gold development và real test độc lập |
| Cohort nghiên cứu 4 lớp dự kiến | 200-400 bệnh nhân, hướng 300; không bảo đảm đủ hay clinical validation | Vài trăm chỉ exploratory; thêm ảnh cùng người không khắc phục thiếu nhóm độc lập | Không tính synthetic thành bệnh nhân, không cộng vào cỡ test |
| Mức chính xác kỳ vọng | Chưa có % đáng tin. Mục tiêu kiểm chứng geometry P90 <=3 px native và vượt baseline cùng split | Không có căn cứ hứa tốt hơn A với dữ liệu này; có thể overfit | Có thể giảm landmark error; phải chứng minh real-only vs real+synthetic |
| Giải thích | Cao: ảnh overlay, tọa độ, vector lệch, uncertainty | Trung bình: có keypoint output; saliency không giải thích nhân quả/chẩn đoán | Tốt cho biết tham số sinh, nhưng độ giống thật không chứng minh nhãn lâm sàng |
| Chi phí compute | CPU cho CV/classifier; GPU chỉ khi A2 | GPU train; runtime phải đo, không suy từ dung lượng model | Gaussian rẻ; render tốn setup/storage |
| Rủi ro chính | Sai glint/ellipse/canthi; calibration chưa đúng; morphology không đủ pseudo | Shortcut theo camera/tuổi/nền; loss class làm sai geometry; lệch train-runtime | Domain gap, mô phỏng sai optics; classifier học dấu vết chỉnh ảnh |

Số người trên là ngân sách pilot đề xuất, **không ngưỡng khoa học tối thiểu đã được chứng minh**. Không có dữ liệu lâm sàng/licensed thì chỉ làm hạ tầng, fixture synthetic và thử định vị theo quyền được duyệt; không đánh giá classifier bệnh.

Ước lượng công sức kỹ thuật riêng sau khi dữ liệu đủ, 1 kỹ sư: A0/A1 khoảng 8-12 ngày làm việc; A2 thêm 5-10; B thêm 10-20; C Gaussian thêm 2-4, render thêm 10-20. Đây là ước lượng sơ bộ, không cam kết tiến độ; không gồm tuyển bệnh nhân, bác sĩ, ethics/permission, gán nhãn hoặc sửa dữ liệu. Chưa có báo giá phòng khám/GPU, nên không bịa chi phí tiền.

## 3. A: pipeline hình học, triển khai theo mức cần thiết

### A0. Kiểm tra trần khả năng của feature trên gold

Chỉ dùng dữ liệu đã có quyền + patient ID và nhãn lâm sàng. Lấy tọa độ gold riêng pupil/limbus/glint/canthi, tính feature rồi logistic regression regularized. “Oracle” ở đây nghĩa là dùng landmarks do người xác nhận, **không phải độ chính xác tối đa tuyệt đối** vì rater và feature vẫn có sai số.

Trên cùng validation fold, thay gold bằng output detector và giữ label/split/classifier-family/grid không đổi. Có hai đối chiếu: (1) classifier train+eval gold so train+eval detector; (2) giữ classifier gold cố định, thay input gold/detector để thấy tác động sai số định vị. Báo rõ đối chiếu (2) có distribution shift, không coi mọi chênh lệch là causal proof. Fit mọi preprocessing trên train fold.

Nếu cohort chưa đủ pseudo: làm thử định vị và classifier **chỉ những lớp thật có gold**, ghi rõ không phải 4 lớp. Không thêm pseudo giả hoặc đánh giá chỉ số pseudo bằng zero mẫu. Với bệnh nhân thiếu liên kết được xác minh, loại khỏi đánh giá patient-generalization, giữ lịch sử audit không xóa dữ liệu.

### A1. Detector rẻ nhất

1. ROI mặt/mắt theo helper repo nếu vượt kiểm tra framing; nếu không, thử landmark library có quyền rõ. ROI chỉ định vị vùng, không coi face-mesh iris là gold pupil/glint. Lưu transform crop/rotation/resize, quy ước OD/OS và mirroring.
2. Crop native-resolution. Boundary/ellipse cho pupil và limbus riêng, fit robust khi đủ nhìn thấy; reject nếu không đủ. Không dùng tâm mask bị mi che như tâm toàn iris.
3. Sinh và xếp hạng glint candidates trong ROI phù hợp, phân biệt flash với tear/mi/đèn. Sau chọn đúng candidate mới refine Gaussian/centroid có kiểm tra saturation. Không fallback bằng vị trí giả hoặc lấy sáng nhất vô điều kiện.
4. Canthi thật + visibility/uncertainty. Ban đầu canthi/epicanthal lấy annotation gold cho ablation; tự động hóa sau khi feature chứng minh hữu ích. Không suy ICD từ pupil distance.
5. Xuất overlay kèm quality flags và tọa độ, không ép class khi geometry lỗi. Reason codes đề xuất: `NO_VALID_GLINT`, `AMBIGUOUS_GLINT`, `IRIS_OCCLUDED`, `POSE_OUT_OF_SCOPE`, `INSUFFICIENT_CONTEXT`, `UNCERTAIN_CLASS`.

### A2. Keypoint/segmentation nhỏ nếu A1 chưa đủ

Heatmap là bản đồ điểm có khả năng nằm ở đâu; boundary head giúp fit ellipse. Model riêng glint và pupil/limbus có thể nhẹ hơn CNN phân loại toàn ảnh; loss chỉ dùng target có nhãn/visibility, không ép occluded thành (0,0). Chọn theo lỗi development, không tự tăng model size.

[EllSeg](https://github.com/RSKothari/EllSeg) là tham khảo ellipse/segmentation; [Byrne CR CNN](https://github.com/dcnieho/Byrneetal_CR_CNN) là tham khảo refine và sinh glint. Phải kiểm tra quyền weights/model và dependency trước dùng. Kết quả subpixel của nguồn khác không thay benchmark native RGB của dự án.

### Feature và classifier

| Nhóm feature | Cách tính/giới hạn |
|---|---|
| Geometry mỗi mắt | `(glint - limbus_center) / limbus_diameter`; giữ pupil-relative feature riêng để ablation |
| So sánh hai mắt | Chuyển về cùng quy ước camera và nasal/temporal; tính binocular horizontal/vertical difference đã định nghĩa rõ, không cộng/trừ theo sign tùy tiện |
| Hình thái | ICD/OCD, ICD/PFL, tỷ lệ sclera nasal/temporal, fold extent; chỉ nếu canthus/mi/gaze đủ quan sát |
| Chất lượng | Missing/visibility, saturation, ellipse residual, candidate ambiguity; ưu tiên quality gate, kiểm tra không dùng capture artifact làm bệnh-label shortcut |
| Physical scale | WTW cá nhân/camera calibration nếu đã xác nhận; thiếu thì giữ dimensionless, không dùng 11,8 mm |

Logistic regression là mô hình nhỏ học trọng số feature; baseline dễ debug. So thêm luật geometry chỉ với threshold học trên development và được bác sĩ xem xét; không bịa medical cutoff. XGBoost/boosted tree không mặc định cần thiết, chỉ thử nếu baseline lộ tương tác phi tuyến và đủ mẫu.

Đề xuất primary task đầu tiên: `manifest horizontal misalignment` vs `not_manifest_in_capture`, theo nhãn ảnh được bác sĩ đối chiếu. Secondary 4-class clinical screening cần target và biểu hiện phù hợp; không coi ảnh thẳng của intermittent exo là normal. Giữ `clinical_diagnosis` riêng `image_manifest_alignment`.

Pseudo không chỉ là “glint thẳng + có fold”. Nếp rẻ quạt có thể tạo vẻ lác nhưng cần khám phân biệt; [AAPOS](https://aapos.org/glossary/pseudostrabismus) là cơ sở thận trọng. Khi thiếu context hoặc thiếu bằng chứng, trả uncertain thay vì khẳng định pseudo/normal.

## 4. B: pretrained CNN + multi-task, đối chứng có kiểm soát

Đề xuất thử một backbone nhỏ như ResNet18, không thay bằng model rất lớn. Dùng [Torchvision weights API](https://docs.pytorch.org/vision/stable/models.html) nếu quyền weights thích hợp được xác minh; lưu enum/version/hash. Không dùng `pretrained=False` ngẫu nhiên rồi gọi là transfer learning, không âm thầm fallback khi load lỗi.

Input gồm eye detail riêng từng mắt và bilateral/periocular context chứa canthi/fold. Không resize ảnh mặt thành 224 rồi trông chờ flash vài px còn đủ. Nhánh keypoint dùng ROI đủ chi tiết, xuất tọa độ với transform inverse về native; context branch có thể nhỏ hơn. Channel/normalization/OD-OS phải giống train/eval/runtime.

Head class + pupil/limbus/glint heatmap hoặc boundary. Loss đề xuất: class cross-entropy + masked landmark/boundary losses; weight các loss chọn trong inner development split. Freeze encoder trước, fine-tune một phần sau nếu validation ủng hộ. Khi class head tăng nhưng P90 landmark xấu, không gọi multi-task thành công.

Minibatch sampling và class weights chỉ từ train, không cho một người nhiều ảnh chi phối. Dùng cùng patient cohort/folds/nhãn/metric với A. Ablation: frozen embedding + linear head; class-only vs multi-task; detail-only vs thêm context. Nếu không vượt A hoặc CI chưa phân biệt, chọn A ít phức tạp hơn.

Augmentation: chỉ train, biến đổi geometry đúng cho mọi label/OD-OS; giới hạn xoay và brightness trong phạm vi capture thật. Không elastic warp, không dịch glint độc lập giữ nhãn bệnh, không blur phá glint rồi giữ label chắc chắn; horizontal flip chỉ nếu remap anatomy/sign đầy đủ, mặc định tắt. Không xóa/sửa red-eye ở vùng dùng đo.

## 5. C: dữ liệu tổng hợp, bổ trợ định vị

**C1 khuyến nghị:** Gaussian glint trên background tạo thuần synthetic, biến thiên noise/PSF/saturation/intensity/pixel phase, lưu tâm liên tục làm nhãn. Nếu dùng background ảnh người, ảnh đó vẫn phải có quyền và chỉ từ train groups. Thêm distractor để phân biệt candidate và đánh giá refine riêng; không chỉ sinh một chấm giữa nền tối dễ.

**C2 tùy chọn:** render mắt với camera intrinsics, flash offset và pose cụ thể; [UnityEyes2](https://github.com/alexanderdsmith/UnityEyes2) là đầu mối kỹ thuật. Chỉ dùng sau khi kiểm tra asset/texture/output license và annotation 2D. Không giả định 3D vị trí đèn trong JSON chính là 2D glint ground truth.

Mô phỏng chuyển glint/iris có thể tạo **bài toán detector synthetic**, nhưng không tự tạo bệnh nhân eso/exo/pseudo. Corneal reflection phụ thuộc optics, camera-light-eye geometry và angle-kappa; dịch chấm bằng Photoshop có thể tạo vật lý sai. Pseudo cần morphology/context/khám, không được sinh bằng giữ glint thẳng và nhãn pseudo.

Thử real-only vs synthetic-pretrain+real-finetune trên cùng real validation; target chưa có real labels thì ghi chưa đánh giá. Synthetic validation không tính clinical accuracy. Test chỉ ảnh thật độc lập, không tune generator theo private test. Split background và identity trước sinh; synthetic từ train không xuất hiện ở val/test.

Chọn C khi real validation landmark error/coverage tốt hơn mà không có subgroup deterioration đáng kể được xác định trước; nếu chỉ synthetic score tăng thì bỏ. Dùng C như nguồn pretrain, không đưa synthetic class examples vào test hay số bệnh nhân.

## 6. Tận dụng 62 annotation hiện có, không ghi đè

Nguồn đọc: [hirschberg_manual_annotations.csv](../../processed/hirschberg_manual_annotations.csv), gồm `relative_path`, `class_label`, pupil/reflex OD/OS x-y và notes. Đây là tọa độ trên file crop hiện có; **không phải tọa độ ảnh gốc high-resolution** và chưa có limbus/canthi/bác sĩ xác nhận.

Quy trình đề xuất PHASE 4:
1. Import bản chỉ đọc sang manifest/schema mới với annotation_origin=`legacy`, original_file_hash, image_width/height và coordinate_space=`legacy_crop`. Không sửa CSV gốc.
2. Chỉ dùng ảnh sau kiểm tra quyền, consent scope và patient link. Unknown -> quarantine; không train kể cả phi thương mại. Cùng bệnh nhân phải liên kết được, không đoán patient_id từ dHash.
3. Hai rater độc lập kiểm tra glint/pupil theo quy tắc mới, bác sĩ xác nhận nhãn bệnh; bổ sung limbus/canthi khi nhìn thấy. Lưu A/B/adjudicated/version. Nhãn thư mục không thành nhãn clinical tự động.
4. Có ảnh native + transform crop thật mới map annotation; thiếu thì không upscale để giả native precision. Reannotate ảnh native khi có quyền. Giữ 224x224 legacy thành domain phụ, không hòa P90 legacy với native.
5. Mẫu đã audit/tuning là development, không nâng thành private test mới. Có thể dùng để regression sau quyền được xác minh; tạo test prospectively từ người mới.

Annotation tool ban đầu ưu tiên helper đã có nếu lưu được tọa độ/metadata an toàn; nếu không đủ, chọn CVAT/Label Studio bản local sau kiểm tra docs/version/quyền và dữ liệu lưu. Không upload sang dịch vụ hosted mặc định. Chưa chọn/cài công cụ trong phase này.

## 7. Dữ liệu hợp pháp và quyền phi thương mại

Phi thương mại phù hợp để **xem xét** nguồn NC, không miễn consent, attribution, share-alike/DUA và điều khoản weights. Nếu mục đích đổi, phải xem lại trước tái sử dụng. Không tuyên bố mọi model train từ NC được phát hành tự do hoặc tự động mang một license cụ thể; cần xét từng nguồn/điều khoản.

Đề xuất bảng quyền cho mỗi source: owner/source URL, license exact version/file scope, allowed purposes, consent confirmation, patient/session fields, acquisition date, permitted storage, redistribution/model-release restrictions và trạng thái `approved`/`hold`/`denied`. Không lưu consent định danh trong Git.

LPW còn cần xác minh consent phù hợp; OpenEDS/Night Eyes cần DUA ảnh gốc; Toronto/Roboflow/figshare/flash portrait chưa xác minh vẫn HOLD theo PHASE 2. NC không biến các nguồn này thành approved. Không tải mirror, không scrape bệnh nhân, không gửi email khi chưa được chủ dự án ủy quyền.

Thu mới tại phòng khám: dùng protocol/consent/email PHASE 2 sau được cơ sở duyệt. Bác sĩ khám đối chiếu, hai người gán độc lập, ghi patient ID, camera/capture, WTW nếu có và trạng thái nhãn. Xin vài trăm người, không “vài trăm ảnh” như proxy số bệnh nhân. Tuyển có chủ đích pseudo/góc nhỏ/intermittent cho nghiên cứu; không lấy prevalence/binary PPV của cohort cân bằng áp sang dân số sàng lọc.

## 8. Split và đánh giá trung thực

Ví dụ ngân sách **300 người**, mục tiêu 75 người/lớp nếu tuyển được: 240 development, 60 private test mới, khoảng 15/lớp test. Đây là pilot enrich-class, không prevalence dân số. Không đủ bằng chứng chứng minh độ nhạy cao với CI hẹp; cần tính cỡ mẫu riêng trước nghiên cứu xác nhận lâm sàng.

Development: outer 5-fold patient-level khi từng lớp có đủ người trong mọi fold; inner 3-fold chọn hyperparameters/thresholds nếu tuning. Nếu ít nhóm, giảm folds và ghi lý do; không ép stratification hay tạo người giả. Dùng [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) với group patient_id, kiểm tra fold composition vì stratification chỉ cố gắng giữ tỷ lệ. Toàn bộ visits/eyes/burst/crops của người nằm cùng group; train preprocessing/imputation/calibration threshold chỉ trong train fold. Validation nhân tạo từ cùng bệnh nhân không phải patient-independent.

Nếu dùng detector học từ ảnh bệnh nhân, detector cũng phải train trong outer train groups; không pretrain nó trên toàn cohort rồi cross-validate classifier. Gold detector val/test phải là labels độc lập; không gọi prediction/pseudo-label là gold. Dataset public pretrained có thể dùng khi được phép và không trùng người test; record source và kiểm tra overlap khi có thể.

Giữ private test bởi người quản lý khác nếu khả thi; không xem ảnh/labels để debug hay chọn model. Freeze pipeline, split IDs, weights/hash, thresholds, aggregation và metric trước chạy test một lần. Test fail thì báo fail; muốn iteration mới cần development và test mới, không tune lại test cũ rồi giữ tên “untouched”.

Đơn vị primary classifier: bệnh nhân. Pilot dùng ảnh frontal đầu tiên theo quy tắc capture đã chốt, không chọn ảnh đoán đúng nhất; nếu ảnh đầu lỗi thì abstain. Secondary có thể xét bộ 3 ảnh với aggregation pre-registered và toàn patient chung split. Image-level chỉ là phân tích phụ, CI cluster theo bệnh nhân; số mắt không phải số người.

### Metric bắt buộc

- Detector: Euclidean mean/median/P90/P95 native px riêng glint/pupil/limbus/canthi, candidate-ID errors, iris px và blur/occlusion/camera subgroups; successful-measurement coverage. P90 conditional phải kèm tỷ lệ <=3 px và tỷ lệ “đủ điểm + đúng <=3 px” trên toàn đầu vào.
- Bốn lớp: confusion 4 hàng thật, 5 cột dự đoán gồm abstain; recall-all từng lớp=`đúng lớp/N lớp thật`, precision/F1, macro recall-all; có thể thêm BA/macroF1 trên covered nhưng label rõ conditional, không làm headline duy nhất. Không gọi metric custom có abstain là BA chuẩn mà không ghi định nghĩa.
- Binary research screening: sensitivity với eso/exo, specificity trên pseudo/normal, CI95% và false-negative review. Đánh giá true clinical diagnosis riêng image manifest condition, không nhập hai target thành một số.
- Abstain là output cần khám/lấy lại ảnh theo workflow đã phê duyệt, **không normal**. Báo referral sensitivity với abstain tính refer **cùng referral specificity/rate**; báo class recall-all với abstain không tính đúng. “Refer tất cả” không phải model tốt.
- Confidence/quality threshold chọn development; báo calibration nếu đủ mẫu. PPV/NPV phụ thuộc prevalence nên cohort cân bằng không đại diện quần thể.
- CI/bootstrap resample bệnh nhân, so sánh A/B/C paired trên cùng người/split. Không coi 5 lần CV của một ảnh là 5 ảnh độc lập. Fold thiếu lớp -> metric NA và count, không điền 0 hay giấu fold.

Mục tiêu thử nghiệm classifier chưa chốt: ví dụ sensitivity referral >=0,90 **đồng thời** specificity referral >=0,70 trên development, với CI và coverage; để bác sĩ xem xét và pre-register trước chạy, không tiêu chuẩn đưa vào clinical use. Không điều chỉnh mục tiêu sau test. Chưa có đủ cohort để khẳng định sẽ đạt.

## 9. Lộ trình và điều kiện hoàn thành

| Mốc | Công việc | Điều kiện hoàn thành đo được | Khi chưa đạt |
|---|---|---|---|
| M0. Rights + protocol | Approval/consent, nguồn, schema và patient linking | 100% ảnh được train/eval có source, rights approved, scope và patient ID; unknown bị quarantine. Private test plan chốt | Chỉ xây infrastructure bằng fixture synthetic; không train clinical |
| M1. Gold pilot | 40-60 người development nếu khả thi; hai rater + bác sĩ; giữ native | Có audit A/B; clinical raw agreement/kappa/CI; landmark P90 <=2 px trên ảnh đo được là mục tiêu pilot. Lớp thiếu ghi rõ, không coi pilot đủ 4-class | Sửa hướng dẫn/capture và đo lại; không dùng model để ép đồng thuận |
| M2. A0 + eval harness | Oracle features, dummy baseline, luật development và logistic regression; fold cố định | Unit test leakage/coordinate/sign/missing/metric denominator pass; có OOF predictions + confusion/counts; mọi số tái lập được trên cùng config | Nếu gold không hơn dummy trong giới hạn CI: xem target/morphology/label, chưa tăng CNN |
| M3. A1 detector | Baseline CV native; paired gold-vs-detector trên development | Mục tiêu P90 <=3 px glint/pupil/limbus **riêng target**, coverage đo geometry >=80% pilot; subgroup/failure/CI đầy đủ, không fake point. Canthus metric báo riêng | Phân loại lỗi capture/candidate/refine/boundary; chỉ thêm A2 ở phần sai |
| M4. Class/context | Thêm gold canthi/fold rồi tự động nếu hữu ích; cohort đủ lớp | Clinical labels đủ 4 lớp trong folds; morphology ablation, recall-all/covered và referral metrics rõ; thresholds chốt development | Không có pseudo -> chỉ scope lớp thật và localization, không phát hành 4-class |
| M5. B/C có điều kiện | Thử ít configs, cùng cohort và budget đã ghi | B so A; C real-only so synthetic+real; paired CI, landmark/coverage/subgroups; giữ phương án đơn giản nếu chưa có lợi ích rõ | B/C không cải thiện real validation thì bỏ, không kéo dài tuning vô hạn |
| M6. Freeze + private test | Pin version/hash, model card; evaluator độc lập | 0 overlap patient/source-derived images; test chạy một lần với toàn mẫu/counts/CI; không loại abstain khỏi headline | Báo kết quả/giới hạn trung thực; không promote model bằng test-tuning |

Mục tiêu M1/M3 về pixel/coverage là thỏa thuận pilot cần xác nhận, không certification. Canary runtime/Render chỉ xét ở PHASE 4 riêng sau approval; một unit test pass không chứng minh độ chính xác lâm sàng. Không tự deploy model nghiên cứu.

## 10. Kế hoạch file và tái lập PHASE 4, chưa tạo

Nếu chọn A, triển khai trên branch mới prefix `codex/` và thư mục nghiên cứu mới, không sửa/ghi đè dataset, CSV, artifact/report cũ. Xác minh trạng thái Git và môi trường trước thay đổi. Dự kiến layout:

```text
research/hirschberg_phase4/
  README.md
  configs/                 # seed, input contract, source rights manifest refs
  annotation/              # schema/converters, không consent định danh
  detectors/               # native ROI/ellipse/glint; không runtime swap mặc định
  features/                # dimensionless geometry/context
  train.py
  evaluate.py              # tách khỏi train, không fit/tune
  tests/                   # synthetic fixtures, không ảnh bệnh nhân public
  runs/<run_id>/           # version/config/hash/metrics, riêng mỗi run
```

Ảnh thật/consent lưu ở nơi private được duyệt, không trong public Git. Lưu split digest, config, seed, code commit, environment lock, rights manifest version, preprocessing, label schema, pretrained source/hash, model artifact/hash, OOF/test predictions và failure counts. Ảnh overlay nhận dạng không đưa vào report public. Artifact thiếu weights/dependency phải fail rõ, không tự chuyển random backbone.

Regression tests đề xuất: transform roundtrip; OD/OS/mirror sign; resize-native error; missing không (0,0)/fake glint; scale thiếu WTW không xuất mm; không physical PD nếu chưa calibration; canthi missing không suy từ pupil; patient overlap assert; no fitting in evaluator; abstain numerator/denominator; duplicate/near-duplicate review, không auto-delete.

Nguyên tắc commit nhỏ: schema/evaluator -> baseline A0 -> detector A1 -> optional A2/context -> model card. Không gọi training script cũ với output mặc định vì có thể ghi đè. Chỉ đổi runtime khi người dùng duyệt rõ và train/eval contract đã được kiểm tra.

## 11. Tóm tắt 5 dòng và lựa chọn

1. Khuyến nghị **A0 rồi A1**, classifier nhỏ + geometry native; A2 chỉ nếu cần và không hứa accuracy.
2. **B** là đối chứng sau khi có clinical labels/group split; vài trăm người chưa bảo đảm CNN tốt hơn.
3. **C** ưu tiên pretrain glint/detector, không sinh nhãn pseudo thay người bệnh thật.
4. Phi thương mại không bỏ rights/consent; tận dụng 62 annotation qua import/version riêng sau xác minh.
5. M0-M6 có gold agreement, pixel/coverage, patient-CV và private test; chưa triển khai/train/push/deploy.

**Dừng PHASE 3. Bạn chọn A (khuyến nghị), B hay C để chuyển PHASE 4?** Chọn A cho phép bắt đầu hạ tầng/schema/evaluator và fixture không có ảnh người; train/eval ảnh thật vẫn đợi M0, không tự coi dữ liệu cũ đã được phép.
