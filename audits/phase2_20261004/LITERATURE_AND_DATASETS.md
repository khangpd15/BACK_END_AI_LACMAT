# PHASE 2: Tài liệu và dữ liệu theo audit

Ngày kiểm tra nguồn: 2026-10-04. Cơ sở: [AUDIT_REPORT.md](../phase1_20261004/AUDIT_REPORT.md).

**Chỉ phục vụ nghiên cứu sàng lọc, không chẩn đoán và không thay thế khám mắt.** Không tải ảnh, train, sửa runtime, commit, push hoặc deploy trong PHASE 2. Mọi tài liệu dưới đây mới được tạo ở thư mục riêng. Quy trình và tiêu chí đề xuất chưa được xác nhận lâm sàng.

## 1. Kết luận cần quyết định

1. Trong phạm vi nguồn kiểm tra, **chưa xác minh được dataset công khai đáp ứng đồng thời 4 lớp, nhãn lâm sàng, license và quyền sử dụng ảnh người**. Không có nghĩa là dataset như vậy không tồn tại.
2. Có nguồn phù hợp để nghiên cứu định vị, nhưng license không thương mại và khác biệt camera là rào cản. Không coi ảnh IR trong headset là ảnh flash RGB nhi khoa.
3. **Chưa có bằng chứng đã kiểm tra đủ để cam kết P90 <3 px trên ảnh flash điện thoại gốc của dự án.** Đây phải là mục tiêu thực nghiệm, không phải kết quả có sẵn.
4. Không thay `11.8 mm` bằng một đường kính trung bình theo tuổi khác. Muốn mm phải đo kích thước của chính người đó; thiếu phép đo thì chỉ báo tỷ lệ không đơn vị.
5. Với vài trăm bệnh nhân: xác minh quyền và nhãn chuẩn là điều kiện bắt đầu; sửa detector là ưu tiên kỹ thuật; thêm bệnh nhân có nhãn chuẩn, đặc biệt pseudo, là bước tiếp theo.

## 2. Cách tìm và mức xác minh

Tìm có mục tiêu theo ba nhóm: ảnh lác có chẩn đoán; pupil/iris/glint có annotation; ảnh flash gốc độ phân giải cao. Ưu tiên bài tác giả, trang dataset chính thức và LICENSE trong repo; khảo sát thêm đầu mối GitHub, arXiv, PMC/PubMed, Zenodo, DaRUS, figshare và Roboflow. Đây là rà soát có mục tiêu, **không phải tổng quan hệ thống bao phủ mọi kho dữ liệu**. Không tải archive ảnh hoặc sao chép ảnh minh họa vào dataset.

Phân biệt ba quyền: license bài báo, license code/weights, và quyền dùng ảnh/dữ liệu người. Một license mở trên trang re-upload không chứng minh nguồn gốc hay đồng thuận. Crop mắt vẫn có thể chứa dữ liệu sinh trắc; không mặc nhiên vô danh. Không tự suy luận quyền thương mại từ việc được phép nghiên cứu. Chỉ tiếp nhận ảnh thật sau khi xác minh điều khoản và căn cứ đồng thuận phù hợp với mục đích này.

Trạng thái:
- **Có license xác minh, có điều kiện:** chỉ cân nhắc đúng phạm vi; không có nghĩa đã được duyệt tải/train.
- **HOLD:** thiếu điều khoản, provenance, consent hoặc nội dung gốc; không tiếp nhận/train.
- **Tham khảo phương pháp:** đọc kỹ thuật, không lấy ảnh từ bài báo làm train.

## 3. Dataset, theo thứ tự ưu tiên

### A. Strabismus/pseudostrabismus có nhãn lâm sàng

| Nguồn | Nhãn, quy mô | Ảnh gốc hay crop | Quyền, truy cập, định danh | Kết luận |
|---|---|---|---|---|
| [Joo et al., PLOS ONE 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303355) | 900 normal +896 strabismus; khảo sát 10 chuyên gia, chọn ảnh ít nhất 5 người cùng nhãn; không phải 4 lớp | Chỉ vùng hai mắt sau crop trong nghiên cứu; không xác minh quyền truy cập ảnh gốc | Bài CC BY **không cấp quyền dataset**. Dữ liệu không công khai do riêng tư; xin IRB bệnh viện qua +82-055-360-4722. Ảnh người, không coi crop là vô danh | HOLD; đầu mối hợp tác, không tải. Đồng thuận của chuyên gia trên ảnh không đồng nghĩa mọi nhãn đã có cover test |
| [Nine cardinal positions, figshare](https://figshare.com/articles/figure/Nine_cardinal_positions_of_gaze_high-resolution_image_dataset_for_fine-grained_strabismus_diagnosis/29302409) | Đầu mối tìm kiếm, chưa xác minh nội dung gốc | Chưa xác minh; không được gọi là bộ ảnh full-resolution đã sẵn sàng | Mở trang bị 403; API không truy cập được. Không xác nhận license/consent/nhãn từ snippet | HOLD, không dùng |
| [STB front open clinical, Roboflow](https://universe.roboflow.com/strabismus-tml8m/stb-front-open-clinical) | Đầu mối tìm kiếm, chưa xác minh người gán/khám | Chưa xác minh original hay export resize | Mở trang timeout. Badge license trong kết quả tìm kiếm chưa đủ chứng minh quyền ảnh nguồn và consent | HOLD, không dùng |

**Riêng pseudo:** không biến ảnh có nếp rẻ quạt thành nhãn pseudo bằng mắt thường. Cần bệnh nhân có biểu hiện giả lác nhưng được bác sĩ xác nhận bằng khám phù hợp. Dataset hiện có của dự án có 0 ảnh pseudo; không thể giải quyết bằng oversampling lớp không tồn tại. Bằng chứng nội bộ: audit mục 1 và kiểm kê dữ liệu.

### B. Pretrain pupil/iris/glint

| Nguồn | Quy mô/annotation | Độ phân giải, phạm vi ảnh | License/truy cập/consent | Mức phù hợp |
|---|---|---|---|---|
| [LPW, DaRUS chính thức](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-3237) | 22 người, 66 video, 130.856 frame; tọa độ tâm pupil; không xác minh nhãn glint | 640x480 video vùng mắt từ camera đeo đầu; không phải ảnh nguyên mặt/RAW điện thoại | **CC BY-NC-SA 4.0**; metadata có dữ liệu cá nhân đã pseudonymize; file public. Chưa xác minh riêng phạm vi consent cho nghiên cứu này, cần hỏi chủ dữ liệu trước tiếp nhận | Trung bình: pupil; thấp cho nhãn lác/flash RGB. Có license xác minh, chưa duyệt tải |
| [OpenEDS 2019, bài gốc](https://arxiv.org/html/1905.03702) | 152 người; 12.759 ảnh có mask pupil/iris/sclera; còn video và ảnh không nhãn | 400x640 theo bài; ảnh near-eye IR trong VR, không nguyên mặt | Bài ghi written consent phát hành ảnh; truy cập qua yêu cầu tới tác giả. Trang phân phối gốc hiện không mở được. License ảnh gốc chưa đọc độc lập được; không dùng license arXiv thay thế | HOLD tới khi nhận DUA/license. Hữu ích segmentation, domain khác dự án |
| [Night Eyes v3, Zenodo](https://zenodo.org/records/19335499) | Nhãn glint theo danh tính LED cho OpenEDS; **tạo bằng thuật toán**, còn lỗi định vị/correspondence | Annotation mở rộng, không bộ ảnh gốc kèm theo | Record ghi **CC BY-NC 4.0** cho annotation, tuyên bố kế thừa OpenEDS; không thay việc xác minh điều khoản ảnh gốc | HOLD cho train ảnh; tham khảo cách phân biệt nhiều phản xạ. Không lấy nhãn sinh tự động làm gold test |
| [Eye Tracking XR, Toronto](https://www.eecg.utoronto.ca/~jayar/datasets/xreyetrack.html) | 15 người tự thu + nhãn cho 10 người NVIDIA; pupil center/boundary/ellipse, glint x-y và LED ID | 640x480 near-eye VR, không ảnh mặt flash | Không thấy license riêng/consent trong trang đã đọc. Ảnh NVIDIA phải lấy từ nguồn NVIDIA do hạn chế license. Liên hệ soumil.chugh@gmail.com | HOLD dù annotation rất phù hợp |
| [UnityEyes2](https://github.com/alexanderdsmith/UnityEyes2), [LICENSE](https://github.com/alexanderdsmith/UnityEyes2/blob/main/LICENSE) | Generator ảnh tổng hợp; tùy camera/đèn/kích thước pupil; số ảnh do người chạy sinh | Resolution cấu hình được; ảnh render, **không ảnh bệnh nhân gốc** | Code **MIT**. Cần kiểm tra assets/texture phụ thuộc và quyền outputs trước sử dụng; không tự coi MIT bao mọi asset. Repo mô tả vị trí nguồn sáng, chưa xác minh nhãn glint 2D turnkey | Cao để thử pretrain không dùng ảnh người; không thay dữ liệu lác thật |

### C. Flash độ phân giải cao

| Nguồn | Ảnh/độ phân giải | Gốc hay xử lý | License/consent/quyết định |
|---|---|---|---|
| [FAID, trang tác giả](https://yaksoy.github.io/flashambient/) | Flash/ambient; trang thông báo không còn host dataset | Không xác minh file gốc hiện được cung cấp | Chưa xác minh license ảnh/consent người. HOLD; không tìm mirror để lách |
| [ExtendedFAID, tài liệu tác giả](https://yaksoy.github.io/papers/CVPR23-IntrinsicFlash-Dataset.pdf) | Mô tả FAID >2700 cặp 1440x1080; DPD 495 cặp/101 người, chụp 3120x4160 | Bản DPD xử lý cuối crop **512x512**, căn chỉnh và sửa red-eye; không xác nhận full-resolution được phát hành. Tài liệu ghi bỏ 65 cặp thiếu consent | License dữ liệu chưa xác minh. HOLD; không dùng ảnh đã chỉnh mắt để đánh giá glint. Việc có ảnh được chụp lớn không có nghĩa ảnh gốc được tải hợp pháp |

**Kết luận nhóm C:** chưa xác minh được nguồn flash ảnh người high-resolution đủ điều kiện tiếp nhận. Hướng có thể kiểm soát nhất là thu mới tại phòng khám với consent riêng, giữ ảnh native. Không tải ảnh bệnh nhân từ bài báo, mạng xã hội, Google Images hay kho re-upload.

## 4. Phương pháp định vị: bằng chứng và giới hạn

| Nguồn | Phương pháp, dữ liệu, metric | Code/weights và quyền | Dùng cho dự án |
|---|---|---|---|
| [Byrne et al. 2023](https://arxiv.org/html/2304.05673), [repo tác giả](https://github.com/dcnieho/Byrneetal_CR_CNN) | Định vị thô glint, sau đó CNN trên patch; train Gaussian tổng hợp. Validation tổng hợp: 0,085 px. Real video IR: đo precision/lặp lại, **không chứng minh absolute error trên ảnh flash** | Có code sinh/train/eval và weights; **CC BY-NC-SA 4.0** cho code/model. Real images không kèm repo, xin tác giả | Cao để thử tinh chỉnh subpixel sau chọn đúng glint; không chữa lỗi chọn nhầm điểm sáng |
| [EllSeg, repo tác giả](https://github.com/RSKothari/EllSeg), [license đọc được](https://raw.githubusercontent.com/RSKothari/EllSeg/master/License.md) | Segmentation + ellipse cho pupil/iris, xử lý vùng bị mi che; weights qua nhiều bộ eye-tracking. Metric segmentation/detection không tương đương P90 tọa độ native trên ảnh dự án | Code **MIT**, có pretrained model; quyền dữ liệu/weights cần kiểm tra riêng trước dùng | Cao cho hướng ellipse; trung bình cho chuyển IR sang RGB. Không lấy centroid phần mask còn nhìn thấy làm tâm toàn iris |
| [EyeTurn, Pundlik et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6369861/) | Dùng limbus center và flash điện thoại, thí nghiệm fixation 25 người, so prism cover 66 người. RMSE 2,4 **PD**, không px; so clinical có sai khác RMS 6,8 PD. Có nghiên cứu HR cá nhân | Bài không phát hành dataset/model có quyền tái sử dụng được xác minh | Cao: cơ sở hình học/capture; không sao chép hệ số population thành hằng số runtime |

### Thiết kế kiểm chứng được đề xuất, chưa triển khai

1. Detector mặt/mắt chỉ tìm ROI. Giữ crop ở pixel native với ma trận biến đổi về ảnh gốc; không nén hai mắt vào 224x224 rồi mong lấy lại flash bằng upscale.
2. Định vị pupil và limbus thành **hai target riêng**: mask/boundary -> ellipse robust -> tâm và confidence. Không so iris output với pupil label rồi coi đó là lỗi cùng loại. Khi che quá nhiều, báo không đo được; không ép fit.
3. Trong ROI giác mạc, lấy **nhiều candidate glint**, phân biệt phản xạ flash với mi, tear film và đèn phòng. Dùng dữ liệu thu theo camera/flash biết trước để học candidate đúng; không mặc định điểm sáng nhất đúng.
4. Sau chọn candidate đúng: Gaussian 2D + nền cục bộ, ellipse/intensity model hoặc CNN nhỏ refine. Bỏ pixel lõi bão hòa khi fit phần rìa nếu đủ thông tin; nếu không đủ hoặc multi-glint không phân biệt được thì từ chối đo. Các bước này là đề xuất kỹ thuật, không bảo đảm Gaussian phù hợp mọi flash.
5. Kiểm tra chéo các lần chụp và confidence, nhưng không lấy trung bình để che lỗi hệ thống. Không sinh glint giả ±0,5 mm khi không tìm thấy.

**Định nghĩa mục tiêu <~3 px:** khoảng cách Euclidean tới nhãn đã adjudicate trên hệ tọa độ ảnh gốc, riêng pupil, limbus và glint; mục tiêu pilot **P90 <=3 px**, kèm mean/median/P95, tỷ lệ sai candidate, tỷ lệ không đo được và CI theo bệnh nhân. Không cam kết mọi frame <3 px. Báo native width/height và đường kính iris theo px, vì 3 px không có ý nghĩa giống nhau ở mọi camera.

Ví dụ tự dựng: crop resize 0,25 lần -> lỗi 3 px crop thành 12 px native. Upscale không sinh thêm thông tin. Rater cũng phải đủ nhất quán để phân biệt 3 px; nếu hai người còn lệch 5 px thì chưa thể kết luận detector đạt chuẩn đó.

## 5. Hiệu chuẩn mm, không hằng số đường kính

**Đề xuất phép đo, không chứng nhận độ chính xác vật lý:** bác sĩ/thiết bị sinh trắc đo horizontal white-to-white (WTW) riêng OD và OS; lưu giá trị, thiết bị, thời điểm và độ lặp lại. Trên ảnh frontal, đo đường kính limbus tương ứng, không trộn pupil diameter với corneal WTW. Visible iris/limbus ảnh không hoàn toàn là phép đo giải phẫu WTW, nên cần đối chiếu hệ thống trước sử dụng scale.

```text
D_e = WTW cá nhân đo tại phòng khám (mm), e thuộc OD/OS
d_e = đường kính limbus tương ứng trong ảnh đã sửa distortion (px)
s_e = D_e / d_e                         # mm/px, xấp xỉ tại mặt phẳng tham chiếu
delta_e = s_e * (g_e - c_e)              # displacement chiếu trong ảnh
q_e = (g_e - c_e) / d_e                  # tỷ lệ không đơn vị khi chưa có D_e
```

`c_e` là tâm limbus nếu sử dụng phương pháp limbus; đồng thời lưu pupil center để đánh giá riêng. Quy ước trục nasal/temporal theo OD/OS, không theo trái/phải màn hình. Không gọi displacement này là khoảng cách giải phẫu trên bề mặt cong giác mạc. Sai số ellipse, phối cảnh, corneal magnification và glint chiếu ở độ sâu khác phải được kiểm chứng, không bỏ qua bằng công thức tỷ lệ.

- Calibrate intrinsics/distortion theo đúng camera, lens, zoom và resolution. [OpenCV calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) cung cấp công cụ camera; **không tự hiệu chuẩn giải phẫu mắt**.
- Nếu ảnh nghiêng: ellipse giúp phát hiện pose nhưng không tự giải quyết projection/refraction. Pilot giới hạn frontal, kiểm tra thực nghiệm; ngoài phạm vi thì không báo mm.
- Thiếu WTW riêng: giữ `q_e`, không fallback 11,8 mm hay bảng tuổi. Pupil diameter thay đổi theo ánh sáng, không làm thước cố định.
- Truyền bất định: với `delta = D*u/d`, xấp xỉ `sigma_delta² = (u/d)²*sigma_D² + (D/d)²*sigma_u² + (D*u/d²)²*sigma_d²`, chỉ khi bỏ qua covariance hợp lý. Phải tính covariance/bootstrap nếu đại lượng cùng fit.
- **mm -> PD là hiệu chuẩn khác**: cần HR/offset angle-kappa được xác định hoặc mô hình đã được xác nhận ở quần thể và setup tương ứng. Không lấy 22 PD/mm hoặc hệ số từ bài thành hằng số. Không có calibration được xác nhận thì không xuất PD. Thử nghiệm fixation/calibration chỉ thực hiện theo protocol bác sĩ phê duyệt.

## 6. Hai khóe mắt trong thật và epicanthal fold

Khóe mắt trong (medial canthus) là chỗ bờ mi trên và dưới gặp nhau về phía mũi; không phải tâm pupil, mép iris hay tâm caruncle. Schema annotation cần `medial_canthus_OD/OS` và `lateral_canthus_OD/OS`, với visible/occluded/uncertain. Nếu nếp da che chỗ gặp nhau thì **không bịa landmark** từ mép nếp da.

```text
ICD_px = norm(medial_OD - medial_OS)
OCD_px = norm(lateral_OD - lateral_OS)
PFL_e_px = norm(medial_e - lateral_e)
canthal_index = ICD_px / OCD_px
```

Đây là **khoảng cách chiếu 2D** giữa canthi thật. Muốn ICD mm giải phẫu, ưu tiên đo trực tiếp bởi nhân viên được huấn luyện/thiết bị được phòng khám chấp thuận và đối chiếu ảnh. Không nhân ICD với scale iris như thể hai mặt phẳng trùng nhau. Head yaw, chiều sâu sống mũi và distortion có thể làm sai. Không dùng `pupil_distance * 0.45` như hiện tại.

Đặc trưng đề xuất để bác sĩ đánh giá: mask/đường nếp rẻ quạt, tỷ lệ che vùng canthus và sclera phía mũi, độ bất đối xứng khe mi, ICD/OCD, ICD/PFL từng mắt, tỷ lệ sclera nasal/temporal sau điều chỉnh gaze. **Chưa có ngưỡng được xác nhận cho dự án**; không dùng tuổi/dân tộc hoặc “mũi tẹt” làm nhãn.

[AAPOS](https://aapos.org/glossary/pseudostrabismus) giải thích nếp rẻ quạt/sống mũi có thể tạo vẻ lác, và cần light/cover test để phân biệt; giả lác không bảo đảm không xuất hiện lác thật sau này. Vì vậy output nghiên cứu nên nói “hình thái gợi ý”, không khẳng định pseudo/normal chỉ từ morphology.

Đầu mối [orbitofacial anthropometry 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10697275/) tìm thấy qua metadata/text tìm kiếm, nhưng mở full text bị CAPTCHA và DOI không truy cập được trong lượt này. **Chưa đọc toàn văn độc lập**; không lấy ngưỡng hay số đo từ nguồn này để quyết định pipeline.

## 7. Chụp, nhãn chuẩn, đồng thuận

Xem [CAPTURE_AND_ANNOTATION_PROTOCOL.md](CAPTURE_AND_ANNOTATION_PROTOCOL.md), [CONSENT_TEMPLATE.md](CONSENT_TEMPLATE.md) và [CLINIC_CONTACT_EMAIL.md](CLINIC_CONTACT_EMAIL.md). Đây là bản dự thảo cần phòng khám/hội đồng đạo đức và người phụ trách bảo vệ dữ liệu duyệt. Không phải giấy tờ pháp lý dùng ngay; không có email nào được gửi.

Điểm bắt buộc: giữ ảnh native và nguyên provenance; hai rater độc lập trước adjudication; bác sĩ xác nhận nhãn bằng khám chứ không chỉ đoán ảnh; patient-level split; consent lưu riêng khỏi Git; private test không dùng chọn ngưỡng. Đo Cohen kappa cho lớp không thứ tự, Euclidean disagreement cho landmark và agreement của phép đo vật lý, không thay tất cả bằng accuracy.

## 8. Thêm dữ liệu, sửa detector hay sửa nhãn?

**Không có xếp hạng phổ quát. Theo audit này:**

| Việc | Vì sao/điều kiện |
|---|---|
| 0. Quyền dữ liệu, patient ID và nhãn chuẩn | Chưa xác minh consent/license; 0 pseudo; clinician_confirmed=false. Không thể học đúng bài toán bằng nhãn chưa xác thực |
| 1. Bộ gold nhỏ: sửa/xác nhận nhãn | Hai rater + bác sĩ để biết detector thực sự sai bao nhiêu. Đây là điều kiện đo, không khẳng định nhãn hiện tại chắc chắn sai |
| 2. Sửa detector/capture, giữ native | Glint P90 **37,22 px** trên annotation hiện có; chênh iris/pupil khác target nên không được diễn giải như pupil error. Lỗi định vị có thể phá feature trước classifier |
| 3. Thêm bệnh nhân, không chỉ ảnh | Ưu tiên pseudo được khám xác nhận, góc lác nhỏ/intermittent, nhiều camera và điều kiện thực tế. 20 ảnh cùng người không tương đương 20 bệnh nhân |

Oracle landmark trên 62 ảnh có BA 0,665 nhưng **khác mẫu** với detector cũ, nên chưa chứng minh causal gain do sửa detector. Không suy luận sẽ đạt 95% sau sửa; metric headline audit đã có train contamination/abstention bias. Bằng chứng: audit và `evidence.json` PHASE 1.

Thí nghiệm để ra quyết định sau khi được phép triển khai: cùng bệnh nhân/split, cùng classifier nhỏ và preprocessing, so nhãn landmark gold với detector, giữ nhãn bệnh đã xác nhận. Chênh lớn -> ưu tiên detector; cả hai thấp -> xem lại target, lâm sàng/nhãn và feature; learning curve tiếp tục tăng với bệnh nhân độc lập -> ưu tiên bổ sung dữ liệu. Chỉ ablation trên development, không dùng private test để chọn.

Một cohort 300 người chỉ là ví dụ ngân sách, không bảo đảm đủ: có thể giữ 60 người test, 240 development; ít mẫu từng lớp khiến CI rộng. Nếu một lớp test 15 người, mỗi lỗi thay recall khoảng 6,7 điểm %. Tăng độ đa dạng và chất lượng nhãn quan trọng hơn nhân số crop. Chưa tính power/sample-size lâm sàng chính thức.

## 9. Top 5 nguồn đáng đọc/dùng đúng điều kiện

1. **EyeTurn**: cơ sở clinical geometry và giới hạn PD; tham khảo phương pháp, không lấy ảnh.
2. **Byrne CR CNN**: subpixel refinement và synthetic generation; NC, không tự đưa weights lên production.
3. **EllSeg**: iris/pupil ellipse; code MIT, xác minh weights riêng.
4. **UnityEyes2**: tạo dữ liệu không dùng ảnh bệnh nhân; kiểm tra asset/output và nhãn xuất ra.
5. **LPW**: nguồn pupil có license chính thức; chỉ xem xét nghiên cứu NC sau kiểm tra consent/mục đích.

URL và quyền nằm tại bảng phía trên; đây không phải năm bộ clinical dataset đã được phép train. OpenEDS/Night Eyes là ứng viên tiếp theo khi điều khoản ảnh được xác minh. Nguồn bị lỗi truy cập chỉ là đầu mối, không bằng chứng sử dụng.

**Dừng tại PHASE 2.** Cần chủ dự án xác nhận chuyển PHASE 3 và cho biết mục tiêu nghiên cứu phi thương mại hay hướng sản phẩm thương mại để lọc license phù hợp.
