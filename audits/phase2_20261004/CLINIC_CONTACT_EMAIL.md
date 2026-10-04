# Email dự thảo, chưa gửi

**Nghiên cứu sàng lọc, không chẩn đoán hoặc thay thế khám mắt.** Không đính kèm ảnh bệnh nhân/hồ sơ định danh trong email đầu tiên. Mọi hợp tác cần approval, consent và thỏa thuận dữ liệu trước chuyển ảnh.

## 1. Liên hệ phòng khám

**Tiêu đề:** Đề nghị trao đổi hợp tác nghiên cứu sàng lọc lác từ ảnh có flash

Kính gửi bác sĩ [Tên]/[Phòng khám],

Tôi là [Tên, đơn vị, thông tin liên hệ], đang nghiên cứu phần mềm hỗ trợ sàng lọc lác từ ảnh mắt có flash. Hệ thống hiện chưa được xác nhận lâm sàng và không thay thế khám chuyên khoa. Tôi mong được trao đổi về tính khả thi của một nghiên cứu pilot, không xin nhận ảnh bệnh nhân sẵn có khi chưa xác minh quyền sử dụng.

Dự kiến khảo sát vài trăm người thuộc các nhóm lác trong, lác ngoài, giả lác và không thấy lác qua khám. Nhãn cần bác sĩ xác nhận; ảnh phải được chụp có đồng thuận riêng, giữ chất lượng gốc và thu theo quy trình cơ sở phê duyệt. Pilot nhỏ trước sẽ kiểm tra an toàn chụp, khả năng hợp tác của trẻ và mức đồng thuận giữa hai người gán.

Xin bác sĩ góp ý về tiêu chí tuyển, phương pháp khám đối chiếu, điều kiện flash/fixation phù hợp và khả năng đo WTW cá nhân để nghiên cứu hiệu chuẩn. Nếu khả thi, chúng tôi sẽ chuẩn bị đề cương cho hội đồng đạo đức, thông tin nghiên cứu/consent/assent và thỏa thuận bảo vệ dữ liệu. Danh tính và khóa nối mã bệnh nhân nên do cơ sở giữ; kỹ sư chỉ nhận bản theo scope được phê duyệt.

Mục đích dự kiến: [nghiên cứu phi thương mại / hướng thương mại, ghi trung thực]. Chúng tôi không mặc nhiên được công bố ảnh, chia sẻ dataset hoặc dùng ảnh cho sản phẩm. Ngân sách, nhân sự gán nhãn, quyền sở hữu, công bố và retention sẽ được thỏa thuận trước, không gây ảnh hưởng chăm sóc thường quy.

Phòng khám có thể cho một buổi trao đổi [thời gian] và chỉ định đầu mối nghiên cứu/bảo vệ dữ liệu không? Tôi có thể gửi bản audit và protocol dự thảo không chứa ảnh người.

Trân trọng,
[Tên, đơn vị, email, điện thoại]

## 2. Xin xác minh quyền dataset/code với tác giả

**Subject:** Permission and provenance clarification for [dataset/model], eye-alignment screening research

Dear [Name],

I am [name/affiliation], investigating photographic Hirschberg screening. Our prototype is not clinically validated and will not replace an eye examination. Before acquiring any human-eye images or using pretrained weights, we would like to clarify the permitted scope of your [dataset/model].

Our intended use is [noncommercial research / potential commercial development, explicitly state which]. Could you confirm:

1. The exact license/data-use agreement and which files it covers: images, annotations, code and pretrained weights, including third-party components.
2. Whether participant consent permits release and the proposed ML training/evaluation use, including any restrictions for minors, biometric identification, commercial use or model redistribution. We do not request copies of identifiable consent forms; a documented institutional confirmation is preferable.
3. Whether original native-resolution images are available or only resized/cropped/processed exports; camera modality, flash/LED setup and any red-eye editing.
4. Annotation provenance, clinician verification if applicable, uncertainty, and stable participant/session IDs for leakage-free evaluation.
5. Registration/ethics approval requirements, storage location, retention, withdrawal handling, publication and redistribution restrictions.

We will not scrape mirrors, publish patient images or treat an article/code license as image permission. If access requires a formal agreement, please let us know the appropriate institutional contact and procedure.

Thank you,
[Name, affiliation, contact details]

## 3. Đầu mối đã thấy trực tiếp trong nguồn

| Nguồn | Liên hệ | Việc cần hỏi |
|---|---|---|
| [Byrne repo](https://github.com/dcnieho/Byrneetal_CR_CNN) | sean.byrne@imtlucca.it hoặc dcnieho@gmail.com | NC code/weights; quyền khác nếu hướng thương mại; real data cần permission riêng |
| [Toronto Eye Tracking XR](https://www.eecg.utoronto.ca/~jayar/datasets/xreyetrack.html) | soumil.chugh@gmail.com | License ảnh/annotation, consent, phần NVIDIA và native resolution |
| [PLOS 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303355) | Pusan National University Yangsan Hospital IRB Center: +82-055-360-4722 | Đầu mối cơ sở, không phải email; hỏi thủ tục data agreement, không xin ảnh qua kênh không bảo mật |

Chưa gửi hay đặt lịch. Không bịa email phòng khám, không tự liên hệ thay chủ dự án.
