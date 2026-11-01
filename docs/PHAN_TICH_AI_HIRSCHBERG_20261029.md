# Phân tích AI và cơ sở đánh giá ảnh Hirschberg

Ngày kiểm tra: 29/10/2026. Dự án: RemiCare Hirschberg AI Backend.

Mốc mã nguồn: `685839a9a9c51545be05bb360b7906db21dc7305`.

## 1 Kết luận chính

Dự án hiện có một pipeline thị giác máy tính sử dụng nhiều mạng nơ ron để tìm các cấu trúc của mắt, sau đó kết hợp ảnh với hình học Hirschberg để đề xuất một trong ba lớp: NORMAL, ESOTROPIA và EXOTROPIA. Mô hình phân loại chính là ResNet50 kết hợp đặc trưng phản xạ sáng và đặc trưng vị trí đồng tử trong khe mi. Không thấy lời gọi GPT, Gemini hay dịch vụ AI tạo sinh trong luồng suy luận hiện hành. [S1–S5]

Cơ sở đưa ra đề xuất gồm ba nhóm thông tin: hình ảnh hai mắt; vector từ tâm đồng tử đến phản xạ sáng được chuẩn hóa theo bán kính mống mắt; và vị trí đồng tử tương đối với vùng mở mí mắt. Đây là một mô hình học từ dữ liệu có thêm thông tin hình học. Kết luận cuối không được tính chỉ bằng một ngưỡng độ lệch cố định. [S4–S5]

Trạng thái thực tế vẫn là công cụ sàng lọc nghiên cứu cần người xem xét. Khi thiếu điều kiện chụp hoặc điểm mốc không đạt kiểm tra, hệ thống trả INCONCLUSIVE và không gọi bộ phân loại. Khi bộ phân loại được gọi thành công, trạng thái vẫn là HUMAN_REVIEW, không phải chẩn đoán đã xác nhận. [S3, S6–S8]

Chưa có bằng chứng trong bộ runtime về độ nhạy, độ đặc hiệu hoặc độ chính xác trên tập kiểm định độc lập của cấu hình đang chạy. Chỉ số accuracy và macro F1 bằng 1,0 là TRAIN FIT ONLY trên 112 ảnh; không thể diễn giải thành độ chính xác 100% khi sử dụng thực tế. [S9–S11]

Phạm vi kiểm tra là mã nguồn, cấu hình, checkpoint và kiểm thử tại workspace này. Kết quả không xác nhận phiên bản đang được triển khai trên máy chủ hay chất lượng lâm sàng của sản phẩm ngoài thực địa.

## 2 Dự án đang sử dụng những AI nào

| Thành phần | Công nghệ và phiên bản | Vai trò trong luồng hiện tại |
| --- | --- | --- |
| Phát hiện mặt và vùng mắt | MediaPipe Face Mesh với refine_landmarks=True | Tìm một khuôn mặt, vùng hai mắt, viền mí và các điểm mống mắt tham chiếu. Không trực tiếp quyết định lác trong hay lác ngoài. |
| Phát hiện mắt dự phòng | OpenCV Haar cascade eye_tree_eyeglasses | Tìm cặp mắt khi không tìm được mặt trong chế độ auto, hoặc khi nhận ảnh bilateral. Đây là detector cổ điển đã học, không phải CNN phân loại bệnh. |
| Định vị phản xạ sáng | CNN encoder decoder kiểu U Net, crop256_human_review_20261010_v1/final.pt | Sinh heatmap đồng tử và phản xạ; tọa độ phản xạ được lấy từ cực đại heatmap. Tọa độ đồng tử sau đó được thay bằng đầu ra mô hình chuyên biệt. |
| Định vị đồng tử | CNN heatmap có prior mống mắt, pupil_color_features_20261012_v1/control_final.pt | Dự đoán tâm đồng tử bằng expected center. Dùng RGB, prior mềm từ mống mắt và tọa độ không gian. Nhánh được nạp là control, không phải color. |
| Định vị mống mắt | CNN hồi quy iris_four_reviewed_20261011_v1/iris_four_final.pt | Dự đoán bốn điểm trái, phải, trên, dưới; suy ra tâm và đường kính mống mắt. Không phải phân đoạn toàn bộ mống mắt. |
| Phân đoạn khe mi | EyelidUNet, eyelid_reviewed_20261011_v1/candidate_final.pt | Sinh mask vùng mở mắt và đường viền, phục vụ kiểm tra phản xạ và tính vị trí đồng tử trong khe mi. |
| Phân loại ba lớp | ScreeningEyelidBranch, screening_eyelid_branch_20261011_v1/candidate_final.pt | Kết hợp ResNet50, 11 đặc trưng hình học và 10 đặc trưng khe mi để đề xuất normal, esotropia, exotropia. |

Các mô hình tùy biến chạy bằng PyTorch trên CPU; runtime đặt số luồng CPU mặc định là 1. requirements.txt khai báo torch 2.7.1+cpu, mediapipe 0.10.14 và opencv-contrib-python 4.14.0.94. Đây là phiên bản khai báo trong dự án, không phải khẳng định cấu hình máy chủ đã cài đúng các phiên bản đó. [S1–S5, S12]

Manifest còn chứa một số checkpoint dùng làm seed, kiến trúc khởi tạo hoặc đối chiếu hash. Việc một tệp .pt xuất hiện trong manifest không có nghĩa nó là một bộ phân loại độc lập đang chạy song song. Không nên đếm số checkpoint để suy ra số mô hình chẩn đoán. [S2–S3]

## 3 Ảnh được xử lý như thế nào

Luồng chính: POST /api/v1/hirschberg/predict → predict_trained_hirschberg → Engine.process → định vị cấu trúc mắt → kiểm tra đầu vào và điểm mốc → ScreeningEyelidBranch → kết quả nghiên cứu. [S1–S3]

1. Kiểm tra tệp: chấp nhận JPEG, PNG, WebP; tối đa 10 MB, 25 megapixel và ít nhất 16 pixel mỗi chiều. Đọc ảnh RGB trên hệ tọa độ pixel lưu trong tệp. EXIF yêu cầu phép xoay hoặc lật khác mặc định sẽ khiến đầu vào không đủ điều kiện kết luận; runtime không tự chuẩn hóa EXIF trước khi đo.
2. Tìm vùng mắt: dùng Face Mesh khi có đúng một mặt; fallback sang Haar cascade theo chế độ đầu vào. Với ảnh có SHA-256 trùng danh mục frozen_rois.json, hệ thống dùng lại các ROI đã đóng băng. Đây là sử dụng vùng cắt lưu sẵn, không phải lấy nhãn bệnh từ tên ảnh.
3. Chuẩn hóa vùng cắt: crop mắt, letterbox và lật crop phía phải ảnh theo quy ước nội bộ. Tọa độ dự đoán được chuyển lại về ảnh nguồn. Việc lật crop nội bộ này khác với xác nhận ảnh gốc có bị mirror hay không.
4. Định vị: dự đoán phản xạ sáng, bốn điểm mống mắt, tâm đồng tử và mask khe mi. Vùng mống mắt hỗ trợ mô hình đồng tử qua prior mềm.
5. Kiểm tra chất lượng: xác nhận ROI, tọa độ, trạng thái nhìn thấy, hướng ảnh, hình học hai mắt, vị trí phản xạ và thông tin định thị. Nếu không đạt thì dừng trước bộ phân loại.
6. Phân loại: ghép ảnh hai mắt thành panel 224 × 224; tính đặc trưng hình học và khe mi; chạy mô hình để lấy ba điểm softmax rồi chọn lớp có điểm lớn nhất.
7. Trả kết quả: xuất lớp đề xuất, trạng thái, tọa độ, thông tin QC và phiên bản mô hình. Nếu save_session=true, API lưu metadata và kết quả JSON cục bộ; không lưu ảnh upload trong luồng này. [S1–S8, S13]

Điểm cần lưu ý khi đánh giá khả năng tổng quát hóa: ảnh có frozen ROI và ảnh mới phải qua detector trực tuyến có thể có độ khó khác nhau. Bộ test độc lập cần ghi nhận roiSource, kiểm tra ảnh trùng và tách theo người tham gia. Hash hợp lệ chỉ chứng minh tính toàn vẹn của bộ runtime, không chứng minh không rò rỉ dữ liệu hoặc độ chính xác y khoa.

## 4 Cơ sở hình học và cách mô hình đưa ra phán đoán

### 4.1 Nguyên lý Hirschberg

Nghiệm pháp Hirschberg quan sát vị trí phản xạ ánh sáng trên giác mạc so với tâm đồng tử ở hai mắt khi người được kiểm tra nhìn vào nguồn sáng. Tính tương ứng và đối xứng của phản xạ là cơ sở đánh giá hướng trục mắt. Phản xạ ở mắt bình thường không nhất thiết nằm đúng tâm đồng tử; góc kappa và khác biệt giải phẫu có thể làm sai lệch cách diễn giải. [M1–M3]

Trong điều kiện chụp và quy ước giải phẫu phù hợp, phản xạ lệch phía thái dương có thể gợi ý lác trong, lệch phía mũi có thể gợi ý lác ngoài. Tuy nhiên, hướng nhìn, vị trí nguồn sáng và góc kappa đều ảnh hưởng phép đo. Không thể áp dụng quy tắc trái hoặc phải trên ảnh mà bỏ qua lịch sử mirror và mắt OD hay OS. [M1–M3]

Đây là cơ sở y học của phép quan sát; nó không tự chứng minh các checkpoint trong dự án đã đạt chất lượng lâm sàng.

### 4.2 Mười một đặc trưng Hirschberg

Gọi P là tâm đồng tử, C là phản xạ sáng, I_left và I_right là hai điểm rìa ngang mống mắt. Runtime tính:

R = distance(I_left, I_right) / 2

v = (C − P) / R = (dx/R, dy/R)

Vector này có đơn vị bán kính mống mắt, không phải milimet, độ hay prism diopter. Chuẩn hóa giúp giảm ảnh hưởng thay đổi kích thước ảnh nhưng không loại hết ảnh hưởng phối cảnh và sai số điểm mốc. [S4]

Vector đầu vào gồm: dx và dy của mắt trái ảnh; dx và dy của mắt phải ảnh; hai thành phần trung bình common = (v_left + v_right)/2; hai thành phần difference = v_left − v_right; tỷ lệ min(R_left,R_right)/max(R_left,R_right); hai cờ hình học hợp lệ. Tổng cộng 11 giá trị. Chín giá trị đầu được chuẩn hóa bằng mean và std lưu trong checkpoint, có masking theo hai cờ hợp lệ. [S4, S11]

Ví dụ minh họa phép tính, không phải ca bệnh: R_left=20 px và C_left−P_left=(2,1) px thì v_left=(0,10;0,05). Nếu v_right=(−0,05;0,05), common=(0,025;0,05), difference=(0,15;0). Những số này là đầu vào mô hình; không thể tự suy ra một nhãn bệnh khi chưa chạy các nhánh còn lại.

### 4.3 Mười đặc trưng khe mi

Với mỗi mắt, hệ thống tìm trục chính PCA của đường viền khe mi; lấy hai đầu mút theo trục đó; tính vị trí đồng tử dọc trục u, vuông góc v và tỷ lệ chiều cao trên chiều rộng. Sáu giá trị của hai mắt cộng trung bình u, chênh lệch u và hai cờ hợp lệ tạo thành 10 đặc trưng. Các đầu mút này là cực trị của mask CNN, không phải khóe mắt đã được người xác nhận. [S5]

### 4.4 Quyết định cuối là phép học kết hợp

ResNet50 tạo vector ảnh 2048 chiều. Nhánh hình học biến 11 đặc trưng thành 32 chiều; ghép thành 2080 chiều rồi qua lớp 128 và lớp đầu ra 3. Nhánh khe mi chạy qua 10 → 32 → 3; logits của nhánh này chỉ được cộng khi cả hai mắt có đặc trưng khe mi hợp lệ. Sau đó softmax và argmax chọn lớp đề xuất. [S4–S5]

Vì ảnh pixel trực tiếp đi vào ResNet50, quyết định không được giải thích hoàn toàn bằng vector Hirschberg. Các yếu tố khác trong ảnh cũng có thể ảnh hưởng mô hình. Runtime chưa cung cấp bằng chứng quy kết theo từng ca rằng nhãn được quyết định bởi một đặc trưng hình học cụ thể. Muốn kiểm chứng cần thí nghiệm loại bỏ từng nhánh và đánh giá sai số trên dữ liệu độc lập.

## 5 Các điều kiện cho phép hoặc ngăn kết luận

| Điều kiện trong mã | Hệ quả và ý nghĩa |
| --- | --- |
| orientation phải là unmirrored hoặc mirrored | auto được chuyển thành unknown và input contract không đạt. Hệ thống không tự chứng minh lịch sử mirror từ ảnh tĩnh. |
| fixation phải là confirmed | unknown hoặc offaxis không được qua capture guard. confirmed là thông tin do bên gọi API xác nhận, không phải kết quả nhận biết định thị từ pixel. |
| Hai mắt có ROI và điểm mốc hợp lệ | Thiếu, không hữu hạn, ngoài ảnh hoặc ngoài ROI thì không đủ điều kiện chạy phân loại. Occluded và not_visible bị chặn. |
| EXIF orientation bằng 1 và không có nhiều mặt | Đầu vào khác quy ước này bị chặn; cần chuẩn hóa ảnh hoặc chụp lại. |
| Khoảng cách đồng tử đến phản xạ không vượt 0,4R | Đây là gate QC nghiên cứu; vượt gate khiến dừng, không phải bằng chứng bình thường hay một ngưỡng phân loại lác đã xác thực. |
| Phản xạ ở trong vùng tìm kiếm hợp lệ | Vùng này là giao của đĩa mống mắt, polygon khe mi và đĩa bán kính 0,4R quanh đồng tử. Thiếu viền mí, vùng rỗng hoặc cường độ phẳng bị loại. |
| Chênh đường kính hai mống mắt không quá 15% | Công thức abs(D_left−D_right)/max(D_left,D_right); vượt ngưỡng khiến QC hai mắt không đạt. |
| Nếu có hình học iris MediaPipe, tỷ số trục nằm trong 0,85 đến 1,0 | Ngoài khoảng thì bị đánh dấu Invalid Landmark / Low Confidence. Đây là tỷ lệ từ điểm mốc, không phải ellipse segmentation đã fit. |

Các ngưỡng nêu trên được mã đánh dấu chưa xác thực lâm sàng. Gate 0,4R có thể loại cả ảnh lệch lớn hoặc landmark sai; cần đo tỷ lệ từ chối theo từng nhóm bệnh để biết hệ thống bỏ sót nhóm nào. [S3, S6–S8]

QC còn tìm đốm sáng bằng ngưỡng xám >230, phép mở hình thái 3 px và thành phần kích thước 3–10 px. Các ứng viên được xếp theo tương phản và khoảng cách tới tâm mống mắt; chúng không thay tọa độ phản xạ CNN và không tự xác nhận đúng nguồn sáng. Không nên mô tả bước này là đã xác minh phản xạ giác mạc thật. [S6]

Cờ same_half_plane chỉ biểu thị tích vô hướng hai vector dương. Chính sách hiện tại không dùng cờ này để kết luận nhìn xéo, bình thường hay để chặn classifier. [S2, S7]

Kết quả có hai trạng thái chính: INCONCLUSIVE nghĩa là chưa đủ điều kiện đánh giá; HUMAN_REVIEW nghĩa là có đề xuất cần người xem xét. NORMAL chỉ có nghĩa mô hình chưa ghi nhận dấu hiệu rõ trong ảnh đạt điều kiện; không phải giấy xác nhận không có bệnh. Hai lớp còn lại là đề xuất lác trong hoặc lác ngoài. Không có lớp lác đứng, lác ẩn hoặc giả lác trong classifier đang chạy. clinicalDiagnosis và probabilities đều là null; confidence nếu có chỉ là điểm softmax lớn nhất chưa hiệu chuẩn. [S1, S5]

## 6 Bằng chứng huấn luyện và giới hạn độ tin cậy

| Bằng chứng có trong dự án | Diễn giải đúng |
| --- | --- |
| 54 ảnh normal, 39 esotropia, 19 exotropia | Tổng 112 ảnh được chọn để train; phân bố lệch lớp. Không được suy ra số người tham gia từ số ảnh. |
| split ghi all selected unique images TRAIN và subject overlap unresolved | Chưa có đánh giá tách người độc lập trong cấu hình này. |
| accuracy=1,0 và macro_f1=1,0 của candidate | Khớp tập train; không phải hiệu năng trên người mới hoặc ảnh mới. |
| Nhãn ghi doctor verification reported by owner ngày 11/10/2026 | Nguồn gốc xác nhận được khai báo trong cấu hình; chưa tương đương hồ sơ thẩm định nhãn độc lập có thể truy vết từng ca. |
| pupil control mean_px khoảng 1,868 trên 179 điểm visible | Sai số TRAIN FIT; không phải độ chính xác chẩn đoán. Hồ sơ cũng ghi false_visible_on_hidden=63, cảnh báo đầu visibility chưa đáng tin. |
| updated_pupil_classifier_pair_independently_evaluated=False | Đồng tử đang dùng khác checkpoint dùng khi huấn luyện classifier; cấu hình kết hợp mới chưa được đánh giá độc lập. |
| calibrated_probabilities=False và researchOnly=True | Điểm 0,98 không có nghĩa xác suất đúng 98%. Không có hiệu chuẩn hoặc ngưỡng quyết định lâm sàng đã chứng minh. |

Nguồn trực tiếp: completion của classifier, completion của pupil, config hình học và Engine.model_info. Một số metadata lịch sử còn ghi runtime_integrated=false hoặc runtime_promoted=false; việc đã được tích hợp hiện nay được xác định từ đường gọi thực tế và manifest, không suy ra chỉ từ các cờ lịch sử đó. [S2–S3, S9–S11]

Sai số các bước có thể truyền sang nhau: crop sai làm landmark sai; iris sai làm bán kính và prior đồng tử sai; đồng tử hoặc phản xạ sai làm vector sai; mask khe mi sai làm đặc trưng opening sai. Khi đó một điểm softmax lớn vẫn có thể đi kèm một kết luận sai. Các kiểm tra QC làm giảm một số tình huống đầu vào không hợp lệ nhưng chưa chứng minh hiệu quả loại bỏ mọi sai số.

## 7 Những điểm cần sửa hoặc làm rõ trong dự án

1. Luồng research hiện không truyền điều kiện chụp tới AI. measure_hirschberg gọi adapter predict_hirschberg; adapter gọi predict_trained_hirschberg chỉ với bytes, nên dùng orientation=auto và fixation=unknown. Theo input contract và capture guard, aiPrediction của luồng này sẽ không đạt điều kiện phân loại. Cần truyền metadata đã được xác nhận hoặc giải thích rõ đây là phép đo chưa có kết luận AI. Đây là suy luận trực tiếp từ đường gọi và giá trị mặc định, không phải kết quả thử mọi ảnh. [S1, S8, S14]
2. Không nhầm hai hệ đo hình học. API research có phép đo h_iris và h_pupil theo đường kính iris, có quy ước dấu nasal; classifier chính dùng vector theo bán kính iris trên trục ảnh. Với cùng độ dời, chia cho R tạo giá trị gấp hai lần chia cho D. Không dùng chung ngưỡng hoặc diễn giải hai trường này như nhau. [S4, S14]
3. Quy tắc giả lác chưa nằm trong đường gọi API chính. Class ResearchMeasurementService có rule 0,40 mm, bất đối xứng 0,50 mm và tỷ lệ củng mạc 0,70; nhưng router research gọi measure_research_request, không khởi tạo class này. Rule cần cornea_diameter_mm được cung cấp để có số đo mm. Không dùng đoạn mã đó làm bằng chứng hệ thống đang chẩn đoán giả lác; classifier chính ghi pseudostrabismus_supported=False. [S5, S14–S15]
4. Tài liệu cũ chưa đồng bộ. HIRSCHBERG_RUNTIME.md còn nói /api/v1/strabismus/predict tiếp tục dùng ONNX. app/main.py hiện chỉ đăng ký router Hirschberg và research; test_hirschberg_only xác nhận strabismus trả 404. ONNX nằm ở research/legacy_onnx; các candidate nghiên cứu và mô hình transfer không được suy ra là đang phục vụ suy luận. [S12, S15–S16]
5. Giao diện cần giữ trạng thái nghiên cứu. Không chỉ hiển thị predictedClass rồi bỏ qua status. Với HUMAN_REVIEW, hiển thị đây là dấu hiệu đề xuất cần kiểm tra; với INCONCLUSIVE, hiển thị lý do hoặc hướng dẫn chụp lại. Không biểu diễn confidence như tỷ lệ chắc chắn chẩn đoán.

Ưu tiên tiếp theo là đánh giá đúng cấu hình classifier và pupil đang triển khai trên tập độc lập tách theo người, kiểm tra ảnh trùng và frozen ROI, lấy nhãn đối chứng từ thăm khám phù hợp, báo cáo độ nhạy và độ đặc hiệu kèm khoảng tin cậy. Phải báo cáo cả tỷ lệ INCONCLUSIVE trên toàn bộ ảnh và theo từng lớp, thay vì chỉ tính hiệu năng trên ảnh đã qua gate. Cần kiểm tra độ lặp lại và khác biệt theo thiết bị, ánh sáng, độ tuổi, kính và điều kiện nguồn sáng. Đây là đề xuất kiểm chứng, không phải công việc đã hoàn thành.

## 8 Kiểm tra thực hiện và nguồn đối chiếu

Đã chạy scripts/verify_runtime_bundle.py: verified=true, đủ 40 tệp, tổng 101.074.925 byte. SHA-256 và kích thước khớp manifest của bộ runtime. Kiểm tra này không đo hiệu năng chẩn đoán.

Đã chạy 11 kiểm thử trong test_trained_hirschberg.py, test_hirschberg_only.py và test_production_api_runtime.py: 11 passed trong 17,61 giây. Các kiểm thử bao gồm nạp checkpoint thật, từ chối ảnh trống, từ chối artifact sai hash, kiểm tra đầu ra ba lớp, validation upload, xác nhận API strabismus không còn được cung cấp và hành vi runtime API. Có một cảnh báo deprecation của thư viện TestClient; không có test thất bại. Đây là kiểm thử phần mềm, không phải thử nghiệm độ chính xác lâm sàng. Phân tích không thay đổi weights, ngưỡng QC hoặc logic suy luận.

### Nguồn mã trong workspace

Các đường dẫn bên dưới tính từ thư mục gốc dự án; tên hàm và class là vị trí đối chiếu ổn định tại commit đã nêu. Trong S9–S11, RUN là thư mục app/hirschberg_runtime/research/hirschberg_localization/runs/.

- S1: app/services/trained_hirschberg_service.py — load_trained_hirschberg, predict_trained_hirschberg, screening_message.
- S2: app/hirschberg_runtime/manifest.json — modelId, pupilVersion, capturePolicy, danh mục hash.
- S3: app/hirschberg_runtime/scripts/hirschberg_engine.py — Engine.__init__, Engine.process, screen_with_guards.
- S4: app/hirschberg_runtime/research/hirschberg_localization/screening_fusion.py — geometry, image_panel, ScreeningFusion.
- S5: app/hirschberg_runtime/research/hirschberg_localization/screening_eyelid_branch.py — opening_features, ScreeningEyelidBranch, infer.
- S6: app/hirschberg_runtime/research/hirschberg_localization/reflex_roi_qc.py — CONFIG, diagnose, guarded_infer.
- S7: app/hirschberg_runtime/research/hirschberg_localization/binocular_guard.py và iris_radius_qc.py — compare_eyes, capture_guard, compare_diameters.
- S8: app/hirschberg_runtime/research/hirschberg_localization/input_contract.py — validate_input, human_review_only.
- S9: RUN/screening_eyelid_branch_20261011_v1/completion.json.
- S10: RUN/pupil_color_features_20261012_v1/completion.json; direct_pupil_inference.py trong localization — predict.
- S11: RUN/screening_verified_clear_directpupil_iris4_20261011_v1/config.json.
- S12: app/main.py, requirements.txt, README.md; automatic_upload.py, iris_four_points.py, pupil_iris_prior.py, eyelid_segmentation.py trong thư mục localization; scripts/hirschberg_localizer.py và export_hirschberg_review163_steps.py trong runtime.
- S13: app/api/hirschberg.py và app/services/hirschberg_session_service.py.
- S14: app/services/research_measurement_service.py — measure_hirschberg, _measure_eye, ResearchMeasurementService; app/services/hirschberg_ai_service.py — predict_hirschberg.
- S15: app/api/research_measurement.py và docs/backend/HIRSCHBERG_RUNTIME.md.
- S16: tests/model/test_hirschberg_only.py, test_trained_hirschberg.py, test_production_api_runtime.py; scripts/verify_runtime_bundle.py.

### Nguồn y khoa đối chiếu

- M1: American Academy of Ophthalmology, Clinical Examination of Ocular Alignment and Binocular Vision in Infants Under Six Months of Age, mục Strabismus Angle Assessment. https://www.aao.org/education/disease-review/51e6ab03-d57f-4e79-a314-3279c6ce662b — nguyên lý đối xứng phản xạ so với tâm đồng tử và ảnh hưởng của góc kappa.
- M2: Moran CORE, University of Utah, Alignment Assessment Hirschberg. https://morancore.utah.edu/basic-ophthalmology-review/alignment-assessment-hirschberg/ — cách thực hiện, hướng lệch phản xạ và hạn chế khi phân biệt giả lác.
- M3: AAO EyeWiki, Pseudostrabismus. https://eyewiki.aao.org/Pseudostrabismus — ảnh hưởng của giải phẫu và góc kappa tới biểu hiện giả lác trong hoặc giả lác ngoài.

Các nguồn y khoa được đối chiếu ngày 29/10/2026; chúng hỗ trợ giải thích nguyên lý, không xác thực mô hình RemiCare.
