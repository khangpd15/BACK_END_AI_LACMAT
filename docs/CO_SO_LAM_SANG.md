# CƠ SỞ LÂM SÀNG & CƠ CHẾ SINH HỌC THỊ GIÁC
## Dự án RemiCare Eye / Strabismus AI

---

## MỤC LỤC
1. [Tổng quan Y khoa về Bệnh lý Lác mắt (Strabismus)](#1-tổng-quan-y-khoa-về-bệnh-lý-lác-mắt-strabismus)
2. [Tiêu chuẩn vàng Lâm sàng: Nghiệm pháp Che mắt (Cover Test Protocol)](#2-tiêu-chuẩn-vàng-lâm-sàng-nghiệm-pháp-che-mắt-cover-test-protocol)
3. [Cơ sở Xử lý Tín hiệu Sinh học: Động học Nháy mắt (Blink Dynamics & EAR)](#3-cơ-sở-xử-lý-tín-hiệu-sinh-học-động-học-nháy-mắt-blink-dynamics--ear)
4. [Động học Vận nhãn & Trích xuất Đặc trưng Toán học (Oculomotor Biomechanics)](#4-động-học-vận-nhãn--trích-xuất-đặc-trưng-toán-học-oculomotor-biomechanics)
5. [Cơ sở Chuyển giao Tri thức & Thách thức Độ trễ Tần số (10–15 FPS vs. 60 Hz Lab Tracker)](#5-cơ-sở-chuyển-giao-tri-thức--thách-thức-độ-trễ-tần-số-1015-fps-vs-60-hz-lab-tracker)
6. [Đạo đức Y khoa & Bảo vệ Quyền riêng tư Dữ liệu Sinh trắc học](#6-đạo-đức-y-khoa--bảo-vệ-quyền-riêng-tư-dữ-liệu-sinh-trắc-học)
7. [Ý nghĩa Ứng dụng Lâm sàng & Khuyến cáo Giới hạn Y tế](#7-ý-nghĩa-ứng-dụng-lâm-sàng--khuyến-cáo-giới-hạn-y-tế)

---

## 1. TỔNG QUAN Y KHOA VỀ BỆNH LÝ LÁC MẮT (STRABISMUS)

### 1.1. Định nghĩa Lâm sàng
**Lác mắt (Strabismus / Heterotropia)** là tình trạng mất đồng trục nhãn cầu của hai mắt khi cùng nhìn về một vật tiêu (ocular misalignment). 
Trong trạng thái sinh lý bình thường (Orthotropia / Orthophoria), hai trục thị giác giao nhau chính xác tại điểm cố định trên hoàng điểm (Fovea centralis) của cả hai võng mạc, cho phép vỏ não chẩm tích hợp hai hình ảnh thành một cảm nhận không gian ba chiều duy nhất — gọi là **Thị giác hai mắt (Binocular Vision)** và **Hợp thị (Fusion)**.

Khi xảy ra lác mắt:
* **Ức chế vỏ não & Nhược thị (Cortical Suppression & Amblyopia):** Ở trẻ nhỏ (trước 7–8 tuổi), vỏ não sẽ tự động ức chế hình ảnh từ mắt lác để tránh hiện tượng song thị (nhìn đôi) hoặc lẫn hình (confusion). Tình trạng ức chế kéo dài không được can thiệp sớm dẫn đến **Nhược thị chức năng (Lazy Eye)** — giảm thị lực vĩnh viễn không thể điều chỉnh bằng kính thông thường.
* **Song thị & Rối loạn thị giác không gian:** Ở người trưởng thành, hệ thần kinh thị giác đã cố định, sự lệch trục đột ngột gây song thị dữ dội, chóng mặt và mất khả năng ước lượng chiều sâu / khoảng cách.

### 1.2. Phân loại Lâm sàng Trọng tâm
Hệ thống RemiCare tập trung phát hiện các dạng lệch trục vận nhãn cơ bản được phản ánh qua chuyển động phục hồi mống mắt:

| Phân loại | Tên Y khoa | Hướng lệch trục sinh lý | Biểu hiện khi che/bỏ che |
| :--- | :--- | :--- | :--- |
| **Lác trong** | Esotropia (ET) | Trục nhãn cầu lệch vào phía mũi | Mắt lác sẽ giật ra ngoài (outward saccade) để bắt lại vật tiêu khi mắt lành bị che |
| **Lác ngoài** | Exotropia (XT) | Trục nhãn cầu lệch ra phía thái dương | Mắt lác sẽ giật vào trong (inward saccade) để bắt lại vật tiêu khi mắt lành bị che |
| **Lác trên / Lác dưới** | Hypertropia (HT) / Hypotropia (HoT) | Trục nhãn cầu lệch lên trên hoặc xuống dưới | Mắt giật theo trục dọc (vertical corrective saccade) để lấy lại tiêu điểm |
| **Lác ẩn** | Heterophoria | Chỉ lệch khi cơ chế hợp thị bị phá vỡ | Mắt giữ nguyên trục khi nhìn hai mắt, chỉ lệch khi che và tái định thị khi bỏ che |
| **Lác hiện** | Heterotropia | Lệch trục liên tục hoặc từng lúc | Xuất hiện ngay trong điều kiện mở cả hai mắt |

---

## 2. TIÊU CHUẨN VÀNG LÂM SÀNG: NGHIỆM PHÁP CHE MẮT (COVER TEST PROTOCOL)

Trong thực hành nhãn khoa hàn lâm (theo Viện Nhãn khoa Hoa Kỳ - AAO), **Nghiệm pháp Che mắt (Cover Test)** là tiêu chuẩn vàng khách quan duy nhất để phân biệt và định lượng độ lác mắt.

### 2.1. Nghiệm pháp Che - Bỏ che (Cover-Uncover Test)
* **Pha Che (Cover Phase):** Bác sĩ đặt bản che trước mắt định thị (Fixating Eye). Khi mắt định thị bị mất thị trường, nếu mắt đối bên có lác hiện (Tropia), mắt đối bên bắt buộc phải thực hiện một chuyển động vận nhãn định thị lại — gọi là **Vận động sửa sai / Cố định lại (Refixation Saccade / Redressment movement)** để đưa hoàng điểm về trục thị giác.
* **Pha Bỏ che (Uncover Phase):** Khi lấy bản che ra, mắt vừa được bỏ che sẽ có xu hướng hồi phục về vị trí ban đầu nếu bệnh nhân có lác ẩn (Phoria) hoặc tiếp tục lệch trục nếu cơ chế cố định bị suy yếu.

### 2.2. Giao thức 3 Chu kỳ Chuẩn hóa của RemiCare (3-Cycle Standard Protocol)
Nhằm loại bỏ nhiễu ngẫu nhiên và kiểm tra tính tái lập (Repeatability & Consistency) theo chuẩn nghiên cứu lâm sàng, RemiCare triển khai giao thức 3 chu kỳ liên tiếp:

```
[Chu kỳ 1] -> Nhìn thẳng (Baseline) -> Che mắt Trái -> Bỏ che -> Che mắt Phải -> Bỏ che
[Chu kỳ 2] -> Nhìn thẳng (Baseline) -> Che mắt Trái -> Bỏ che -> Che mắt Phải -> Bỏ che
[Chu kỳ 3] -> Nhìn thẳng (Baseline) -> Che mắt Trái -> Bỏ che -> Che mắt Phải -> Bỏ che
```

Mỗi chu kỳ bao gồm các pha động học rõ rệt:
1. **BASELINE (Ổn định):** 1.0–1.5 giây đầu nhìn thẳng vào điểm cố định (Fixation Target) trên camera. Đo lường tọa độ trung vị làm chuẩn đối chiếu.
2. **COVER (Che mắt):** Bản che che kín một mắt, phá vỡ hợp thị một bên.
3. **UNCOVER (Bỏ che):** Khoảnh khắc bỏ che (t = 0 ms). Hệ thống mở cửa sổ quan sát vàng (Golden Diagnostic Window: 0–800 ms) để đo tốc độ và biên độ dịch chuyển mống mắt.
4. **TRACKING (Tái ổn định):** Ghi nhận trạng thái hội tụ hoặc lệch trục duy trì.

---

## 3. CƠ SỞ XỬ LÝ TÍN HIỆU SINH HỌC: ĐỘNG HỌC NHÁY MẮT (BLINK DYNAMICS & EAR)

Một trong những thách thức kỹ thuật lớn nhất khi đưa Cover Test từ phòng lab ra camera thông thường là hiện tượng **Nhiễu Nháy Mắt (Blink Artifacts)**.

### 3.1. Hiện tượng Bell (Bell's Phenomenon) & Rủi ro Chẩn đoán Sai
Về mặt sinh lý thần kinh học, khi con người chớp mắt (thời gian diễn ra khoảng 100–400 ms), mí mắt trên hạ xuống đồng thời kích hoạt **Hiện tượng Bell (Bell's Phenomenon)**: *Nhãn cầu có xu hướng phản xạ tự nhiên xoay giật lên trên và ra ngoài (upward and outward rotation)*.
* **Hậu quả nếu không lọc nháy mắt:** Thuật toán thị giác máy tính sẽ nhận diện chuyển động giật sinh lý của Hiện tượng Bell thành một chuyển động bất thường, dẫn đến chẩn đoán **dương tính giả (False Positive)** cho bệnh lý Lác trên (Hypertropia) hoặc Lác ngoài (Exotropia).

### 3.2. Thuật toán Eye Aspect Ratio (EAR) theo Soukupová & Čech (2016)
RemiCare tích hợp thuật toán phân tích tỷ lệ mở mí mắt **EAR (Eye Aspect Ratio)** dựa trên các mốc hình học mí mắt của MediaPipe:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

*Trong đó:*
* $p_1, p_4$: Tọa độ khóe mắt trong và khóe mắt ngoài (chiều ngang mí mắt).
* $p_2, p_6$ và $p_3, p_5$: Hai cặp tọa độ bờ mí trên và bờ mí dưới (chiều dọc mí mắt).

```
          p2       p3
          •--------•
     p1 •            • p4
          •--------•
          p6       p5
```

### 3.3. Cơ chế Khử nhiễu Sinh học của RemiCare
1. **Ngưỡng nháy mắt (Blink Thresholding):**
   * Mắt mở bình thường: $EAR \approx 0.28 - 0.38$.
   * Pha mí mắt khép / chớp mắt: $EAR < 0.20 - 0.22$.
2. **Lọc khung hình mất mống mắt (Iris Occlusion Filtering):** Khi $EAR < Threshold$, các tọa độ mống mắt (Iris Landmarks) trong khoảng thời gian này bị đánh dấu nhãn `INVALID_SAMPLE` và loại bỏ hoàn toàn khỏi chuỗi tính toán chuyển vị.
3. **Phân tách Saccade Thật vs. Chớp Mắt:** Nếu sự thay đổi tọa độ mống mắt trùng với thời điểm sụt giảm của đường cong EAR, hệ thống xác định đó là chớp mắt và không tính vào chỉ số lệch lác.

---

## 4. ĐỘNG HỌC VẬN NHÃN & TRÍCH XUẤT ĐẶC TRƯNG TOÁN HỌC (OCULOMOTOR BIOMECHANICS)

### 4.1. Vận động Nhãn cầu: Saccade vs. Fixation Drift
Cơ quan vận nhãn gồm 6 cơ ngoại nhãn (4 cơ thẳng: trên, dưới, trong, ngoài; 2 cơ chéo: lớn, bé) chịu sự điều khiển của các dây thần kinh sọ III, IV và VI.
* Chuyển động Redressment khi mắt lác định thị lại là một **chuyển động giật nhanh (Fixation Saccade)** với vận tốc đỉnh ($V_{peak}$) rất cao (thường từ $150^\circ/s$ đến $400^\circ/s$ tùy biên độ lệch).
* Mắt bình thường (không lác) khi bỏ che chỉ có các dao động vi giật sinh lý (Microsaccades $< 1^\circ$) và chuyển động trôi nhẹ (Fixation Drift).

### 4.2. Các Đặc trưng Toán học Chuẩn hóa Không Phụ thuộc Phần cứng (30 Shared Features)
Để loại bỏ sự sai lệch giữa các thiết bị khác nhau, RemiCare chuẩn hóa các độ dời chuyển động theo tỉ lệ giải phẫu học của chính bệnh nhân:

1. **Chuẩn hóa Tọa độ theo Khe mi & Khoảng cách Đồng tử:**
   * Mọi độ lệch $\Delta X, \Delta Y$ của tâm mống mắt đều được chia cho khoảng cách giữa hai khóe mắt ngoài (Palpebral Fissure Width) hoặc khoảng cách liên đồng tử (Interpupillary Distance - IPD). Điều này giúp kết quả không bị thay đổi dù bệnh nhân ngồi gần hay xa camera.
2. **Động học Vector Chuyển vị (Displacement Vectors):**
   * $signedDx, signedDy$: Đo lường hướng chuyển động (+: Sang phải/Lên trên, -: Sang trái/Xuống dưới).
   * $absDx, absDy$: Biên độ chuyển vị tuyệt đối của mống mắt trong pha Uncover.
3. **Động học Vận tốc & Gia tốc (Kinematics):**
   * Vận tốc đỉnh ($peakVelocity$): Phản ánh xung lực cơ vận nhãn trong pha tái định thị.
   * Thời gian đạt đỉnh ($timeToPeak$): Độ trễ sinh lý từ lúc bỏ che đến khi mắt bắt đầu giật định thị.
4. **Chỉ số Tái lập Đa chu kỳ (Cycle Consistency):**
   * Sai số chuẩn giữa 3 chu kỳ lặp lại. Bệnh lý lác mắt thực thể sẽ có hướng và biên độ chuyển động tương đồng giữa các chu kỳ ($Consistency > 0.85$), trong khi cử động liếc mắt ngẫu nhiên sẽ có độ phân tán lớn.

---

## 5. CƠ SỞ CHUYỂN GIAO TRI THỨC & THÁCH THỨC ĐỘ TRỄ TẦN SỐ (10–15 FPS vs. 60 HZ LAB TRACKER)

### 5.1. Giới hạn Định lý Lấy mẫu Nyquist-Shannon trên Webcam Phổ thông
* **Phòng nghiên cứu Lab (Mô hình Hàn Quốc):** Sử dụng máy ghi hình mắt hồng ngoại chuyên dụng 60 Hz (60 khung hình/giây). Một chuyển động Saccade 150 ms được ghi lại qua 9–10 frame chi tiết, vẽ nên đường cong vận tốc parabol hoàn hảo.
* **Thực tế khám từ xa / Webcam học đường (RemiCare):** Thiết bị phổ thông chỉ đạt 10–15 FPS (khoảng 66–100 ms giữa 2 frame liên tiếp). Một chuyển động Saccade chỉ kịp xuất hiện trên 1–3 frame.
* **Hiện tượng Lệch miền Dữ liệu (Domain Shift):** Nếu đưa trực tiếp mô hình huấn luyện từ máy 60 Hz vào phân tích video 10–15 FPS, các đặc trưng phụ thuộc thời gian (như `sampleCount`, `meanIntervalMs`, `estimatedHz`) sẽ gây sụp đổ độ chính xác.

### 5.2. Giải pháp Hợp đồng Đặc trưng Khách quan & Bộ Tổng hợp Đồng thuận (Consensus Aggregator)
1. **Lược bỏ các đặc trưng nhạy cảm với tần số:** Tuyệt đối không sử dụng số lượng mẫu hoặc khoảng cách thời gian thô làm đầu vào mô hình.
2. **Trích xuất đặc trưng hình thái tĩnh & tỉ lệ tương đối:** Tận dụng độ dịch chuyển tổng thể giữa vị trí trước che (Baseline) và vị trí sau bỏ che (Post-uncover steady state) — đây là thông số bất biến với tốc độ khung hình.
3. **Mô hình Huấn luyện Tương thích 10–15 FPS (`remicare_15fps_candidate.joblib`):** Được tối ưu hóa trên không gian đặc trưng thích ứng với tốc độ quét thấp, kết hợp giải thuật **Temporal Consensus Aggregator** để bầu chọn đa số giữa 3 chu kỳ khám.
4. **Vị trí của Mô hình Hàn Quốc (`korean_shared_model`):** Được bảo lưu trong hệ thống như một thước đo nghiên cứu chuẩn mực đối chứng (`comparison_models` / Transfer Experiment), minh bạch hóa sai lệch miền cho các chuyên gia thị giác máy tính.

---

## 6. ĐẠO ĐỨC Y KHOA & BẢO VỆ QUYỀN RIÊNG TƯ DỮ LIỆU SINH TRẮC HỌC

Tuân thủ nghiêm ngặt các nguyên tắc đạo đức y khoa (Tuyên ngôn Helsinki), Đạo luật HIPAA về thông tin sức khỏe và Nghị định 13/2023/NĐ-CP của Chính phủ Việt Nam về Bảo vệ Dữ liệu Cá nhân:

### 6.1. Nguyên tắc "Privacy-by-Design" (Bảo mật Từ Thiết kế)
* **Tuyệt đối Không Lưu trữ Ảnh Mắt / Khuôn mặt:** Ảnh chụp mắt và khuôn mặt là dữ liệu sinh trắc học cá nhân nhạy cảm nhất (Biometric PII). Hệ thống RemiCare đã **xóa bỏ hoàn toàn** việc lưu trữ file ảnh vào cơ sở dữ liệu (`cover_test_images`) và dịch vụ đám mây (Cloudinary/S3).
* **Số hóa Tọa độ Ẩn danh (De-identified Numeric Time-series):** Toàn bộ hình ảnh camera được xử lý tức thời tại RAM/Client để trích xuất thành các cặp tọa độ số học ($X, Y, Quality, Timestamp$). Sau khi trích xuất, luồng ảnh bị hủy ngay lập tức. Cơ sở dữ liệu chỉ lưu trữ các dãy số tọa độ vô danh, không thể tái tạo lại gương mặt người bệnh.

---

## 7. Ý NGHĨA ỨNG DỤNG LÂM SÀNG & KHUYẾN CÁO GIỚI HẠN Y TẾ

### 7.1. Định vị Ứng dụng: Công cụ Sàng lọc Sớm Cộng đồng (Screening & Triage Tool)
RemiCare được thiết kế như một giải pháp **Sàng lọc & Phân loại Ban đầu (Triage Tool)** phục vụ:
* Khám khúc xạ học đường tại các vùng sâu vùng xa, nơi thiếu bác sĩ nhãn nhi chuyên khoa.
* Hỗ trợ bác sĩ đa khoa và kỹ thuật viên khúc xạ phát hiện sớm các ca nghi ngờ lác ẩn / lác hiện để chuyển tuyến kịp thời trong "giai đoạn vàng" điều trị nhược thị (trước 7 tuổi).
* Theo dõi định kỳ sau phẫu thuật chỉnh lác hoặc sau tập thị giác hai mắt (Vision Therapy).

### 7.2. Tuyên bố Miễn trừ & Giới hạn Y tế (Clinical Disclaimer)
> ⚠️ **LƯU Ý QUAN TRỌNG:**
> Kết quả từ hệ thống trí tuệ nhân tạo RemiCare **KHÔNG PHẢI là kết luận chẩn đoán y khoa chính thức (Not a definitive Medical Diagnosis)** và không thay thế cho quy trình khám mắt toàn diện của Bác sĩ Chuyên khoa Mắt.
> 
> Hệ thống đưa ra cảnh báo không kết luận (`INCONCLUSIVE`) khi:
> 1. Bệnh nhân có tật giật cầu mắt bẩm sinh (Nystagmus) làm sai lệch chuyển động saccade.
> 2. Bệnh nhân bị sụp mi nặng (Severe Ptosis) che khuất mống mắt khiến $EAR$ không ổn định.
> 3. Tín hiệu theo dõi chất lượng thấp do ánh sáng ngược, chuyển động đầu quá mức hoặc sai lệch khoảng cách khám.
> 
> Mọi trường hợp có kết quả `STRABISMUS` hoặc nghi ngờ lâm sàng cần được chuyển tuyến đến bệnh viện mắt để thực hiện nghiệm pháp khám lăng kính (Prism Cover Test - PCT / Krimsky Test) và đo khúc xạ liệt điều tiết.
