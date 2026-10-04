# RemiCare Strabismus AI: Dataset Manifest Schema & Governance Standard

**Document ID:** `REMICARE-DATA-GOV-001`  
**Version:** `1.0.0`  
**Status:** Approved Standard  
**Target Domain:** Ophthalmic Computer Vision (Pediatric & Adult Strabismus Screening, Hirschberg & Pseudo-strabismus Differentiation)  

---

## 1. Mục Đích & Nguyên Tắc Cốt Lõi (Core Principles)

Tài liệu này quy định chuẩn manifest bắt buộc cho toàn bộ dữ liệu hình ảnh được thu thập, lưu trữ, tiền xử lý và sử dụng trong huấn luyện/thẩm định các mô hình AI thuộc hệ thống **RemiCare Strabismus AI**.

### 1.1. Nguyên tắc Không Rò Rỉ Bệnh Nhân (Strict Patient-level Disjoint Split)
- **Tuyệt đối cấm** chia tách tập train/val/test theo cấp độ ảnh (image-level) hoặc phiên chụp (session-level).
- **Mọi ảnh của cùng một bệnh nhân (`participant_id`) bắt buộc phải thuộc về DUY NHẤT một tập split (`train`, `val`, hoặc `test`).**
- Tỷ lệ giao thoa bệnh nhân giữa các split phải là **0.00%**. Mọi manifest vi phạm nguyên tắc này sẽ bị từ chối tự động tại cổng CI/CD Quality Gate.

### 1.2. Nguyên tắc Nhãn Chuẩn Y Khoa (Clinical Ground Truth)
- Nhãn mô hình AI suy luận **không bao giờ** được phép dùng làm ground truth để huấn luyện mô hình tiếp theo.
- Nhãn chỉ hợp lệ khi được bác sĩ nhãn khoa hoặc chuyên viên khúc xạ/chỉnh quang (orthoptist) xác nhận qua quy trình khám trực tiếp hoặc hội chẩn telemedicine đa chuyên gia.

### 1.3. Bảo Mật & Đạo Đức Y Sinh (HIPAA / GDPR / Medical Ethics)
- Dữ liệu định danh cá nhân trực tiếp (Họ tên, SĐT, Địa chỉ, CCCD/CMND, BHYT) bị loại bỏ hoàn toàn tại nguồn.
- Chỉ sử dụng mã định danh giả danh (Pseudonymous `participant_id` và `sample_id` chuẩn UUID v4).
- Phải có trạng thái đồng thuận (consent status) rõ ràng từ người giám hộ/bệnh nhân.

---

## 2. Đặc Tả Chi Tiết Các Trường Metadata (Field Specifications)

Mỗi mẫu ảnh trong manifest tương ứng với một bản ghi (record). Định dạng hỗ trợ: **Parquet** (khuyến nghị cho quy mô lớn) hoặc **JSONL / CSV**.

| Tên Trường (Field Name) | Kiểu Dữ Liệu | Bắt Buộc | Ràng Buộc & Giá Trị Hợp Lệ | Mô Tả Nghiệp Vụ & Ý Nghĩa Lâm Sàng |
| :--- | :--- | :---: | :--- | :--- |
| `sample_id` | `string` (UUIDv4) | **Có** | Định dạng chuẩn RFC 4122 (ví dụ: `c8d19760-449e-4e8c-8f4b-74d1a3c7489e`) | Khóa chính duy nhất cho từng ảnh / mẫu dữ liệu. |
| `participant_id` | `string` | **Có** | Chuỗi định danh giả danh (ví dụ: `P_HN_00124`) | Khóa liên kết bệnh nhân. Dùng làm nhóm phân chia (`group_id`) để đảm bảo không rò rỉ bệnh nhân giữa các split. |
| `image_path` | `string` | **Có** | Đường dẫn tương đối từ gốc dataset hoặc URI lưu trữ | Đường dẫn tới file ảnh gốc hoặc ảnh chuẩn hóa (`.png`, `.jpg`, `.jpeg`). |
| `label` | `string` (Enum) | **Có** | Một trong 5 giá trị: `normal`, `esotropia`, `exotropia`, `pseudostrabismus`, `poor_quality` | Phân loại chẩn đoán của mẫu ảnh. |
| `clinician_confirmed` | `boolean` | **Có** | `true` hoặc `false` | `true` nếu nhãn được bác sĩ chuyên khoa xác nhận; `false` nếu là nhãn sơ bộ hoặc dữ liệu đối chứng. |
| `diagnosis_source` | `string` | **Có** | Chuỗi mô tả nguồn (ví dụ: `orthoptist_cover_test`, `ophthalmologist_clinical_exam`, `telehealth_expert_panel`, `synthetic_control`) | Nguồn gốc và phương pháp xác lập chẩn đoán ground truth. |
| `age_group` | `string` (Enum) | **Có** | `infant_0_1`, `toddler_1_3`, `preschool_3_5`, `school_6_12`, `adolescent_13_18`, `adult_18_plus` | Nhóm tuổi của người tham gia. Rất quan trọng để đánh giá bias và hiệu năng theo lứa tuổi (đặc biệt trẻ em có nếp quạt epicanthus). |
| `capture_condition` | `object` hoặc `string` (JSON) | **Có** | Chứa `flash_present` (bool), `ambient` (string: `standard_clinic`, `dim`, `bright`, `natural_indoor`) | Điều kiện chụp: Bắt buộc ghi nhận có đèn flash hay không (phục vụ phản xạ Hirschberg) và môi trường sáng. |
| `head_pose_status` | `string` (Enum) | **Có** | `optimal`, `mild_tilt`, `rejected_exceeded_tolerance` | Trạng thái tư thế đầu: `optimal` (yaw/pitch < 5°), `mild_tilt` (yaw/pitch <= 10°, roll <= 5°), hoặc vượt ngưỡng. |
| `license` | `string` | **Có** | Ví dụ: `RemiCare_Proprietary_Clinical`, `CC-BY-NC-4.0`, `Research_Only_IRB_Approved` | Bản quyền sở hữu và giấy phép sử dụng của dữ liệu. |
| `consent_status` | `string` (Enum) | **Có** | `guardian_consented_full`, `research_only_deidentified`, `clinical_audit_only`, `withdrawn` | Trạng thái đồng thuận tham gia nghiên cứu y sinh của người giám hộ hoặc bệnh nhân. |
| `split` | `string` (Enum) | **Có** | `train`, `val`, `test` | Tập phân chia dữ liệu cho quá trình huấn luyện và đánh giá mô hình. |

---

## 3. Định Nghĩa 5 Lớp Chẩn Đoán (Disease Classification Ontology)

Hệ thống RemiCare Strabismus AI định nghĩa chính xác 5 class mục tiêu:

### 3.1. `normal` (Chính Thị / Không Lác)
- Hai trục thị giác song song khi nhìn thẳng vào tiêu điểm vô cực hoặc nguồn sáng.
- Phản xạ giác mạc Hirschberg (Corneal Light Reflex - CLR) nằm đối xứng ở cả hai mắt, thường lệch nhẹ về phía trong (nasal) từ +0.1mm đến +0.3mm do góc Kappa sinh lý bình thường.
- Không có bất đối xứng phản xạ ($|\Delta h| < 0.35\text{ mm}$).
- Nghiệm pháp Cover Test: Không có chuyển động điều chỉnh khi che/mở mắt.

### 3.2. `esotropia` (Lác Trong Thật)
- Một hoặc cả hai mắt bị lệch vào trong (về phía sống mũi).
- Phản xạ ánh sáng Hirschberg ở mắt lệch bị dịch chuyển ra phía ngoài (temporal) so với tâm đồng tử ($dx > +0.4\text{ mm}$ hoặc lệch temporal rõ rệt).
- Bất đối xứng Hirschberg giữa 2 mắt vượt ngưỡng bệnh lý.
- Cover Test: Mắt lệch thực hiện chuyển động tái định vị ra ngoài khi mắt lành bị che.

### 3.3. `exotropia` (Lác Ngoài Thật)
- Một hoặc cả hai mắt bị lệch ra ngoài (về phía thái dương).
- Phản xạ ánh sáng Hirschberg ở mắt lệch bị dịch chuyển vào phía trong (nasal) vượt quá góc Kappa sinh lý ($dx < -0.3\text{ mm}$).
- Bất đối xứng Hirschberg rõ rệt.
- Cover Test: Mắt lệch chuyển động vào trong khi mắt lành bị che.

### 3.4. `pseudostrabismus` (Giả Lác - Thường là Giả Lác Trong do Nếp Quạt Epicanthus)
- **Đặc điểm hình thái lâm sàng:** Bề ngoài tạo cảm giác như bị lác trong do nếp quạt mí trong (epicanthal fold) che khuất một phần củng mạc (lòng trắng) phía trong, hoặc do sống mũi trẻ còn bẹt, phẳng và khoảng cách hai góc mắt trong rộng (telecanthus).
- **Đặc điểm phản xạ giác mạc Hirschberg chuẩn:** Phản xạ ánh sáng Hirschberg ở **cả hai mắt hoàn toàn bình thường, cân đối và đồng trục** ($|dx| < 0.4\text{ mm}$, $|\Delta h| < 0.35\text{ mm}$).
- **Tỉ lệ củng mạc trong/ngoài:** Diện tích củng mạc khóe trong hẹp rõ rệt so với khóe ngoài (`nasal_to_temporal_ratio < 0.65`).
- **Ý nghĩa sống còn trong AI:** Phải phân biệt tuyệt đối giữa `pseudostrabismus` và `esotropia`. Chẩn đoán nhầm trẻ giả lác thành lác trong gây lo âu không cần thiết cho phụ huynh và chỉ định phẫu thuật sai; ngược lại, bỏ sót lác trong thật sẽ dẫn đến nhược thị (amblyopia) vĩnh viễn ở trẻ.

### 3.5. `poor_quality` (Ảnh Không Đạt Tiêu Chuẩn Chất Lượng)
- Ảnh bị nhòe mờ nghiêm trọng (Laplacian variance < 100.0).
- Góc xoay đầu vượt ngưỡng dung sai (Pitch/Yaw > 10°, Roll > 5°).
- Mắt bị nhắm, chớp hoặc bị che khuất (occlusion).
- Thiếu phản xạ ánh sáng (không có flash hoặc bị lóa diện rộng).
- Ảnh không đạt sẽ được Quality Gatekeeper gắn nhãn `poor_quality` và yêu cầu chụp lại thay vì đưa ra chẩn đoán sai lệch.

---

## 4. Định Dạng Lưu Trữ & Ví Dụ Manifest

### 4.1. Ví Dụ Bản Ghi JSONL (`datasets/manifest.jsonl`)

```json
{
  "sample_id": "a5e8f492-9c1a-4d78-b118-2e33d0781290",
  "participant_id": "PT_PED_2026_0045",
  "image_path": "images/raw/PT_PED_2026_0045_01.jpg",
  "label": "pseudostrabismus",
  "clinician_confirmed": true,
  "diagnosis_source": "orthoptist_cover_test",
  "age_group": "toddler_1_3",
  "capture_condition": {
    "flash_present": true,
    "ambient": "standard_clinic",
    "device_type": "iphone_13_rear_flash"
  },
  "head_pose_status": "optimal",
  "license": "RemiCare_Proprietary_Clinical",
  "consent_status": "guardian_consented_full",
  "split": "train"
}
```

```json
{
  "sample_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
  "participant_id": "PT_PED_2026_0089",
  "image_path": "images/raw/PT_PED_2026_0089_01.jpg",
  "label": "esotropia",
  "clinician_confirmed": true,
  "diagnosis_source": "ophthalmologist_clinical_exam",
  "age_group": "preschool_3_5",
  "capture_condition": {
    "flash_present": true,
    "ambient": "standard_clinic",
    "device_type": "galaxy_s22_flash"
  },
  "head_pose_status": "optimal",
  "license": "RemiCare_Proprietary_Clinical",
  "consent_status": "guardian_consented_full",
  "split": "test"
}
```

---

## 5. Schema Validation Bằng Pydantic (Python Specification)

```python
from enum import Enum
from typing import Optional, Union, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class DiagnosisLabel(str, Enum):
    NORMAL = "normal"
    ESOTROPIA = "esotropia"
    EXOTROPIA = "exotropia"
    PSEUDOSTRABISMUS = "pseudostrabismus"
    POOR_QUALITY = "poor_quality"


class AgeGroup(str, Enum):
    INFANT_0_1 = "infant_0_1"
    TODDLER_1_3 = "toddler_1_3"
    PRESCHOOL_3_5 = "preschool_3_5"
    SCHOOL_6_12 = "school_6_12"
    ADOLESCENT_13_18 = "adolescent_13_18"
    ADULT_18_PLUS = "adult_18_plus"


class HeadPoseStatus(str, Enum):
    OPTIMAL = "optimal"
    MILD_TILT = "mild_tilt"
    REJECTED_EXCEEDED_TOLERANCE = "rejected_exceeded_tolerance"


class ConsentStatus(str, Enum):
    GUARDIAN_CONSENTED_FULL = "guardian_consented_full"
    RESEARCH_ONLY_DEIDENTIFIED = "research_only_deidentified"
    CLINICAL_AUDIT_ONLY = "clinical_audit_only"
    WITHDRAWN = "withdrawn"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CaptureCondition(BaseModel):
    flash_present: bool = Field(..., description="Có đèn flash trợ sáng để tạo phản xạ giác mạc hay không")
    ambient: str = Field(default="standard_clinic", description="Môi trường ánh sáng xung quanh")
    device_type: Optional[str] = Field(default=None, description="Loại cảm biến/thiết bị chụp")


class ImageSampleManifestSchema(BaseModel):
    sample_id: UUID = Field(..., description="Khóa định danh UUID duy nhất của mẫu ảnh")
    participant_id: str = Field(..., min_length=3, description="Mã người bệnh/chủ thể duy nhất")
    image_path: str = Field(..., min_length=1, description="Đường dẫn file ảnh")
    label: DiagnosisLabel = Field(..., description="1 trong 5 nhãn bệnh lý chuẩn")
    clinician_confirmed: bool = Field(..., description="Được xác nhận bởi bác sĩ/chuyên gia")
    diagnosis_source: str = Field(..., min_length=2, description="Nguồn chẩn đoán lâm sàng")
    age_group: AgeGroup = Field(..., description="Nhóm tuổi của bệnh nhân")
    capture_condition: Union[CaptureCondition, Dict[str, Any], str] = Field(..., description="Điều kiện chụp")
    head_pose_status: HeadPoseStatus = Field(..., description="Trạng thái góc xoay đầu")
    license: str = Field(..., min_length=2, description="Bản quyền/Giấy phép sử dụng")
    consent_status: ConsentStatus = Field(..., description="Trạng thái đồng thuận đạo đức y sinh")
    split: DatasetSplit = Field(..., description="Tập dữ liệu: train, val hoặc test")

    @field_validator("participant_id")
    @classmethod
    def validate_participant_id(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("participant_id cannot be blank")
        return v_clean
```

---

## 6. Tiêu Chuẩn Thẩm Tra Quản Trị Dữ Liệu (Automated Governance Checks)

Mọi manifest trước khi merge vào nhánh chính hoặc đưa vào pipeline huấn luyện phải vượt qua script `scripts/verify_dataset_governance.py` với các tiêu chí:

1. **Schema Check:** 100% bản ghi tuân thủ chặt chẽ Pydantic schema trên, không thiếu trường bắt buộc, không sai định dạng UUID.
2. **Patient Disjoint Check:**
   $$\text{Set}(Participant_{\text{train}}) \cap \text{Set}(Participant_{\text{val}}) = \emptyset$$
   $$\text{Set}(Participant_{\text{train}}) \cap \text{Set}(Participant_{\text{test}}) = \emptyset$$
   $$\text{Set}(Participant_{\text{val}}) \cap \text{Set}(Participant_{\text{test}}) = \emptyset$$
3. **Cryptographic & Perceptual Hash Check:**
   - Không có 2 ảnh trùng lặp SHA-256 nằm ở 2 split khác nhau.
   - Không có 2 ảnh có khoảng cách pHash $\le 4$ nằm ở 2 split khác nhau (ngăn chặn crop biến thể, resize, hoặc ảnh tương đồng rò rỉ vào tập test).
4. **Consent Check:** 100% mẫu trong tập huấn luyện và kiểm thử phải có `consent_status != 'withdrawn'`.
