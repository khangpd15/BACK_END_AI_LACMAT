#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_harschberg.py - Lọc và phân loại bộ dữ liệu train cho phương pháp Hirschberg.

Chức năng:
1. Phát hiện khuôn mặt và mắt bằng MediaPipe Face Mesh (refine_landmarks=True)
   kèm cơ chế Fallback thông minh cho ảnh crop 2 mắt (binocular eye strip 224x224).
2. Phát hiện chấm phản xạ ánh sáng flash trên giác mạc (corneal light reflex / Purkinje spot)
   ở CẢ 2 MẮT (OD và OS).
3. Phân loại ảnh hợp lệ vào 3 thư mục trong `harschberg_data_detect/`:
   - `esotropia_harschberg/`  (Lác trong)
   - `exotropia_harschberg/`  (Lác ngoài)
   - `normal_harschberg/`     (Mắt thẳng, không lác)
4. Sao chép ảnh bị loại vào `harschberg_data_detect/rejected/` kèm lý do cụ thể.
5. Hỗ trợ chế độ `--dry-run`, xuất `report.csv`, và lưu 20 ảnh debug trực quan.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Cấu hình logging và stdout UTF-8 cho Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("filter_harschberg")

# ==============================================================================
# THAM SỐ CẤU HÌNH PHÁT HIỆN CHẤM TRẮNG & MẮT (CÓ THỂ ĐIỀU CHỈNH)
# ==============================================================================
# 1. Ngưỡng độ sáng phản xạ flash giác mạc (0 - 255)
MIN_BRIGHTNESS_THRESHOLD = 205      # Ngưỡng sáng tuyệt đối tối thiểu của điểm phản xạ
TOP_PERCENTILE_THRESHOLD = 98.5     # Ngưỡng bách phân vị luma trong vùng mống mắt
MIN_EYE_MAX_LUMA = 195              # Độ sáng cao nhất trong mắt phải đạt tối thiểu mức này

# 2. Tiêu chuẩn kích thước và hình dạng chấm trắng (pixel trên ảnh 224x224)
MIN_REFLEX_AREA_PX = 1              # Diện tích tối thiểu (pixel)
MAX_REFLEX_AREA_PX = 75             # Diện tích tối đa để loại vùng chói lớn, da bóng, tròng trắng
MAX_REFLEX_ASPECT_RATIO = 3.0       # Tỷ lệ dài/rộng tối đa (loại trừ vệt lóa dài hoặc lông mi)

# 3. Vị trí không gian so với tâm mống mắt / đồng tử
MAX_DIST_TO_IRIS_CENTER_RATIO = 1.25 # Khoảng cách tối đa từ chấm trắng tới tâm mống mắt (so với bán kính mống mắt)
MIN_LOCAL_CONTRAST = 15.0           # Độ tương phản tối thiểu giữa chấm sáng và viền xung quanh

# 4. Ngưỡng phân loại Heuristic độ lệch Hirschberg (khi không có nhãn sẵn)
# Độ lệch = (tọa độ reflex_x - iris_center_x) / iris_radius
# Lưu ý: Đây là gán nhãn tạm thời heuristic, cần chuyên gia lâm sàng thẩm định lại.
HEURISTIC_NORMAL_BOUND = 0.22       # Nếu độ lệch |disp| <= 0.22 ở cả 2 mắt => Normal
HEURISTIC_STRABISMUS_BOUND = 0.25   # Độ lệch vượt ngưỡng sẽ phân biệt Esotropia vs Exotropia

# 5. MediaPipe Landmark Indices
LEFT_IRIS_CENTER = 468   # Mống mắt trái (viewer's right)
RIGHT_IRIS_CENTER = 473  # Mống mắt phải (viewer's left)


def resolve_source_directory(src_arg: str) -> Path:
    """Xác định đường dẫn thư mục nguồn linh hoạt."""
    src_path = Path(src_arg)
    if src_path.exists():
        return src_path

    # Kiểm tra các đường dẫn tương đương trong workspace
    candidates = [
        Path("data_hirschberg/eye-classification/train"),
        Path("data_hirschberg/train"),
        Path("data_harschberg/train"),
        Path("data/hirschberg/train"),
    ]
    for cand in candidates:
        if cand.exists():
            logger.warning("Đường dẫn '%s' không tồn tại. Tự động sử dụng thư mục thay thế: '%s'", src_arg, cand)
            return cand

    raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: '{src_arg}'. Vui lòng kiểm tra lại đường dẫn.")


def collect_images(src_dir: Path) -> List[Tuple[Path, str]]:
    """Quét tất cả ảnh và lấy nhãn ban đầu (nếu có từ subfolder hoặc tên file)."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: List[Tuple[Path, str]] = []

    for path in sorted(src_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in valid_exts:
            # Xác định nhãn ban đầu
            parent_name = path.parent.name.upper()
            file_name = path.stem.upper()
            
            source_label = "UNKNOWN"
            if "ESOTROPIA" in parent_name or "ESOTROPIA" in file_name:
                source_label = "esotropia"
            elif "EXOTROPIA" in parent_name or "EXOTROPIA" in file_name:
                source_label = "exotropia"
            elif "NORMAL" in parent_name or "NORMAL" in file_name:
                source_label = "normal"

            images.append((path, source_label))

    return images


def detect_reflex_in_eye_roi(
    eye_gray: np.ndarray,
    iris_center_roi: Tuple[float, float],
    iris_radius_px: float,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Phát hiện chấm phản xạ flash nhỏ, tròn, rất sáng trong phạm vi mống mắt."""
    h, w = eye_gray.shape[:2]
    cx, cy = iris_center_roi

    # Bán kính vùng tìm kiếm quanh mống mắt
    search_radius = max(8.0, iris_radius_px * MAX_DIST_TO_IRIS_CENTER_RATIO)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(search_radius)), 255, -1)

    masked_gray = cv2.bitwise_and(eye_gray, mask)
    max_val = np.max(masked_gray)
    if max_val < MIN_EYE_MAX_LUMA:
        return False, None, "LOW_PEAK_BRIGHTNESS"

    # Ngưỡng sáng thích ứng: kết hợp ngưỡng tối thiểu và mức sáng đỉnh
    thresh_val = max(MIN_BRIGHTNESS_THRESHOLD, int(max_val - 20))
    _, binary = cv2.threshold(eye_gray, thresh_val, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates: List[Dict[str, Any]] = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        c_x, c_y = float(centroids[i][0]), float(centroids[i][1])

        # Loại trừ kích thước không thỏa mãn
        if not (MIN_REFLEX_AREA_PX <= area <= MAX_REFLEX_AREA_PX):
            continue

        # Kiểm tra tỷ lệ dài/rộng
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > MAX_REFLEX_ASPECT_RATIO:
            continue

        # Khoảng cách tới tâm mống mắt
        dist = np.hypot(c_x - cx, c_y - cy)
        if dist > search_radius:
            continue

        # Kiểm tra độ tương phản cục bộ với viền xung quanh (để loại da bị chói hoặc tròng trắng)
        rx, ry = int(round(c_x)), int(round(c_y))
        box_rad = max(4, int(round(np.sqrt(area) + 3)))
        y1, y2 = max(0, ry - box_rad), min(h, ry + box_rad + 1)
        x1, x2 = max(0, rx - box_rad), min(w, rx + box_rad + 1)
        sub_roi = eye_gray[y1:y2, x1:x2]
        spot_luma = float(np.mean(sub_roi[sub_roi >= thresh_val])) if np.any(sub_roi >= thresh_val) else float(max_val)
        surround_luma = float(np.mean(sub_roi[sub_roi < thresh_val])) if np.any(sub_roi < thresh_val) else 0.0
        contrast = spot_luma - surround_luma
        if contrast < MIN_LOCAL_CONTRAST:
            continue

        candidates.append({
            "x": c_x,
            "y": c_y,
            "area": area,
            "dist": dist,
            "aspect": aspect,
            "contrast": contrast,
            "brightness": float(max_val),
        })

    if not candidates:
        return False, None, "NO_VALID_COMPACT_REFLEX"

    # Chọn ứng viên gần tâm mống mắt nhất
    best_candidate = min(candidates, key=lambda c: c["dist"])
    return True, best_candidate, "OK"


def process_image(
    image_path: Path,
    face_mesh: Optional[Any] = None,
    allow_crop_fallback: bool = True,
) -> Dict[str, Any]:
    """Phân tích một ảnh: phát hiện mặt/mắt, tìm phản xạ 2 mắt, và tính toán độ lệch."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {
            "success": False,
            "rejection_reason": "CANNOT_READ_IMAGE",
            "method": "NONE",
            "has_right": False,
            "has_left": False,
        }

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detected_method = "NONE"
    right_iris: Optional[Tuple[float, float, float]] = None # (cx, cy, radius)
    left_iris: Optional[Tuple[float, float, float]] = None

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 1: Phát hiện bằng MediaPipe Face Mesh
    # --------------------------------------------------------------------------
    if face_mesh is not None:
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            # Mắt phải (OD - viewer's left)
            od_cx = lms[RIGHT_IRIS_CENTER].x * w
            od_cy = lms[RIGHT_IRIS_CENTER].y * h
            # Bán kính mống mắt từ các điểm xung quanh mống mắt phải (474, 476)
            od_r1 = np.hypot((lms[474].x - lms[RIGHT_IRIS_CENTER].x) * w, (lms[474].y - lms[RIGHT_IRIS_CENTER].y) * h)
            od_r2 = np.hypot((lms[476].x - lms[RIGHT_IRIS_CENTER].x) * w, (lms[476].y - lms[RIGHT_IRIS_CENTER].y) * h)
            od_r = max(6.0, float((od_r1 + od_r2) / 2.0))

            # Mắt trái (OS - viewer's right)
            os_cx = lms[LEFT_IRIS_CENTER].x * w
            os_cy = lms[LEFT_IRIS_CENTER].y * h
            os_r1 = np.hypot((lms[469].x - lms[LEFT_IRIS_CENTER].x) * w, (lms[469].y - lms[LEFT_IRIS_CENTER].y) * h)
            os_r2 = np.hypot((lms[471].x - lms[LEFT_IRIS_CENTER].x) * w, (lms[471].y - lms[LEFT_IRIS_CENTER].y) * h)
            os_r = max(6.0, float((os_r1 + os_r2) / 2.0))

            right_iris = (od_cx, od_cy, od_r)
            left_iris = (os_cx, os_cy, os_r)
            detected_method = "FACEMESH_LANDMARKS"

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 2: Fallback cho ảnh Crop 2 mắt (Periocular strip 224x224)
    # --------------------------------------------------------------------------
    if (right_iris is None or left_iris is None) and allow_crop_fallback:
        # Trong ảnh crop 2 mắt chuẩn hóa:
        # Nửa bên trái ảnh (viewer's left) là mắt phải của bệnh nhân (OD)
        # Nửa bên phải ảnh (viewer's right) là mắt trái của bệnh nhân (OS)
        mid_x = w // 2
        
        # Mắt phải (OD): nửa trái
        od_roi_gray = gray[:, :mid_x]
        od_blurred = cv2.GaussianBlur(od_roi_gray, (7, 7), 1.5)
        # Vùng trung tâm mắt (tránh viền ngoài cùng)
        od_inner = od_blurred[int(h * 0.15):int(h * 0.85), int(mid_x * 0.12):int(mid_x * 0.88)]
        _, _, od_min_loc, _ = cv2.minMaxLoc(od_inner)
        od_cx = float(od_min_loc[0] + int(mid_x * 0.12))
        od_cy = float(od_min_loc[1] + int(h * 0.15))
        od_r = float(min(mid_x, h) * 0.28)
        right_iris = (od_cx, od_cy, od_r)

        # Mắt trái (OS): nửa phải
        os_roi_gray = gray[:, mid_x:]
        os_blurred = cv2.GaussianBlur(os_roi_gray, (7, 7), 1.5)
        os_inner = os_blurred[int(h * 0.15):int(h * 0.85), int(mid_x * 0.12):int(mid_x * 0.88)]
        _, _, os_min_loc, _ = cv2.minMaxLoc(os_inner)
        os_cx = float(os_min_loc[0] + int(mid_x * 0.12) + mid_x)
        os_cy = float(os_min_loc[1] + int(h * 0.15))
        os_r = float(min(mid_x, h) * 0.28)
        left_iris = (os_cx, os_cy, os_r)

        detected_method = "PERIOCULAR_CROP_HEURISTIC"

    if right_iris is None or left_iris is None:
        return {
            "success": False,
            "rejection_reason": "NO_FACE_OR_EYES_DETECTED",
            "method": detected_method,
            "has_right": False,
            "has_left": False,
        }

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 3: Phát hiện phản xạ flash ở CẢ 2 MẮT
    # --------------------------------------------------------------------------
    # 1. Mắt phải (OD)
    od_cx, od_cy, od_r = right_iris
    od_ok, od_reflex, od_msg = detect_reflex_in_eye_roi(gray, (od_cx, od_cy), od_r)

    # 2. Mắt trái (OS)
    os_cx, os_cy, os_r = left_iris
    os_ok, os_reflex, os_msg = detect_reflex_in_eye_roi(gray, (os_cx, os_cy), os_r)

    # Đánh giá đạt / không đạt
    if not od_ok and not os_ok:
        rejection_reason = f"MISSING_BOTH_REFLEXES (OD: {od_msg}, OS: {os_msg})"
        success = False
    elif not od_ok:
        rejection_reason = f"MISSING_RIGHT_EYE_REFLEX ({od_msg})"
        success = False
    elif not os_ok:
        rejection_reason = f"MISSING_LEFT_EYE_REFLEX ({os_msg})"
        success = False
    else:
        rejection_reason = ""
        success = True

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 4: Tính toán độ lệch Hirschberg (Displacement Heuristic)
    # --------------------------------------------------------------------------
    od_disp_ratio: Optional[float] = None
    os_disp_ratio: Optional[float] = None
    heuristic_label = "unknown"

    if success and od_reflex is not None and os_reflex is not None:
        # Độ lệch ngang = reflex_x - iris_center_x
        # Chuẩn hóa theo bán kính mống mắt
        od_dx = (od_reflex["x"] - od_cx) / max(1.0, od_r)
        os_dx = (os_reflex["x"] - os_cx) / max(1.0, os_r)
        od_disp_ratio = round(float(od_dx), 4)
        os_disp_ratio = round(float(os_dx), 4)

        # Định hướng giải phẫu:
        # OD (Mắt phải - viewer's left): phía mũi (Nasal) là sang phải (+dx), phía thái dương (Temporal) là sang trái (-dx)
        # OS (Mắt trái - viewer's right): phía mũi (Nasal) là sang trái (-dx), phía thái dương (Temporal) là sang phải (+dx)
        
        # Nếu chấm sáng nằm gần tâm cả 2 mắt => Normal
        if abs(od_dx) <= HEURISTIC_NORMAL_BOUND and abs(os_dx) <= HEURISTIC_NORMAL_BOUND:
            heuristic_label = "normal"
        # Chấm sáng lệch về phía thái dương (OD -dx, OS +dx) => Mắt lác vào trong (Esotropia)
        elif od_dx < -HEURISTIC_STRABISMUS_BOUND or os_dx > HEURISTIC_STRABISMUS_BOUND:
            heuristic_label = "esotropia"
        # Chấm sáng lệch về phía mũi (OD +dx, OS -dx) => Mắt lác ra ngoài (Exotropia)
        elif od_dx > HEURISTIC_STRABISMUS_BOUND or os_dx < -HEURISTIC_STRABISMUS_BOUND:
            heuristic_label = "exotropia"
        else:
            heuristic_label = "normal"

    return {
        "success": success,
        "rejection_reason": rejection_reason,
        "method": detected_method,
        "has_right": od_ok,
        "has_left": os_ok,
        "right_iris": right_iris,
        "left_iris": left_iris,
        "right_reflex": od_reflex,
        "left_reflex": os_reflex,
        "od_disp_ratio": od_disp_ratio,
        "os_disp_ratio": os_disp_ratio,
        "heuristic_label": heuristic_label,
        "image_shape": (h, w),
    }


def draw_debug_visualization(
    img_bgr: np.ndarray,
    result: Dict[str, Any],
    label_text: str,
) -> np.ndarray:
    """Vẽ vòng tròn trực quan quanh chấm trắng và tâm mống mắt để kiểm tra."""
    vis = img_bgr.copy()

    # Vẽ tâm mống mắt (màu đỏ)
    if result.get("right_iris"):
        rcx, rcy, rr = result["right_iris"]
        cv2.circle(vis, (int(round(rcx)), int(round(rcy))), 3, (0, 0, 255), -1)
        cv2.circle(vis, (int(round(rcx)), int(round(rcy))), int(round(rr)), (0, 0, 180), 1)

    if result.get("left_iris"):
        lcx, lcy, lr = result["left_iris"]
        cv2.circle(vis, (int(round(lcx)), int(round(lcy))), 3, (0, 0, 255), -1)
        cv2.circle(vis, (int(round(lcx)), int(round(lcy))), int(round(lr)), (0, 0, 180), 1)

    # Vẽ chấm phản xạ flash (vòng tròn xanh lá đậm nét + tâm vàng)
    if result.get("right_reflex"):
        ref = result["right_reflex"]
        rx, ry = int(round(ref["x"])), int(round(ref["y"]))
        cv2.circle(vis, (rx, ry), 6, (0, 255, 0), 2)
        cv2.circle(vis, (rx, ry), 1, (0, 255, 255), -1)

    if result.get("left_reflex"):
        ref = result["left_reflex"]
        lx, ly = int(round(ref["x"])), int(round(ref["y"]))
        cv2.circle(vis, (lx, ly), 6, (0, 255, 0), 2)
        cv2.circle(vis, (lx, ly), 1, (0, 255, 255), -1)

    # Chú thích nhãn
    cv2.putText(
        vis,
        label_text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0) if result["success"] else (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def get_safe_destination_path(dest_folder: Path, original_filename: str) -> Path:
    """Tạo đường dẫn file đích, tự thêm hậu tố nếu trùng tên để không ghi đè."""
    base_name = Path(original_filename).stem
    suffix = Path(original_filename).suffix
    dest_path = dest_folder / f"{base_name}{suffix}"

    counter = 1
    while dest_path.exists():
        dest_path = dest_folder / f"{base_name}_{counter}{suffix}"
        counter += 1

    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Lọc và phân loại ảnh Hirschberg giác mạc.")
    parser.add_argument(
        "--src",
        type=str,
        default="data_hirschberg/eye-classification/train",
        help="Thư mục train chứa ảnh nguồn (mặc định: data_hirschberg/eye-classification/train)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="harschberg_data_detect",
        help="Thư mục gốc chứa kết quả phân loại (mặc định: harschberg_data_detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chế độ chạy thử nghiệm: chỉ quét, thống kê và xuất báo cáo, KHÔNG copy file",
    )
    parser.add_argument(
        "--max-debug-samples",
        type=int,
        default=20,
        help="Số lượng ảnh debug trực quan cần lưu vào thư mục debug/ (mặc định: 20)",
    )
    parser.add_argument(
        "--strict-facemesh-only",
        action="store_true",
        help="Chỉ dùng MediaPipe Face Mesh thuần túy, không dùng fallback crop 2 mắt",
    )

    args = parser.parse_args()

    # 1. Xác thực thư mục
    src_dir = resolve_source_directory(args.src)
    dest_root = Path(args.dest)
    dry_run = args.dry_run
    allow_crop_fallback = not args.strict_facemesh_only

    logger.info("=" * 70)
    logger.info("HIRSCHBERG DATASET FILTER & CLASSIFIER")
    logger.info("=" * 70)
    logger.info("Thư mục nguồn:   %s", src_dir.resolve())
    logger.info("Thư mục đích:    %s", dest_root.resolve())
    logger.info("Chế độ DRY-RUN:  %s", "BẬT (Không thực hiện copy)" if dry_run else "TẮT (Chạy thật & Copy file)")
    logger.info("Crop Fallback:   %s", "BẬT (Hỗ trợ ảnh crop 2 mắt 224x224)" if allow_crop_fallback else "TẮT")

    # 2. Thiết lập các thư mục con trong harschberg_data_detect/
    folder_esotropia = dest_root / "esotropia_harschberg"
    folder_exotropia = dest_root / "exotropia_harschberg"
    folder_normal = dest_root / "normal_harschberg"
    folder_rejected = dest_root / "rejected"
    folder_debug = dest_root / "debug"

    if not dry_run:
        folder_esotropia.mkdir(parents=True, exist_ok=True)
        folder_exotropia.mkdir(parents=True, exist_ok=True)
        folder_normal.mkdir(parents=True, exist_ok=True)
        folder_rejected.mkdir(parents=True, exist_ok=True)
        folder_debug.mkdir(parents=True, exist_ok=True)
    else:
        # Ở chế độ dry-run vẫn có thể tạo folder debug để người dùng kiểm tra 20 ảnh mẫu
        folder_debug.mkdir(parents=True, exist_ok=True)

    # 3. Quét danh sách ảnh
    image_list = collect_images(src_dir)
    total_images = len(image_list)
    logger.info("Tìm thấy %d ảnh trong thư mục nguồn.", total_images)

    if total_images == 0:
        logger.error("Không tìm thấy ảnh hợp lệ nào trong '%s'!", src_dir)
        sys.exit(1)

    # 4. Khởi tạo MediaPipe Face Mesh
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.2,
    )

    # 5. Xử lý từng ảnh
    passed_count = 0
    rejected_count = 0
    folder_counts = {
        "esotropia_harschberg": 0,
        "exotropia_harschberg": 0,
        "normal_harschberg": 0,
        "rejected": 0,
    }
    rejection_reasons: Dict[str, int] = {}
    report_rows: List[Dict[str, Any]] = []
    saved_debug_samples = 0

    logger.info("Bắt đầu xử lý và lọc ảnh...")

    try:
        for idx, (img_path, source_label) in enumerate(image_list, 1):
            if idx % 50 == 0 or idx == total_images:
                logger.info("Đã xử lý %d/%d ảnh (%.1f%%)...", idx, total_images, (idx / total_images) * 100)

            # Phân tích ảnh
            res = process_image(img_path, face_mesh=face_mesh, allow_crop_fallback=allow_crop_fallback)

            filename = img_path.name
            is_passed = res["success"]

            assigned_label = "unknown"
            dest_folder: Optional[Path] = None
            out_file_path: Optional[Path] = None

            if is_passed:
                passed_count += 1

                # Phân loại: Ưu tiên nhãn có sẵn, nếu không có thì dùng Heuristic
                if source_label in {"esotropia", "exotropia", "normal"}:
                    assigned_label = source_label
                else:
                    assigned_label = res["heuristic_label"]

                if assigned_label == "esotropia":
                    dest_folder = folder_esotropia
                    folder_counts["esotropia_harschberg"] += 1
                elif assigned_label == "exotropia":
                    dest_folder = folder_exotropia
                    folder_counts["exotropia_harschberg"] += 1
                else:
                    dest_folder = folder_normal
                    folder_counts["normal_harschberg"] += 1

                if not dry_run and dest_folder is not None:
                    out_file_path = get_safe_destination_path(dest_folder, filename)
                    shutil.copy2(str(img_path), str(out_file_path))

            else:
                rejected_count += 1
                folder_counts["rejected"] += 1
                reason = res["rejection_reason"]
                base_reason = reason.split(" (")[0]
                rejection_reasons[base_reason] = rejection_reasons.get(base_reason, 0) + 1

                if not dry_run:
                    out_file_path = get_safe_destination_path(folder_rejected, filename)
                    shutil.copy2(str(img_path), str(out_file_path))

            # Lưu ảnh mẫu debug (tối đa max_debug_samples ảnh)
            if is_passed and saved_debug_samples < args.max_debug_samples:
                raw_bgr = cv2.imread(str(img_path))
                if raw_bgr is not None:
                    vis_img = draw_debug_visualization(
                        raw_bgr,
                        res,
                        label_text=f"{assigned_label.upper()} (OD:{res['od_disp_ratio']}, OS:{res['os_disp_ratio']})",
                    )
                    debug_file = folder_debug / f"debug_{filename}"
                    cv2.imwrite(str(debug_file), vis_img)
                    saved_debug_samples += 1

            # Ghi nhận vào báo cáo
            report_rows.append({
                "filename": filename,
                "status": "PASS" if is_passed else "REJECT",
                "assigned_folder": dest_folder.name if dest_folder else "rejected",
                "source_label": source_label,
                "heuristic_label": res.get("heuristic_label", ""),
                "detection_method": res["method"],
                "right_eye_reflex": "YES" if res["has_right"] else "NO",
                "left_eye_reflex": "YES" if res["has_left"] else "NO",
                "od_displacement_ratio": res.get("od_disp_ratio", ""),
                "os_displacement_ratio": res.get("os_disp_ratio", ""),
                "rejection_reason": res["rejection_reason"],
                "destination_file": str(out_file_path) if out_file_path else "",
            })

    finally:
        face_mesh.close()

    # 6. Xuất report.csv
    report_csv_path = dest_root / "report.csv"
    dest_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "status",
        "assigned_folder",
        "source_label",
        "heuristic_label",
        "detection_method",
        "right_eye_reflex",
        "left_eye_reflex",
        "od_displacement_ratio",
        "os_displacement_ratio",
        "rejection_reason",
        "destination_file",
    ]
    with open(report_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    logger.info("Đã xuất báo cáo chi tiết ra: %s", report_csv_path.resolve())

    # 7. In thống kê tổng kết
    print("\n" + "=" * 70)
    print("BÁO CÁO THỐNG KÊ LỌC VÀ PHÂN LOẠI (HIRSCHBERG FILTER REPORT)")
    print("=" * 70)
    print(f"Tổng số ảnh xử lý:             {total_images}")
    print(f"Số ảnh ĐẠT (Cả 2 mắt có chấm): {passed_count} ({passed_count/total_images*100:.1f}%)")
    print(f"Số ảnh BỊ LOẠI:                {rejected_count} ({rejected_count/total_images*100:.1f}%)")
    print("-" * 70)
    print("Phân bố số lượng trong các thư mục:")
    print(f"  + esotropia_harschberg/ :     {folder_counts['esotropia_harschberg']} ảnh")
    print(f"  + exotropia_harschberg/ :     {folder_counts['exotropia_harschberg']} ảnh")
    print(f"  + normal_harschberg/    :     {folder_counts['normal_harschberg']} ảnh")
    print(f"  + rejected/             :     {folder_counts['rejected']} ảnh")
    print("-" * 70)
    print("Thống kê chi tiết các lý do loại:")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {reason:30s}: {count:3d} ảnh ({count/rejected_count*100:.1f}%)")
    print("-" * 70)
    print(f"Đã lưu {saved_debug_samples} ảnh mẫu kiểm tra trực quan tại: {folder_debug.resolve()}")
    if dry_run:
        print("\n[CHÚ Ý]: Đang ở chế độ DRY-RUN (chưa sao chép file vào các folder phân loại).")
        print("Để thực hiện sao chép thật, hãy chạy lệnh KHÔNG CÓ cờ '--dry-run'.")
    else:
        print("\n[HOÀN THÀNH]: Toàn bộ ảnh đã được sao chép an toàn vào các thư mục tương ứng!")
    print("=" * 70)


if __name__ == "__main__":
    main()
