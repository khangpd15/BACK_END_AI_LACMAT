"""Separate capture attestation failures from algorithmic localization failures."""


def screening_feedback(screening):
    if screening.get('status') != 'INCONCLUSIVE':
        normal = str(screening.get('class_proposal', '')).lower() == 'normal'
        return {'category': 'review_result', 'next_action': 'human_review',
                'message': 'Chưa ghi nhận dấu hiệu lác rõ trong ảnh này.' if normal else
                'Có dấu hiệu cần kiểm tra thêm. Bạn nên khám chuyên khoa mắt để được đánh giá.',
                'capture_issues': [], 'localization_issues': []}
    reasons = screening.get('input_reasons', [])
    capture_codes = {'orientation_unverified', 'exif_transform_requires_normalized_reupload', 'multiple_faces',
                     'fixation_not_confirmed', 'orientation_unknown'}
    capture = [r for r in reasons if r in capture_codes]
    localization = [r for r in reasons if r not in capture_codes]
    reason = screening.get('reason')
    if reason in capture_codes:
        capture.append(reason)
    if reason in ('invalid_reflex_landmark', 'missing_binocular_geometry'):
        localization.append(reason)
    if capture and localization:
        category, action = 'localization_and_capture', 'review_localization_and_confirm_capture'
        message = 'Chưa thể đánh giá: hệ thống chưa định vị đủ vùng mắt hoặc điểm mắt; đồng thời cần xác nhận hướng ảnh và điều kiện nhìn nguồn sáng. Cần xem lại ảnh đánh dấu.'
    elif capture:
        category, action = 'capture_conditions', 'confirm_capture_metadata'
        if 'multiple_faces' in capture:
            message = 'Ảnh có nhiều khuôn mặt; hãy chọn ảnh chỉ có một người.'
        elif 'exif_transform_requires_normalized_reupload' in capture:
            message = 'Hướng xoay ảnh chưa được chuẩn hóa; hãy lưu ảnh đúng chiều rồi tải lại.'
        else:
            message = 'Chưa đủ xác nhận về hướng ảnh hoặc việc nhìn nguồn sáng gần camera. Hãy xác nhận điều kiện chụp.'
    elif localization or reason == 'invalid_input_contract':
        category, action = 'localization', 'review_localization_overlay'
        if any(r.endswith(':missing_or_invalid_roi') for r in localization):
            message = 'Hệ thống chưa xác định được cặp hai mắt trong ảnh. Cần kiểm tra vùng mắt được đánh dấu; lỗi này chưa chứng minh ảnh cần chụp lại.'
        else:
            message = 'Hệ thống chưa định vị đủ đồng tử, phản xạ hoặc đường viền mắt hợp lệ. Cần xem lại ảnh đánh dấu và kết quả định vị.'
    else:
        category, action = 'alignment_review', 'human_review'
        message = 'Chưa thể đánh giá chắc chắn từ kết quả hiện tại. Cần người chuyên môn xem lại ảnh và các điểm được đánh dấu.'
    return {'category': category, 'next_action': action, 'message': message,
            'capture_issues': capture, 'localization_issues': localization}
