from app.hirschberg_runtime.research.hirschberg_localization.learned_eye_roi import LearnedEyeROIs
from app.hirschberg_runtime.research.hirschberg_localization.screening_feedback import screening_feedback
"""Frozen inference subset imported from the training workspace."""
import hashlib
import io
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from app.hirschberg_runtime.scripts.hirschberg_all_image_review import digest, mesh_eyes
from app.hirschberg_runtime.scripts.hirschberg_review100_finetune import INITIAL
from app.hirschberg_runtime.research.hirschberg_localization.screening_eyelid_branch import ScreeningEyelidBranch, opening_features, infer as opening_infer
from app.hirschberg_runtime.research.hirschberg_localization.eyelid_segmentation import EyelidUNet, predict_mask
from app.hirschberg_runtime.scripts.hirschberg_crop256_review_pilot import padded_box
from app.hirschberg_runtime.scripts.export_hirschberg_review163_steps import decode
from app.hirschberg_runtime.scripts.hirschberg_iris_geometry_stable_pilot import stable_model
from app.hirschberg_runtime.research.hirschberg_localization.iris_four_points import four_point_model
from app.hirschberg_runtime.research.hirschberg_localization.iris_context import predict_with_context
from app.hirschberg_runtime.research.hirschberg_localization.pupil_iris_prior import make_model
from app.hirschberg_runtime.research.hirschberg_localization.boundary_pupil import predict as predict_direct_pupil, BoundaryModel
from app.hirschberg_runtime.scripts.hirschberg_localizer import network
from app.hirschberg_runtime.research.hirschberg_localization.automatic_upload import capture_assessment, paired_eye_rois
from app.hirschberg_runtime.research.hirschberg_localization.iris_radius_qc import compare_diameters
from app.hirschberg_runtime.research.hirschberg_localization.binocular_guard import compare_eyes, capture_guard
from app.hirschberg_runtime.research.hirschberg_localization.input_contract import validate_input, human_review_only
from app.hirschberg_runtime.research.hirschberg_localization.reflex_roi_qc import diagnose as reflex_diagnose, guarded_infer, CONFIG as REFLEX_QC_CONFIG
CLASS_RUN = ROOT / 'research/hirschberg_localization/runs/screening_eyelid_branch_20261011_v1'
GEOMETRY_RUN = ROOT / 'research/hirschberg_localization/runs/screening_verified_clear_directpupil_iris4_20261011_v1'
EYELID_RUN = ROOT / 'research/hirschberg_localization/runs/eyelid_reviewed_20261011_v1'
PUPIL_RUN = ROOT / 'research/hirschberg_localization/runs/pupil_direct_center_20261011_v1'
PUPIL_DEPLOY_RUN = ROOT / 'research/hirschberg_localization/runs/pupil_color_features_20261012_v1'
PUPIL_DEPLOY_BRANCH = 'control'
PUPIL_DEPLOY_PATH = PUPIL_DEPLOY_RUN / (PUPIL_DEPLOY_BRANCH + '_final.pt')
IRIS_PATH = ROOT / 'research/hirschberg_localization/runs/iris_four_reviewed_20261011_v1/iris_four_final.pt'
PUPIL_BASE = ROOT / 'research/hirschberg_localization/runs/pupil_iris_reviewed_20261011_v1/pupil_final.pt'
LOCAL_RUN = ROOT / 'research/hirschberg_localization/runs/crop256_human_review_20261010_v1'

def screen_with_guards(classifier, record, orientation, fixation, policy='strict'):
    if policy not in ('strict', 'review_warnings'):
        raise ValueError('Unknown screening policy')
    if policy == 'review_warnings':
        import copy
        record = copy.deepcopy(record)
        warnings = []
        soft = {'mediapipe_iris_axis_ratio_outside_0_85_1_0', 'bilateral_iris_radius_difference_over_0_15'}
        for side, eye in record.get('eyes', {}).items():
            qc = eye.get('reflex_qc', {})
            reasons = qc.get('reasons', [])
            warnings.extend((side + ':' + r for r in reasons if r in soft))
            if reasons and all((r in soft for r in reasons)):
                qc['eligible_for_score'] = True
            qc['reasons'] = [r for r in reasons if r not in soft]
        comparison = record.get('binocular_comparison', {})
        if comparison.get('same_half_plane'):
            warnings.append('common_direction_ambiguous')
            comparison['same_half_plane'] = False
        result = screen_with_guards(classifier, record, orientation, fixation, policy='strict')
        result.update(screening_policy=policy, quality_warnings=warnings)
        return result
    contract = record.get('input_contract')
    if contract is not None and (not contract['eligible']):
        return {'status': 'INCONCLUSIVE', 'reason': 'invalid_input_contract', 'input_reasons': contract['reasons'], 'class_proposal': None, 'clinical_diagnosis': None, 'classifier_called': False, 'requires_human_review': True}

    def checked_classifier(*unused):
        captured = capture_guard({'status': 'READY'}, record['binocular_comparison'], fixation)
        if captured['status'] == 'INCONCLUSIVE':
            return {**captured, 'classifier_called': False}
        return human_review_only(classifier())
    return guarded_infer(None, record, orientation, checked_classifier)

class Engine:

    def __init__(self):
        import torch
        self.torch = torch
        torch.set_num_threads(max(1, min(4, int(os.getenv('HIRSCHBERG_CPU_THREADS', '1')))))
        self.device = 'cpu'
        bundle = json.loads((ROOT / 'model_bundle.json').read_text(encoding='utf-8'))
        paths = {}
        for role, artifact in bundle['artifacts'].items():
            path = (ROOT / artifact['path']).resolve()
            if not path.is_relative_to(ROOT.resolve()) or digest(path) != artifact['sha256']:
                raise RuntimeError('Invalid Hirschberg bundle artifact: ' + role)
            paths[role] = path
        def load(role):
            return torch.load(paths[role], map_location=self.device, weights_only=True)
        classifier = load('classifier')
        if classifier['classes'] != ['normal', 'esotropia', 'exotropia']:
            raise RuntimeError('Classifier class order mismatch')
        for role in ('pupil', 'reflex', 'iris'):
            if classifier['config']['localizers'][role]['sha256'] != bundle['artifacts'][role]['sha256']:
                raise RuntimeError('Classifier/localizer provenance mismatch: ' + role)
        if classifier['config']['eyelid_checkpoint_sha256'] != bundle['artifacts']['eyelid']['sha256']:
            raise RuntimeError('Classifier/eyelid provenance mismatch')
        self.classifier_sha256 = bundle['artifacts']['classifier']['sha256']
        self.fusion = ScreeningEyelidBranch().to(self.device)
        self.fusion.load_state_dict(classifier['state_dict'])
        self.fusion.eval()
        self.feature_mean = classifier['geometry_mean']
        self.feature_std = classifier['geometry_std']
        self.opening_mean = classifier['opening_mean']
        self.opening_std = classifier['opening_std']
        self.eyelid = EyelidUNet().to(self.device)
        self.eyelid.load_state_dict(load('eyelid')['state_dict'])
        self.eyelid.eval()
        self.pr = network(torch, spatial_logits=True, targets=2).to(self.device)
        self.pr.load_state_dict(load('reflex')['state_dict'])
        self.pr.eval()
        self.pupil = BoundaryModel().to(self.device)
        self.pupil.load_state_dict(load('pupil')['state_dict'])
        self.pupil.eval()
        self.iris = four_point_model(torch, stable_model(torch, load('iris_seed'))).to(self.device)
        self.iris.load_state_dict(load('iris')['state_dict'])
        self.iris.eval()
        self.checkpoints = {'pupil_direct_center.pt': bundle['artifacts']['pupil']['sha256'],
                            'reflex_crop256.pt': bundle['artifacts']['reflex']['sha256'],
                            'iris_four.pt': bundle['artifacts']['iris']['sha256'],
                            'eyelid_UNet.pt': bundle['artifacts']['eyelid']['sha256'],
                            'roi_seed192.pt': bundle['artifacts']['iris_seed']['sha256']}
        self.model_info = {'classifier_name': 'ResNet50 + Hirschberg + pupil–khe mi',
                           'version': bundle['model_id'], 'classifier_sha256': self.classifier_sha256,
                           'localizers': self.checkpoints, 'pupil_version': bundle['pupil_version'],
                           'pupil_checkpoint': bundle['artifacts']['pupil']['path'],
                           'pupil_branch': 'pupil_boundary_segmentation',
                           'pupil_train_mean_px': bundle['pupil_train_mean_px'],
                           'classifier_training_pupil_sha256': bundle['artifacts']['pupil']['sha256'],
                           'updated_pupil_classifier_pair_independently_evaluated': False,
                           'independent_evaluation': False, 'calibrated_probabilities': False,
                           'training_scope': 'TRAIN ONLY; patient identities unresolved',
                           'visibility_validated': False}
        # Match the latest image_new experiment: use online ROI on every input.
        self.model_info['roi_policy'] = 'bilateral_cascade_consensus_v1'
        self.roi_detector = LearnedEyeROIs(paths['eye_roi'], self.device) if 'eye_roi' in paths else None
        if self.roi_detector is not None:
            self.model_info['roi_policy'] = 'learned_eye_roi_eso_exo_v1'
            self.checkpoints['eye_roi.pt'] = digest(paths['eye_roi'])
        self.frozen_rois = {}

    def infer_current(self, model, image, proposal, mean, std, device):
        return opening_infer(model, image, proposal, mean, std, self.opening_mean, self.opening_std, device)

    def classify(self, image, record):
        proposal = {}
        for side in ('image_left', 'image_right'):
            eye = record['eyes'][side]
            edges = eye.get('iris_edges')
            proposal[side] = {**eye, 'iris_left': edges[0] if edges else None, 'iris_right': edges[1] if edges else None}
        result = self.infer_current(self.fusion, image, proposal, self.feature_mean, self.feature_std, self.device)
        result['classifier_called'] = result['status'] != 'INCONCLUSIVE'
        return result

    def process(self, raw, name, orientation='auto', kind='auto', fixation='unknown', fixation_source='api_request_capture_metadata'):
        from PIL import Image, ImageDraw
        automatic = kind == 'auto'
        if orientation not in ('auto', 'unmirrored', 'mirrored') or kind not in ('auto', 'fullface', 'bilateral'):
            raise ValueError('Chọn loại ảnh và xác minh hướng ảnh trước khi chạy.')
        if fixation not in ('unknown', 'confirmed', 'offaxis'):
            raise ValueError('Trạng thái hướng nhìn không hợp lệ.')
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError('Chỉ nhận JPG, PNG hoặc WebP.')
            if im.width * im.height > 25000000 or min(im.size) < 16:
                raise ValueError('Ảnh phải từ 16 pixel mỗi chiều và không quá 25 megapixel.')
            suffix = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}[im.format]
            image = im.convert('RGB')
            exif_orientation = im.getexif().get(274, 1)
        key = uuid.uuid4().hex
        record = {'case_id': key, 'filename': Path(name).name, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': hashlib.sha256(raw).hexdigest(), 'width': image.width, 'height': image.height, 'orientation': orientation, 'input_kind': kind, 'coordinate_space': 'stored pixels; no rotation/unwarp', 'exif_orientation': exif_orientation, 'checkpoints': self.checkpoints, 'model': self.model_info, 'classifier_sha256': self.classifier_sha256, 'clinical_diagnosis': None, 'requires_human_review': True, 'training_eligible': False, 'eyes': {}}
        boxes = {}
        iris_boxes = {}
        contours = {}
        mesh_geometry = {}
        reason = None
        assessment = None
        if kind in ('auto', 'fullface'):
            import mediapipe as mp
            import numpy as np
            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as mesh:
                result = mesh.process(np.array(image))
            count = len(result.multi_face_landmarks or [])
            record['detected_faces'] = count
            assessment = capture_assessment(image, result.multi_face_landmarks or [], [])
            if automatic and count == 1:
                kind = 'fullface'
            if count == 1:
                eyes = mesh_eyes(result.multi_face_landmarks[0].landmark, image.width, image.height)
                for side, e in zip(('image_left', 'image_right'), eyes):
                    boxes[side] = e['bbox']
                    iris_boxes[side] = e['bbox']
                    contours[side] = e['eyelid_contour']
                    mesh_geometry[side] = e
                assessment = capture_assessment(image, result.multi_face_landmarks, eyes)
            elif count == 0 and automatic:
                fallback, detector_info = paired_eye_rois(image, self.roi_detector)
                record['eye_roi_fallback'] = detector_info
                if fallback:
                    boxes = dict(fallback)
                    iris_boxes = dict(fallback)
                    kind = 'bilateral_detected'
                    assessment.update(input_type='bilateral_detected', eye_count=2, eye_detection=detector_info)
                else:
                    reason = 'Chưa tìm được khuôn mặt hoặc cặp hai mắt hợp lệ. Không tạo tọa độ giả.'
            else:
                reason = 'Không xác định được đúng một khuôn mặt. Hãy dùng ảnh có một người, hai mắt nhìn rõ.'
        if kind == 'bilateral':
            boxes, detector_info = paired_eye_rois(image, self.roi_detector)
            iris_boxes = dict(boxes)
            record['eye_roi_fallback'] = detector_info
            if not boxes:
                reason = 'Không xác định được cặp hai mắt. Cần review hoặc chụp lại.'
        record['automatic_capture'] = assessment or capture_assessment(image, [], [])
        record['input_kind'] = kind
        if orientation == 'auto':
            orientation = 'unknown'
        record['orientation'] = orientation
        frozen = self.frozen_rois.get(record['source_sha256'])
        if frozen:
            boxes = {side: e['bbox'] for side, e in frozen.items()}
            iris_boxes = {side: e.get('iris_bbox') for side, e in frozen.items()}
            for side, e in frozen.items():
                if not contours.get(side) and e.get('eyelid_contour'):
                    contours[side] = e['eyelid_contour']
            record['roi_source'] = 'exact_SHA_matched_frozen_detector_CNN_ROIs'
        else:
            boxes = {side: boxes.get(side) if record.get('eye_roi_fallback', {}).get('roi_padding_applied') else padded_box(iris_boxes.get(side) or boxes.get(side), image.width, image.height) for side in ('image_left', 'image_right')}
            record['roi_source'] = record.get('eye_roi_fallback', {}).get('method', 'paired_eye_cascade') + '+10percent_padding' if kind == 'bilateral_detected' else 'online_detector_CNN_ROI_plus10percent_padding'
        if record.get('detected_faces', 0) > 1 or exif_orientation != 1:
            boxes = {}
            iris_boxes = {}
            contours = {}
        for row, side in enumerate(('image_left', 'image_right')):
            box = boxes.get(side)
            eye, _ = decode(self.pr, image, box, side, self.torch, self.device, size=256, visualize=False)
            iris_box = iris_boxes.get(side)
            four, iris_context = predict_with_context(self.iris, image, iris_box, side, self.torch, self.device)
            edges = [four['iris_left'], four['iris_right']] if four else None
            pupil, _, pupil_raw = predict_direct_pupil(self.pupil, image, box, side, four, self.torch, self.device, visualize=False)
            eye['pupil'] = pupil
            eye['raw_predictions']['pupil'] = pupil_raw
            eye.update(iris_edges=edges, iris_four=four, iris_bbox=iris_context['used_bbox'], iris_context_metadata=iris_context, iris_scale_valid=False, eyelid_contour=contours.get(side, []))
            if four:
                eye.update(four)
                eye.update(iris_center=[(four['iris_left'][0] + four['iris_right'][0]) / 2, (four['iris_top'][1] + four['iris_bottom'][1]) / 2], iris_diameter_px=math.dist(*edges))
            mp_iris = mesh_geometry.get(side)
            if mp_iris:
                mp_width = math.dist(mp_iris['iris_left'], mp_iris['iris_right'])
                mp_height = math.dist(mp_iris['iris_top'], mp_iris['iris_bottom'])
                eye['mediapipe_iris_geometry'] = {'center': mp_iris['iris_center'], 'left': mp_iris['iris_left'], 'right': mp_iris['iris_right'], 'top': mp_iris['iris_top'], 'bottom': mp_iris['iris_bottom'], 'landmark_bbox_axis_ratio': min(mp_width, mp_height) / max(mp_width, mp_height), 'source': 'MediaPipe iris landmarks; points, not segmentation or fitted ellipse'}
            eye['detector_eyelid_contour'] = eye['eyelid_contour']
            if box:
                pts, eye_mask, metadata = predict_mask(self.eyelid, image, box, side, self.device)
                eye['eyelid_branch_contour'] = pts
                eye['eyelid_prediction_metadata'] = metadata
                if not eye['eyelid_contour'] and len(pts) >= 3 and (not metadata.get('touches_crop_edge')):
                    eye['eyelid_contour'] = pts
                    eye['eyelid_qc_source'] = 'CNN_eye_opening_proposal'
                else:
                    eye['eyelid_qc_source'] = 'detector_contour' if eye['eyelid_contour'] else 'unavailable'
                _, opening_details = opening_features({side: {**eye, 'eyelid_contour': pts}})
                eye['opening_geometry'] = opening_details[side]
            eye['reflex_qc'], _ = reflex_diagnose(image, eye, visualize=False)
            axis_ratio = (eye.get('mediapipe_iris_geometry') or {}).get('landmark_bbox_axis_ratio')
            if axis_ratio is not None and (not 0.85 <= axis_ratio <= 1.0):
                eye['reflex_qc'].setdefault('warnings', []).append('mediapipe_iris_axis_ratio_outside_0_85_1_0')
            record['eyes'][side] = eye
        iris_diameters = [record['eyes'][side].get('iris_diameter_px') for side in ('image_left', 'image_right')]
        record['binocular_iris_radius_qc'] = compare_diameters(iris_diameters)
        if record['binocular_iris_radius_qc']['status'] == 'Invalid Landmark / Low Confidence':
            for side in ('image_left', 'image_right'):
                qc = record['eyes'][side]['reflex_qc']
                qc.setdefault('warnings', []).append('bilateral_iris_radius_difference_over_0_15')
        record['fixation_confirmation'] = fixation
        record['fixation_source'] = fixation_source if fixation == 'confirmed' else None
        record['input_contract'] = validate_input(record)
        for side, anatomical in record['input_contract']['anatomical_sides'].items():
            record['eyes'][side]['anatomical_side'] = anatomical
        record['binocular_comparison'] = compare_eyes(record, orientation)
        if orientation == 'unknown':
            record['binocular_comparison'].update(frame='stored image axes; mirror history unknown', anatomical_sides_verified=False)
        record['reflex_qc_config'] = REFLEX_QC_CONFIG.copy()
        record['experimental_classifier_proposal'] = screen_with_guards(lambda: self.classify(image, record), record, orientation, fixation)
        record['screening'] = record['experimental_classifier_proposal']
        record['screening']['quality_warnings'] = [{'side': side, 'code': code} for side, eye in record['eyes'].items() for code in eye.get('reflex_qc', {}).get('warnings', [])]
        record['screening']['quality_policy'] = 'geometry_warnings_with_hard_visibility_and_coordinate_checks_v1'
        if automatic:
            record['automatic_capture']['common_direction_flag'] = record['binocular_comparison'].get('same_half_plane')
            record['automatic_capture']['fixation_status'] = fixation
            record['automatic_capture']['fixation_verified'] = fixation == 'confirmed'
            record['automatic_capture']['fixation_source'] = record['fixation_source']
            record['automatic_capture']['fixation_note'] = 'Attested by the project capture protocol; not inferred from still-image pixels.' if fixation == 'confirmed' else 'Not confirmed by capture metadata.'
        messages = {'fixation_not_confirmed': 'Chưa xác nhận nhìn nguồn sáng gần camera, hoặc ảnh đang nhìn xéo. Không kết luận từ ảnh này.', 'common_direction_ambiguous': 'Hai vector phản xạ có thành phần cùng hướng. Có thể do điều kiện chụp/hướng nhìn hoặc sai điểm; vẫn có thể kèm lệch tương đối. Cần review hoặc ảnh nhìn nguồn sáng gần camera.', 'missing_binocular_geometry': 'Thiếu pupil/reflex nhìn thấy hoặc scale iris ở ít nhất một mắt.', 'invalid_reflex_landmark': 'Invalid Landmark / Low Confidence: reflex không đạt QC ROI hoặc vector, hoặc thiếu vùng mắt nhìn thấy. Không chạy classifier; cần sửa điểm/kiểm tra iris hoặc chụp lại. Đây là QC thử nghiệm, không phải kết luận bình thường.'}
        messages['invalid_input_contract'] = 'Đầu vào chưa đủ điều kiện: ' + ', '.join(record['input_contract']['reasons'])
        record['reason_vi'] = reason or messages.get(record['screening'].get('reason'))
        failed = record['screening'].get('failed_eyes', {})
        if failed and any((e.get('bbox') for e in record['eyes'].values())):
            names = {'missing_or_nonfinite_geometry': 'Thiếu pupil/reflex hoặc iris hợp lệ; chưa tính được vector', 'missing_visible_eye_boundary': 'Chưa xác định được đường viền mí mắt', 'reflex_pupil_vector_exceeds_0_4R': 'Khoảng cách pupil–reflex vượt gate nghiên cứu 0,4R', 'bilateral_iris_radius_difference_over_0_15': 'Kích thước iris hai mắt chênh quá gate nghiên cứu 15%'}
            record['reason_vi'] = '; '.join((('Mắt trái ảnh: ' if side == 'image_left' else 'Mắt phải ảnh: ') + ', '.join((names.get(r, r) for r in reasons)) for side, reasons in failed.items()))
        record['feedback'] = screening_feedback(record['screening'])
        record['reason_vi_diagnostic'] = record['reason_vi']
        if record['screening']['status'] == 'INCONCLUSIVE':
            record['reason_vi'] = record['feedback']['message']
        return record
