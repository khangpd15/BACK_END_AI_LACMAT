import numpy as np
import pytest
from PIL import Image
from app.hirschberg_runtime.research.hirschberg_localization.bilateral_roi import consensus_pair, cropped_eye_rois
from app.hirschberg_runtime.research.hirschberg_localization.screening_feedback import screening_feedback


def votes(box):
    return [{'box': box, 'view': view} for view in ('eye:raw', 'glasses:raw')]


def test_consensus_maps_padded_detections_to_source_pixels():
    boxes, info = consensus_pair(votes([20, -30, 180, 130])+votes([520, -25, 680, 135]), 800, 160)
    assert info['valid_pair_count'] == 1
    assert boxes == {'image_left': [20, 2, 180, 98], 'image_right': [520, 7, 680, 103]}


def test_one_eye_and_single_view_never_create_pair():
    assert not consensus_pair(votes([20, 0, 180, 160]), 800, 200)[0]
    assert not consensus_pair([{'box': [20, 0, 180, 160], 'view': 'eye:raw'},
                               {'box': [520, 0, 680, 160], 'view': 'eye:raw'}], 800, 200)[0]


def test_two_distinct_supported_pairs_are_ambiguous():
    candidates = votes([20, 0, 180, 160])+votes([520, 0, 680, 160])+votes([200, 0, 360, 160])
    boxes, info = consensus_pair(candidates, 800, 200)
    assert not boxes and info['status'] == 'ambiguous'


def test_detections_in_reflected_pixels_do_not_become_source_rois():
    assert not consensus_pair(votes([20, -190, 180, -30])+votes([520, 0, 680, 160]), 800, 200)[0]


@pytest.mark.parametrize('pixels', [np.full((160,800,3),128,np.uint8),
                                  np.random.default_rng(42).integers(0,256,(160,800,3),dtype=np.uint8)])
def test_real_cascades_reject_blank_and_noise(pixels):
    assert not cropped_eye_rois(Image.fromarray(pixels))[0]


def test_localization_message_does_not_request_new_capture():
    feedback = screening_feedback({'status':'INCONCLUSIVE', 'reason':'invalid_input_contract',
                                    'input_reasons':['image_left:pupil_missing_or_invalid']})
    assert feedback['category'] == 'localization'
    assert feedback['next_action'] == 'review_localization_overlay'
    assert 'chụp lại' not in feedback['message']


def test_capture_and_mixed_failure_keep_both_actions():
    base = {'status':'INCONCLUSIVE', 'reason':'invalid_input_contract', 'input_reasons':['orientation_unverified']}
    assert screening_feedback(base)['category'] == 'capture_conditions'
    assert screening_feedback({**base, 'input_reasons':base['input_reasons']+['image_left:pupil_missing_or_invalid']})['category'] == 'localization_and_capture'
    assert screening_feedback({'status':'INCONCLUSIVE','reason':'fixation_not_confirmed'})['category'] == 'capture_conditions'


def test_visible_head_without_mask_has_no_fabricated_pupil():
    import torch
    from app.hirschberg_runtime.research.hirschberg_localization.boundary_pupil import predict
    class EmptyMask:
        def __call__(self, x):
            return torch.full((1,1,192,192),-10.), torch.tensor([[10.,0.,0.]])
    pupil, _, raw = predict(EmptyMask(), Image.new('RGB',(80,40)),[0,0,40,40], 'image_left',{},torch,'cpu')
    assert pupil is None and raw['reason'] == 'empty_pupil_mask'
    assert raw['visibility'] == 'visible' and raw['mask_positive_pixels'] == 0
    assert raw['ellipse_boundary_source'] is None


def test_api_feedback_has_distinct_localization_and_capture_fields(monkeypatch):
    from app.services import trained_hirschberg_service as service
    from app.hirschberg_runtime.research.hirschberg_localization.screening_feedback import screening_feedback
    class Engine:
        def process(self, *args):
            s = {'status': 'INCONCLUSIVE', 'reason': 'invalid_input_contract', 'input_reasons': ['image_left:missing_or_invalid_roi'], 'classifier_called': False}
            return {'screening': s, 'feedback': screening_feedback(s), 'model': {}, 'eyes': {}, 'orientation': 'unmirrored', 'input_kind': 'auto'}
    monkeypatch.setattr(service, 'load_trained_hirschberg', lambda: Engine())
    result = service.predict_trained_hirschberg(b'input')
    assert result['failureCategory'] == 'localization'
    assert result['nextAction'] == 'review_localization_overlay'
    assert result['screening']['classifier_called'] is False
    assert result['feedback']['capture_issues'] == []
