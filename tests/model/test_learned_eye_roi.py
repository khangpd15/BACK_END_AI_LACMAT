import numpy as np
import torch
from PIL import Image
from app.hirschberg_runtime.research.hirschberg_localization.learned_eye_roi import prepare,decode,select_pair,INPUT_SIZE


def test_letterbox_boxes_return_to_exact_source_pixels():
    _, transform = prepare(Image.new('RGB',(1200,300)))
    heat = torch.full((1,1,32,48),-20.)
    distances = torch.zeros(1,4,32,48); offsets = torch.zeros(1,2,32,48)
    heat[0,0,16,10]=20
    distances[0,:,16,10]=torch.tensor([48/384,32/256,48/384,32/256])
    result=decode((heat,distances,offsets),transform,.5)
    assert len(result)==1
    assert np.max(np.abs(np.array(result[0]['bbox'])-[100,50,400,250]))<=1


def test_one_detected_eye_never_synthesizes_other_eye():
    assert not select_pair([{'bbox':[10,20,150,100],'score':.99}],800,160)[0]


def test_multiple_valid_eye_pairs_fail_as_ambiguous():
    candidates=[{'bbox':b,'score':.99} for b in [[10,20,150,100],[300,20,440,100],[600,20,740,100]]]
    pair,info=select_pair(candidates,800,160)
    assert not pair and info['status']=='ambiguous'


def test_boxes_entirely_in_letterbox_padding_are_rejected():
    _, transform=prepare(Image.new('RGB',(1200,300)))
    heat=torch.full((1,1,32,48),-20.);dist=torch.zeros(1,4,32,48);offset=torch.zeros(1,2,32,48)
    heat[0,0,2,10]=20;dist[0,:,2,10]=torch.tensor([.1,.02,.1,.02])
    assert not decode((heat,dist,offset),transform,.5)


def test_learned_detector_has_priority_only_when_explicitly_provided():
    from app.hirschberg_runtime.research.hirschberg_localization.automatic_upload import paired_eye_rois
    class Detector:
        def detect(self,image):
            return {'image_left':[10,20,150,100],'image_right':[500,20,640,100]}, {'method':'learned_test','status':'paired_candidates'}
    pair,info=paired_eye_rois(Image.new('RGB',(800,160)),Detector())
    assert info['method']=='learned_test' and len(pair)==2


def test_installed_detector_is_loaded_and_hash_verified():
    from app.services.trained_hirschberg_service import load_trained_hirschberg
    engine=load_trained_hirschberg()
    assert engine.roi_detector is not None
    assert engine.model_info['roi_policy']=='learned_eye_roi_eso_exo_v1'
    assert 'eye_roi.pt' in engine.checkpoints
    assert not engine.roi_detector.detect(Image.new('RGB',(800,200),(128,128,128)))[0]
