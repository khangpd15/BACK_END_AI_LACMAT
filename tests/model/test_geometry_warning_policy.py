import math
from PIL import Image
from app.hirschberg_runtime.research.hirschberg_localization.reflex_roi_qc import diagnose
from app.hirschberg_runtime.research.hirschberg_localization.input_contract import validate_input

def test_large_reflex_vector_is_warning_inside_visible_eye():
 image=Image.new('RGB',(100,80),(40,40,40));image.putpixel((60,40),(255,255,255));image.putpixel((45,40),(20,20,20))
 eye={'bbox':[10,10,90,70],'pupil':[45,40],'reflex':[60,40],'iris_center':[50,40],'iris_diameter_px':50,'eyelid_contour':[[10,10],[90,10],[90,70],[10,70]]}
 qc,_=diagnose(image,eye,visualize=False)
 assert qc['eligible_for_score']
 assert 'reflex_pupil_vector_exceeds_0_4R' in qc['warnings']
 assert 'cnn_reflex_outside_search_roi' in qc['warnings']
 eye['reflex']=[99,40];qc,_=diagnose(image,eye,visualize=False)
 assert not qc['eligible_for_score']
 assert 'cnn_reflex_outside_visible_eye' in qc['reasons']

def test_missing_geometry_stays_blocked():
 qc,_=diagnose(Image.new('RGB',(100,80)),{'pupil':None},visualize=False)
 assert not qc['eligible_for_score']
 assert 'missing_or_nonfinite_geometry' in qc['reasons']

def test_orientation_cannot_be_replaced_by_geometry_warning():
 r={'width':100,'height':80,'orientation':'unknown','eyes':{}}
 assert 'orientation_unverified' in validate_input(r)['reasons']

