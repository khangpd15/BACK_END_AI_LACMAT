"""Exercise the real API on local images; save aggregate metrics, never uploads."""
import argparse
import collections
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.hirschberg import router
from app.services import trained_hirschberg_service as service

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reference=json.loads(args.reference.read_text(encoding='utf-8'))
    app=FastAPI();app.include_router(router);client=TestClient(app)
    counts=collections.Counter();errors=collections.defaultdict(list);compared=0;inconclusive=0;coverage_changes=[]
    bundle=json.loads((service.ROOT/'model_bundle.json').read_text(encoding='utf-8'))
    for ref in reference:
        path=args.images/ref['filename']
        response=client.post('/api/v1/hirschberg/predict',files={'image':(path.name,path.read_bytes(),'image/jpeg')},data={'orientation':'unmirrored','fixation':'confirmed','kind':'auto','save_session':'false'})
        assert response.status_code==200,response.text[:500]
        actual=response.json();assert actual['modelId']==bundle['model_id']
        assert actual['model']['classifier_sha256']==bundle['artifacts']['classifier']['sha256']
        assert actual['saved'] is False and actual['researchOnly'] is True
        assert actual['probabilities'] is None and actual['clinicalDiagnosis'] is None
        counts[actual['predictedClass']]+=1
        for side in ('image_left','image_right'):
            old,new=ref['eyes'][side],actual['measurements'][side]
            for key in ('pupil','reflex','iris_left','iris_right','iris_top','iris_bottom'):
                a,b=old.get(key),new.get(key)
                if a is not None and b is not None:errors[key].append(math.dist(a,b))
                elif (a is None)!=(b is None):coverage_changes.append({'case_index':compared+1,'side':side,'landmark':key,'reference_present':a is not None,'api_present':b is not None})
        compared+=1;inconclusive+=actual['predictedClass']=='INCONCLUSIVE'
    result={'api_checks_passed':True,'images':compared,'modelId':bundle['model_id'],'counts':dict(counts),
            'api_policy':'existing production research_capture_v2, strict landmark QC',
            'reference_policy_difference':'Source strict rejects common-direction; production intentionally does not. Do not compare policy counts as accuracy.',
            'coverage_changes':coverage_changes,
            'max_coordinate_difference_px':{k:max(v) for k,v in errors.items()},
            'coordinate_pairs':{k:len(v) for k,v in errors.items()},
            'source_pixel_parity_tolerance':.25,'image_uploads_saved':False,
            'clinical_accuracy':None}
    result['numerical_parity_passed']=not coverage_changes and all(max(v)<=.25 for v in errors.values())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    assert result['api_checks_passed'],'API verification failed'

if __name__=='__main__':main()
