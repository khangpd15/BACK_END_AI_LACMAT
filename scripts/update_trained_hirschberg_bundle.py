"""Install the explicitly selected 2026-10-30 inference bundle, without datasets."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

MODEL_ID = 'classifier_retrain_20261030_v2'
RUNS = 'research/hirschberg_localization/runs/'
CV = RUNS + 'cv_visibility_prior_20261030_v1/'
SOURCES = {
    'classifier': (CV + 'classifier_retrain_v2/candidate_final.pt', 'checkpoint_sha256'),
    'pupil': (CV + 'successful_500_target15_refine_v2/block_03/pupil_final.pt', 'pupil_checkpoint_sha256'),
    'reflex': (CV + 'successful_500_target15_refine_v2/reflex_03/reflex_final.pt', 'checkpoint_sha256'),
    'iris': (CV + 'all_old_direct_v1/iris_pupil/iris_four_final.pt', 'iris_checkpoint_sha256'),
    'eyelid': (RUNS + 'eyelid_reviewed_20261011_v1/candidate_final.pt', None),
    'pupil_base': (RUNS + 'pupil_iris_reviewed_20261011_v1/pupil_final.pt', None),
    'iris_seed': (RUNS + 'eye_crop_192_v1/best.pt', None),
}

INIT = '''    def __init__(self):
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
        self.pupil = make_model(torch, load('pupil_base')['state_dict']).to(self.device)
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
                           'pupil_branch': 'source_pixel_refinement',
                           'pupil_train_mean_px': bundle['pupil_train_mean_px'],
                           'classifier_training_pupil_sha256': bundle['artifacts']['pupil']['sha256'],
                           'updated_pupil_classifier_pair_independently_evaluated': False,
                           'independent_evaluation': False, 'calibrated_probabilities': False,
                           'training_scope': 'TRAIN ONLY; patient identities unresolved',
                           'visibility_validated': False}
        # Match the latest image_new experiment: use online ROI on every input.
        self.frozen_rois = {}
'''

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def install_bundle(source, dest):
    source, dest = Path(source).resolve(), Path(dest).resolve()
    manifest_path = dest / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    artifacts = {}
    for role, (relative, completion_key) in SOURCES.items():
        src = source / relative
        source_hash = sha(src)
        if completion_key:
            completion = json.loads((src.parent / 'completion.json').read_text(encoding='utf-8'))
            if not completion['complete'] or completion[completion_key] != source_hash:
                raise RuntimeError('Incomplete or invalid source checkpoint: ' + role)
        # Retain immutable initializer dependencies at existing paths; version new weights.
        target_relative = ('models/' + MODEL_ID + '/' + role + '.pt') if completion_key else relative
        target = dest / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or sha(target) != source_hash:
            shutil.copy2(src, target)
        if sha(target) != source_hash:
            raise RuntimeError('Copy verification failed: ' + role)
        artifacts[role] = {'path': target_relative, 'sha256': source_hash,
                           'source_relative_path': relative, 'bytes': target.stat().st_size}
    bundle = {'schema_version': 1, 'model_id': MODEL_ID,
              'pupil_version': 'successful_500_target15_refine_v2/block_03',
              'pupil_train_mean_px': 1.7979000772618392,
              'research_only': True, 'independent_validation': False,
              'roi_policy': 'online_only', 'artifacts': artifacts}
    (dest / 'model_bundle.json').write_text(json.dumps(bundle, indent=2) + '\n', encoding='utf-8')
    engine_path = dest / 'scripts/hirschberg_engine.py'
    code = engine_path.read_text(encoding='utf-8')
    tree = ast.parse(code)
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
    init = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    lines = code.splitlines(keepends=True)
    code = ''.join(lines[:init.lineno - 1]) + INIT + ''.join(lines[init.end_lineno:])
    ast.parse(code)
    engine_path.write_text(code, encoding='utf-8')
    inventory = {item['path']: item for item in manifest['files']}
    for item in artifacts.values():
        inventory[item['path']] = {k: item[k] for k in ('path', 'sha256', 'bytes')}
    for path in [dest / 'model_bundle.json', *dest.rglob('*.py')]:
        rel = path.relative_to(dest).as_posix()
        inventory[rel] = {'path': rel, 'sha256': sha(path), 'bytes': path.stat().st_size}
    manifest.update(modelId=MODEL_ID, pupilVersion=bundle['pupil_version'],
                    bundle='model_bundle.json', files=sorted(inventory.values(), key=lambda x: x['path']),
                    roiPolicy='online_only', independentValidation=False,
                    integrationScope='latest completed classifier/iris/pupil/reflex; no datasets or uploads')
    manifest.pop('excludedCandidate', None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Installed ' + MODEL_ID + ' with verified source checkpoint hashes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('D:/REMICARE_TRAIN_AI/HIRSCHBERG_METHOD'))
    parser.add_argument('--dest', type=Path, default=Path(__file__).resolve().parents[1] / 'app/hirschberg_runtime')
    args = parser.parse_args()
    install_bundle(args.source, args.dest)
