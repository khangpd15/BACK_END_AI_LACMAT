"""Vendor the active training runtime with source hashes; exclude uploads and training data."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

SOURCE = Path('D:/REMICARE_TRAIN_AI/HIRSCHBERG_METHOD')
DEST = Path(__file__).resolve().parents[1] / 'app/hirschberg_runtime'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prune_runtime():
    """Keep only inference definitions and their statically referenced dependencies."""
    modules = {}
    prefix = 'app.hirschberg_runtime.'
    for path in DEST.rglob('*.py'):
        if path.name == '__init__.py':
            continue
        module = prefix + path.relative_to(DEST).with_suffix('').as_posix().replace('/', '.')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        definitions = {}
        for node in tree.body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                names = [n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.asname or a.name.split('.')[0] for a in node.names]
            for name in names:
                definitions[name] = node
        modules[module] = (path, tree, definitions, set(), set())

    def require(module, name):
        if module not in modules:
            return
        path, tree, definitions, kept, imported = modules[module]
        node = definitions.get(name)
        if node is None:
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if name in imported:
                return
            imported.add(name)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix):
                alias = next(a for a in node.names if (a.asname or a.name) == name)
                require(node.module, alias.name)
            return
        if id(node) in kept:
            return
        kept.add(id(node))
        for ref in ast.walk(node):
            if isinstance(ref, ast.Name) and isinstance(ref.ctx, ast.Load):
                require(module, ref.id)
            elif isinstance(ref, ast.Attribute) and isinstance(ref.value, ast.Name):
                imp = definitions.get(ref.value.id)
                if isinstance(imp, ast.ImportFrom) and imp.module and imp.module.startswith(prefix):
                    alias = next((a for a in imp.names if (a.asname or a.name) == ref.value.id), None)
                    if alias:
                        require(imp.module + '.' + alias.name, ref.attr)
            elif isinstance(ref, ast.ImportFrom) and ref.module and ref.module.startswith(prefix):
                for alias in ref.names:
                    require(ref.module, alias.name)

    require(prefix + 'scripts.hirschberg_engine', 'Engine')
    require(prefix + 'scripts.hirschberg_engine', 'screen_with_guards')
    for module, (path, tree, definitions, kept, imported) in modules.items():
        body = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                node.names = [a for a in node.names if (a.asname or a.name.split('.')[0]) in imported]
                if node.names:
                    body.append(node)
            elif id(node) in kept:
                body.append(node)
        if body:
            tree.body = body
            path.write_text('"""Frozen inference subset imported from the training workspace."""\n' + ast.unparse(tree) + '\n', encoding='utf-8')
        else:
            assert path.resolve().is_relative_to(DEST.resolve())
            path.unlink()



def optimize_runtime():
    """Remove API-unused preview/visualization work without changing numeric inference."""
    import textwrap

    def rewrite(relative, transform):
        path = DEST / relative
        path.write_text(transform(path.read_text(encoding='utf-8')), encoding='utf-8')

    def replace_once(code, old, new):
        if code.count(old) != 1:
            raise RuntimeError(f'Unexpected source structure for production optimization: {old[:90]}')
        return code.replace(old, new, 1)

    def engine(code):
        start = code.index('        self.seed = network(')
        end = code.index('\n    def infer_current', start)
        code = code[:start] + code[end:]
        start = code.index('            proposal = {side:', code.index('        if automatic:', code.index('def process(')))
        end = code.index("            record['automatic_capture']['common_direction_flag']", start)
        code = code[:start] + code[end:]
        code = replace_once(code, 'eye, maps = decode(self.pr, image, box, side, self.torch, self.device, size=256)',
                            'eye, _ = decode(self.pr, image, box, side, self.torch, self.device, size=256, visualize=False)')
        code = replace_once(code, 'pupil, pupil_maps, pupil_raw = predict_direct_pupil(self.pupil, image, box, side, four, self.torch, self.device)',
                            'pupil, _, pupil_raw = predict_direct_pupil(self.pupil, image, box, side, four, self.torch, self.device, visualize=False)')
        code = replace_once(code, '            maps.update(pupil_maps)\n', '')
        code = replace_once(code, "eye['reflex_qc'], search_mask = reflex_diagnose(image, eye)",
                            "eye['reflex_qc'], _ = reflex_diagnose(image, eye, visualize=False)")
        return code

    def decoder(code):
        code = replace_once(code, 'def decode(pr_model, image, box, side, torch, device, size=192):',
                            'def decode(pr_model, image, box, side, torch, device, size=192, *, visualize=True):')
        start = code.index('        values = prob[0, j].cpu().numpy()')
        end = code.index("        result['heatmap_transforms'][kind]", start)
        return code[:start] + '        if visualize:\n' + textwrap.indent(code[start:end], '    ') + code[end:]

    def pupil(code):
        code = replace_once(code, 'def predict(model, image, box, side, iris_four, torch, device):',
                            'def predict(model, image, box, side, iris_four, torch, device, *, visualize=True):')
        start = code.index('        values = heat[0, 0].masked_fill(')
        end = code.index('        visibility = flags', start)
        code = code[:start] + '        if visualize:\n' + textwrap.indent(code[start:end], '    ') + code[end:]
        start = code.index('    gray = Image.fromarray(')
        end = code.index('    metadata = ', start)
        code = code[:start] + '    maps = {}\n    if visualize:\n' + textwrap.indent(code[start:end], '    ') + "        maps = {'pupil': mapped, 'pupil_model256': gray}\n" + code[end:]
        return replace_once(code, "{'pupil': mapped, 'pupil_model256': gray}, metadata)", 'maps, metadata)')

    def reflex(code):
        code = replace_once(code, 'def diagnose(image, eye):', 'def diagnose(image, eye, *, visualize=True):')
        return replace_once(code, "    mask_image = Image.fromarray(mask.astype('uint8') * 255)",
                            "    mask_image = Image.fromarray(mask.astype('uint8') * 255) if visualize else None")

    def binocular(code):
        # Direction alone is not an alignment verdict. Keep the measured flag
        # for audit; let the trained classifier assess eligible binocular input.
        code = replace_once(code,
                            "    elif comparison.get('available') and comparison['same_half_plane']:\n        reason = 'common_direction_ambiguous'\n",
                            '')
        return code.replace('research_common_direction_v1', 'research_capture_v2')

    rewrite('scripts/hirschberg_engine.py', engine)
    rewrite('scripts/export_hirschberg_review163_steps.py', decoder)
    rewrite('research/hirschberg_localization/direct_pupil_inference.py', pupil)
    rewrite('research/hirschberg_localization/reflex_roi_qc.py', reflex)
    rewrite('research/hirschberg_localization/binocular_guard.py', binocular)


def main():
    global SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    DEST.mkdir(parents=True, exist_ok=True)
    original = (SOURCE / 'scripts/hirschberg_real_screening_page.py').read_text(encoding='utf-8')
    engine = original[:original.index('\ndef result_file(')]
    # Remove the page, saved uploads, and rendering; inference and QC stay identical.
    engine = engine.replace("key=uuid.uuid4().hex;folder=STORE/key;folder.mkdir(parents=True)", "key=uuid.uuid4().hex")
    engine = re.sub(r"^        \(folder/\('original'.*\n", '', engine, flags=re.M)
    start = engine.index("        process=Image.new(")
    end = engine.index("        for row,side", start)
    engine = engine[:start] + engine[end:]
    start = engine.index("                from PIL import Image as PILImage")
    end = engine.index("                if not eye['eyelid_contour']", start)
    engine = engine[:start] + engine[end:]
    start = engine.index("            if eye['reflex_qc'].get('glint_candidates')")
    end = engine.index("            axis_ratio=", start)
    engine = engine[:start] + engine[end:]
    start = engine.index("            if search_mask is not None:")
    end = engine.index("            record['eyes'][side]=eye", start)
    engine = engine[:start] + engine[end:]
    start = engine.index("        full.save(")
    end = engine.index("        return record", start)
    engine = engine[:start] + engine[end:]
    start = engine.index("        snapshot=json.loads((LOCAL_RUN/'snapshot.json')")
    end = engine.index("        self.pr=network", start)
    engine = engine[:start] + "        self.frozen_rois=json.loads((ROOT/'frozen_rois.json').read_text(encoding='utf-8'))\n" + engine[end:]
    engine = engine.replace("orientation='unmirrored'", "orientation='auto'")
    engine = engine.replace("fixation='confirmed'", "fixation='unknown'")
    engine = engine.replace("fixation_source='owner_attested_straight_gaze_upload_protocol_2026-10-15'", "fixation_source='api_request_capture_metadata'")
    engine = engine.replace('import math\n', 'import math\nimport os\n')
    engine = engine.replace('torch.set_num_threads(4)', 'torch.set_num_threads(max(1, min(4, int(os.getenv("HIRSCHBERG_CPU_THREADS", "1")))))')
    engine = engine.replace("self.device='cuda' if torch.cuda.is_available() else 'cpu'", "self.device='cpu'")
    queue = [('scripts/hirschberg_engine.py', engine)]
    visited = set()
    inventory = []
    while queue:
        relative, code = queue.pop()
        if relative in visited:
            continue
        visited.add(relative)
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(('scripts.', 'research.')):
                dep = node.module.replace('.', '/') + '.py'
                if (SOURCE / dep).is_file():
                    queue.append((dep, (SOURCE / dep).read_text(encoding='utf-8-sig')))
        code = re.sub(r'\bfrom (scripts|research)(?=[. ])', r'from app.hirschberg_runtime.\1', code)
        code = re.sub(r'^sys.path.insert\(0,\s*str\(ROOT\)\)\s*$', '', code, flags=re.M)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding='utf-8')
        for parent in [DEST, *target.parents]:
            if not parent.is_relative_to(DEST):
                break
            (parent / '__init__.py').touch()
    files = [
        'screening_eyelid_branch_20261011_v1/candidate_final.pt',
        'screening_eyelid_branch_20261011_v1/completion.json',
        'screening_verified_clear_directpupil_iris4_20261011_v1/config.json',
        'eyelid_reviewed_20261011_v1/candidate_final.pt',
        'pupil_direct_center_20261011_v1/candidate_final.pt',
        'pupil_color_features_20261012_v1/control_final.pt',
        'pupil_color_features_20261012_v1/completion.json',
        'iris_four_reviewed_20261011_v1/iris_four_final.pt',
        'pupil_iris_reviewed_20261011_v1/pupil_final.pt',
        'crop256_human_review_20261010_v1/final.pt',
        'eye_crop_192_v1/best.pt',
    ]
    for relative in files:
        relative = 'research/hirschberg_localization/runs/' + relative
        src, target = SOURCE / relative, DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or sha(src) != sha(target):
            shutil.copy2(src, target)
        assert sha(src) == sha(target)
        inventory.append({'path': relative, 'sha256': sha(target), 'bytes': target.stat().st_size})
    runs = SOURCE / 'research/hirschberg_localization/runs'
    rois = {r['item']['sha256']: r['eyes'] for r in json.loads((runs / 'crop256_human_review_20261010_v1/snapshot.json').read_text(encoding='utf-8'))}
    for row in json.loads((runs / 'screening_eyelid_branch_20261011_v1/snapshot.json').read_text(encoding='utf-8'))['records']:
        rois[row['sha256']] = {side: row['proposal'][side] for side in ('image_left', 'image_right')}
    rois = {key: {side: {k: eye[k] for k in ('bbox', 'iris_bbox', 'eyelid_contour') if k in eye} for side, eye in pair.items()} for key, pair in rois.items()}
    (DEST / 'frozen_rois.json').write_text(json.dumps(rois), encoding='utf-8')
    (DEST / 'manifest.json').write_text(json.dumps({'source': str(SOURCE), 'modelId': 'screening_eyelid_branch_20261011_v1', 'pupilVersion': 'pupil_color_features_20261012_v1', 'researchOnly': True, 'files': inventory, 'modules': sorted(visited), 'sourceRuntimeSha256': sha(SOURCE / 'scripts/hirschberg_real_screening_page.py'), 'excludedCandidate': {'run': 'pupil_iris_continued_20261015_v3', 'reason': 'Source completion marks recommended_for_runtime=false; keep active validated configuration.'}}, indent=2), encoding='utf-8')
    prune_runtime()
    optimize_runtime()
    prune_runtime()
    manifest = json.loads((DEST / 'manifest.json').read_text(encoding='utf-8'))
    manifest['runtimeOptimizations'] = {'version': 'production_inference_v1', 'mirrorPreview': False, 'visualizationImages': False, 'unusedSeedInstance': False, 'checkpointWeightsChanged': False}
    manifest['capturePolicy'] = {'version': 'research_capture_v2', 'sameHalfPlaneIsDiagnostic': False, 'sameHalfPlaneBlocksClassifier': False, 'requiresFixationAndLandmarkQC': True}
    manifest['modules'] = [p.relative_to(DEST).as_posix() for p in sorted(DEST.rglob('*.py')) if p.name != '__init__.py']
    for path in [DEST / 'frozen_rois.json', *sorted(DEST.rglob('*.py'))]:
        manifest['files'].append({'path': path.relative_to(DEST).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size})
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Imported {len(visited)} modules and {len(files)} verified artifacts into {DEST}')
    # Default import selects the current completed bundle, not historical weights.
    from update_trained_hirschberg_bundle import install_bundle
    install_bundle(SOURCE, DEST)


if __name__ == '__main__':
    main()
