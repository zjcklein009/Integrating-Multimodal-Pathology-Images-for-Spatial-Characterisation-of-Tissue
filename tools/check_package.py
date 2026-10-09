from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path


def inspect(root: Path) -> dict:
    errors = []
    files = sorted(path for path in root.rglob('*') if path.is_file())
    notebooks = sorted(root.glob('notebooks/**/*.ipynb'))
    code_cells = 0
    python_files = 0
    forbidden = {'.qptiff', '.qptif', '.svs', '.ndpi', '.tif', '.tiff', '.pth', '.pt',
                 '.ckpt', '.safetensors', '.parquet', '.npy', '.npz', '.joblib', '.pkl', '.pickle'}
    secret = re.compile(r'\bhf_[A-Za-z0-9]{20,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b')
    cells_by_file = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in forbidden:
            errors.append(f'Data or weight file: {relative}')
        if '__pycache__' in path.parts or '.ipynb_checkpoints' in path.parts:
            errors.append(f'Generated cache: {relative}')
        if path.suffix in {'.py', '.ipynb', '.yaml', '.md', '.txt', '.toml', '.json'}:
            text = path.read_text(encoding='utf-8-sig')
            if secret.search(text):
                errors.append(f'Possible access token: {relative}')
        if path.suffix == '.py':
            python_files += 1
            try:
                compile(path.read_text(encoding='utf-8-sig'), relative, 'exec')
            except SyntaxError as error:
                errors.append(f'{relative}: {error}')
    for path in notebooks:
        relative = path.relative_to(root).as_posix()
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError) as error:
            errors.append(f'{relative}: {error}')
            continue
        if payload.get('nbformat') != 4 or payload.get('nbformat_minor') != 5:
            errors.append(f'Unexpected notebook format: {relative}')
        ids = []
        sources = []
        for index, cell in enumerate(payload.get('cells', [])):
            ids.append(cell.get('id'))
            source = cell.get('source', '')
            source = ''.join(source) if isinstance(source, list) else source
            if cell.get('cell_type') != 'code':
                continue
            code_cells += 1
            if cell.get('execution_count') is not None or cell.get('outputs'):
                errors.append(f'Executed cell remains: {relative}:{index}')
            try:
                tree = ast.parse(source)
                compile(source, f'{relative}:{index}', 'exec')
                sources.append(source)
            except SyntaxError as error:
                errors.append(f'{relative}:{index}: {error}')
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
                    errors.append(f'Debug print remains: {relative}:{index}')
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if re.search(r'[C-H]:[\\/]', node.value):
                        errors.append(f'Hard-coded drive path: {relative}:{index}')
                    if re.search(r'\bTP\d{2}-\d+', node.value):
                        errors.append(f'Case identifier literal: {relative}:{index}')
                    if re.search(r'[\u4e00-\u9fff]', node.value):
                        errors.append(f'Untranslated code string: {relative}:{index}')
        if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
            errors.append(f'Invalid or duplicate cell identifiers: {relative}')
        cells_by_file[path.stem] = sources
    expected = ['01_Global_Registration', '02_Local_Refinement', '03_Tissue_Segmentation',
                '04_KRONOS2_Extraction', '05_Feature_QC_and_Case_Profiles', '06_Virchow2_Extraction',
                '07_Spatial_Analysis', '08_Modality_Comparison', '09_Report_Figures']
    if sorted(cells_by_file) != expected:
        errors.append('Notebook stage list does not match report scope')
    contracts = {
        '02_Local_Refinement': ['PATHS.registration_global', 'registration_summary_v5.csv', 'v5_transform_package.npz'],
        '03_Tissue_Segmentation': ['PATHS.registration', 'segmentation_inventory_v2.csv', 'class_mask_level.zarr'],
        '05_Feature_QC_and_Case_Profiles': ['case_level_features_final.parquet', 'patch_index_final.parquet'],
        '06_Virchow2_Extraction': ['PATHS.registration', 'registration_pairs.csv', 'tokens[:, 5:]'],
        '07_Spatial_Analysis': ['case_level_features_final.parquet', 'patch_index_final.parquet', 'spatial_motif_interactions.csv'],
        '08_Modality_Comparison': ['case_level_features_final.parquet', 'virchow2_he', 'mif_he_fused'],
    }
    for name, fragments in contracts.items():
        combined = '\n'.join(cells_by_file.get(name, []))
        for fragment in fragments:
            if fragment not in combined:
                errors.append(f'Missing cross-stage contract {name}: {fragment}')
    try:
        import yaml

        for path in (root / 'config').glob('*.yaml'):
            value = yaml.safe_load(path.read_text(encoding='utf-8'))
            if not isinstance(value, dict):
                errors.append(f'Configuration is not a mapping: {path.name}')
        mapping = (root / 'config' / 'marker_mapping.csv').read_text(encoding='utf-8-sig').splitlines()
        if len(mapping) != 9:
            errors.append('Expected eight channel mapping rows')
    except ImportError:
        errors.append('PyYAML is needed to validate configuration files')
    sources = json.loads((root / 'docs' / 'source_manifest.json').read_text(encoding='utf-8'))
    unchanged_vendor_files = 0
    for item in sources['files']:
        path = root / item['packaged_file']
        if not path.is_file():
            errors.append(f'Missing source-tracked file: {item["packaged_file"]}')
        if item['packaged_file'].startswith('vendor/'):
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['source_sha256']:
                errors.append(f'Vendor source modified: {item["packaged_file"]}')
            unchanged_vendor_files += 1
    return {'status': 'passed' if not errors else 'failed', 'scope': 'static only; no notebook execution or model loading',
            'notebooks': len(notebooks), 'code_cells': code_cells, 'python_files': python_files,
            'unchanged_vendor_files': unchanged_vendor_files, 'files': len(files), 'errors': errors}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    report = inspect(root)
    (root / 'docs' / 'static_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    sys.stdout.write(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['status'] == 'passed' else 1)
