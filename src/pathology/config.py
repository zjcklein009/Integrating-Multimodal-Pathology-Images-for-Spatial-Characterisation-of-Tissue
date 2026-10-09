from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    he_dir: Path
    mif_dir: Path
    clinical_labels: Path
    segmentation_checkpoint: Path
    kronos2_model: Path
    virchow2_model: Path
    output_root: Path
    run_name: str

    @property
    def registration_global(self) -> Path:
        return self.output_root / 'registration_global'

    @property
    def registration(self) -> Path:
        return self.output_root / 'registration'

    @property
    def segmentation(self) -> Path:
        return self.output_root / 'segmentation'

    @property
    def features_root(self) -> Path:
        return self.output_root / 'features'

    @property
    def run(self) -> Path:
        return self.features_root / self.run_name


def load_paths(root: Path) -> ProjectPaths:
    root = root.resolve()
    settings = yaml.safe_load((root / 'config' / 'project.yaml').read_text(encoding='utf-8'))
    local = root / 'config' / 'local.yaml'
    if local.is_file():
        overrides = yaml.safe_load(local.read_text(encoding='utf-8')) or {}
        unexpected = set(overrides) - set(settings)
        if unexpected:
            raise ValueError(f'Unknown configuration keys: {sorted(unexpected)}')
        settings.update(overrides)
    path_keys = ['he_dir', 'mif_dir', 'clinical_labels', 'segmentation_checkpoint',
                 'kronos2_model', 'virchow2_model', 'output_root']
    paths = {}
    for key in path_keys:
        path = Path(settings[key]).expanduser()
        paths[key] = (root / path).resolve() if not path.is_absolute() else path.resolve()
    name = str(settings['run_name'])
    if not name or name in {'.', '..'} or any(character in name for character in '/\\:'):
        raise ValueError('run_name must be a single directory name')
    for key in ['he_dir', 'mif_dir', 'kronos2_model', 'virchow2_model']:
        if paths['output_root'] == paths[key] or paths[key] in paths['output_root'].parents:
            raise ValueError(f'Output directory must not be inside {key}')
    return ProjectPaths(root=root, **paths, run_name=name)


def kronos_config(root: Path, paths: ProjectPaths) -> dict:
    config = yaml.safe_load((root / 'config' / 'kronos2.yaml').read_text(encoding='utf-8'))
    config['inputs'].update({
        'registration_pairs_csv': str(paths.registration / 'registration_pairs.csv'),
        'registration_summary_csv': str(paths.registration / 'registration_summary_v5.csv'),
        'registration_results_dir': str(paths.registration),
        'segmentation_inventory_csv': str(paths.segmentation / 'segmentation_inventory_v2.csv'),
        'segmentation_summary_csv': str(paths.segmentation / 'segmentation_summary_v2.csv'),
        'segmentation_results_dir': str(paths.segmentation),
        'marker_mapping_csv': str(root / 'config' / 'marker_mapping.csv'),
    })
    config['outputs']['root_dir'] = str(paths.features_root)
    config['outputs']['run_name'] = paths.run_name
    config['kronos2']['model_dir'] = str(paths.kronos2_model)
    return config
