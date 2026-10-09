from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import cv2
import numpy as np
import pandas as pd
import tifffile
import yaml
import zarr
from PIL import Image, ImageDraw
from scipy.interpolate import RBFInterpolator
CLASS_COLUMNS = {0: 'tumour_fraction', 1: 'tumour_associated_stroma_fraction', 2: 'inflamed_stroma_fraction', 3: 'other_fraction'}

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f'Configuration must be a mapping: {path}')
    return data

def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)

def save_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None

def safe_slug(text: str) -> str:
    return re.sub('[^0-9A-Za-z._-]+', '_', str(text)).strip('._') or 'case'

def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    out = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return out[:, :2] / np.maximum(np.abs(out[:, 2:3]), 1e-12)

def class_statistics(values: np.ndarray, ignore_label: int=255) -> dict[str, Any]:
    flat = np.asarray(values).reshape(-1)
    valid = flat != ignore_label
    tissue_count = int(valid.sum())
    total = int(flat.size)
    result: dict[str, Any] = {'tissue_fraction': tissue_count / max(total, 1), 'valid_pixel_count': tissue_count, 'total_pixel_count': total}
    fractions = []
    for class_id, column in CLASS_COLUMNS.items():
        fraction = float((flat[valid] == class_id).mean()) if tissue_count else 0.0
        result[column] = fraction
        fractions.append(fraction)
    present = np.asarray(fractions) > 0.01
    result['class_count'] = int(present.sum())
    if tissue_count:
        dominant = int(np.bincount(flat[valid].astype(np.int64), minlength=4).argmax())
    else:
        dominant = ignore_label
    result['dominant_class'] = dominant
    result['non_tumour_fraction'] = max(0.0, 1.0 - result['tumour_fraction'])
    probabilities = np.asarray(fractions, dtype=np.float64)
    probabilities = probabilities[probabilities > 0]
    result['class_entropy'] = float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.0
    result['tumour_context_flag'] = bool(result['tumour_fraction'] >= 0.05 and result['tumour_associated_stroma_fraction'] + result['inflamed_stroma_fraction'] + result['other_fraction'] >= 0.05)
    return result

def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = (max(ax, bx), max(ay, by))
    ix1, iy1 = (min(ax + aw, bx + bw), min(ay + ah, by + bh))
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - intersection
    return intersection / max(union, 1)

def choose_rois(mask: Any, case_id: str, cfg: dict[str, Any], base_seed: int) -> list[dict[str, Any]]:
    roi_cfg = cfg['roi']
    ignore = int(cfg['segmentation']['ignore_label'])
    size = int(roi_cfg['size_px'])
    height, width = map(int, mask.shape)
    if width < size or height < size:
        raise ValueError(f'ROI size {size} exceeds mask shape {width}x{height}')
    case_seed = (base_seed + int(hashlib.sha256(case_id.encode('utf-8')).hexdigest()[:8], 16)) % 2 ** 32
    rng = random.Random(case_seed)
    stride = max(1, int(roi_cfg['sampling_stride_px']))
    candidates: list[dict[str, Any]] = []
    for _ in range(int(roi_cfg['candidate_trials'])):
        x = rng.randrange(0, width - size + 1)
        y = rng.randrange(0, height - size + 1)
        x = min(x // stride * stride, width - size)
        y = min(y // stride * stride, height - size)
        sampled = np.asarray(mask[y:y + size:stride, x:x + size:stride])
        stats = class_statistics(sampled, ignore)
        stats.update({'roi_x': x, 'roi_y': y, 'roi_width': size, 'roi_height': size})
        stats['score'] = 2.0 * stats['tissue_fraction'] + stats['class_entropy'] + 2.0 * min(stats['tumour_fraction'], stats['non_tumour_fraction'])
        stats['strict_eligible'] = bool(stats['tissue_fraction'] >= float(roi_cfg['min_tissue_fraction']) and stats['tumour_fraction'] >= float(roi_cfg['min_tumour_fraction']) and (stats['non_tumour_fraction'] >= float(roi_cfg['min_non_tumour_fraction'])) and (stats['class_count'] >= int(roi_cfg['min_class_count'])))
        candidates.append(stats)
    candidates.sort(key=lambda row: (row['strict_eligible'], row['score']), reverse=True)
    selected: list[dict[str, Any]] = []
    target = int(roi_cfg['count_per_wsi'])
    max_iou = float(roi_cfg['max_pairwise_iou'])
    for candidate in candidates:
        if not candidate['strict_eligible']:
            continue
        box = (candidate['roi_x'], candidate['roi_y'], size, size)
        if all((box_iou(box, (r['roi_x'], r['roi_y'], size, size)) <= max_iou for r in selected)):
            selected.append(candidate)
        if len(selected) == target:
            break
    if len(selected) < target and bool(roi_cfg.get('allow_relaxed_fallback', False)):
        for candidate in candidates:
            if candidate in selected or candidate['tissue_fraction'] < 0.3:
                continue
            box = (candidate['roi_x'], candidate['roi_y'], size, size)
            if all((box_iou(box, (r['roi_x'], r['roi_y'], size, size)) <= max_iou for r in selected)):
                candidate = dict(candidate)
                candidate['selection_mode'] = 'relaxed_fallback'
                selected.append(candidate)
            if len(selected) == target:
                break
    for index, roi in enumerate(selected):
        roi['case_id'] = case_id
        roi['roi_id'] = f'ROI_{index + 1:02d}'
        roi.setdefault('selection_mode', 'strict')
        roi['random_seed'] = case_seed
    if len(selected) < target:
        raise RuntimeError(f'Only {len(selected)}/{target} non-overlapping tissue ROIs could be selected')
    return selected

@dataclass
class RegistrationTransform:
    enabled: bool
    base_fixed_registration: np.ndarray
    delta_fixed_registration: np.ndarray
    affine_registration: np.ndarray
    fixed_low_to_full: np.ndarray
    moving_full_to_low: np.ndarray
    max_displacement: float
    smoothing: float
    neighbors: int
    rbf: RBFInterpolator | None

    @classmethod
    def load(cls, path: Path) -> 'RegistrationTransform':
        with np.load(path) as package:
            enabled = bool(package['enabled'][0])
            base = package['base_fixed_registration'].astype(np.float64)
            delta = package['delta_fixed_registration'].astype(np.float64)
            affine = package['affine_registration'].astype(np.float64)
            fixed_low_to_full = package['fixed_low_to_full'].astype(np.float64)
            moving_full_to_low = package['moving_full_to_low'].astype(np.float64)
            max_displacement = float(package['max_displacement'][0])
            smoothing = float(package['smoothing'][0])
            neighbors = int(package['neighbors'][0])
        rbf = None
        if enabled and len(base):
            rbf = RBFInterpolator(base, delta, kernel='thin_plate_spline', smoothing=smoothing, neighbors=neighbors if neighbors < len(base) else None)
        return cls(enabled, base, delta, affine, fixed_low_to_full, moving_full_to_low, max_displacement, smoothing, neighbors, rbf)

    def displacement(self, fixed_base_points: np.ndarray) -> np.ndarray:
        if self.rbf is None:
            return np.zeros_like(fixed_base_points, dtype=np.float64)
        delta = self.rbf(fixed_base_points)
        norm = np.linalg.norm(delta, axis=1)
        scale = np.minimum(1.0, self.max_displacement / np.maximum(norm, 1e-06))
        return delta * scale[:, None]

    def fixed_full_to_moving_full(self, points: np.ndarray, inverse_iterations: int=5) -> np.ndarray:
        fixed_registration = transform_points(np.linalg.inv(self.fixed_low_to_full), points)
        affine_base = fixed_registration.copy()
        if self.rbf is not None:
            for _ in range(max(1, inverse_iterations)):
                affine_base = fixed_registration - self.displacement(affine_base)
        moving_registration = transform_points(np.linalg.inv(self.affine_registration), affine_base)
        return transform_points(np.linalg.inv(self.moving_full_to_low), moving_registration)

class MIFReader:

    def __init__(self, path: Path, cfg: dict[str, Any]):
        self.path = path
        self.cfg = cfg
        self.store = None
        self.array = None
        self.full_shape: tuple[int, int] | None = None
        self.level_shape: tuple[int, int] | None = None
        self.downsample_x = 1.0
        self.downsample_y = 1.0
        self.channel_names: list[str] = []

    def __enter__(self) -> 'MIFReader':
        series_index = int(self.cfg['mif']['series_index'])
        level = int(self.cfg['mif']['preferred_level'])
        with tifffile.TiffFile(self.path) as tif:
            series = tif.series[series_index]
            if level >= len(series.levels):
                level = len(series.levels) - 1
            full = series.levels[0]
            chosen = series.levels[level]
            self.full_shape = (int(full.shape[-2]), int(full.shape[-1]))
            self.level_shape = (int(chosen.shape[-2]), int(chosen.shape[-1]))
            channel_count = int(full.shape[0])
            for index in range(channel_count):
                description = tif.pages[index].description or ''
                match = re.search('<Name>(.*?)</Name>', description, flags=re.IGNORECASE | re.DOTALL)
                self.channel_names.append(match.group(1).strip() if match else f'channel_{index}')
        self.downsample_y = self.full_shape[0] / self.level_shape[0]
        self.downsample_x = self.full_shape[1] / self.level_shape[1]
        self.store = tifffile.imread(self.path, series=series_index, level=level, aszarr=True)
        self.array = zarr.open(self.store, mode='r')
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.store is not None and hasattr(self.store, 'close'):
            self.store.close()

    def extract_registered_patch(self, transform: RegistrationTransform, patch_x: int, patch_y: int, patch_size: int, he_full_downsample: float, inverse_iterations: int, channel_indices: list[int] | None=None) -> tuple[np.ndarray, dict[str, Any]]:
        if self.array is None or self.level_shape is None:
            raise RuntimeError('MIFReader must be used as a context manager')
        yy, xx = np.mgrid[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
        fixed_full = np.column_stack([(xx.reshape(-1) + 0.5) * he_full_downsample - 0.5, (yy.reshape(-1) + 0.5) * he_full_downsample - 0.5])
        moving_full = transform.fixed_full_to_moving_full(fixed_full, inverse_iterations)
        map_x = (moving_full[:, 0] / self.downsample_x).reshape(patch_size, patch_size).astype(np.float32)
        map_y = (moving_full[:, 1] / self.downsample_y).reshape(patch_size, patch_size).astype(np.float32)
        valid = (map_x >= 0) & (map_x <= self.level_shape[1] - 1) & (map_y >= 0) & (map_y <= self.level_shape[0] - 1)
        if not valid.any():
            channels = channel_indices or list(range(int(self.array.shape[0])))
            return (np.zeros((len(channels), patch_size, patch_size), dtype=self.array.dtype), {'mif_valid_fraction': 0.0, 'mif_full_center_x': float('nan'), 'mif_full_center_y': float('nan')})
        pad = 3
        x0 = max(0, int(math.floor(float(map_x[valid].min()))) - pad)
        x1 = min(self.level_shape[1], int(math.ceil(float(map_x[valid].max()))) + pad + 1)
        y0 = max(0, int(math.floor(float(map_y[valid].min()))) - pad)
        y1 = min(self.level_shape[0], int(math.ceil(float(map_y[valid].max()))) + pad + 1)
        channels = channel_indices or list(range(int(self.array.shape[0])))
        source = np.asarray(self.array[channels, y0:y1, x0:x1])
        local_x = map_x - x0
        local_y = map_y - y0
        output = np.empty((len(channels), patch_size, patch_size), dtype=np.float32)
        for index, plane in enumerate(source):
            output[index] = cv2.remap(plane.astype(np.float32), local_x, local_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if np.issubdtype(source.dtype, np.integer):
            output = np.rint(output).clip(np.iinfo(source.dtype).min, np.iinfo(source.dtype).max).astype(source.dtype)
        center_index = patch_size // 2 * patch_size + patch_size // 2
        center = moving_full[center_index]
        corners = moving_full[[0, patch_size - 1, len(moving_full) - patch_size, len(moving_full) - 1]]
        info = {'mif_valid_fraction': float(valid.mean()), 'mif_full_center_x': float(center[0]), 'mif_full_center_y': float(center[1]), 'mif_full_min_x': float(corners[:, 0].min()), 'mif_full_max_x': float(corners[:, 0].max()), 'mif_full_min_y': float(corners[:, 1].min()), 'mif_full_max_y': float(corners[:, 1].max())}
        return (output, info)

def patch_rows(mask: Any, case_row: pd.Series, rois: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    patch_cfg = cfg['patch']
    patch_size = int(patch_cfg['size_px'])
    stride = int(patch_cfg['stride_px'])
    min_tissue = float(patch_cfg['min_tissue_fraction'])
    ignore = int(cfg['segmentation']['ignore_label'])
    he_downsample = float(case_row['selected_downsample'])
    rows: list[dict[str, Any]] = []
    for roi in rois:
        for y in range(int(roi['roi_y']), int(roi['roi_y'] + roi['roi_height'] - patch_size + 1), stride):
            for x in range(int(roi['roi_x']), int(roi['roi_x'] + roi['roi_width'] - patch_size + 1), stride):
                values = np.asarray(mask[y:y + patch_size, x:x + patch_size])
                stats = class_statistics(values, ignore)
                if stats['tissue_fraction'] < min_tissue:
                    continue
                row: dict[str, Any] = {'case_id': str(case_row['case_id']), 'wsi_name': Path(str(case_row['he_path'])).name, 'registration_status': str(case_row.get('registration_status', 'unknown')), 'registration_provisional_for_manual_tre': bool(case_row.get('registration_provisional_for_manual_tre', True)), 'roi_id': roi['roi_id'], 'roi_x': int(roi['roi_x']), 'roi_y': int(roi['roi_y']), 'roi_width': int(roi['roi_width']), 'roi_height': int(roi['roi_height']), 'roi_selection_mode': roi['selection_mode'], 'patch_x': x, 'patch_y': y, 'patch_size': patch_size, 'patch_stride': stride, 'he_full_x': float(x * he_downsample), 'he_full_y': float(y * he_downsample), 'he_full_width': float(patch_size * he_downsample), 'he_full_height': float(patch_size * he_downsample), 'segmentation_mpp': float(case_row['selected_mpp']), 'registration_transform_package': str(case_row['registration_transform_package'])}
                row.update(stats)
                rows.append(row)
    for index, row in enumerate(rows):
        row['patch_row'] = index
        row['patch_id'] = f'{row['case_id']}__{row['roi_id']}__x{row['patch_x']:06d}_y{row['patch_y']:06d}'
    return rows

def load_mapping(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {'source_channel_index', 'source_channel_name', 'protein_marker_name', 'enabled'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing marker mapping columns: {sorted(missing)}')
    frame['enabled'] = frame['enabled'].astype(str).str.lower().isin({'1', 'true', 'yes', 'y'})
    return frame

def marker_match_key(name: str) -> str:
    table = str.maketrans({'-': '_', ' ': '_', ':': '_', 'α': 'a', '(': '_', ')': '_', '/': ''})
    return str(name).lower().translate(table).strip().replace('_', '').replace('.', '')

def load_kronos2_vocabulary(model_dir: Path) -> tuple[pd.DataFrame, set[str]]:
    marker_csv = model_dir / 'marker_metadata.csv'
    if not marker_csv.is_file():
        raise FileNotFoundError(f'KRONOS2 marker metadata is missing: {marker_csv}')
    metadata = pd.read_csv(marker_csv)
    required = {'marker_name', 'mean', 'std'}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f'KRONOS2 marker metadata is missing columns: {sorted(missing)}')
    vocabulary = {marker_match_key(value) for value in metadata['marker_name'].dropna().astype(str)}
    return (metadata, vocabulary)

def validate_mapping(mapping: pd.DataFrame, actual_channels: list[str], require_complete: bool, marker_vocabulary: set[str] | None=None, minimum_enabled_markers: int=1) -> list[str]:
    errors: list[str] = []
    for row in mapping.itertuples(index=False):
        index = int(row.source_channel_index)
        if index < 0 or index >= len(actual_channels):
            errors.append(f'channel index {index} is outside QPTIFF channel count {len(actual_channels)}')
            continue
        if str(row.source_channel_name).strip().lower() != actual_channels[index].strip().lower():
            errors.append(f"mapping channel {index} says '{row.source_channel_name}', QPTIFF says '{actual_channels[index]}'")
        if bool(row.enabled):
            marker = '' if pd.isna(row.protein_marker_name) else str(row.protein_marker_name).strip()
            if not marker:
                errors.append(f'enabled channel {index} is missing protein_marker_name')
                continue
            source = str(row.source_channel_name).strip()
            if source.lower().startswith('opal') and marker_match_key(marker) == marker_match_key(source):
                errors.append(f"enabled channel {index} uses dye name '{source}' as a protein marker; fill the verified laboratory Opal-to-protein panel")
            if marker_vocabulary is not None and marker_match_key(marker) not in marker_vocabulary:
                errors.append(f"enabled channel {index} marker '{marker}' is absent from KRONOS2 marker_metadata.csv; do not use an alias or guessed marker name")
    if require_complete:
        enabled_count = int(mapping['enabled'].sum())
        if enabled_count == 0:
            errors.append('no mIF channels are enabled for KRONOS2')
        elif enabled_count < minimum_enabled_markers:
            errors.append(f'only {enabled_count} marker channel(s) enabled; at least {minimum_enabled_markers} verified channels are required for the formal multi-marker run')
    return errors

def build_inventory(cfg: dict[str, Any]) -> pd.DataFrame:
    inputs = cfg['inputs']
    pairs = pd.read_csv(inputs['registration_pairs_csv'])
    inventory = pd.read_csv(inputs['segmentation_inventory_csv'])
    segmentation_summary = pd.read_csv(inputs['segmentation_summary_csv'])
    keep = [column for column in ('case_id', 'status', 'device', 'checkpoint', 'inferred_patches') if column in segmentation_summary.columns]
    segmentation_summary = segmentation_summary[keep].rename(columns={'status': 'segmentation_status'})
    merged = inventory.merge(pairs[['case_id', 'fixed', 'moving', 'verified_same_case']], on='case_id', how='left')
    merged = merged.merge(segmentation_summary, on='case_id', how='left')
    registration_summary_path = inputs.get('registration_summary_csv')
    if registration_summary_path:
        registration_summary = pd.read_csv(registration_summary_path)
        registration_keep = [column for column in ('case_id', 'status', 'provisional_for_manual_tre', 'accepted_for_cell_level', 'warnings') if column in registration_summary.columns]
        registration_summary = registration_summary[registration_keep].rename(columns={'status': 'registration_status', 'provisional_for_manual_tre': 'registration_provisional_for_manual_tre', 'accepted_for_cell_level': 'registration_accepted_for_cell_level', 'warnings': 'registration_warnings'})
        merged = merged.merge(registration_summary, on='case_id', how='left')
        allowed_statuses = {str(value).strip().lower() for value in cfg.get('registration_qc', {}).get('allowed_statuses', ['ok', 'warning'])}
        merged = merged[merged['registration_status'].astype(str).str.strip().str.lower().isin(allowed_statuses)].copy()
    merged['he_path'] = merged['fixed'].fillna(merged['he_path'])
    merged['mif_path'] = merged['moving']
    root = Path(inputs['registration_results_dir'])
    merged['registration_transform_package'] = merged['case_id'].map(lambda case_id: str(root / 'pairs' / f'{safe_slug(case_id)}__mIF_to_HE' / 'v5_transform_package.npz'))
    seg_root = Path(inputs['segmentation_results_dir'])
    merged['class_mask_zarr'] = merged['case_id'].map(lambda case_id: str(seg_root / safe_slug(case_id) / 'class_mask_level.zarr'))
    return merged.sort_values('case_id').reset_index(drop=True)

def draw_roi_preview(case_dir: Path, case_id: str, rois: list[dict[str, Any]], mask_shape: tuple[int, int], cfg: dict[str, Any]) -> None:
    seg_root = Path(cfg['inputs']['segmentation_results_dir'])
    source = seg_root / case_id / 'segmentation_overlay_thumbnail.jpg'
    if not source.exists():
        source = seg_root / case_id / 'class_mask_thumbnail.png'
    if not source.exists():
        return
    image = Image.open(source).convert('RGB')
    draw = ImageDraw.Draw(image)
    sx = image.width / mask_shape[1]
    sy = image.height / mask_shape[0]
    colors = [(255, 0, 255), (0, 255, 255), (50, 255, 80), (255, 140, 0)]
    for index, roi in enumerate(rois):
        x0, y0 = (roi['roi_x'] * sx, roi['roi_y'] * sy)
        x1 = (roi['roi_x'] + roi['roi_width']) * sx
        y1 = (roi['roi_y'] + roi['roi_height']) * sy
        draw.rectangle((x0, y0, x1, y1), outline=colors[index % len(colors)], width=5)
        draw.text((x0 + 5, y0 + 5), roi['roi_id'], fill=colors[index % len(colors)])
    image.save(case_dir / 'roi_selection_preview.png', quality=95)

def save_mif_preview(path: Path, patch: np.ndarray, channel_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if patch.shape[0] == 1:
        rgb = np.repeat(patch[0][..., None], 3, axis=2)
    else:
        picks = [0, min(3, patch.shape[0] - 1), min(6, patch.shape[0] - 1)]
        planes = []
        for index in picks:
            plane = patch[index].astype(np.float32)
            positive = plane[plane > 0]
            high = float(np.percentile(positive, 99.5)) if len(positive) else 1.0
            planes.append(np.clip(plane / max(high, 1.0), 0, 1))
        rgb = (np.stack(planes, axis=-1) * 255).astype(np.uint8)
    Image.fromarray(rgb.astype(np.uint8)).save(path)

class Kronos2Runner:

    def __init__(self, cfg: dict[str, Any], mapping: pd.DataFrame):
        os.environ.setdefault('XFORMERS_DISABLED', '1')
        import torch
        from transformers import AutoModel
        self.torch = torch
        kcfg = cfg['kronos2']
        device_setting = str(kcfg.get('device', 'auto'))
        if device_setting == 'auto':
            device_setting = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_setting)
        self.model_dir = Path(str(kcfg['model_dir']))
        self.model = AutoModel.from_pretrained(str(self.model_dir), trust_remote_code=True, local_files_only=True, device=str(self.device))
        self.model.to(self.device).eval()
        enabled = mapping[mapping['enabled']].copy()
        self.channel_indices = enabled['source_channel_index'].astype(int).tolist()
        self.marker_names = enabled['protein_marker_name'].astype(str).str.strip().tolist()
        self.intensity_max = float(cfg['mif']['intensity_max'])
        self.batch_size = int(kcfg['batch_size'])
        self.preferred_dapi = str(kcfg.get('preferred_dapi', 'DAPI'))
        self.embedding_dim = int(kcfg.get('expected_feature_dim', 768))

    def _embed_once(self, patches: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        scaled = np.stack(patches).astype(np.float32) / self.intensity_max
        normalized = self.model.preprocess(scaled, self.marker_names, preferred_dapi=self.preferred_dapi)
        batch = torch.from_numpy(np.asarray(normalized, dtype=np.float32)).to(self.device)
        with torch.inference_mode():
            patch_embeddings = self.model(batch, self.marker_names)
        if patch_embeddings.ndim != 2 or int(patch_embeddings.shape[1]) != self.embedding_dim:
            raise RuntimeError(f'Unexpected KRONOS2 output shape {tuple(patch_embeddings.shape)}; expected (batch, {self.embedding_dim})')
        return patch_embeddings.detach().cpu().float().numpy()

    def embed(self, patches: list[np.ndarray]) -> np.ndarray:
        try:
            return self._embed_once(patches)
        except self.torch.cuda.OutOfMemoryError:
            if self.device.type != 'cuda' or len(patches) <= 1:
                raise
            fallback = 8 if len(patches) > 8 else max(1, len(patches) // 2)
            self.batch_size = min(self.batch_size, fallback)
            self.torch.cuda.empty_cache()
            return np.concatenate([self.embed(patches[index:index + fallback]) for index in range(0, len(patches), fallback)], axis=0)

def process_case(case_row: pd.Series, cfg: dict[str, Any], output_root: Path, stage: str, mapping: pd.DataFrame, runner: Kronos2Runner | None) -> dict[str, Any]:
    started = time.time()
    case_id = str(case_row['case_id'])
    case_dir = output_root / 'cases' / safe_slug(case_id)
    preview_dir = case_dir / 'preview_patches'
    case_dir.mkdir(parents=True, exist_ok=True)
    mask_path = Path(str(case_row['class_mask_zarr']))
    transform_path = Path(str(case_row['registration_transform_package']))
    mif_path = Path(str(case_row['mif_path']))
    for required in (Path(str(case_row['he_path'])), mif_path, mask_path, transform_path):
        if not required.exists():
            raise FileNotFoundError(required)
    mask = zarr.open(mask_path, mode='r')
    rois = choose_rois(mask, case_id, cfg, int(cfg['project']['random_seed']))
    pd.DataFrame(rois).to_csv(case_dir / 'roi_metadata.csv', index=False, encoding='utf-8-sig')
    draw_roi_preview(case_dir, case_id, rois, tuple(mask.shape), cfg)
    rows = patch_rows(mask, case_row, rois, cfg)
    if not rows:
        raise RuntimeError('No patches passed the tissue threshold')
    candidate_patch_count = len(rows)
    max_patches = cfg.get('_runtime_max_patches_per_case')
    if max_patches is not None and candidate_patch_count > int(max_patches):
        selected = np.linspace(0, candidate_patch_count - 1, int(max_patches), dtype=int)
        rows = [dict(rows[int(index)]) for index in selected]
        for new_index, row in enumerate(rows):
            row['candidate_patch_row'] = int(row['patch_row'])
            row['patch_row'] = new_index
    transform = RegistrationTransform.load(transform_path)
    he_downsample = float(case_row['selected_downsample'])
    centres_fixed_full = np.asarray([[(float(row['patch_x']) + float(row['patch_size']) / 2.0) * he_downsample, (float(row['patch_y']) + float(row['patch_size']) / 2.0) * he_downsample] for row in rows], dtype=np.float64)
    centres_mif_full = transform.fixed_full_to_moving_full(centres_fixed_full, int(cfg['mif']['inverse_tps_iterations']))
    for row, centre in zip(rows, centres_mif_full):
        row['mif_full_center_x'] = float(centre[0])
        row['mif_full_center_y'] = float(centre[1])
    require_model = stage == 'extract'
    feature_batches: list[np.ndarray] = []
    processed_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    pending_patches: list[np.ndarray] = []
    pending_rows: list[dict[str, Any]] = []
    preview_limit = int(cfg['patch']['preview_patches_per_case'])
    with MIFReader(mif_path, cfg) as reader:
        _, marker_vocabulary = load_kronos2_vocabulary(Path(str(cfg['kronos2']['model_dir'])))
        mapping_errors = validate_mapping(mapping, reader.channel_names, require_complete=require_model, marker_vocabulary=marker_vocabulary, minimum_enabled_markers=1)
        if require_model and mapping_errors:
            raise RuntimeError('Marker mapping validation failed: ' + '; '.join(mapping_errors))
        selected_channels = runner.channel_indices if runner is not None else None
        if require_model:
            rows_to_sample: Iterable[dict[str, Any]] = rows
        else:
            preview_indices = np.linspace(0, len(rows) - 1, min(preview_limit, len(rows)), dtype=int)
            rows_to_sample = [rows[int(index)] for index in preview_indices]
        coordinate_info: dict[int, dict[str, Any]] = {}
        for sample_index, row in enumerate(rows_to_sample):
            patch, info = reader.extract_registered_patch(transform=transform, patch_x=int(row['patch_x']), patch_y=int(row['patch_y']), patch_size=int(row['patch_size']), he_full_downsample=float(case_row['selected_downsample']), inverse_iterations=int(cfg['mif']['inverse_tps_iterations']), channel_indices=selected_channels)
            coordinate_info[int(row['patch_row'])] = info
            signal_fraction = float((np.max(patch, axis=0) > 0).mean())
            info['mif_signal_fraction'] = signal_fraction
            if sample_index < preview_limit:
                preview_dir.mkdir(parents=True, exist_ok=True)
                stem = safe_slug(row['patch_id'])
                if bool(cfg['outputs'].get('save_preview_patch_npz', True)):
                    np.savez_compressed(preview_dir / f'{stem}.npz', image=patch, channel_names=np.asarray(reader.channel_names if selected_channels is None else [reader.channel_names[i] for i in selected_channels]), **{key: np.asarray(value) for key, value in info.items()})
                save_mif_preview(preview_dir / f'{stem}.png', patch, reader.channel_names)
            if require_model:
                row.update(info)
                if signal_fraction < float(cfg['mif'].get('min_signal_fraction', 0.0)):
                    row['extraction_status'] = 'skipped_low_mif_signal'
                    skipped_rows.append(dict(row))
                    continue
                row['extraction_status'] = 'embedded'
                pending_patches.append(patch)
                pending_rows.append(row)
                if len(pending_patches) >= runner.batch_size:
                    feature_batches.append(runner.embed(pending_patches))
                    processed_rows.extend(pending_rows)
                    pending_patches, pending_rows = ([], [])
        if require_model and pending_patches:
            feature_batches.append(runner.embed(pending_patches))
            processed_rows.extend(pending_rows)
    if require_model:
        if not feature_batches:
            raise RuntimeError('No patches retained after mIF signal filtering')
        features = np.concatenate(feature_batches, axis=0).astype(np.float32)
        if len(features) != len(processed_rows):
            raise RuntimeError('Feature/metadata row count mismatch')
        metadata = pd.DataFrame(processed_rows)
        np.save(case_dir / 'patch_embeddings.npy', features)
        metadata.to_parquet(case_dir / 'patch_metadata.parquet', index=False)
        if bool(cfg['outputs'].get('write_csv_metadata', True)):
            metadata.to_csv(case_dir / 'patch_metadata.csv', index=False, encoding='utf-8-sig')
        if bool(cfg['outputs'].get('write_wide_parquet', True)):
            feature_columns = pd.DataFrame(features, columns=[f'feature_{i:04d}' for i in range(features.shape[1])])
            pd.concat([metadata.reset_index(drop=True), feature_columns], axis=1).to_parquet(case_dir / 'features.parquet', index=False)
        feature_dim = int(features.shape[1])
        final_rows = processed_rows
        if skipped_rows:
            skipped = pd.DataFrame(skipped_rows)
            skipped.to_parquet(case_dir / 'skipped_patches.parquet', index=False)
            skipped.to_csv(case_dir / 'skipped_patches.csv', index=False, encoding='utf-8-sig')
    else:
        for row in rows:
            row.update(coordinate_info.get(int(row['patch_row']), {}))
        metadata = pd.DataFrame(rows)
        metadata.to_parquet(case_dir / 'patch_metadata.parquet', index=False)
        if bool(cfg['outputs'].get('write_csv_metadata', True)):
            metadata.to_csv(case_dir / 'patch_metadata.csv', index=False, encoding='utf-8-sig')
        feature_dim = 0
        final_rows = rows
    return {'case_id': case_id, 'status': 'ok', 'registration_status': str(case_row.get('registration_status', 'unknown')), 'registration_provisional_for_manual_tre': bool(case_row.get('registration_provisional_for_manual_tre', True)), 'stage': stage, 'roi_count': len(rois), 'patch_count': len(final_rows), 'candidate_patch_count': candidate_patch_count, 'skipped_patch_count': len(skipped_rows), 'feature_dim': feature_dim, 'tps_enabled': transform.enabled, 'output_dir': str(case_dir), 'elapsed_seconds': time.time() - started, 'roi_rows': rois, 'patch_rows': final_rows}

def resume_provenance(preflight: dict[str, Any]) -> dict[str, Any]:
    return {'stage': preflight['stage'], 'config_sha256': preflight['config_sha256'], 'marker_mapping_sha256': preflight['marker_mapping_sha256'], 'weights_bytes': preflight['weights_bytes'], 'max_patches_per_case': preflight['max_patches_per_case'], 'analysis_scope': preflight['analysis_scope']}

def save_case_checkpoint(result: dict[str, Any], preflight: dict[str, Any]) -> None:
    summary = {key: value for key, value in result.items() if key not in {'roi_rows', 'patch_rows'}}
    case_dir = Path(str(summary['output_dir']))
    save_json_atomic(case_dir / 'case_manifest.json', {'checkpoint_version': 1, 'complete': True, 'provenance': resume_provenance(preflight), 'summary': summary})

def load_case_checkpoint(case_row: pd.Series, output_root: Path, stage: str, cfg: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any] | None:
    case_id = str(case_row['case_id'])
    case_dir = output_root / 'cases' / safe_slug(case_id)
    checkpoint_path = case_dir / 'case_manifest.json'
    if not checkpoint_path.is_file():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if checkpoint.get('checkpoint_version') != 1 or checkpoint.get('complete') is not True:
            return None
        if checkpoint.get('provenance') != resume_provenance(preflight):
            return None
        summary = checkpoint.get('summary')
        if not isinstance(summary, dict):
            return None
        if summary.get('case_id') != case_id or summary.get('status') != 'ok' or summary.get('stage') != stage:
            return None
        required = [case_dir / 'roi_metadata.csv', case_dir / 'patch_metadata.parquet']
        if stage == 'extract':
            required.append(case_dir / 'patch_embeddings.npy')
            if bool(cfg['outputs'].get('write_wide_parquet', True)):
                required.append(case_dir / 'features.parquet')
        if any((not path.is_file() for path in required)):
            return None
        roi_frame = pd.read_csv(case_dir / 'roi_metadata.csv')
        patch_frame = pd.read_parquet(case_dir / 'patch_metadata.parquet')
        if len(roi_frame) != int(summary['roi_count']) or len(patch_frame) != int(summary['patch_count']):
            return None
        if stage == 'extract':
            embeddings = np.load(case_dir / 'patch_embeddings.npy', mmap_mode='r')
            if embeddings.ndim != 2:
                return None
            if embeddings.shape != (int(summary['patch_count']), int(summary['feature_dim'])):
                return None
        return {**summary, 'roi_rows': roi_frame.to_dict(orient='records'), 'patch_rows': patch_frame.to_dict(orient='records')}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Matched ROI sampling and KRONOS2 extraction')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('preflight', 'prepare', 'extract'), default='preflight')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--output-root', type=Path, default=None, help='Override outputs.root_dir (useful for a local smoke test)')
    parser.add_argument('--case-id', action='append', default=[])
    parser.add_argument('--limit-cases', type=int, default=None)
    parser.add_argument('--resume', action='store_true', help='Reuse complete per-case checkpoints and rerun incomplete cases.')
    parser.add_argument('--max-patches-per-case', type=int, default=None, help='Evenly subsample accepted patches for a bounded technical smoke test; omit for full extraction.')
    parser.add_argument('--allow-dapi-only-smoke', action='store_true', help='Allow a one-marker DAPI-only technical extraction smoke test; never use for the formal run.')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_yaml(config_path)
    if args.max_patches_per_case is not None:
        if args.max_patches_per_case <= 0:
            raise ValueError('--max-patches-per-case must be positive')
        cfg['_runtime_max_patches_per_case'] = int(args.max_patches_per_case)
    marker_path = Path(cfg['inputs']['marker_mapping_csv'])
    if not marker_path.is_absolute():
        marker_path = config_path.parent / marker_path
    mapping = load_mapping(marker_path)
    run_name = args.run_name or str(cfg['outputs']['run_name'])
    output_base = args.output_root if args.output_root is not None else Path(cfg['outputs']['root_dir'])
    output_root = output_base.resolve() / safe_slug(run_name)
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(cfg)
    if args.case_id:
        inventory = inventory[inventory['case_id'].astype(str).isin(set(args.case_id))]
    if args.limit_cases is not None:
        inventory = inventory.head(args.limit_cases)
    if inventory.empty:
        raise RuntimeError('No cases selected')
    inventory.to_csv(output_root / 'inventory.csv', index=False, encoding='utf-8-sig')
    model_dir = Path(str(cfg['kronos2']['model_dir']))
    model_files = {name: model_dir / name for name in ('config.json', 'configuration_kronos2.py', 'modeling_kronos2.py', 'marker_metadata.csv', 'marker_utils.py', 'kronos2_vitb16_teacher.pth')}
    try:
        marker_metadata, marker_vocabulary = load_kronos2_vocabulary(model_dir)
    except Exception:
        marker_metadata, marker_vocabulary = (pd.DataFrame(), set())
    enabled_count = int(mapping['enabled'].sum())
    minimum_markers = 1 if args.allow_dapi_only_smoke else int(cfg['kronos2'].get('minimum_enabled_markers', 2))
    preflight = {'stage': args.stage, 'selected_cases': int(len(inventory)), 'registration_status_counts': inventory['registration_status'].value_counts(dropna=False).to_dict() if 'registration_status' in inventory.columns else {}, 'config': str(config_path), 'config_sha256': sha256_file(config_path), 'marker_mapping': str(marker_path), 'marker_mapping_sha256': sha256_file(marker_path), 'model_dir': str(model_dir), 'model_files': {name: path.is_file() for name, path in model_files.items()}, 'weights_bytes': model_files['kronos2_vitb16_teacher.pth'].stat().st_size if model_files['kronos2_vitb16_teacher.pth'].is_file() else 0, 'dinov2_present': (model_dir / 'dinov2' / 'models' / 'inference.py').is_file(), 'kronos2_marker_vocabulary_size': int(len(marker_metadata)), 'enabled_marker_count': enabled_count, 'max_patches_per_case': args.max_patches_per_case, 'resume_enabled': bool(args.resume), 'analysis_scope': 'dapi_only_technical_smoke' if args.allow_dapi_only_smoke else 'verified_multimarker', 'scientific_note': 'Automatic registration QC is provisional and is not cell-level validation.', 'runtime': {'python_executable': sys.executable, 'python_version': sys.version, 'packages': {name: package_version(name) for name in ('numpy', 'pandas', 'pyarrow', 'tifffile', 'zarr', 'scipy', 'opencv-python-headless', 'torch', 'transformers', 'timm', 'omegaconf', 'huggingface-hub', 'safetensors')}}}
    with MIFReader(Path(str(inventory.iloc[0]['mif_path'])), cfg) as first_reader:
        preflight['qptiff_channels'] = first_reader.channel_names
        preflight['mapping_errors'] = validate_mapping(mapping, first_reader.channel_names, require_complete=args.stage == 'extract', marker_vocabulary=marker_vocabulary, minimum_enabled_markers=minimum_markers)
        preflight['mif_level_shape'] = first_reader.level_shape
        preflight['mif_level_downsample'] = [first_reader.downsample_x, first_reader.downsample_y]
    blockers = list(preflight['mapping_errors'])
    if args.stage == 'extract':
        for name, present in preflight['model_files'].items():
            if not present:
                blockers.append(f'KRONOS2 model file is missing: {model_files[name]}')
        if not preflight['dinov2_present']:
            blockers.append(f'KRONOS2 dinov2 package is incomplete: {model_dir / 'dinov2'}')
        if 0 < int(preflight['weights_bytes']) < 400000000:
            blockers.append('KRONOS2 weight file is unexpectedly small or incomplete')
    preflight['blockers'] = blockers
    save_json(output_root / 'preflight.json', preflight)
    if args.stage == 'preflight':
        print(json.dumps(json_ready(preflight), ensure_ascii=False, indent=2))
        return 2 if blockers else 0
    if args.stage == 'extract' and blockers:
        for blocker in blockers:
            pass
        return 2
    runner: Kronos2Runner | None = None
    case_summaries: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    patch_rows_all: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    resumed_case_count = 0
    last_completed_case: str | None = None
    run_started = time.time()
    for position, (_, case_row) in enumerate(inventory.iterrows(), start=1):
        case_id = str(case_row['case_id'])
        try:
            result = None
            if args.resume:
                result = load_case_checkpoint(case_row, output_root, args.stage, cfg, preflight)
            if result is not None:
                resumed_case_count += 1
            else:
                if args.stage == 'extract' and runner is None:
                    runner = Kronos2Runner(cfg, mapping)
                result = process_case(case_row, cfg, output_root, args.stage, mapping, runner)
                save_case_checkpoint(result, preflight)
            case_summaries.append({key: value for key, value in result.items() if key not in {'roi_rows', 'patch_rows'}})
            roi_rows.extend(result['roi_rows'])
            patch_rows_all.extend(result['patch_rows'])
            last_completed_case = case_id
            save_json_atomic(output_root / 'progress.json', {'run_name': run_name, 'stage': args.stage, 'cases_requested': int(len(inventory)), 'cases_completed': len(case_summaries), 'cases_resumed': resumed_case_count, 'cases_failed': len(failures), 'last_completed_case': last_completed_case, 'completed_case_ids': [row['case_id'] for row in case_summaries]})
        except Exception as exc:
            failure = {'case_id': case_id, 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
            failures.append(failure)
            save_json_atomic(output_root / 'progress.json', {'run_name': run_name, 'stage': args.stage, 'cases_requested': int(len(inventory)), 'cases_completed': len(case_summaries), 'cases_resumed': resumed_case_count, 'cases_failed': len(failures), 'last_completed_case': last_completed_case, 'completed_case_ids': [row['case_id'] for row in case_summaries]})
    pd.DataFrame(case_summaries).to_csv(output_root / 'wsi_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(roi_rows).to_csv(output_root / 'roi_summary.csv', index=False, encoding='utf-8-sig')
    patch_index = pd.DataFrame(patch_rows_all)
    if not patch_index.empty:
        patch_index.to_parquet(output_root / 'patch_index.parquet', index=False)
        if bool(cfg['outputs'].get('write_csv_metadata', True)):
            patch_index.to_csv(output_root / 'patch_index.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(failures).to_csv(output_root / 'failures.csv', index=False, encoding='utf-8-sig')
    manifest = {**preflight, 'run_name': run_name, 'output_root': str(output_root), 'cases_requested': int(len(inventory)), 'cases_succeeded': len(case_summaries), 'cases_failed': len(failures), 'cases_resumed': resumed_case_count, 'roi_count': len(roi_rows), 'patch_count': len(patch_rows_all), 'feature_dim': int(case_summaries[0]['feature_dim']) if case_summaries else 0, 'elapsed_seconds': time.time() - run_started, 'failures': [{'case_id': row['case_id'], 'error_type': row['error_type'], 'error': row['error']} for row in failures]}
    save_json(output_root / 'run_manifest.json', manifest)
    print(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2))
    return 1 if failures else 0
if __name__ == '__main__':
    raise SystemExit(main())
