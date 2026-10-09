from __future__ import annotations
from pathlib import Path
import pandas as pd

def normalise_label(value) -> int | None:
    if pd.isna(value) or str(value).strip() == '':
        return None
    text = str(value).strip().lower().replace('_', '-')
    if text in {'1', '1.0', 'pcr', 'yes', 'true', 'responder'}:
        return 1
    if text in {'0', '0.0', 'non-pcr', 'nonpcr', 'no', 'false', 'non-responder', 'nonresponder'}:
        return 0
    raise ValueError(f'Unsupported pcr_label value: {value!r}')

def load_case_features(features_root: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    files = sorted((features_root / 'cases').glob('*/features.parquet'))
    if not files:
        raise FileNotFoundError(f'No cases/*/features.parquet files under {features_root}')
    case_frames = []
    roi_frames = []
    feature_columns: list[str] | None = None
    for path in files:
        frame = pd.read_parquet(path)
        current = [column for column in frame.columns if column.startswith('feature_')]
        if not current:
            raise ValueError(f'No feature columns in {path}')
        if feature_columns is None:
            feature_columns = current
        elif current != feature_columns:
            raise ValueError(f'Feature columns differ across cases: {path}')
        if 'case_id' not in frame or 'roi_id' not in frame:
            raise ValueError(f'case_id/roi_id metadata missing from {path}')
        case_id = str(frame['case_id'].iloc[0])
        case_mean = frame[feature_columns].mean(axis=0).to_frame().T
        case_mean.insert(0, 'patch_count', len(frame))
        case_mean.insert(0, 'case_id', case_id)
        case_frames.append(case_mean)
        roi_mean = frame.groupby(['case_id', 'roi_id'], as_index=False)[feature_columns].mean()
        roi_counts = frame.groupby(['case_id', 'roi_id']).size().rename('patch_count').reset_index()
        roi_frames.append(roi_mean.merge(roi_counts, on=['case_id', 'roi_id'], how='left'))
    return (pd.concat(case_frames, ignore_index=True), feature_columns or [], pd.concat(roi_frames, ignore_index=True))
