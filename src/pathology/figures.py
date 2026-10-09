from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
import zarr
from PIL import Image


def save(fig, destination: Path) -> None:
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def image_panel(paths: list[Path], titles: list[str], destination: Path, columns: int = 2) -> None:
    rows = int(np.ceil(len(paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False)
    for ax, path, title in zip(axes.flat, paths, titles):
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            ax.imshow(np.asarray(image))
        ax.set_title(title)
        ax.axis('off')
    for ax in axes.flat[len(paths):]:
        ax.axis('off')
    save(fig, destination)


def registration_figures(paths, destination: Path, case_id: str) -> None:
    pair = paths.registration / 'pairs' / f'{case_id}__mIF_to_HE'
    image_panel([pair / 'fixed_registration.png', pair / 'moving_structural_registration.png'],
                ['H&E fixed image', 'mIF structural view'], destination / 'registration_inputs.png')
    image_panel([pair / 'qc_global_contours.png', pair / 'qc_matches_inlier_outlier.png',
                 pair / 'qc_matches_affine_residual.png', pair / 'qc_nonrigid_contours.png'],
                ['Global alignment', 'Local correspondences', 'Affine refinement', 'Non-rigid alignment'],
                destination / 'registration_stages.png')
    summary = pd.read_csv(paths.registration / 'registration_summary_v5.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = summary.status.value_counts().reindex(['ok', 'warning', 'failed'], fill_value=0)
    axes[0].bar(counts.index, counts.values, color=['#2F9C95', '#FFA600', '#E45756'])
    axes[0].set_ylabel('Image pairs')
    axes[0].set_title('Automatic registration QC')
    columns = ['global_dice', 'affine_dice', 'nonrigid_dice']
    values = summary[columns].apply(pd.to_numeric, errors='coerce')
    axes[1].boxplot([values[column].dropna() for column in columns])
    axes[1].set_xticks([1, 2, 3], ['Global', 'Affine', 'Final'])
    axes[1].set_ylabel('Tissue-mask Dice')
    axes[1].set_title('Descriptive alignment checks')
    save(fig, destination / 'registration_cohort_qc.png')
    image_panel([pair / 'qc_nonrigid_checkerboard.png', pair / 'qc_control_residuals.png'],
                ['Checkerboard inspection', 'Fitted control-point residuals'], destination / 'registration_cases.png')


def segmentation_figures(paths, destination: Path, case_id: str) -> None:
    case = paths.segmentation / case_id
    metadata = json.loads((case / 'full_run_metadata.json').read_text(encoding='utf-8'))
    he_path = Path(metadata['he_path'])
    with tifffile.TiffFile(he_path) as tif:
        candidates = [(np.prod(series.shape[:2]), i) for i, series in enumerate(tif.series)
                      if series.axes == 'YXS' and series.shape[-1] >= 3]
        if not candidates:
            raise ValueError('No RGB H&E TIFF series')
        series_index = max(candidates)[1]
        series = tif.series[series_index]
        level = min(series.levels, key=lambda item: abs(np.log(max(item.shape[:2]) / 1200)))
        thumbnail = level.asarray()[..., :3]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(thumbnail)
    with Image.open(case / 'segmentation_overlay_thumbnail.jpg') as overlay_image:
        axes[1].imshow(np.asarray(overlay_image))
    for ax, title in zip(axes, ['Native H&E tissue context', 'Four-class tissue overlay']):
        ax.set_title(title)
        ax.axis('off')
    save(fig, destination / 'segmentation_context.png')
    summary = pd.read_csv(paths.segmentation / 'segmentation_summary_v2.csv').sort_values('case_id')
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(summary))
    axes[0].bar(x, summary.inferred_patches, color='#4C78A8')
    axes[0].set_ylabel('Inferred tissue patches')
    axes[1].bar(x, summary.elapsed_seconds / 60, color='#2F9C95')
    axes[1].set_ylabel('Elapsed minutes')
    axes[1].set_xticks(x, summary.case_id, rotation=90, fontsize=6)
    save(fig, destination / 'segmentation_run_summary.png')
    mask = zarr.open(str(case / 'class_mask_level.zarr'), mode='r')
    height, width = mask.shape
    step = max(1, int(np.ceil(max(height, width) / 1200)))
    sample = np.asarray(mask[::step, ::step])
    boundaries = np.zeros_like(sample, dtype=bool)
    boundaries[1:, :] |= sample[1:, :] != sample[:-1, :]
    boundaries[:, 1:] |= sample[:, 1:] != sample[:, :-1]
    boundaries &= sample != 255
    yy, xx = np.nonzero(boundaries)
    if not len(xx):
        raise ValueError('No tissue-class boundary for the representative ROI')
    middle = len(xx) // 2
    size = min(1024, height, width)
    x0 = int(np.clip(xx[middle] * step - size // 2, 0, width - size))
    y0 = int(np.clip(yy[middle] * step - size // 2, 0, height - size))
    crop = np.asarray(mask[y0:y0 + size, x0:x0 + size])
    store = tifffile.imread(he_path, series=series_index, level=int(metadata['selected_level']), aszarr=True)
    try:
        image = np.asarray(zarr.open(store, mode='r')[y0:y0 + size, x0:x0 + size, :3])
    finally:
        store.close()
    palette = np.asarray([[255, 0, 0], [180, 120, 0], [255, 255, 0], [160, 160, 160]], dtype=np.uint8)
    colour = np.zeros((*crop.shape, 3), dtype=np.uint8)
    valid = crop != 255
    colour[valid] = palette[crop[valid]]
    overlay = image.copy()
    overlay[valid] = (0.7 * image[valid] + 0.3 * colour[valid]).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, view, title in zip(axes, [image, colour, overlay], ['Boundary-rich H&E ROI', 'Four tissue classes', 'Light overlay']):
        ax.imshow(view)
        ax.set_title(title)
        ax.axis('off')
    save(fig, destination / 'segmentation_roi.png')


def feature_figures(paths, destination: Path, case_id: str) -> None:
    case = paths.run / 'cases' / case_id
    image_panel([case / 'roi_selection_preview.png'], ['Selected 4096-pixel ROIs'],
                destination / 'kronos_roi_selection.png', columns=1)
    previews = sorted((case / 'preview_patches').glob('*.png'))[:4]
    if not previews:
        raise FileNotFoundError('No saved mIF patch previews')
    image_panel(previews, [f'Matched mIF patch {i + 1}' for i in range(len(previews))],
                destination / 'kronos_patch_montage.png')


def spatial_interaction_figure(paths, destination: Path) -> None:
    source = paths.run / 'feature_characterisation' / 'spatial_community_graph' / 'data'
    table = pd.read_csv(source / 'spatial_motif_interactions.csv')
    number = int(max(table.motif_a.max(), table.motif_b.max()))
    matrix = np.full((number, number), np.nan)
    annotations = np.full((number, number), '', dtype=object)
    graph = nx.Graph()
    graph.add_nodes_from(range(1, number + 1))
    for row in table.itertuples(index=False):
        effect = np.log2(max(float(row.fold_enrichment), 1e-8))
        left, right = int(row.motif_a), int(row.motif_b)
        matrix[left - 1, right - 1] = matrix[right - 1, left - 1] = effect
        if row.fdr_bh < 0.05:
            annotations[left - 1, right - 1] = annotations[right - 1, left - 1] = '*'
            if left != right:
                graph.add_edge(left, right, effect=effect)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = [f'M{i}' for i in range(1, number + 1)]
    sns.heatmap(matrix, center=0, cmap='RdBu_r', xticklabels=labels, yticklabels=labels,
                annot=annotations, fmt='', ax=axes[0], cbar_kws={'label': 'log2 observed / expected edges'})
    axes[0].set_title('Within-ROI motif interactions; * BH-adjusted p < 0.05')
    position = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, position, ax=axes[1], node_color='#4C78A8', node_size=650)
    nx.draw_networkx_labels(graph, position, labels={i: f'M{i}' for i in graph}, ax=axes[1], font_color='white')
    for enriched, colour, style in [(True, '#E45756', 'solid'), (False, '#4C78A8', 'dashed')]:
        edges = [(left, right) for left, right, data in graph.edges(data=True) if (data['effect'] > 0) == enriched]
        if edges:
            nx.draw_networkx_edges(graph, position, edgelist=edges, ax=axes[1], edge_color=colour,
                                   style=style, width=[min(4, 0.7 + abs(graph[a][b]['effect'])) for a, b in edges], alpha=0.7)
    axes[1].set_title('Enriched and depleted motif-pair edges')
    axes[1].axis('off')
    save(fig, destination / 'statistics_spatial_interactions.png')


def modality_figure(paths, destination: Path) -> None:
    source = paths.run / 'feature_characterisation' / 'virchow2_spatial_comparison'
    modes = ['mif_only', 'he_only', 'mif_he_fused']
    rows = []
    for mode in modes:
        spatial = pd.read_csv(source / mode / 'motif_spatial_pvalues.csv')
        clinical = pd.read_csv(source / mode / 'pcr_case_level_pvalues.csv')
        rows.append({'mode': mode, 'local_continuity': spatial.observed_edges.sum() / spatial.expected_edges.sum(),
                     'minimum_raw_p': clinical.p_value.min(), 'minimum_adjusted_p': clinical.fdr_bh.min()})
    summary = pd.DataFrame(rows)
    summary.to_csv(destination / 'modality_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(modes))
    axes[0].bar(x, summary.local_continuity, color=['#4C78A8', '#2F9C95', '#58508D'])
    axes[0].axhline(1, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Observed / expected same-motif edges')
    axes[0].set_title('Descriptive local continuity')
    axes[1].bar(x - 0.18, summary.minimum_raw_p, width=0.36, label='Minimum raw p', color='#4C78A8')
    axes[1].bar(x + 0.18, summary.minimum_adjusted_p, width=0.36, label='Minimum BH-adjusted p', color='#58508D')
    axes[1].axhline(0.05, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel('Exploratory case-level p-value')
    axes[1].set_title('pCR comparisons within each modality setting')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, ['mIF', 'H&E', 'Fusion'])
    save(fig, destination / 'modality_spatial_comparison.png')


def copy_spatial_panels(paths, destination: Path) -> None:
    import shutil

    source = paths.run / 'feature_characterisation' / 'spatial_community_graph' / 'figures'
    mapping = {
        '02_cluster_selection_stability.png': 'statistics_cluster_selection.png',
        '04_hierarchy_and_tissue_profiles.png': 'statistics_motif_profiles.png',
        '06_3D_spatial_landscapes.png': 'statistics_roi_landscapes.png',
        '07_patient_similarity_heatmap.png': 'statistics_patient_similarity.png',
        '09_pCR_effect_size_forest.png': 'statistics_pcr_effects.png',
    }
    for original, final in mapping.items():
        shutil.copy2(source / original, destination / final)
