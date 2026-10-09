# Multimodal pathology image integration and spatial analysis

Code accompanying *Integrating Multimodal Pathology Images for Spatial Characterisation of Tissue*.

The workflow aligns mIF images to H&E, segments H&E into four tissue classes, samples matched ROIs, extracts KRONOS2 and Virchow2 features, and performs exploratory spatial and case-level analyses. The analysis unit is a patch or an independent case, not a segmented cell.

## Repository layout

```text
config/                 Shared paths, sampling settings and verified marker panel
environment/            Dependencies by workflow stage
notebooks/
  01_registration/      Global matching and local affine/TPS refinement
  02_segmentation/      Four-class H&E inference
  03_features/          KRONOS2, feature QC, and matched Virchow2 extraction
  04_statistics/        Spatial characterisation and matched-modality comparison
src/pathology/          Configuration and reusable extraction utilities
vendor/tiger/           Minimal model definitions needed to load the tissue checkpoint
docs/                   Input specifications and report-to-code mapping
tools/                  Static verification
```

## Inputs and models

No images, clinical records, embeddings, model weights, notebook outputs or existing analysis results are distributed. Supply H&E and mIF QPTIFF directories, an optional clinical CSV, the four-class tissue checkpoint, and complete local KRONOS2 and Virchow2 repositories. Use authorised model downloads and respect their respective licences. Do not infer protein identities from Opal labels; update `config/marker_mapping.csv` using the panel documentation for a different dataset.

`config/project.yaml` contains portable defaults. Override input paths in `config/local.yaml`, which is ignored by Git. The local copy points to the existing BS6211 inputs through relative paths. Outputs default to this repository's `outputs/` directory; no existing project outputs are overwritten.

## Environments

Use Python 3.10 or 3.11. Install the appropriate requirements file in the selected notebook environment. For CUDA stages, install the matched PyTorch 2.6 / torchvision 0.21 CUDA build appropriate for the machine before installing the remaining requirements. Registration can use CPU; the supplied segmentation and Virchow2 configurations require CUDA. Statistics does not require a GPU. Dependency ranges are compatibility constraints, not an archived environment lockfile.

```sh
python -m pip install -r environment/features.txt
python -m pip install -e .
```

Start Jupyter inside the repository and run notebooks in their numbered order. Use `environment/registration.txt` for notebooks 01–02, `segmentation.txt` for 03, `features.txt` for 04 and 06, `statistics.txt` for 05 and 07–08, and `figures.txt` for 09. Restart the kernel when switching environments. Notebooks do not install or upgrade packages automatically.

## Analysis settings

- H&E is fixed; mIF is moving. Transform metadata records both pyramid scales.
- Segmentation uses 512-pixel patches with 256-pixel stride at approximately 0.5 µm/pixel.
- Each eligible case supplies three 4096-pixel ROIs. Feature patches are 256 × 256 pixels with 50% minimum tissue and 10% minimum mIF signal.
- KRONOS2 returns 768 features. Virchow2 concatenates its 1280-dimensional class token with the mean of image patch tokens, excluding four register tokens, to return 2560 features.
- Original spatial analysis uses standardisation, 32 PCA components, tissue fractions, two graph-smoothing steps, and repeated motif clustering.
- The matched-modality comparison standardises and reduces each modality separately, combines equally weighted feature blocks, and fits each modality setting independently without tissue-fraction inputs or graph smoothing.
- pCR tests use cases as independent observations. Raw and Benjamini–Hochberg-adjusted p-values remain available; displaying raw p-values does not replace multiple-testing correction.

Automatic registration QC is not an independent accuracy measurement. Tissue masks and patch motifs are not validated cell phenotypes. Coarse and patient partitions that fail size criteria remain explicitly flagged.
