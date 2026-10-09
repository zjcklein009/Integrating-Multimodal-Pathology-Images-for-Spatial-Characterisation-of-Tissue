# Report-to-code mapping

| Report component | Notebook | Implementation |
| --- | --- | --- |
| Global cross-modality matching | 01_Global_Registration | MatchAnything-ELoFTR, bidirectional filtering, spatial balancing, robust geometry and global QC |
| Local registration | 02_Local_Refinement | Reuse global transforms; tile matches, affine update, TPS, topology checks and exports |
| Four-class segmentation | 03_Tissue_Segmentation | Trusted VAN/UPer-style checkpoint, overlapping patch inference, rolling probability fusion and indexed masks |
| ROI sampling and mIF features | 04_KRONOS2_Extraction | Registration-eligible sampling, verified markers, inverse transform sampling, KRONOS2 extraction and per-case checkpoints |
| Feature coverage and analysis inputs | 05_Feature_QC_and_Case_Profiles | Finite/unique-coordinate QC, arithmetic patch means per case and ROI, optional clinical labels |
| H&E features | 06_Virchow2_Extraction | Frozen H&E encoder at the same physical patch coordinates |
| Original spatial statistics | 07_Spatial_Analysis | PCA/tissue context, graph representation, motif selection, permutation tests, heterogeneity, patient-partition audit, case-level pCR tests |
| Matched modalities | 08_Modality_Comparison | mIF-only, H&E-only and equal-weight feature fusion without graph smoothing or appended tissue fractions |
| Report presentation | 09_Report_Figures | Output-based registration/segmentation/QC panels, spatial-interaction plots and modality summary |

The separate global-registration notebook is retained because the reported local refinement reuses its matrices. The feature-profile notebook replaces earlier statistical launchers that were only needed to prepare the downstream input tables. It preserves patch-weighted arithmetic aggregation and explicit missing clinical labels, but omits unused predictive experiments and individual-embedding significance searches.

Core scientific settings in the registration, inference, extraction and retained statistical analyses follow their source implementations. Changes are limited to repository-relative path handling, cleared execution state, concise headings, removed installation/debug output, removal of unused plot panels, and a minimal preprocessing input-export notebook. Model weights and result files are deliberately absent.

Excluded: Mesmer/Nimbus; marker/nuclei detection and phenotype experiments; predictive-model tuning; older statistical-presentation notebooks; cross-modal tissue-transition experiments; unused multiscale/interactive/dashboard plots; manual TRE routines not executed in the report; unrelated TIGER challenge training/detection/validation entry points.

The original spatial motifs and the matched-modality motifs are independently fitted representations and must not be treated as numerically interchangeable. Existing report figures assembled outside the original notebooks are rebuilt from their native output files; their panel styling is not guaranteed to be pixel-identical to the submitted report.
