# Input contract

## Slide pairing

Provide H&E and mIF QPTIFF files whose basenames encode the same de-identified case identifier. The pairing expression is `<case_id>_ScanN.qptiff`. Each modality must contain one unique slide per case. Registration verifies pairing before inference; the mIF directory may contain subdirectories.

The workflow uses the same restained tissue section in the two modalities. It does not assume serial sections or identical pyramid dimensions. The recorded coordinate direction is mIF → H&E; KRONOS2 sampling uses the corresponding inverse mapping.

## Clinical labels

The private CSV requires `case_id,pcr_label`, with one row per case. `1` denotes pCR, `0` denotes non-pCR, and a blank denotes an unknown outcome. Missing labels remain missing, not negative. Do not place a populated CSV in the public repository. Labels are optional for extraction and feature-QC preparation. Notebooks 07–08 reproduce the report's combined spatial/outcome analyses and require at least two labelled cases in each group; other cases remain eligible for unsupervised fitting even without labels.

## Model locations

- Tissue checkpoint: the trusted epoch-20 four-class model object used in the report. It is loaded with `weights_only=False` for compatibility with its original serialization; never load an untrusted checkpoint. `vendor/tiger` preserves the original import paths required by that object.
- KRONOS2: complete authorised repository, including configuration, custom Python files, marker metadata, `dinov2/`, and the teacher `.pth` weight.
- Virchow2: authorised local repository with `config.json` and `model.safetensors`.

Model architectures and weights are not interchangeable. The supplied tissue training configuration is a method reference; the package does not claim to reconstruct the original private training split, cached metadata or checkpoint-selection procedure.

## Generated outputs

Stages write only under the configured output root: `registration_global/`, `registration/`, `segmentation/` and `features/<run_name>/`. The intermediate filenames retain the original contracts so that transforms, class masks, ROI metadata, embeddings and clinical profiles connect correctly. Extraction and segmentation retain completed-case handling; the spatial-analysis caches remain separate from the original project.
