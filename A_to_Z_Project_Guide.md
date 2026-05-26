# Ombria (A → Z) Project Guide

This document provides a complete A→Z overview for the Ombria repository: purpose, data layout, notebooks, models, reproduction steps, and suggested next steps.

**Purpose**
- **Project:** Flood mapping research dataset and notebooks accompanying the OmbriaNet paper ("OmbriaNet—Supervised Flood Mapping via Convolutional Neural Networks Using Multitemporal Sentinel-1 and Sentinel-2 Data Fusion").
- **Goal:** Provide dataset tiles, baseline models (UNet, OmbriaNet fusion, SVM), preprocessing, and evaluation scripts to benchmark flood detection using Sentinel-1 and Sentinel-2 imagery.

**Repository Snapshot**
- **README.md:** Dataset description and provenance.
- **Notebooks:** High-level analysis & model implementations: `OmbriaNet.ipynb`, `UNET.ipynb`, `SVM.ipynb`, `Evaluation.ipynb`, `Reconstruct.ipynb`, `patches.ipynb`, `Create 3D Images.ipynb`, `Otsu Threshold.ipynb`.
- **Data folders:** `2021/` (country subfolders), `OmbriaS1/` and `OmbriaS2/` (prepared train/test splits).

**Data Layout (detailed)**
- Root dataset variants:
  - `2021/<COUNTRY>/Sentinel1/BEFORE`, `AFTER`, `MASK`
  - `2021/<COUNTRY>/Sentinel2/BEFORE`, `AFTER`, `MASK`
  - `OmbriaS1/train|test/BEFORE,AFTER,MASK` and `OmbriaS2/train|test/BEFORE,AFTER,MASK`
- **File types:** PNG image tiles: `imbefore_*.png`, `imafter_*.png`, `gt_*.png`. Tiles are the unit used for training and evaluation.
- **Ground truth:** Binary masks from EMS Rapid Mapping (see README). Masks are used for pixel-wise supervised training.

**Primary Notebooks & What They Do**
- **`OmbriaNet.ipynb`**: Implements the OmbriaNet fusion CNN (two-input network merging features from Sentinel-1 and Sentinel-2). Includes model definition, training loop, and prediction snippets. Colab-first (uses `drive.mount`).
- **`UNET.ipynb`**: Standard single-input U‑Net architecture and training utilities (binary segmentation). Useful baseline.
- **`SVM.ipynb`**: Pixel-wise classification baseline using SVMs and hand-crafted features. Includes utility functions to create X/Y from image stacks.
- **`Evaluation.ipynb`**: Visualize training history; compute IoU, FW IoU, pixel accuracy, ROC/AUC; compares predicted tiles with ground truth.
- **`patches.ipynb`**, **`Reconstruct.ipynb`**: Code to extract image patches and reassemble predictions into larger mosaics / full scenes.
- **`Otsu Threshold.ipynb`**: Simple threshold-based flood detection baseline for comparison.
- **`Create 3D Images.ipynb`**: Utilities for stacking bands or creating visualization artifacts (3D visualization of patches).

**Models implemented**
- **OmbriaNet (fusion):** Two-branch encoder (Sentinel-1 and Sentinel-2) that concatenates mid-level features and decodes via U‑Net-like upsampling. Loss: binary crossentropy. Optimizers: RMSprop / Adam variants in notebooks.
- **U‑Net:** Classical encoder-decoder with skip connections; uses LeakyReLU activations and binary segmentation head.
- **SVM baseline:** Pixel features taken from image bands; trained with scikit-learn SVM/LinearSVC.

**Preprocessing & Assumptions**
- **Normalization:** Notebooks commonly scale image arrays by dividing by 255 (see `load_images_from_folder` helper). Ensure consistent normalization when combining Sentinel-1 (single-band or multi-band) and Sentinel-2 inputs.
- **Mask handling:** Masks are treated as binary; notebooks threshold masks (mask > 0.5 -> 1).
- **Tile size:** Models expect square tiles (example: 256×256) — verify dimensions before training.

**How to Reproduce (Colab)**
1. Open a notebook in Google Colab.
2. Mount Drive: `drive.mount('/content/drive')` and place `OmbriaS1` / `OmbriaS2` in Drive paths used by the notebook.
3. Run cells in order: data loaders → model definition → training loop. Modify save/load paths to point to Drive.

**How to Reproduce (Local)**
1. Create a Python virtual environment (recommend Python 3.8+):
   - `python -m venv venv` then activate it.
2. Install core packages (example):
   - `pip install numpy scikit-image scikit-learn matplotlib pillow tensorflow keras` (pin versions as needed).
3. Make paths configurable: replace `'/content/drive/MyDrive/...'` with local dataset paths (e.g., `./OmbriaS2/test/MASK`).
4. Run notebook cells or convert core cells to scripts (`models.py`, `train.py`, `predict.py`).

**Suggested Minimal `requirements.txt`**
- tensorflow (or tensorflow-cpu) 2.x
- keras (if separate from tensorflow.keras)
- scikit-image
- scikit-learn
- matplotlib
- pillow
- numpy

**Practical Tips & Pitfalls**
- Convert Colab `drive.mount` paths early to avoid file-not-found issues.
- Ensure Sentinel-1 and Sentinel-2 channels align (channel order) before concatenation in the fusion model.
- Monitor class imbalance: flood pixels often rare — consider weighted loss or class-balancing strategies.
- For larger scenes, use `patches.ipynb` + `Reconstruct.ipynb` workflow to tile and reassemble predictions.

**Evaluation & Metrics**
- Notebooks compute: **Pixel accuracy**, **IoU (Jaccard)**, **Frequency-Weighted IoU**, **ROC / AUC**. Use the same thresholds as in the notebooks for consistent comparisons.

**Reproducible Experiment Checklist**
- Fix random seed for numpy and TensorFlow.
- Record library versions (`pip freeze > requirements.txt`).
- Save model checkpoints and training `History` objects for later plotting in `Evaluation.ipynb`.

**Next Steps / Recommendations**
- Add a `requirements.txt` or `environment.yml` for reproducibility.
- Create small `scripts/` with `train.py`, `evaluate.py`, `predict.py` that accept CLI args for data paths and hyperparameters.
- Add a README section with minimal reproduction steps and example commands.
- Optionally convert notebooks to `.py` modules for CI or batch runs.

**References**
- OmbriaNet paper: IEEE (see README).

---
_This guide was generated to summarize and document the contents of the repository and to provide practical reproduction instructions._
