# OMBRIA: Multimodal & Multitemporal Supervised Flood Mapping

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official repository for **OMBRIA**, a multimodal and multitemporal satellite dataset, and **OmbriaNet**, a deep learning architecture designed for supervised flood mapping.

This repository accompanies the paper:  
> **"OmbriaNet—Supervised Flood Mapping via Convolutional Neural Networks Using Multitemporal Sentinel-1 and Sentinel-2 Data Fusion"**  
> *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*  
> [Read the Paper on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9723593)

---

## 🌟 Key Highlights

* **Multimodal Data Fusion**: Integrates Sentinel-1 SAR (Synthetic Aperture Radar) and Sentinel-2 MSI (Multispectral Optical) imagery.
* **Bitemporal Change Detection**: Utilizes pre-event and post-event imagery to distinguish permanent water bodies from temporary floods.
* **Supervised Binary Classification**: Features high-quality binary masks (1 = flooded, 0 = non-flooded) aligned with Copernicus Emergency Management Service (EMS) rapid mapping vectors.
* **Generalization**: Spans **23 historical flood events** globally from 2017 to 2021.

---

## 📂 Directory & File Layout

* **[main.py](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/main.py)**: Command-line interface (CLI) to execute training, evaluation, inference, and run baseline models.
* **[models.py](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/models.py)**: PyTorch implementations of U-Net baseline, Bitemporal OmbriaNet, and Multimodal OmbriaNet architectures.
* **[train.py](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/train.py)**: Training loop utility incorporating optimization, validation splitting, and automatic checkpointing.
* **[metrics.py](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/metrics.py)**: Metric calculators for Pixel Accuracy, Mean IoU, and Frequency-Weighted IoU.
* **[baselines.py](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/baselines.py)**: Hand-crafted baselines (Otsu thresholding and Linear Support Vector Machines).
* **[gee_parser.py](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/gee_parser.py)**: Simulation of GEE pipelines (filtering, reprojection, and splitting).
* **[test_checklist.py](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/test_checklist.py)**: Repository self-test suite checking data splits, metrics, and parameters.
* **[OMBRIA_Detailed_Documentation.md](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/OMBRIA_Detailed_Documentation.md)**: Comprehensive guide detailing the datasets, models, metrics, and GEE pipelines.
* **[KAGGLE_INSTRUCTIONS.md](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/KAGGLE_INSTRUCTIONS.md)**: Setup and run guide tailored for training models on Kaggle GPUs.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Your Setup
Run the test checklist suite to make sure directories are correctly placed and PyTorch features are functional:
```bash
python test_checklist.py
```

### 3. Run Training (Local or CLI)
Train the multimodal sensor-fusion model on the OMBRIA dataset:
```bash
python main.py --mode train --model multimodal --epochs 50 --batch_size 8 --data_dir_s1 OmbriaS1 --data_dir_s2 OmbriaS2
```

---

## 📊 Benchmarks

| Method | Modality | Pixel Accuracy (PA) | Mean IoU |
| :--- | :--- | :---: | :---: |
| Otsu MNDWI | Sentinel-2 | 0.7930 | 0.3330 |
| Linear SVM | S1 & S2 | 0.8504 | 0.6245 |
| U-Net | Sentinel-2 | 0.8251 | 0.5418 |
| Bitemporal OmbriaNet | Sentinel-2 | 0.8733 | 0.6457 |
| **Multimodal OmbriaNet (Ours)** | **S1 & S2 Fused** | **0.9010** | **0.7236** |

---

## 📖 Further Documentation

* **Detailed Guide**: Read [OMBRIA_Detailed_Documentation.md](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/OMBRIA_Detailed_Documentation.md) for data specifications, architecture details, formulas, and FAQs.
* **Kaggle GPU Guide**: Read [KAGGLE_INSTRUCTIONS.md](file:///c:/Users/Zui/Downloads/OMBRIA-master/OMBRIA-master/KAGGLE_INSTRUCTIONS.md) to run the code using Kaggle's free GPU instances.
