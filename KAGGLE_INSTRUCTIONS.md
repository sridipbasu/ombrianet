# Running OMBRIA on Kaggle with GPU

This guide explains how to set up, run, and train the OMBRIA models in a **Kaggle Notebook** using GPU acceleration.

---

## Step 1: Upload the Data and Code to Kaggle

To run the repository on Kaggle, you need to import the datasets (`OmbriaS1`, `OmbriaS2`) and the source code files.

### Option A: Uploading as a Kaggle Dataset (Recommended)
1. Zip your local project directory (excluding large caches like `__pycache__` or previous `.pth` checkpoints).
2. Go to [Kaggle Datasets](https://www.kaggle.com/datasets) and click **"New Dataset"**.
3. Name your dataset (e.g., `ombria-code-and-data`) and upload:
   - The project ZIP file (or individual scripts: `main.py`, `models.py`, `train.py`, `metrics.py`, `baselines.py`, `gee_parser.py`, `test_checklist.py`, `requirements.txt`).
   - The folders `OmbriaS1` and `OmbriaS2`.
4. Click **Create**.

### Option B: Clone/Upload Code and Data separately
- You can upload the data as one dataset, and create/clone the scripts directly inside the Kaggle notebook's working directory (`/kaggle/working`).

---

## Step 2: Create a Kaggle Notebook and Enable GPU

1. Open Kaggle and click **"New Notebook"**.
2. Click **"Add Input"** in the right-hand panel and search for your uploaded dataset (e.g., `ombria-code-and-data`). Add it to the notebook.
3. In the right sidebar under **Settings**:
   - Locate the **Accelerator** option.
   - Select **GPU T4 x2** or **GPU P100** (GPU T4 x2 is highly recommended for faster bitemporal/multimodal training).
   - Ensure the notebook environment is set to the latest version.

---

## Step 3: Copy Code to Working Directory and Install Dependencies

Since `/kaggle/input` is read-only, you should copy the Python scripts to the writeable `/kaggle/working` directory so that training logs, plots, and checkpoints (`best_model.pth`) can be saved.

In the first cell of your Kaggle notebook, run:

```python
# 1. Copy the code files and dataset folders from Input to Working directory
import os
import shutil

# Replace 'ombria-code-and-data' with the exact name of your dataset on Kaggle
input_dir = "/kaggle/input/ombria-code-and-data" 

# Copy python files
for file in os.listdir(input_dir):
    if file.endswith(".py") or file == "requirements.txt":
        shutil.copy(os.path.join(input_dir, file), "/kaggle/working/")
        print(f"Copied {file} to working directory.")

# Copy dataset folders
if not os.path.exists("/kaggle/working/OmbriaS1"):
    shutil.copytree(os.path.join(input_dir, "OmbriaS1"), "/kaggle/working/OmbriaS1")
    print("Copied OmbriaS1 to working directory.")
if not os.path.exists("/kaggle/working/OmbriaS2"):
    shutil.copytree(os.path.join(input_dir, "OmbriaS2"), "/kaggle/working/OmbriaS2")
    print("Copied OmbriaS2 to working directory.")

# Verify files copied
!ls -la /kaggle/working
```

In the next cell, install any required packages:

```bash
!pip install -r requirements.txt
```

---

## Step 4: Verify GPU and Run Checklist Tests

Run a simple check to verify that PyTorch is using the Kaggle GPU accelerator:

```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
```

Run the automated test checklist suite to ensure code compatibility and directory structure alignment:

```bash
!python test_checklist.py
```

---

## Step 5: Execute Training and Evaluation on Kaggle GPU

You can run training, evaluation, and baseline commands using the CLI interface. Since the datasets were copied into `/kaggle/working`, they will be automatically resolved by default, but you can also specify them explicitly.

### 1. Train the Multimodal OMBRIA Model on GPU
```bash
!python main.py --mode train --model multimodal --epochs 50 --batch_size 8 --data_dir_s1 OmbriaS1 --data_dir_s2 OmbriaS2
```

*Note: After training finishes, the script automatically generates and saves:*
- **`learning_curves.png`**: Plot of training & validation loss and accuracy across epochs.
- **`val_predictions.png`**: Visual comparison grid displaying the input, ground truth, and predicted flood masks for validation set samples.

To display these plots directly inside your Kaggle notebook cell:
```python
from IPython.display import Image, display
display(Image("learning_curves.png"))
display(Image("val_predictions.png"))
```

### 2. Evaluate the Trained Model on the Test Set
```bash
!python main.py --mode evaluate --model multimodal --data_dir_s1 OmbriaS1 --data_dir_s2 OmbriaS2 --checkpoint best_model.pth
```

### 3. Generate Predictions and Visualizations
To make a prediction and plot the input, ground truth, and predicted flood mask:
```bash
!python main.py --mode predict --model multimodal --pred_index 10 --output_image prediction_10.png --data_dir_s1 OmbriaS1 --data_dir_s2 OmbriaS2
```

To display the saved plot inside your Kaggle notebook:
```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("prediction_10.png")
plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.axis("off")
plt.show()
```

### 4. Run Classical Baselines
You can also run Otsu thresholding or train the SVM classifier:
```bash
# Otsu MNDWI Baseline
!python main.py --mode otsu --data_dir_s1 OmbriaS1 --data_dir_s2 OmbriaS2

# SVM Baseline (takes ~2 minutes to fit 40M pixels on Kaggle CPU/GPU)
!python main.py --mode svm --data_dir_s1 OmbriaS1 --data_dir_s2 OmbriaS2
```
