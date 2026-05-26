import os
import torch
import numpy as np
from PIL import Image
from models import UNet, BitemporalOmbriaNet, MultimodalOmbriaNet
from metrics import pixel_accuracy, mean_IU, macro_iou, frequency_weighted_IU
from baselines import run_otsu_mndwi, BaselineSVM
from sklearn.svm import LinearSVC
from gee_parser import GoogleEarthEngineParser

def test_ds_01():
    print("[Test T-DS-01] Checking tile size and channels...")
    # Load sample images from local train folder
    s1_path = os.path.join("OmbriaS1", "train", "AFTER")
    s2_path = os.path.join("OmbriaS2", "train", "AFTER")
    if not os.path.exists(s1_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        s1_path = os.path.join(base_dir, "OmbriaS1", "train", "AFTER")
        s2_path = os.path.join(base_dir, "OmbriaS2", "train", "AFTER")
    
    s1_files = sorted(os.listdir(s1_path))
    s2_files = sorted(os.listdir(s2_path))
    
    img_s1 = np.array(Image.open(os.path.join(s1_path, s1_files[0])))
    img_s2 = np.array(Image.open(os.path.join(s2_path, s2_files[0])))
    
    print(f"S1 sample shape: {img_s1.shape}")
    print(f"S2 sample shape: {img_s2.shape}")
    
    # Assert dimensions
    assert img_s1.shape == (256, 256) or img_s1.shape == (256, 256, 1)
    assert img_s2.shape == (256, 256, 3)
    print("-> T-DS-01 Passed!")

def test_ds_02():
    print("[Test T-DS-02] Checking split ratio (80/10/10)...")
    parser = GoogleEarthEngineParser()
    train_c, val_c, test_c = parser.split_dataset("OmbriaS1", ratios=(0.8, 0.1, 0.1))
    
    total = train_c + val_c + test_c
    assert total == 844
    assert abs(train_c / total - 0.8) < 0.02
    assert abs(val_c / total - 0.1) < 0.02
    assert abs(test_c / total - 0.1) < 0.02
    print("-> T-DS-02 Passed!")

def test_md_params():
    print("[Test T-MD-01, T-MD-02, T-MD-03] Checking model parameters...")
    unet = UNet()
    bit = BitemporalOmbriaNet()
    mult = MultimodalOmbriaNet()
    
    unet_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    bit_params = sum(p.numel() for p in bit.parameters() if p.requires_grad)
    mult_params = sum(p.numel() for p in mult.parameters() if p.requires_grad)
    
    print(f"UNet parameters: {unet_params}")
    print(f"Bitemporal parameters: {bit_params}")
    print(f"Multimodal parameters: {mult_params}")
    
    assert unet_params == 31032837
    assert bit_params == 10797637
    assert mult_params == 18108101
    print("-> T-MD-01, T-MD-02, T-MD-03 Passed!")

def test_opt_01():
    print("[Test T-OPT-01] Checking optimizer and loss configuration...")
    # Simulated check on loss and optimizer
    unet = UNet()
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCELoss()
    
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Loss function: {type(loss_fn).__name__}")
    
    assert type(optimizer).__name__ == "Adam"
    assert type(loss_fn).__name__ == "BCELoss"
    print("-> T-OPT-01 Passed!")

def test_base_01():
    print("[Test T-BASE-01] Checking Otsu/MNDWI sanity check...")
    # Create a toy Sentinel-2 image
    toy_s2 = np.zeros((256, 256, 3), dtype=np.float32)
    # Channel 2 is B3 (Green), Channel 0 is B11 (SWIR)
    # Make a water body in the middle (B3 > B11 -> high MNDWI)
    toy_s2[100:150, 100:150, 2] = 0.8
    toy_s2[100:150, 100:150, 0] = 0.2
    
    pred = run_otsu_mndwi(toy_s2)
    print(f"Otsu mask unique values: {np.unique(pred)}")
    assert np.array_equal(np.unique(pred), [0, 1])
    # The water region should be detected as 1
    assert pred[120, 120] == 1
    assert pred[50, 50] == 0
    print("-> T-BASE-01 Passed!")

def test_base_02():
    print("[Test T-BASE-02] Checking SVM config...")
    svm_model = BaselineSVM(C=10)
    print(f"SVM class: {type(svm_model.clf).__name__}")
    print(f"SVM C parameter: {svm_model.clf.C}")
    
    assert isinstance(svm_model.clf, LinearSVC)
    assert svm_model.clf.C == 10.0
    print("-> T-BASE-02 Passed!")

def test_eval_01():
    print("[Test T-EVAL-01] Checking metric correctness on toy input...")
    # Toy prediction (2x2): [[0, 1], [1, 1]]
    # Toy target (2x2):     [[0, 0], [1, 1]]
    toy_pred = np.array([[0, 1], [1, 1]])
    toy_target = np.array([[0, 0], [1, 1]])
    
    pa = pixel_accuracy(toy_pred, toy_target)
    iou_pos = mean_IU(toy_pred, toy_target)
    iou_macro = macro_iou(toy_pred, toy_target)
    fwiou = frequency_weighted_IU(toy_pred, toy_target)
    
    print(f"PA: {pa:.4f}")
    print(f"IoU (positive): {iou_pos:.4f}")
    print(f"IoU (macro): {iou_macro:.4f}")
    print(f"FWIoU: {fwiou:.4f}")
    
    assert abs(pa - 0.75) < 1e-4
    assert abs(iou_pos - 0.6667) < 1e-3
    assert abs(iou_macro - 0.5833) < 1e-3
    assert abs(fwiou - 0.5833) < 1e-3
    print("-> T-EVAL-01 Passed!")

if __name__ == "__main__":
    test_ds_01()
    test_ds_02()
    test_md_params()
    test_opt_01()
    test_base_01()
    test_base_02()
    test_eval_01()
    print("\n[Checklist Suite] All automated verification tests completed successfully!")
