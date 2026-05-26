import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from models import UNet, BitemporalOmbriaNet, MultimodalOmbriaNet
from metrics import pixel_accuracy, mean_IU, macro_iou, frequency_weighted_IU
from baselines import run_otsu_mndwi, BaselineSVM
from train import train_model, set_seeds

class OmbriaDataset(Dataset):
    def __init__(self, s2_dir, s1_dir, split="train", mode="multimodal"):
        self.split = split
        self.mode = mode
        
        self.mask_dir = os.path.join(s2_dir, split, "MASK")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found at {self.mask_dir}")
            
        self.mask_files = sorted(os.listdir(self.mask_dir))
        
        self.s2_after_dir = os.path.join(s2_dir, split, "AFTER")
        self.s2_before_dir = os.path.join(s2_dir, split, "BEFORE")
        self.s1_after_dir = os.path.join(s1_dir, split, "AFTER")
        self.s1_before_dir = os.path.join(s1_dir, split, "BEFORE")
        
    def __len__(self):
        return len(self.mask_files)
        
    def __getitem__(self, idx):
        mask_name = self.mask_files[idx]
        
        # Load mask
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")
        mask_np = np.array(mask) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        mask_tensor = torch.tensor(mask_np).unsqueeze(0)  # (1, H, W)
        
        # Load inputs depending on mode
        if self.mode == "unet":
            s2_after_file = mask_name.replace("_mask_", "_after_")
            s2_after = Image.open(os.path.join(self.s2_after_dir, s2_after_file)).convert("RGB")
            s2_after_np = np.array(s2_after, dtype=np.float32).transpose(2, 0, 1) / 255.0
            return torch.tensor(s2_after_np), mask_tensor
            
        elif self.mode == "bitemporal":
            s2_after_file = mask_name.replace("_mask_", "_after_")
            s2_before_file = mask_name.replace("_mask_", "_before_")
            s2_after = Image.open(os.path.join(self.s2_after_dir, s2_after_file)).convert("RGB")
            s2_before = Image.open(os.path.join(self.s2_before_dir, s2_before_file)).convert("RGB")
            s2_after_np = np.array(s2_after, dtype=np.float32).transpose(2, 0, 1) / 255.0
            s2_before_np = np.array(s2_before, dtype=np.float32).transpose(2, 0, 1) / 255.0
            return [torch.tensor(s2_after_np), torch.tensor(s2_before_np)], mask_tensor
            
        elif self.mode == "multimodal":
            s2_after_file = mask_name.replace("_mask_", "_after_")
            s2_before_file = mask_name.replace("_mask_", "_before_")
            s1_after_file = mask_name.replace("_mask_", "_after_").replace("S2_", "S1_")
            s1_before_file = mask_name.replace("_mask_", "_before_").replace("S2_", "S1_")
            
            s2_after = Image.open(os.path.join(self.s2_after_dir, s2_after_file)).convert("RGB")
            s2_before = Image.open(os.path.join(self.s2_before_dir, s2_before_file)).convert("RGB")
            s1_after = Image.open(os.path.join(self.s1_after_dir, s1_after_file)).convert("L")
            s1_before = Image.open(os.path.join(self.s1_before_dir, s1_before_file)).convert("L")
            
            s2_after_np = np.array(s2_after, dtype=np.float32).transpose(2, 0, 1) / 255.0
            s2_before_np = np.array(s2_before, dtype=np.float32).transpose(2, 0, 1) / 255.0
            s1_after_np = np.array(s1_after, dtype=np.float32)[np.newaxis, :, :] / 255.0
            s1_before_np = np.array(s1_before, dtype=np.float32)[np.newaxis, :, :] / 255.0
            
            return [torch.tensor(s2_after_np), torch.tensor(s2_before_np), torch.tensor(s1_after_np), torch.tensor(s1_before_np)], mask_tensor

def plot_learning_curves(history, save_path="learning_curves.png"):
    epochs = len(history["train_loss"])
    epoch_range = range(1, epochs + 1)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss subplot
    ax[0].plot(epoch_range, history["train_loss"], label="Train Loss", color="royalblue", linewidth=2)
    ax[0].plot(epoch_range, history["val_loss"], label="Val Loss", color="tomato", linewidth=2)
    ax[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax[0].set_xlabel("Epochs", fontsize=12)
    ax[0].set_ylabel("Loss", fontsize=12)
    ax[0].legend(fontsize=11)
    ax[0].grid(True, linestyle="--", alpha=0.6)
    
    # Accuracy subplot
    ax[1].plot(epoch_range, history["train_acc"], label="Train Acc", color="royalblue", linewidth=2)
    ax[1].plot(epoch_range, history["val_acc"], label="Val Acc", color="tomato", linewidth=2)
    ax[1].set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax[1].set_xlabel("Epochs", fontsize=12)
    ax[1].set_ylabel("Accuracy", fontsize=12)
    ax[1].legend(fontsize=11)
    ax[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualization] Learning curves saved to {save_path}")

def plot_sample_predictions(model, dataset, device, save_path="val_predictions.png", num_samples=3):
    model.eval()
    indices = list(range(min(num_samples, len(dataset))))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            inputs, mask = dataset[idx]
            
            # Forward pass
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs_device = [x.unsqueeze(0).to(device) for x in inputs]
                output = model(*inputs_device)
                s2_after = inputs[0].numpy().transpose(1, 2, 0)
            else:
                inputs_device = inputs.unsqueeze(0).to(device)
                output = model(inputs_device)
                s2_after = inputs.numpy().transpose(1, 2, 0)
                
            pred_mask = (output > 0.5).cpu().numpy().squeeze()
            target_mask = mask.numpy().squeeze()
            
            # Plot
            ax_row = axes[i] if num_samples > 1 else axes
            
            ax_row[0].imshow(s2_after)
            ax_row[0].set_title(f"Sample {idx} - S2 After", fontsize=11)
            ax_row[0].axis("off")
            
            ax_row[1].imshow(target_mask, cmap="gray")
            ax_row[1].set_title(f"Sample {idx} - Ground Truth", fontsize=11)
            ax_row[1].axis("off")
            
            ax_row[2].imshow(pred_mask, cmap="gray")
            ax_row[2].set_title(f"Sample {idx} - Model Pred", fontsize=11)
            ax_row[2].axis("off")
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualization] Sample predictions saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="OMBRIA Model Training, Evaluation, and Inference")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "predict", "otsu", "svm"], help="Execution mode")
    parser.add_argument("--model", type=str, default="multimodal", choices=["unet", "bitemporal", "multimodal"], help="Deep learning model variant")
    parser.add_argument("--data_dir_s1", type=str, default="OmbriaS1", help="Path to Sentinel-1 folder")
    parser.add_argument("--data_dir_s2", type=str, default="OmbriaS2", help="Path to Sentinel-2 folder")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Checkpoint file path")
    parser.add_argument("--pred_index", type=int, default=0, help="Index of test image to run predictions on")
    parser.add_argument("--output_image", type=str, default="prediction_plot.png", help="Filename for visualization output")
    
    args = parser.parse_args()
    set_seeds(7)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ------------------ CLASSICAL BASELINES ------------------
    if args.mode == "otsu":
        print("[Baseline] Running Otsu MNDWI thresholding baseline on test set...")
        test_dataset = OmbriaDataset(args.data_dir_s2, args.data_dir_s1, split="test", mode="multimodal")
        
        pa_list, iou_list, fwiou_list = [], [], []
        for idx in range(len(test_dataset)):
            inputs, mask = test_dataset[idx]
            # inputs[0] is s2_after_np, shape (3, H, W). Transpose back to (H, W, 3)
            img_s2 = inputs[0].numpy().transpose(1, 2, 0)
            pred = run_otsu_mndwi(img_s2)
            target = mask.squeeze(0).numpy()
            
            pa_list.append(pixel_accuracy(pred, target))
            iou_list.append(mean_IU(pred, target))
            fwiou_list.append(frequency_weighted_IU(pred, target))
            
        print(f"Otsu Average PA: {np.mean(pa_list):.4f}")
        print(f"Otsu Average IoU (positive): {np.mean(iou_list):.4f}")
        print(f"Otsu Average FWIoU: {np.mean(fwiou_list):.4f}")
        
    elif args.mode == "svm":
        print("[Baseline] Training Linear SVM C=10 baseline on train set pixels...")
        train_dataset = OmbriaDataset(args.data_dir_s2, args.data_dir_s1, split="train", mode="multimodal")
        test_dataset = OmbriaDataset(args.data_dir_s2, args.data_dir_s1, split="test", mode="multimodal")
        
        # Helper to construct pixel feature matrix
        def get_svm_features(dataset):
            X_list, Y_list = [], []
            for idx in range(len(dataset)):
                inputs, mask = dataset[idx]
                # inputs: [s2_after, s2_before, s1_after, s1_before]
                s2_after = inputs[0].numpy().transpose(1, 2, 0).reshape(-1, 3)
                s2_before = inputs[1].numpy().transpose(1, 2, 0).reshape(-1, 3)
                s1_after = inputs[2].numpy().reshape(-1, 1)
                s1_before = inputs[3].numpy().reshape(-1, 1)
                
                features = np.column_stack((s2_after, s2_before, s1_after, s1_before))
                X_list.append(features)
                Y_list.append(mask.numpy().reshape(-1))
            return np.vstack(X_list), np.hstack(Y_list)
            
        print("Extracting train features...")
        X_train, Y_train = get_svm_features(train_dataset)
        print(f"Train features shape: {X_train.shape}")
        
        svm_model = BaselineSVM(C=10.0)
        svm_model.train(X_train, Y_train)
        
        print("Extracting test features & evaluating...")
        X_test, Y_test = get_svm_features(test_dataset)
        preds = svm_model.predict(X_test)
        
        # Reshape predictions back to image tiles for spatial metrics
        H, W = 256, 256
        n_test = len(test_dataset)
        preds_tiles = preds.reshape(n_test, H, W)
        targets_tiles = Y_test.reshape(n_test, H, W)
        
        pa_list, iou_list, fwiou_list = [], [], []
        for idx in range(n_test):
            pa_list.append(pixel_accuracy(preds_tiles[idx], targets_tiles[idx]))
            iou_list.append(macro_iou(preds_tiles[idx], targets_tiles[idx]))
            fwiou_list.append(frequency_weighted_IU(preds_tiles[idx], targets_tiles[idx]))
            
        print(f"SVM Average PA: {np.mean(pa_list):.4f}")
        print(f"SVM Average IoU (macro): {np.mean(iou_list):.4f}")
        print(f"SVM Average FWIoU: {np.mean(fwiou_list):.4f}")
        
    # ------------------ DEEP LEARNING MODEL MODES ------------------
    else:
        # Instantiate correct model
        if args.model == "unet":
            model = UNet(in_channels=3)
        elif args.model == "bitemporal":
            model = BitemporalOmbriaNet(in_channels=3)
        elif args.model == "multimodal":
            model = MultimodalOmbriaNet(in_ch_s2=3, in_ch_s1=1)
            
        if args.mode == "train":
            print(f"[Deep Learning] Starting training pipeline for {args.model} model...")
            train_dataset = OmbriaDataset(args.data_dir_s2, args.data_dir_s1, split="train", mode=args.model)
            
            # Subsample for testing/quick verification purposes
            if os.environ.get("TEST_RUN") == "1":
                train_dataset = torch.utils.data.Subset(train_dataset, range(8))
                
            # Create a 90/10 train/validation split locally
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            if val_size == 0 and len(train_dataset) > 1:
                train_size = len(train_dataset) - 1
                val_size = 1
            train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
            
            history = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device, save_path=args.checkpoint)
            
            print("[Visualization] Generating learning curves...")
            plot_learning_curves(history, save_path="learning_curves.png")
            
            print("[Visualization] Generating validation sample predictions...")
            plot_sample_predictions(model, val_subset, device=device, save_path="val_predictions.png")
        elif args.mode == "evaluate":
            print(f"[Deep Learning] Starting evaluation pipeline on test split for {args.model} model...")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            model.to(device)
            model.eval()
            
            test_dataset = OmbriaDataset(args.data_dir_s2, args.data_dir_s1, split="test", mode=args.model)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            pa_list, iou_list, fwiou_list = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    if isinstance(batch[0], list) or isinstance(batch[0], tuple):
                        inputs = [x.to(device) for x in batch[0]]
                        targets = batch[1].to(device)
                        outputs = model(*inputs)
                    else:
                        inputs = batch[0].to(device)
                        targets = batch[1].to(device)
                        outputs = model(inputs)
                        
                    preds = (outputs > 0.5).cpu().numpy().squeeze(1)
                    targets_np = targets.cpu().numpy().squeeze(1)
                    
                    for idx in range(preds.shape[0]):
                        pa_list.append(pixel_accuracy(preds[idx], targets_np[idx]))
                        iou_list.append(macro_iou(preds[idx], targets_np[idx]))
                        fwiou_list.append(frequency_weighted_IU(preds[idx], targets_np[idx]))
                        
            print(f"Model Average PA: {np.mean(pa_list):.4f}")
            print(f"Model Average IoU (macro): {np.mean(iou_list):.4f}")
            print(f"Model Average FWIoU: {np.mean(fwiou_list):.4f}")
            
        elif args.mode == "predict":
            print(f"[Deep Learning] Making prediction for index {args.pred_index} using {args.model} model...")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            model.to(device)
            model.eval()
            
            test_dataset = OmbriaDataset(args.data_dir_s2, args.data_dir_s1, split="test", mode=args.model)
            inputs, mask = test_dataset[args.pred_index]
            
            with torch.no_grad():
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    inputs_device = [x.unsqueeze(0).to(device) for x in inputs]
                    output = model(*inputs_device)
                else:
                    output = model(inputs.unsqueeze(0).to(device))
                    
            pred_mask = (output > 0.5).cpu().numpy().squeeze()
            target_mask = mask.numpy().squeeze()
            
            # Plotting S2 AFTER image (first input channel 0, 1, 2)
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                s2_after = inputs[0].numpy().transpose(1, 2, 0)
            else:
                s2_after = inputs.numpy().transpose(1, 2, 0)
                
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(s2_after)
            ax[0].set_title("Input Sentinel-2 (AFTER)")
            ax[0].axis("off")
            
            ax[1].imshow(target_mask, cmap="gray")
            ax[1].set_title("Ground Truth (MASK)")
            ax[1].axis("off")
            
            ax[2].imshow(pred_mask, cmap="gray")
            ax[2].set_title("Model Prediction")
            ax[2].axis("off")
            
            plt.savefig(args.output_image)
            print(f"Saved visualization plot to {args.output_image}")

if __name__ == "__main__":
    main()
