import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def set_seeds(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_accuracy(preds, targets):
    # preds: probabilities from sigmoid, shape (B, 1, H, W)
    # targets: binary ground truth, shape (B, 1, H, W)
    preds_bin = (preds > 0.5).float()
    correct = (preds_bin == targets).float().sum()
    total = targets.numel()
    return float(correct / total)

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, device="cpu", save_path="best_model.pth"):
    """
    Trains a PyTorch model using Adam optimizer and BCE loss.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch in train_loader:
            # handle single vs multiple inputs
            if isinstance(batch[0], list) or isinstance(batch[0], tuple):
                inputs = [x.to(device) for x in batch[0]]
                targets = batch[1].to(device)
                optimizer.zero_grad()
                outputs = model(*inputs)
            else:
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            train_acc += compute_accuracy(outputs, targets) * targets.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch[0], list) or isinstance(batch[0], tuple):
                    inputs = [x.to(device) for x in batch[0]]
                    targets = batch[1].to(device)
                    outputs = model(*inputs)
                else:
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device)
                    outputs = model(inputs)

                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                val_acc += compute_accuracy(outputs, targets) * targets.size(0)

        # Average loss and accuracy
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        
        epoch_train_loss = train_loss / n_train
        epoch_train_acc = train_acc / n_train
        epoch_val_loss = val_loss / n_val
        epoch_val_acc = val_acc / n_val

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Save checkpoint if best validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model checkpoint to {save_path}")

    return history
