# test.py (Final Corrected Version)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import numpy as np
import h5py # <-- Import h5py

# Make sure these can be imported from your project structure
from data_loader import MultiInputSignalDataset
from model import MultiInputModel

# --- Utility functions and classes (copied from training) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = logits
        intersection = (probs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        probs = torch.sigmoid(logits)
        dice = self.dice_loss(probs, targets)
        return self.bce_weight * bce + self.dice_weight * dice

def dice_coefficient(probs, targets, smooth=1e-6):
    intersection = (probs * targets).sum()
    return (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)


def main(args):
    """
    Model evaluation/inference main function.
    """
    # --- 1. Configuration ---
    CONFIG = {
        "data_path": args.data_path,
        "batch_size": 16,
        "SAMPLING_RATE": 100_000_000,
        "START_FREQ": 2400.0,
        "END_FREQ": 2500.0,
        "NUM_BINS": 100000,
        "STFT_NPERSEG": 1024,
        "STFT_NOVERLAP": 512,
    }
    CONFIG['FREQS_1D'] = np.linspace(CONFIG['START_FREQ'], CONFIG['END_FREQ'], CONFIG['NUM_BINS'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # !! CORE FIX: Manually check for labels before creating the Dataset !!
    try:
        with h5py.File(args.data_path, 'r') as f:
            run_evaluation = 'labels' in f
        if run_evaluation:
            print("Labels dataset found in HDF5 file. Running in evaluation mode.")
        else:
            print("Warning: 'labels' dataset not found in HDF5 file. Running in inference-only mode.")
    except (FileNotFoundError, OSError) as e:
        print(f"Error: Could not open or read HDF5 file at {args.data_path}. Details: {e}")
        return

    # --- 2. Data Loading ---
    try:
        test_dataset = MultiInputSignalDataset(h5_path=CONFIG['data_path'], config=CONFIG)
    except Exception as e:
        print(f"Error: Failed to initialize MultiInputSignalDataset. Details: {e}")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- 3. Model Loading ---
    model = MultiInputModel().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model weights loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # --- 4. Evaluation or Inference Loop ---
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    criterion = CombinedLoss()

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for inputs, targets in progress_bar:
            for key in inputs:
                inputs[key] = inputs[key].to(device)

            logits = model(inputs)

            if run_evaluation:
                if targets.numel() > 0:
                    targets = targets.to(device)
                    loss = criterion(logits, targets)
                    probs = torch.sigmoid(logits)
                    dice = dice_coefficient(probs, targets)

                    total_loss += loss.item()
                    total_dice += dice.item()
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")
                else:
                    # This case should not happen if run_evaluation is true, but as a safeguard:
                    progress_bar.set_postfix(status="Labels expected but not found in batch")
            else:
                progress_bar.set_postfix(status="Inference only")

    # --- 5. Output Results ---
    print("\n--- Test Results ---")
    if run_evaluation:
        if len(test_loader) > 0:
            avg_loss = total_loss / len(test_loader)
            avg_dice = total_dice / len(test_loader)
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Average Dice Coefficient: {avg_dice:.4f}")
        else:
            print("Test loader was empty. No metrics to calculate.")
    else:
        print("Test run completed in inference-only mode.")
        print("No metrics were calculated.")
    print("--------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained Multi-Input Signal Model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model checkpoint (.pth file).")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the test data HDF5 file (.h5 file).")
    args = parser.parse_args()
    main(args)