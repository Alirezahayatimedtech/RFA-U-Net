
# -*- coding: utf-8 -*-
"""RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

This script implements a Vision Transformer (ViT) encoder pre-trained with RETFound
weights and an Attention U-Net decoder for segmenting the choroid in OCT images.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from timm.layers import drop_path, to_2tuple, trunc_normal_
import models_vit
from util.pos_embed import interpolate_pos_embed
import argparse
import gdown
import sys
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from dataset import OCTDataset



def download_retfound_weights_hf(repo_id: str, filename: str, cache_dir: str = "weights"):
    """
    Download `filename` from the HF repo `repo_id` into cache_dir and return local path.
    """
    os.makedirs(cache_dir, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        print(f"✅ Downloaded {filename} from HF repo {repo_id} to {path}")
        return path
    except RepositoryNotFoundError as e:
        raise FileNotFoundError(f"Could not find {repo_id}/{filename} on HF. {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="RFA-U-Net for OCT Choroid Segmentation")
    parser.add_argument('--image_dir', type=str, required=False, help='Path to the directory containing OCT images (required for training)')
    parser.add_argument('--mask_dir', type=str, required=False, help='Path to the directory containing mask images (required for training)')
    parser.add_argument('--weights_path', type=str, default='weights/rfa_unet_best.pth',
                        help='Path to the pre-trained weights file (used if weights_type is retfound or rfa-unet)')
    parser.add_argument('--weights_type', type=str, default='none', choices=['none', 'retfound', 'rfa-unet'],
                        help='Type of weights to load: "none" for random initialization, "retfound" for RETFound weights (training from scratch), "rfa-unet" for pre-trained RFA-U-Net weights (inference/fine-tuning)')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--test_only', action='store_true', help='Run inference on external data without training')
    parser.add_argument('--test_image_dir', type=str, default=None, help='Path to external test images (required if --test_only)')
    parser.add_argument('--test_mask_dir', type=str, default=None, help='Path to external test masks (required if --test_only)')
    parser.add_argument('--segment_dir', type=str, default=None, help='Path to directory containing images to segment (no masks needed)')
    parser.add_argument('--output_dir', type=str, default='segment_results', help='Directory to save segmentation results')
    parser.add_argument('--save_overlay', action='store_true', help='Save overlay images with segmentation boundaries')
    parser.add_argument('--pixel_size_micrometers', type=float, default=10.35, help='Pixel size in micrometers for boundary error computation')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binarizing predicted masks')
    return parser.parse_args()
# Parse arguments
args = parse_args()

# Configuration based on command-line arguments
config = {
    "image_size": args.image_size,
    "hidden_dim": 1024,
    "patch_size": 16,
    "num_channels": 3,
    "num_classes": 2,
    "retfound_weights_path": args.weights_path
}

# Weights file paths
RETFOUND_WEIGHTS_PATH = "weights/RETFound_oct_weights.pth"
RFA_UNET_WEIGHTS_PATH  = "weights/rfa_unet_best.pth"

# URL for downloading RFA-U-Net weights
RFA_UNET_WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=1q2giAcI8ASe2qnA9L69Mqb01l2qKjTV0"

# Function to download RFA-U-Net weights (unchanged)
def download_weights(weights_path, url):
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}. Downloading...")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        gdown.download(url, weights_path, quiet=False)
        print(f"Weights downloaded to {weights_path}")
    else:
        print(f"Weights file already exists at {weights_path}")

# Determine which weights to load based on weights_type
if args.weights_type == 'retfound':
    print("Using RETFound weights for training from scratch")
    # ensure the folder exists
    os.makedirs(os.path.dirname(RETFOUND_WEIGHTS_PATH), exist_ok=True)

    if os.path.exists(RETFOUND_WEIGHTS_PATH):
        config["retfound_weights_path"] = RETFOUND_WEIGHTS_PATH
    else:
        config["retfound_weights_path"] = download_retfound_weights_hf(
            repo_id="YukunZhou/RETFound_mae_natureOCT",
            filename="RETFound_mae_natureOCT.pth",
            cache_dir="weights"
        )
    print(f"→ RETFound weights at {config['retfound_weights_path']}")
    
elif args.weights_type == 'rfa-unet':
    if os.path.exists(args.weights_path):
        config["retfound_weights_path"] = args.weights_path
        print(f"Using pre-trained RFA-U-Net weights from user-provided path: {args.weights_path}")
    else:
        # fallback to Google-Drive download
        config["retfound_weights_path"] = RFA_UNET_WEIGHTS_PATH
        download_weights(RFA_UNET_WEIGHTS_PATH, RFA_UNET_WEIGHTS_URL)
        print("Using pre-trained RFA-U-Net weights for inference or fine-tuning")

elif args.weights_type == 'none':
    print("No pre-trained weights specified. Initializing model with random weights.")

# Convolutional block for decoder
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Attention gate for skip connections
class AttentionGate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        if Wg.shape[-2:] != Ws.shape[-2:]:
            Ws = nn.functional.interpolate(Ws, size=Wg.shape[-2:], mode='bilinear', align_corners=True)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        if out.shape[-2:] != s.shape[-2:]:
            s = nn.functional.interpolate(s, size=out.shape[-2:], mode='bilinear', align_corners=True)
        return out * s

# Decoder block with attention
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, reduce_skip=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.reduce_channels_x = nn.Conv2d(in_c[0], out_c, kernel_size=1)
        self.reduce_channels_s = nn.Conv2d(in_c[1], out_c, kernel_size=1) if reduce_skip else nn.Identity()
        self.ag = AttentionGate([out_c, out_c], out_c)
        self.c1 = ConvBlock(out_c + out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        x = self.reduce_channels_x(x)
        s = self.reduce_channels_s(s)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x

# Main RFA-U-Net model
class AttentionUNetViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize RETFound-based ViT encoder
        self.encoder = models_vit.RETFound_mae(
            num_classes=config["num_classes"],
            drop_path_rate=0.2,
            global_pool=True,
        )
        # Modify patch embedding for 3-channel input
        self.encoder.patch_embed.proj = nn.Conv2d(3, 1024, kernel_size=(16, 16), stride=(16, 16))
        # Load weights if specified
        if args.weights_type in ['retfound', 'rfa-unet']:
            # ensure file exists
            if not os.path.exists(config["retfound_weights_path"]):
                raise FileNotFoundError(
                    f"{args.weights_type}-weights file not found: "
                    f"{config['retfound_weights_path']}. Please download it and place it under `weights/`."
                )

            # 1) load full checkpoint (may include optimizer, epoch, etc.)
            raw_ckpt = torch.load(
                config["retfound_weights_path"],
                map_location='cpu',
                weights_only=False
            )
            print("Checkpoint keys:", list(raw_ckpt.keys()))

            # 2) extract just the model state dict (under model_state_dict or model)
            state_dict = raw_ckpt.get(
                'model_state_dict',
                raw_ckpt.get('model', raw_ckpt)
            )
            

            # 3) sanity‐check only the tensor values
            for k, v in state_dict.items():
                if torch.is_tensor(v):
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        raise ValueError(f"Checkpoint tensor '{k}' contains NaN or Inf")

            # 4) move tensors to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in state_dict.items()}


            if args.weights_type == 'retfound':
                # remove incompatible keys, interpolate positional embeddings
                own = self.encoder.state_dict()
                for k in ['patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
                    if k in state_dict and state_dict[k].shape != own[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del state_dict[k]
                interpolate_pos_embed(self.encoder, state_dict)
                msg = self.encoder.load_state_dict(state_dict, strict=False)
                print("Loaded RETFound weights, missing keys:", msg.missing_keys)
                trunc_normal_(self.encoder.head.weight, std=2e-5)
            else:
                # pure RFA-U-Net checkpoint
                msg = self.load_state_dict(state_dict, strict=False)
                print("Loaded RFA-U-Net weights, missing keys:", msg.missing_keys)


        # Decoder blocks
        self.d1 = DecoderBlock([1024, 1024], 512)
        self.d2 = DecoderBlock([512, 1024], 256)
        self.d3 = DecoderBlock([256, 1024], 128)
        self.d4 = DecoderBlock([128, 1024], 64)
        self.output = nn.Conv2d(64, config["num_classes"], kernel_size=1, padding=0)

    def forward(self, x):
        x = self.encoder.patch_embed(x)
        batch_size, num_patches, embed_dim = x.shape
        pos_embed = self.encoder.pos_embed[:, 1:, :]
        if num_patches != pos_embed.shape[1]:
            pos_embed = interpolate_pos_embed(self.encoder, {'pos_embed': pos_embed})
        x = x + pos_embed
        x = self.encoder.pos_drop(x)
        skip_connections = []
        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)
            if i in [5, 11, 17, 23]:
                skip_connections.append(x)
        z6, z12, z18, z24 = skip_connections
        z6 = z6.transpose(1, 2).reshape(batch_size, 1024, 14, 14)
        z12 = z12.transpose(1, 2).reshape(batch_size, 1024, 14, 14)
        z18 = z18.transpose(1, 2).reshape(batch_size, 1024, 14, 14)
        z24 = z24.transpose(1, 2).reshape(batch_size, 1024, 14, 14)
        x = self.d1(z24, z18)
        x = self.d2(x, z12)
        x = self.d3(x, z6)
        x = self.d4(x, z6)
        output = self.output(x)
        return output

# Tversky Loss (Aligned with root code)
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth

    def forward(self, outputs, targets):
        # outputs: (B,2,H,W), targets: one‐hot (B,2,H,W)
        probs = torch.sigmoid(outputs[:,1,:,:])         # only choroid channel
        t      = targets[:,1,:,:]                       # only choroid channel
        p_flat = probs.contiguous().view(-1)
        t_flat = t.contiguous().view(-1)

        tp = (p_flat * t_flat).sum()
        fn = ((1 - p_flat) * t_flat).sum()
        fp = (p_flat * (1 - t_flat)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky


# Dice Loss (Defined but not used, for compatibility with root code)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs).contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (outputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# Dice Score (Updated to compute both combined and choroid-specific Dice scores)
def dice_score(outputs, targets, smooth=1e-6):
    # Compute combined Dice score (across both classes, matching root code)
    outputs_combined = torch.sigmoid(outputs).contiguous().view(-1)
    targets_combined = targets.contiguous().view(-1)
    intersection_combined = (outputs_combined * targets_combined).sum()
    dice_combined = (2. * intersection_combined) / (outputs_combined.sum() + targets_combined.sum())

    # Compute choroid-specific Dice score (channel 1)
    outputs_choroid = torch.sigmoid(outputs[:, 1, :, :]).contiguous().view(-1)
    targets_choroid = targets[:, 1, :, :].contiguous().view(-1)
    intersection_choroid = (outputs_choroid * targets_choroid).sum()
    dice_choroid = (2. * intersection_choroid + smooth) / (outputs_choroid.sum() + targets_choroid.sum() + smooth)

    return dice_combined.item(), dice_choroid.item()

# Boundary Detection and Error Computation
def find_boundaries(mask):
    upper_boundaries = []
    lower_boundaries = []
    for col in range(mask.shape[1]):
        non_zero_indices = np.where(mask[:, col] > 0)[0]
        if len(non_zero_indices) > 0:
            upper_boundaries.append(non_zero_indices[0])
            lower_boundaries.append(non_zero_indices[-1])
        else:
            upper_boundaries.append(None)
            lower_boundaries.append(None)
    return upper_boundaries, lower_boundaries

def compute_errors(pred_boundaries, gt_boundaries, pixel_size):
    signed_errors = []
    unsigned_errors = []
    for pred, gt in zip(pred_boundaries, gt_boundaries):
        if pred is None or gt is None:
            continue
        signed_error = (pred - gt) * pixel_size
        unsigned_error = abs(signed_error)
        signed_errors.append(signed_error)
        unsigned_errors.append(unsigned_error)
    if not signed_errors:  # If no valid boundary pairs, return 0.0
        print("Warning: No valid boundary pairs found for error computation. Returning 0.0.")
        return 0.0, 0.0
    return np.mean(signed_errors), np.mean(unsigned_errors)

# Visualization Function
def plot_boundaries(images, true_masks, predicted_masks, threshold):
    """
    Plots the original image, true mask, predicted mask, and boundaries.
    Also computes and displays the mean signed and unsigned errors in micrometers.

    Parameters:
    images (torch.Tensor): Batch of original images.
    true_masks (torch.Tensor): Batch of true masks.
    predicted_masks (torch.Tensor): Batch of predicted masks.
    threshold (float): Threshold for binarizing predicted masks.
    """
    batch_size = images.size(0)

    for i in range(batch_size):
        image = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format for RGB
        true_mask = true_masks[i, 1].cpu().numpy()  # True choroid mask
        predicted_mask = predicted_masks[i, 1].cpu().numpy()  # Predicted choroid mask

        predicted_mask_binary = (predicted_mask > threshold).astype(np.uint8)  # FIXED: use parameter
        # we know true_mask is already 0/1, so just test for ==1
        true_mask_binary = (true_mask == 1).astype(np.uint8)

        # Get boundaries
        pred_upper, pred_lower = find_boundaries(predicted_mask_binary)
        gt_upper, gt_lower = find_boundaries(true_mask_binary)

        # Compute errors in micrometers
        upper_signed_error, upper_unsigned_error = compute_errors(pred_upper, gt_upper, args.pixel_size_micrometers)
        lower_signed_error, lower_unsigned_error = compute_errors(pred_lower, gt_lower, args.pixel_size_micrometers)

        # Print errors
        print(f"Image {i + 1}:")
        print(f"Upper Boundary Signed Error: {upper_signed_error:.2f} μm")
        print(f"Upper Boundary Unsigned Error: {upper_unsigned_error:.2f} μm")
        print(f"Lower Boundary Signed Error: {lower_signed_error:.2f} μm")
        print(f"Lower Boundary Unsigned Error: {lower_unsigned_error:.2f} μm")

        # Create boundary visualization
        combined_image = image.copy()
        for col in range(len(pred_upper)):
            if gt_upper[col] is not None:
                combined_image[gt_upper[col], col] = [1, 0, 0]  # Red for true upper boundary
            if gt_lower[col] is not None:
                combined_image[gt_lower[col], col] = [0, 1, 0]  # Green for true lower boundary
            if pred_upper[col] is not None:
                combined_image[pred_upper[col], col] = [0, 0, 1]  # Blue for predicted upper boundary
            if pred_lower[col] is not None:
                combined_image[pred_lower[col], col] = [1, 1, 0]  # Yellow for predicted lower boundary

        # Plotting
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(true_mask_binary, cmap='gray')
        plt.title('True Mask')
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(predicted_mask_binary, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(combined_image)
        plt.title('Boundaries\n(True Upper: Red, True Lower: Green, Pred Upper: Blue, Pred Lower: Yellow)')
        plt.axis('off')
        plt.show()
# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(
        size=(args.image_size, args.image_size),
        scale=(0.8, 1.0)
    ),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])
def save_segmentation_results(images, filenames, original_sizes, predicted_masks, output_dir, save_overlay=False, threshold=0.5):
    """
    Save segmentation results as masks and optionally as overlay images
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    
    if save_overlay:
        overlay_dir = os.path.join(output_dir, 'overlays')
        os.makedirs(overlay_dir, exist_ok=True)
    
    for i, (image, filename, original_size) in enumerate(zip(images, filenames, original_sizes)):
        base_name = os.path.splitext(filename)[0]
        
        # Get predicted mask for choroid class
        pred_mask = predicted_masks[i, 1].cpu().numpy()
        pred_mask_binary = (pred_mask > threshold).astype(np.uint8) * 255
        
        # Resize mask to original image size - FIXED
        pred_mask_pil = Image.fromarray(pred_mask_binary)
        
        # Ensure original_size is in correct format (width, height)
        if isinstance(original_size, (list, tuple)) and len(original_size) == 2:
            # original_size is already (width, height)
            target_size = original_size
        else:
            # Fallback: use a default size or try to extract from image
            print(f"Warning: Unexpected original_size format: {original_size}. Using fallback.")
            target_size = (224, 224)  # Default fallback
        
        pred_mask_pil = pred_mask_pil.resize(target_size, Image.NEAREST)
        
        # Save mask
        mask_path = os.path.join(mask_dir, f'{base_name}_mask.png')
        pred_mask_pil.save(mask_path)
        
        if save_overlay:
            # Create overlay image
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            # Denormalize if needed (assuming image is in [0,1])
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            # Create overlay
            overlay = create_overlay_image(image_np, pred_mask_binary, target_size)
            overlay_path = os.path.join(overlay_dir, f'{base_name}_overlay.png')
            overlay.save(overlay_path)
            
        print(f"Saved segmentation for {filename} to {mask_path}")



def create_overlay_image(image_np, mask_binary, original_size):
    """
    Create an overlay image with segmentation boundaries
    """
    # Resize mask to original size
    mask_pil = Image.fromarray(mask_binary)
    mask_resized = mask_pil.resize(original_size, Image.NEAREST)
    mask_np = np.array(mask_resized)
    
    # Find boundaries
    upper_boundaries, lower_boundaries = find_boundaries((mask_np > 0).astype(np.uint8))
    
    # Create RGB image from grayscale if needed
    if len(image_np.shape) == 2:
        image_rgb = np.stack([image_np] * 3, axis=-1)
    else:
        image_rgb = image_np
    
    # Draw boundaries
    overlay_image = image_rgb.copy()
    for col in range(len(upper_boundaries)):
        if upper_boundaries[col] is not None:
            row = min(upper_boundaries[col], overlay_image.shape[0]-1)
            overlay_image[row, col] = [255, 0, 0]  # Red for upper boundary
        if lower_boundaries[col] is not None:
            row = min(lower_boundaries[col], overlay_image.shape[0]-1)
            overlay_image[row, col] = [0, 255, 0]  # Green for lower boundary
    
    return Image.fromarray(overlay_image)
# Training and Evaluation Functions
def train_fold(train_loader, valid_loader, test_loader, model, criterion, optimizer, device, num_epochs, scaler, threshold):
    # track best validation choroid‐dice
    best_choroid = -1.0

    for epoch in range(num_epochs):
        # — training pass —
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                if torch.isnan(outputs).any():
                    print("Warning: Model outputs contain nan values, skipping batch")
                    continue
                loss = criterion(outputs, masks)
                if torch.isnan(loss):
                    print("Warning: Loss is nan, skipping batch")
                    continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        # — validation pass —
        model.eval()
        dice_choroid_scores = []
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                _, dch = dice_score(outputs, masks)
                dice_choroid_scores.append(dch)

        avg_choroid = np.mean(dice_choroid_scores)
        print(f"  → Validation Choroid Dice: {avg_choroid:.4f}")

        # — save best checkpoint —
        if avg_choroid > best_choroid:
            best_choroid = avg_choroid
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_choroid_dice': best_choroid
            }
            os.makedirs("weights", exist_ok=True)
            torch.save(ckpt, "weights/best_rfa_unet.pth")
            print(f"💾 New best checkpoint saved: weights/best_rfa_unet.pth")

    # — after all epochs, run full evaluation on validation & test sets as before —

    model.eval()
    dice_combined_scores, dice_choroid_scores = [], []
    upper_signed_errors, upper_unsigned_errors, lower_signed_errors, lower_unsigned_errors = [], [], [], []
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_combined, dice_choroid = dice_score(outputs, masks)
            dice_combined_scores.append(dice_combined)
            dice_choroid_scores.append(dice_choroid)
            predicted_masks = torch.sigmoid(outputs).cpu().numpy()
            true_masks = masks.cpu().numpy()
            for i in range(images.size(0)):
                pm = (predicted_masks[i,1] > threshold).astype(np.uint8)
                tm = (true_masks[i,1] == 1).astype(np.uint8)
                pu, pl = find_boundaries(pm)
                gu, gl = find_boundaries(tm)
                us, uu = compute_errors(pu, gu, args.pixel_size_micrometers)
                ls, lu = compute_errors(pl, gl, args.pixel_size_micrometers)
                upper_signed_errors.append(us)
                upper_unsigned_errors.append(uu)
                lower_signed_errors.append(ls)
                lower_unsigned_errors.append(lu)

    avg_dice_combined = np.mean(dice_combined_scores)
    avg_dice_choroid = np.mean(dice_choroid_scores)
    avg_upper_signed_error = np.mean(upper_signed_errors)
    avg_upper_unsigned_error = np.mean(upper_unsigned_errors)
    avg_lower_signed_error = np.mean(lower_signed_errors)
    avg_lower_unsigned_error = np.mean(lower_unsigned_errors)

    # Visualize the first batch from the test set
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
            plot_boundaries(images, masks, outputs, threshold)
            break

    return avg_dice_combined, avg_dice_choroid, avg_upper_signed_error, avg_upper_unsigned_error, avg_lower_signed_error, avg_lower_unsigned_error
class OCTSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, image_size, transform=None):
        self.image_size = image_size
        self.transform = transform
        
        # Supported extensions
        exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        
        # Build list of image paths
        self.image_paths = []
        for fname in os.listdir(image_dir):
            _, ext = os.path.splitext(fname)
            if ext.lower() in exts:
                self.image_paths.append(os.path.join(image_dir, fname))
        
        # Sort for consistent ordering
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # This returns (width, height)
        
        if self.transform:
            image = self.transform(image)
            
        return image, os.path.basename(img_path), original_size



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AttentionUNetViT(config).to(device)
    
    # Freeze early layers if using pre-trained weights
    if args.weights_type in ['retfound', 'rfa-unet']:
        for i, blk in enumerate(model.encoder.blocks):
            if i < 23:
                for p in blk.parameters():
                    p.requires_grad = False

    # ===== SEGMENTATION MODE (No masks needed) =====
    if args.segment_dir:
        print(f"Running segmentation on images in: {args.segment_dir}")
        
        # Create segmentation dataset
        segment_dataset = OCTSegmentationDataset(
            args.segment_dir,
            args.image_size,
            transform=val_test_transform
        )
        
        segment_loader = DataLoader(
            segment_dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        # Load model weights
        if os.path.exists(config['retfound_weights_path']):
            checkpoint = torch.load(config['retfound_weights_path'], map_location=device, weights_only=False)
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded weights from {config['retfound_weights_path']}")
        else:
            print(f"❌ Error: Weight file not found at {config['retfound_weights_path']}")
            sys.exit(1)
        
        model.eval()
        
        all_images = []
        all_filenames = []
        all_original_sizes = []
        all_predicted_masks = []
        
        print(f"Processing {len(segment_dataset)} images...")
        with torch.no_grad():
            for batch_idx, (images, filenames, original_sizes) in enumerate(segment_loader):
                images = images.to(device)
                outputs = model(images)
                predicted_masks = torch.sigmoid(outputs)
                
                all_images.append(images.cpu())
                all_filenames.extend(filenames)
                all_original_sizes.extend(original_sizes)
                all_predicted_masks.append(predicted_masks.cpu())
                
                print(f"Processed batch {batch_idx + 1}/{len(segment_loader)}")
        
        # Concatenate all batches
        all_images = torch.cat(all_images, dim=0)
        all_predicted_masks = torch.cat(all_predicted_masks, dim=0)
        
        # Save results
        save_segmentation_results(
            all_images, 
            all_filenames, 
            all_original_sizes, 
            all_predicted_masks, 
            args.output_dir, 
            args.save_overlay, 
            args.threshold
        )
        
        print(f"✅ Segmentation completed! Results saved to {args.output_dir}")
        print(f"   - Masks saved to: {os.path.join(args.output_dir, 'masks')}")
        if args.save_overlay:
            print(f"   - Overlays saved to: {os.path.join(args.output_dir, 'overlays')}")
        sys.exit(0)


    # ===== TRAINING MODE =====
    assert args.image_dir and args.mask_dir, (
        '--image_dir and --mask_dir are required for training'
    )
    
    # Load pre-trained weights if specified
    if args.weights_type in ['retfound', 'rfa-unet'] and os.path.exists(config["retfound_weights_path"]):
        checkpoint = torch.load(config["retfound_weights_path"], map_location=device, weights_only=False)
        
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded weights from {config['retfound_weights_path']}")
    
    # Setup training components
    criterion = TverskyLoss(alpha=0.7, beta=0.3, smooth=1e-6).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = GradScaler('cuda')
    
    # Prepare datasets
    full_dataset = OCTDataset(args.image_dir, args.mask_dir, args.image_size, transform=val_test_transform, num_classes=2)
    train_size = int(0.7 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"📊 Dataset Info:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(valid_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Starting training for {args.num_epochs} epochs...")
    
    # Train the model
    dice_combined, dice_choroid, upper_signed, upper_unsigned, lower_signed, lower_unsigned = train_fold(
        train_loader, valid_loader, test_loader, model, criterion, optimizer, device, args.num_epochs, scaler, args.threshold
    )
    
    # Print final results
    print(f"\n🎯 Final Validation Results:")
    print(f"   Dice (Combined): {dice_combined:.4f}")
    print(f"   Dice (Choroid): {dice_choroid:.4f}")
    print(f"   Upper Signed Error: {upper_signed:.2f} μm")
    print(f"   Upper Unsigned Error: {upper_unsigned:.2f} μm")
    print(f"   Lower Signed Error: {lower_signed:.2f} μm")
    print(f"   Lower Unsigned Error: {lower_unsigned:.2f} μm")
