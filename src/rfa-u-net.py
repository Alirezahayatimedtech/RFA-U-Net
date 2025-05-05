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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from . import models_vit
from .util.pos_embed import interpolate_pos_embed
import argparse
import gdown

# Command-line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="RFA-U-Net for OCT Choroid Segmentation")
    parser.add_argument('--image_dir', type=str, default='data/images',
                        help='Path to the directory containing OCT images')
    parser.add_argument('--mask_dir', type=str, default='data/masks',
                        help='Path to the directory containing mask images')
    parser.add_argument('--weights_path', type=str, default='weights/rfa_unet_best.pth',
                        help='Path to the pre-trained weights file (used if weights_type is retfound or rfa-unet)')
    parser.add_argument('--weights_type', type=str, default='none', choices=['none', 'retfound', 'rfa-unet'],
                        help='Type of weights to load: "none" for random initialization, "retfound" for RETFound weights (training from scratch), "rfa-unet" for pre-trained RFA-U-Net weights (inference/fine-tuning)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--pixel_size_micrometers', type=float, default=10.35,
                        help='Pixel size in micrometers for boundary error computation')
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
    "retfound_weights_path": args.weights_path  # Will be updated based on weights_type
}

# Weights file paths
RETFOUND_WEIGHTS_PATH = "weights/RETFound_oct_weights.pth"
RFA_UNET_WEIGHTS_PATH = "weights/rfa_unet_best.pth"

# URL for downloading RFA-U-Net weights
RFA_UNET_WEIGHTS_URL = "https://drive.google.com/uc?id=1q2giAcI8ASe2qnA9L69Mqb01l2qKjTV0"

# Function to download weights if not present
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
    config["retfound_weights_path"] = RETFOUND_WEIGHTS_PATH
    print("Using RETFound weights for training from scratch")
elif args.weights_type == 'rfa-unet':
    # Use the user-provided weights_path if it exists, otherwise fall back to default and auto-download
    if os.path.exists(args.weights_path):
        config["retfound_weights_path"] = args.weights_path
        print(f"Using pre-trained RFA-U-Net weights from user-provided path: {args.weights_path}")
    else:
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
            if not os.path.exists(config["retfound_weights_path"]):
                if args.weights_type == 'retfound':
                    raise FileNotFoundError(f"RETFound weights file not found: {config['retfound_weights_path']}. Please download RETFound_oct_weights.pth from https://github.com/rmaphoh/RETFound_MAE and place it in the weights/ directory.")
                else:
                    raise FileNotFoundError(f"RFA-U-Net weights file not found: {config['retfound_weights_path']}. Please download rfa_unet_best.pth and place it in the weights/ directory.")
            checkpoint = torch.load(config["retfound_weights_path"], map_location='cuda', weights_only=False)
            if args.weights_type == 'retfound':
                checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
                state_dict = self.encoder.state_dict()
                for k in ['patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
                interpolate_pos_embed(self.encoder, checkpoint_model)
                msg = self.encoder.load_state_dict(checkpoint_model, strict=False)
                print("Loaded RETFound weights, missing keys:", msg.missing_keys)
                trunc_normal_(self.encoder.head.weight, std=2e-5)
            else:
                # For RFA-U-Net weights, load the entire model state dict
                model_state_dict = self.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint and checkpoint[k].shape != model_state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint[k]
                msg = self.load_state_dict(checkpoint, strict=False)
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

# Tversky Loss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs).contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        true_pos = (outputs * targets).sum()
        false_neg = ((1 - outputs) * targets).sum()
        false_pos = (outputs * (1 - targets)).sum()
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky

# Dice Loss
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

# Dice Score
def dice_score(outputs, targets):
    outputs = torch.sigmoid(outputs).contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (outputs * targets).sum()
    dice = (2. * intersection) / (outputs.sum() + targets.sum())
    return dice.item()

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
    return np.mean(signed_errors), np.mean(unsigned_errors)

# Visualization Function
def plot_boundaries(images, true_masks, predicted_masks):
    """
    Plots the original image, true mask, predicted mask, and boundaries.
    Also computes and displays the mean signed and unsigned errors in micrometers.

    Parameters:
    images (torch.Tensor): Batch of original images.
    true_masks (torch.Tensor): Batch of true masks.
    predicted_masks (torch.Tensor): Batch of predicted masks.
    """
    batch_size = images.size(0)

    for i in range(batch_size):
        image = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format for RGB
        true_mask = true_masks[i, 1].cpu().numpy()  # True choroid mask
        predicted_mask = predicted_masks[i, 1].cpu().numpy()  # Predicted choroid mask

        predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)
        true_mask_binary = (true_mask > 0.5).astype(np.uint8)

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

# Dataset Class
class OCTDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, num_classes=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes
        self.images = [img for img in os.listdir(image_dir) if img.endswith(('.jpg', '.JPG', '.tif'))]
        self.images = self.images[:1000]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '_mask.png').replace('.JPG', '_mask.png').replace('.tif', '_mask.png')
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.dim() == 2:
            mask = torch.nn.functional.one_hot(mask, num_classes=self.num_classes).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        return image, mask

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=(args.image_size, args.image_size), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

# Training and Evaluation Functions
def train_fold(train_loader, valid_loader, test_loader, model, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    dice_scores, upper_signed_errors, upper_unsigned_errors, lower_signed_errors, lower_unsigned_errors = [], [], [], [], []
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice = dice_score(outputs, masks)
            dice_scores.append(dice)
            predicted_masks = torch.sigmoid(outputs).cpu().numpy()
            true_masks = masks.cpu().numpy()
            for i in range(images.size(0)):
                predicted_mask_binary = (predicted_masks[i, 1] > 0.5).astype(np.uint8)
                true_mask_binary = (true_masks[i, 1] > 0.5).astype(np.uint8)
                pred_upper, pred_lower = find_boundaries(predicted_mask_binary)
                gt_upper, gt_lower = find_boundaries(true_mask_binary)
                upper_signed, upper_unsigned = compute_errors(pred_upper, gt_upper, args.pixel_size_micrometers)
                lower_signed, lower_unsigned = compute_errors(pred_lower, gt_lower, args.pixel_size_micrometers)
                upper_signed_errors.append(upper_signed)
                upper_unsigned_errors.append(upper_unsigned)
                lower_signed_errors.append(lower_signed)
                lower_unsigned_errors.append(lower_unsigned)
    avg_dice = np.mean(dice_scores)
    avg_upper_signed_error = np.mean(upper_signed_errors)
    avg_upper_unsigned_error = np.mean(upper_unsigned_errors)
    avg_lower_signed_error = np.mean(lower_signed_errors)
    avg_lower_unsigned_error = np.mean(lower_unsigned_errors)

    # Visualize the first batch from the test set
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
            predicted_masks = (outputs > 0.5).float()
            plot_boundaries(images, masks, predicted_masks)
            break  # Visualize only the first batch

    return avg_dice, avg_upper_signed_error, avg_upper_unsigned_error, avg_lower_signed_error, avg_lower_unsigned_error

# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNetViT(config).to(device)
    # Load pre-trained weights (either RETFound or RFA-U-Net based on weights_type)
    if args.weights_type in ['retfound', 'rfa-unet'] and os.path.exists(config["retfound_weights_path"]):
        checkpoint = torch.load(config["retfound_weights_path"], map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights from {config['retfound_weights_path']}")
    criterion = DiceLoss(smooth=1e-6).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.006622586228339055)
    full_dataset = OCTDataset(args.image_dir, args.mask_dir, transform=val_test_transform, num_classes=2)
    train_size = int(0.7 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])
    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dice, upper_signed, upper_unsigned, lower_signed, lower_unsigned = train_fold(
        train_loader, valid_loader, test_loader, model, criterion, optimizer, device, args.num_epochs
    )
    print(f"Validation Dice: {dice:.4f}, Upper Signed Error: {upper_signed:.2f} μm, Upper Unsigned Error: {upper_unsigned:.2f} μm, "
          f"Lower Signed Error: {lower_signed:.2f} μm, Lower Unsigned Error: {lower_unsigned:.2f} μm")
