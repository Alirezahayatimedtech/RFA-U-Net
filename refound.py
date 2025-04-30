# -*- coding: utf-8 -*-
"""RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

This script implements a Vision Transformer (ViT) encoder pre-trained with RETFound
weights and an Attention U-Net decoder for segmenting the choroid in OCT images.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from timm.models.layers import trunc_normal_
import models_vit
from util.pos_embed import interpolate_pos_embed

# Constants
PIXEL_SIZE_MICROMETERS = 10.35  # Pixel size in micrometers
NUM_EPOCHS = 50
BATCH_SIZE = 8

# Configuration
config = {
    "image_size": 224,
    "hidden_dim": 1024,
    "patch_size": 16,
    "num_channels": 3,
    "num_classes": 2,
    "retfound_weights_path": "weights/RETFound_oct_weights.pth"
}

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
        # Load pre-trained weights
        if not os.path.exists(config["retfound_weights_path"]):
            raise FileNotFoundError(f"Weights file not found: {config['retfound_weights_path']}")
        checkpoint = torch.load(config["retfound_weights_path"], map_location='cuda', weights_only=False)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = self.encoder.state_dict()
        for k in ['patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(self.encoder, checkpoint_model)
        msg = self.encoder.load_state_dict(checkpoint_model, strict=False)
        print("Loaded weights, missing keys:", msg.missing_keys)
        trunc_normal_(self.encoder.head.weight, std=2e-5)

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
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
                upper_signed, upper_unsigned = compute_errors(pred_upper, gt_upper, PIXEL_SIZE_MICROMETERS)
                lower_signed, lower_unsigned = compute_errors(pred_lower, gt_lower, PIXEL_SIZE_MICROMETERS)
                upper_signed_errors.append(upper_signed)
                upper_unsigned_errors.append(upper_unsigned)
                lower_signed_errors.append(lower_signed)
                lower_unsigned_errors.append(lower_unsigned)
    avg_dice = np.mean(dice_scores)
    avg_upper_signed_error = np.mean(upper_signed_errors)
    avg_upper_unsigned_error = np.mean(upper_unsigned_errors)
    avg_lower_signed_error = np.mean(lower_signed_errors)
    avg_lower_unsigned_error = np.mean(lower_unsigned_errors)
    return avg_dice, avg_upper_signed_error, avg_upper_unsigned_error, avg_lower_signed_error, avg_lower_unsigned_error

# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNetViT(config).to(device)
    criterion = DiceLoss(smooth=1e-6).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.006622586228339055)
    image_dir = '/content/drive/MyDrive/Alireza/data'
    mask_dir = '/content/drive/MyDrive/Alireza/mask png'
    full_dataset = OCTDataset(image_dir, mask_dir, transform=val_test_transform, num_classes=2)
    train_size = int(0.7 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])
    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    dice, upper_signed, upper_unsigned, lower_signed, lower_unsigned = train_fold(
        train_loader, valid_loader, test_loader, model, criterion, optimizer, device, NUM_EPOCHS
    )
    print(f"Validation Dice: {dice:.4f}, Upper Signed Error: {upper_signed:.2f} μm, Upper Unsigned Error: {upper_unsigned:.2f} μm, "
          f"Lower Signed Error: {lower_signed:.2f} μm, Lower Unsigned Error: {lower_unsigned:.2f} μm")
