# -*- coding: utf-8 -*-
"""RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

This script implements a Vision Transformer (ViT) encoder pre-trained with RETFound
weights and an Attention U-Net decoder for segmenting the choroid in OCT images.
"""

import os
import sys
import argparse
gdown
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


def parse_args():
    parser = argparse.ArgumentParser(description="RFA-U-Net for OCT Choroid Segmentation")
    parser.add_argument('--image_dir', type=str, help='Path to the directory containing OCT images')
    parser.add_argument('--mask_dir', type=str, help='Path to the directory containing mask images')
    parser.add_argument('--weights_path', type=str, default='weights/rfa_unet_best.pth',
                        help='Path to the pre-trained weights file')
    parser.add_argument('--weights_type', type=str, default='none', choices=['none', 'retfound', 'rfa-unet'],
                        help='Type of weights to load')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--test_only', action='store_true', help='Run inference on external data without training')
    parser.add_argument('--test_image_dir', type=str, help='Path to external test images')
    parser.add_argument('--test_mask_dir', type=str, help='Path to external test masks')
    parser.add_argument('--pixel_size_micrometers', type=float, default=10.35,
                        help='Pixel size in micrometers for boundary error computation')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binarizing predicted masks')
    return parser.parse_args()

# Parse arguments\args = parse_args()

# Configuration
config = {
    "image_size": args.image_size,
    "hidden_dim": 1024,
    "patch_size": 16,
    "num_channels": 3,
    "num_classes": 2,
    "retfound_weights_path": args.weights_path
}

RETFOUND_WEIGHTS_PATH = "weights/RETFound_oct_weights.pth"
RFA_UNET_WEIGHTS_PATH = "weights/rfa_unet_best.pth"
RFA_UNET_WEIGHTS_URL = "https://drive.google.com/uc?id=1q2giAcI8ASe2qnA9L69Mqb01l2qKjTV0"

def download_weights(weights_path, url):
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}. Downloading...")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        gdown.download(url, weights_path, quiet=False)
        print(f"Weights downloaded to {weights_path}")
    else:
        print(f"Weights file already exists at {weights_path}")

# Determine weights type
if args.weights_type == 'retfound':
    config["retfound_weights_path"] = RETFOUND_WEIGHTS_PATH
elif args.weights_type == 'rfa-unet':
    if os.path.exists(args.weights_path):
        config["retfound_weights_path"] = args.weights_path
    else:
        config["retfound_weights_path"] = RFA_UNET_WEIGHTS_PATH
        download_weights(RFA_UNET_WEIGHTS_PATH, RFA_UNET_WEIGHTS_URL)


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


class AttentionGate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        if Wg.shape[-2:] != Ws.shape[-2:]:
            Ws = F.interpolate(Ws, size=Wg.shape[-2:], mode='bilinear', align_corners=True)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        if out.shape[-2:] != s.shape[-2:]:
            s = F.interpolate(s, size=out.shape[-2:], mode='bilinear', align_corners=True)
        return out * s


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, reduce_skip=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.reduce_channels_x = nn.Conv2d(in_c[0], out_c, kernel_size=1)
        self.reduce_channels_s = nn.Conv2d(in_c[1], out_c, kernel_size=1) if reduce_skip else nn.Identity()
        self.ag = AttentionGate([out_c, out_c], out_c)
        self.c1 = ConvBlock(out_c*2, out_c)
    def forward(self, x, s):
        x = self.up(x)
        x = self.reduce_channels_x(x)
        s = self.reduce_channels_s(s)
        s = self.ag(x, s)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)
        return x


class AttentionUNetViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = models_vit.RETFound_mae(
            num_classes=config["num_classes"], drop_path_rate=0.2, global_pool=True)
        self.encoder.patch_embed.proj = nn.Conv2d(3, config["hidden_dim"], kernel_size=(16,16), stride=(16,16))
        if args.weights_type in ['retfound','rfa-unet']:
            if not os.path.exists(config["retfound_weights_path"]):
                raise FileNotFoundError(f"Weights not found: {config['retfound_weights_path']}")
            checkpoint = torch.load(config["retfound_weights_path"], map_location='cpu')
            # ... loading logic ...
        self.d1 = DecoderBlock([config["hidden_dim"], config["hidden_dim"]], 512)
        self.d2 = DecoderBlock([512, config["hidden_dim"]], 256)
        self.d3 = DecoderBlock([256, config["hidden_dim"]], 128)
        self.d4 = DecoderBlock([128, config["hidden_dim"]], 64)
        self.output = nn.Conv2d(64, config["num_classes"], kernel_size=1)

    def forward(self, x):
        x = self.encoder.patch_embed(x)
        batch_size, num_patches, embed_dim = x.shape
        pos_embed = self.encoder.pos_embed[:,1:,:]
        if num_patches != pos_embed.shape[1]:
            pos_embed = interpolate_pos_embed(self.encoder, {'pos_embed': pos_embed})
        x = x + pos_embed
        x = self.encoder.pos_drop(x)
        skips = []
        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)
            if i in [5,11,17,23]:
                skips.append(x)
        z6, z12, z18, z24 = skips
        def to_feature(tensor):
            b, n, e = tensor.shape
            return tensor.transpose(1,2).reshape(b, e, int(e**0.5), int(e**0.5))
        z6, z12, z18, z24 = map(to_feature, [z6, z12, z18, z24])
        x = self.d1(z24, z18)
        x = self.d2(x, z12)
        x = self.d3(x, z6)
        x = self.d4(x, z6)
        return self.output(x)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs).view(-1)
        targets = targets.view(-1)
        tp = (outputs * targets).sum()
        fn = ((1-outputs)*targets).sum()
        fp = (outputs*(1-targets)).sum()
        tversky = (tp+self.smooth)/(tp+self.alpha*fn+self.beta*fp+self.smooth)
        return 1 - tversky

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth=smooth
    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs).view(-1)
        targets = targets.view(-1)
        inter = (outputs*targets).sum()
        dice = (2*inter+self.smooth)/(outputs.sum()+targets.sum()+self.smooth)
        return 1-dice


def dice_score(outputs, targets, smooth=1e-6):
    out_comb = torch.sigmoid(outputs).view(-1)
    tgt_comb = targets.view(-1)
    inter = (out_comb*tgt_comb).sum()
    dice_comb = (2*inter)/(out_comb.sum()+tgt_comb.sum())
    out_ch = torch.sigmoid(outputs[:,1]).view(-1)
    tgt_ch = targets[:,1].view(-1)
    inter_ch = (out_ch*tgt_ch).sum()
    dice_ch = (2*inter_ch+smooth)/(out_ch.sum()+tgt_ch.sum()+smooth)
    return dice_comb.item(), dice_ch.item()


def find_boundaries(mask):
    ups, lows = [], []
    for c in range(mask.shape[1]):
        idx = np.where(mask[:,c]>0)[0]
        if len(idx): ups.append(idx[0]); lows.append(idx[-1])
        else: ups.append(None); lows.append(None)
    return ups, lows


def compute_errors(pred, gt, pixel_size):
    s_err, u_err = [], []
    for p, g in zip(pred, gt):
        if p is None or g is None: continue
        err = (p-g)*pixel_size
        s_err.append(err); u_err.append(abs(err))
    if not s_err: return 0.0, 0.0
    return np.mean(s_err), np.mean(u_err)


def plot_boundaries(images, true_masks, pred_masks, threshold):
    batch = images.size(0)
    for i in range(batch):
        img = images[i].cpu().numpy().transpose(1,2,0)
        tm = true_masks[i,1].cpu().numpy()
        pm = pred_masks[i,1].cpu().numpy()
        pm_bin = (pm>threshold).astype(np.uint8)
        tm_bin = (tm>0.5).astype(np.uint8)
        pu, pl = find_boundaries(pm_bin)
        gu, gl = find_boundaries(tm_bin)
        us, uu = compute_errors(pu, gu, args.pixel_size_micrometers)
        ls, lu = compute_errors(pl, gl, args.pixel_size_micrometers)
        print(f"Image {i+1}: Upper Signed: {us:.2f} μm, Unsigned: {uu:.2f} μm; Lower Signed: {ls:.2f} μm, Unsigned: {lu:.2f} μm")
        comb = img.copy()
        for col in range(len(pu)):
            if gu[col] is not None: comb[gu[col],col]=[1,0,0]
            if gl[col] is not None: comb[gl[col],col]=[0,1,0]
            if pu[col] is not None: comb[pu[col],col]=[0,0,1]
            if pl[col] is not None: comb[pl[col],col]=[1,1,0]
        plt.figure(figsize=(16,8))
        for j,(dt,title) in enumerate(zip([img, tm_bin, pm_bin, comb],
                                         ['Original','True Mask','Pred Mask','Boundaries'])):
            plt.subplot(1,4,j+1); plt.imshow(dt); plt.title(title); plt.axis('off')
        plt.show()


class OCTDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_size=224, num_classes=2):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.img_tf    = image_transform
        self.mask_size = mask_size
        self.num_classes = num_classes
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.tif'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.rsplit('.',1)[0] + '.tif'

        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask  = Image.open(os.path.join(self.mask_dir,  mask_name))

        if self.img_tf:
            image = self.img_tf(image)

        # --- MASK: purely resize + nearest, then numpy mapping ---
        mask = mask.resize((self.mask_size, self.mask_size), resample=Image.NEAREST)
        mask_np = np.array(mask)
        mask_np = np.where(mask_np == 3,   0, mask_np)
        mask_np = np.where(mask_np == 249, 1, mask_np)
        mask_np = mask_np.astype(np.uint8)

        mask = torch.from_numpy(mask_np)
        mask = F.one_hot(mask, num_classes=self.num_classes)
        mask = mask.permute(2,0,1).float()

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
def train_fold(train_loader, valid_loader, test_loader, model, criterion, optimizer, device, num_epochs, scaler, threshold):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                # Check for nan in outputs
                if torch.isnan(outputs).any():
                    print("Warning: Model outputs contain nan values")
                    continue
                loss = criterion(outputs, masks)
                if torch.isnan(loss):
                    print("Warning: Loss is nan, skipping batch")
                    continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    dice_combined_scores, dice_choroid_scores = [], []
    upper_signed_errors, upper_unsigned_errors, lower_signed_errors, lower_unsigned_errors = [], [], [], []
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_combined, dice_choroid = dice_score(outputs, masks)  # Compute both Dice scores
            dice_combined_scores.append(dice_combined)
            dice_choroid_scores.append(dice_choroid)
            predicted_masks = torch.sigmoid(outputs).cpu().numpy()
            true_masks = masks.cpu().numpy()
            for i in range(images.size(0)):
                predicted_mask = predicted_masks[i, 1]  # Choroid channel
                true_mask = true_masks[i, 1]  # Choroid channel
               # print(f"Image {i+1} - True mask sum: {true_mask.sum()}, Predicted mask sum: {predicted_mask.sum()}")
               # print(f"Predicted mask max: {predicted_mask.max()}, min: {predicted_mask.min()}")
                predicted_mask_binary = (predicted_mask > threshold).astype(np.uint8)
                true_mask_binary = (true_mask > 0.5).astype(np.uint8)
                #print(f"Image {i+1} - True mask binary sum: {true_mask_binary.sum()}, Predicted mask binary sum: {predicted_mask_binary.sum()}")
                pred_upper, pred_lower = find_boundaries(predicted_mask_binary)
                gt_upper, gt_lower = find_boundaries(true_mask_binary)
                upper_signed, upper_unsigned = compute_errors(pred_upper, gt_upper, args.pixel_size_micrometers)
                lower_signed, lower_unsigned = compute_errors(pred_lower, gt_lower, args.pixel_size_micrometers)
                upper_signed_errors.append(upper_signed)
                upper_unsigned_errors.append(upper_unsigned)
                lower_signed_errors.append(lower_signed)
                lower_unsigned_errors.append(lower_unsigned)
    avg_dice_combined = np.mean(dice_combined_scores)
    avg_dice_choroid = np.mean(dice_choroid_scores)
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
            predicted_masks = outputs  # Already sigmoid applied, no thresholding here for visualization
            plot_boundaries(images, masks, predicted_masks, threshold)
            break  # Visualize only the first batch

    return avg_dice_combined, avg_dice_choroid, avg_upper_signed_error, avg_upper_unsigned_error, avg_lower_signed_error, avg_lower_unsigned_error

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNetViT(config).to(device)
    for i, blk in enumerate(model.encoder.blocks):
        if i < 23:
            for p in blk.parameters():
                p.requires_grad = False

    if args.test_only:
        assert args.test_image_dir and args.test_mask_dir, (
            '--test_only requires --test_image_dir and --test_mask_dir'
        )
        test_ds = OCTDataset(
            args.test_image_dir, args.test_mask_dir,
            transform=val_test_transform, num_classes=2
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size,
            shuffle=False, num_workers=2, pin_memory=True
        )
        cp = torch.load(config['retfound_weights_path'], map_location=device)
        model.load_state_dict(cp, strict=False)
        model.eval()
        all_dice, all_upper, all_lower = [], [], []
        with torch.no_grad():
            for imgs, msks in test_loader:
                imgs, msks = imgs.to(device), msks.to(device)
                outs = model(imgs)
                print(f"Output shape: {outs.shape}, Min: {outs.min().item()}, Max: {outs.max().item()}")
                print(f"Sigmoid output min/max: {torch.sigmoid(outs).min().item()}/{torch.sigmoid(outs).max().item()}")
                _, dch = dice_score(outs, msks)
                all_dice.append(dch)
                preds = torch.sigmoid(outs).cpu().numpy()
                gts = msks.cpu().numpy()
                for i in range(imgs.size(0)):
                    pm = (preds[i,1] > args.threshold).astype(np.uint8)
                    tm = (gts[i,1] > 0.5).astype(np.uint8)
                    pu, pl = find_boundaries(pm)
                    gu, gl = find_boundaries(tm)
                    us, uu = compute_errors(pu, gu, args.pixel_size_micrometers)
                    ls, lu = compute_errors(pl, gl, args.pixel_size_micrometers)
                    all_upper.append((us, uu))
                    all_lower.append((ls, lu))
        print(f"Choroid Dice on external data: {np.mean(all_dice):.4f}")
        usm = np.mean([u for u,_ in all_upper])
        uum = np.mean([u for _,u in all_upper])
        lsm = np.mean([l for l,_ in all_lower])
        lum = np.mean([l for _,l in all_lower])
        print(f"Upper signed/unsigned error: {usm:.2f}/{uum:.2f} μm")
        print(f"Lower signed/unsigned error: {lsm:.2f}/{lum:.2f} μm")
        sys.exit(0)

    # Training logic (if not test_only)
    assert args.image_dir and args.mask_dir, (
        '--image_dir and --mask_dir are required for training'
    )
    if args.weights_type in ['retfound', 'rfa-unet'] and os.path.exists(config["retfound_weights_path"]):
        checkpoint = torch.load(config["retfound_weights_path"], map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights from {config['retfound_weights_path']}")
    criterion = TverskyLoss(alpha=0.7, beta=0.3, smooth=1e-6).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = GradScaler('cuda')
    full_dataset = OCTDataset(args.image_dir, args.mask_dir, transform=val_test_transform, num_classes=2)
    train_size = int(0.7 * len(full_dataset))
    valid_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])
    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dice_combined, dice_choroid, upper_signed, upper_unsigned, lower_signed, lower_unsigned = train_fold(
        train_loader, valid_loader, test_loader, model, criterion, optimizer, device, args.num_epochs, scaler, args.threshold
    )
    print(f"Validation Dice (Combined): {dice_combined:.4f}, Validation Dice (Choroid): {dice_choroid:.4f}, "
          f"Upper Signed Error: {upper_signed:.2f} μm, Upper Unsigned Error: {upper_unsigned:.2f} μm, "
          f"Lower Signed Error: {lower_signed:.2f} μm, Lower Unsigned Error: {lower_unsigned:.2f} μm")
