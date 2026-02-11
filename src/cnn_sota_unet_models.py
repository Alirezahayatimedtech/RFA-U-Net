#!/usr/bin/env python3
"""
Unified CNN + SOTA U-Net runner with CLI flags.

This script consolidates the model/metric patterns from:
  - rfa/CNN/cnn-u-net-models.ipynb
  - rfa/SOTA/Copy_of_Pgkd.ipynb
  - rfa/SOTA/Deep_GPET.ipynb
  - rfa/SOTA/UNet_BEM.ipynb

It provides:
  - --mode train|eval|predict
  - --model-id for 7 CNN models + 3 SOTA families
  - notebook-aligned metric profiles and default losses
  - checkpoint save/load
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    import timm  # used by AttentionUNetViT
except Exception:
    timm = None


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


CNN_MODEL_IDS = [
    "cnn_attention_unet_vit",
    "cnn_inception_unet",
    "cnn_alexnet_unet",
    "cnn_unet_vgg16",
    "cnn_unet_vgg19",
    "cnn_unet_resnet50",
    "cnn_unet_resnet101",
]

SOTA_MODEL_IDS = [
    "sota_pgkd",
    "sota_deepgpet",
    "sota_unet_bem",
]

ALL_MODEL_IDS = CNN_MODEL_IDS + SOTA_MODEL_IDS


MODEL_DESCRIPTIONS = {
    "cnn_attention_unet_vit": "ViT-large encoder + attention decoder, class-1 Dice/Jaccard + column boundary errors.",
    "cnn_inception_unet": "Inception-U-Net with weighted CE defaults and class-1 Dice/Jaccard + column boundary errors.",
    "cnn_alexnet_unet": "AlexNet-U-Net with weighted CE defaults and class-1 Dice/Jaccard + column boundary errors.",
    "cnn_unet_vgg16": "VGG16-U-Net with weighted CE defaults and class-1 Dice/Jaccard + column boundary errors.",
    "cnn_unet_vgg19": "VGG19-U-Net with Dice-loss default and class-1 Dice/Jaccard + column boundary errors.",
    "cnn_unet_resnet50": "ResNet50-U-Net with weighted CE defaults and class-1 Dice/Jaccard + column boundary errors.",
    "cnn_unet_resnet101": "ResNet101-U-Net with weighted CE defaults and class-1 Dice/Jaccard + column boundary errors.",
    "sota_pgkd": "PGKD ResDeTransDoubleUnet (binary output) + Otsu Dice/IoU + column boundary errors by default.",
    "sota_deepgpet": "DeepGPET model wrapper (single-channel/binary) + fixed-threshold Dice/IoU + global boundary errors by default.",
    "sota_unet_bem": "UNetBEM (2-class output) + class-1 Dice/Jaccard + column boundary errors.",
}


PROFILE_BY_MODEL_ID = {
    "cnn_attention_unet_vit": "cnn_class1",
    "cnn_inception_unet": "cnn_class1",
    "cnn_alexnet_unet": "cnn_class1",
    "cnn_unet_vgg16": "cnn_class1",
    "cnn_unet_vgg19": "cnn_class1",
    "cnn_unet_resnet50": "cnn_class1",
    "cnn_unet_resnet101": "cnn_class1",
    "sota_pgkd": "sota_dynamic_otsu",
    "sota_deepgpet": "sota_fixed",
    "sota_unet_bem": "cnn_class1",
}


LOSS_AUTO_BY_MODEL_ID = {
    "cnn_attention_unet_vit": "tversky_multiclass",
    "cnn_inception_unet": "weighted_ce",
    "cnn_alexnet_unet": "weighted_ce",
    "cnn_unet_vgg16": "weighted_ce",
    "cnn_unet_vgg19": "dice_multiclass",
    "cnn_unet_resnet50": "weighted_ce",
    "cnn_unet_resnet101": "weighted_ce",
    "sota_pgkd": "bce_logits",
    "sota_deepgpet": "bce_logits",
    "sota_unet_bem": "weighted_ce",
}


AUGMENT_STYLE_CHOICES = {"none", "cnn_joint", "pgkd_flip", "bem_joint"}
IMAGE_MODE_CHOICES = {"rgb", "gray"}
METRIC_AREA_CHOICES = {"full", "gt_choroid_columns", "union_choroid_columns"}


@dataclass(frozen=True)
class NotebookPreprocessPreset:
    image_size: int
    image_mode: str
    normalize: bool
    augment_style: str


NOTEBOOK_PREPROCESS_BY_MODEL_ID = {
    "cnn_attention_unet_vit": NotebookPreprocessPreset(
        image_size=224,
        image_mode="rgb",
        normalize=False,
        augment_style="cnn_joint",
    ),
    "cnn_inception_unet": NotebookPreprocessPreset(
        image_size=224,
        image_mode="rgb",
        normalize=False,
        augment_style="cnn_joint",
    ),
    "cnn_alexnet_unet": NotebookPreprocessPreset(
        image_size=224,
        image_mode="rgb",
        normalize=False,
        augment_style="cnn_joint",
    ),
    "cnn_unet_vgg16": NotebookPreprocessPreset(
        image_size=224,
        image_mode="rgb",
        normalize=False,
        augment_style="cnn_joint",
    ),
    "cnn_unet_vgg19": NotebookPreprocessPreset(
        image_size=224,
        image_mode="rgb",
        normalize=False,
        augment_style="cnn_joint",
    ),
    "cnn_unet_resnet50": NotebookPreprocessPreset(
        image_size=224,
        image_mode="rgb",
        normalize=False,
        augment_style="cnn_joint",
    ),
    "cnn_unet_resnet101": NotebookPreprocessPreset(
        image_size=224,
        image_mode="rgb",
        normalize=False,
        augment_style="cnn_joint",
    ),
    "sota_pgkd": NotebookPreprocessPreset(
        image_size=224,
        image_mode="rgb",
        normalize=False,
        augment_style="pgkd_flip",
    ),
    "sota_deepgpet": NotebookPreprocessPreset(
        image_size=224,
        image_mode="gray",
        normalize=False,
        augment_style="none",
    ),
    "sota_unet_bem": NotebookPreprocessPreset(
        image_size=256,
        image_mode="rgb",
        normalize=False,
        augment_style="bem_joint",
    ),
}


def expected_input_channels(model_id: str) -> int:
    if model_id == "sota_deepgpet":
        return 1
    return 3


def augment_params_for_style(style: str) -> Dict[str, Any]:
    if style == "none":
        return {
            "random_resized_crop": False,
            "horizontal_flip_prob": 0.0,
            "vertical_flip_prob": 0.0,
            "rotation_degrees": 0.0,
            "color_jitter": False,
        }
    if style == "pgkd_flip":
        return {
            "random_resized_crop": False,
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.5,
            "rotation_degrees": 0.0,
            "color_jitter": False,
        }
    if style in {"cnn_joint", "bem_joint"}:
        return {
            "random_resized_crop": True,
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.0,
            "rotation_degrees": 15.0,
            "color_jitter": True,
        }
    raise ValueError(f"unsupported augment style: {style}")


def resolve_preprocess_config(args: argparse.Namespace) -> Dict[str, Any]:
    preset = NOTEBOOK_PREPROCESS_BY_MODEL_ID[args.model_id]

    if args.notebook_defaults:
        image_size = int(args.image_size) if args.image_size is not None else int(preset.image_size)
        normalize = bool(args.normalize) if args.normalize is not None else bool(preset.normalize)
        image_mode = args.image_mode if args.image_mode != "auto" else preset.image_mode
        augment_style = args.augment_style if args.augment_style != "auto" else preset.augment_style
    else:
        image_size = int(args.image_size) if args.image_size is not None else 224
        normalize = bool(args.normalize) if args.normalize is not None else True
        image_mode = args.image_mode if args.image_mode != "auto" else "rgb"
        augment_style = args.augment_style if args.augment_style != "auto" else "cnn_joint"

    if image_mode not in IMAGE_MODE_CHOICES:
        raise ValueError(f"unsupported image mode: {image_mode}")
    if augment_style not in AUGMENT_STYLE_CHOICES:
        raise ValueError(f"unsupported augment style: {augment_style}")

    channel_count = 1 if image_mode == "gray" else 3
    expected_channels = expected_input_channels(args.model_id)
    if channel_count != expected_channels:
        raise ValueError(
            f"model {args.model_id} expects {expected_channels}-channel input, "
            f"but image-mode={image_mode} provides {channel_count} channel(s)."
        )

    cfg = {
        "image_size": image_size,
        "normalize": normalize,
        "image_mode": image_mode,
        "augment_style": augment_style,
        "mask_threshold": float(args.mask_threshold),
    }
    cfg.update(augment_params_for_style(augment_style))
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_class_weights(raw: Optional[str], num_classes: int, device: torch.device) -> Optional[torch.Tensor]:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    values = [float(p) for p in parts]
    if len(values) == 1:
        values = values * num_classes
    if len(values) != num_classes:
        raise ValueError(f"class-weights requires {num_classes} values, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32, device=device)


def to_device(batch, device: torch.device):
    if isinstance(batch, (tuple, list)):
        return [to_device(x, device) for x in batch]
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


class RunningAverage:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@dataclass
class EvalResult:
    loss: float
    dice: float
    jaccard: float
    upper_signed_um: float
    upper_unsigned_um: float
    lower_signed_um: float
    lower_unsigned_um: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss": self.loss,
            "dice": self.dice,
            "jaccard": self.jaccard,
            "upper_signed_um": self.upper_signed_um,
            "upper_unsigned_um": self.upper_unsigned_um,
            "lower_signed_um": self.lower_signed_um,
            "lower_unsigned_um": self.lower_unsigned_um,
        }


class JointSegmentationTransform:
    """
    Apply synchronized image/mask transforms.
    """

    def __init__(
        self,
        image_size: int = 224,
        augment: bool = False,
        normalize: bool = True,
        binary_mask: bool = False,
        image_channels: int = 3,
        mask_threshold: float = 0.5,
        random_resized_crop: bool = False,
        horizontal_flip_prob: float = 0.0,
        vertical_flip_prob: float = 0.0,
        rotation_degrees: float = 0.0,
        color_jitter: bool = False,
    ) -> None:
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.normalize = bool(normalize)
        self.binary_mask = bool(binary_mask)
        self.image_channels = int(image_channels)
        self.mask_threshold = float(mask_threshold)
        self.random_resized_crop = bool(random_resized_crop)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.vertical_flip_prob = float(vertical_flip_prob)
        self.rotation_degrees = float(rotation_degrees)
        self.color_jitter_enabled = bool(color_jitter)
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

        if self.image_channels == 1:
            self.mean = [0.5]
            self.std = [0.5]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image, mask: Optional[Image.Image] = None):
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        if mask is not None:
            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )

        if self.augment and mask is not None:
            if self.random_resized_crop:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image,
                    scale=(0.8, 1.0),
                    ratio=(1.0, 1.0),
                )
                image = TF.resized_crop(
                    image,
                    i,
                    j,
                    h,
                    w,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                )
                mask = TF.resized_crop(
                    mask,
                    i,
                    j,
                    h,
                    w,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.NEAREST,
                )

            if self.horizontal_flip_prob > 0 and random.random() < self.horizontal_flip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if self.vertical_flip_prob > 0 and random.random() < self.vertical_flip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if self.rotation_degrees > 0:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

            if self.color_jitter_enabled:
                image = self.color_jitter(image)

        image_t = TF.to_tensor(image)
        if self.normalize:
            image_t = TF.normalize(image_t, mean=self.mean, std=self.std)

        if mask is None:
            return image_t

        mask_t = TF.to_tensor(mask)  # [1,H,W]
        if self.binary_mask:
            mask_t = (mask_t > self.mask_threshold).float()
        else:
            mask_t = (mask_t > self.mask_threshold).long().squeeze(0)  # [H,W]
        return image_t, mask_t


class PairSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: JointSegmentationTransform,
        binary_mask: bool,
        image_mode: str = "rgb",
        allow_all_empty_masks: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.binary_mask = bool(binary_mask)
        self.image_mode = image_mode.lower().strip()
        if self.image_mode not in IMAGE_MODE_CHOICES:
            raise ValueError(f"unsupported image-mode: {image_mode}")

        if not self.image_dir.exists():
            raise FileNotFoundError(f"image-dir not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"mask-dir not found: {self.mask_dir}")

        files = [
            p
            for p in sorted(self.image_dir.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
        if not files:
            raise RuntimeError(f"no image files found in {self.image_dir}")

        pairs = []
        missing = []
        for image_path in files:
            mask_path = self.mask_dir / f"{image_path.stem}_mask.png"
            if mask_path.exists():
                pairs.append((image_path, mask_path))
            else:
                missing.append(mask_path.name)

        if not pairs:
            raise RuntimeError(
                f"no image/mask pairs found. expected masks like <name>_mask.png in {self.mask_dir}"
            )
        if missing:
            print(
                f"[WARN] {len(missing)} images missing masks; first examples: {missing[:5]}",
                file=sys.stderr,
            )

        self.pairs = pairs
        non_empty_masks = 0
        for _, mask_path in self.pairs:
            mask = Image.open(mask_path).convert("L")
            if mask.getbbox() is not None:
                non_empty_masks += 1
        if non_empty_masks == 0:
            message = (
                "all masks are empty in dataset "
                f"{self.mask_dir}. Check mask generation/split or use --allow-empty-masks to bypass."
            )
            if allow_all_empty_masks:
                print(f"[WARN] {message}", file=sys.stderr)
            else:
                raise RuntimeError(message)
        elif non_empty_masks < len(self.pairs):
            print(
                f"[WARN] sparse masks detected: {non_empty_masks}/{len(self.pairs)} non-empty in {self.mask_dir}",
                file=sys.stderr,
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]
        pil_mode = "L" if self.image_mode == "gray" else "RGB"
        image = Image.open(image_path).convert(pil_mode)
        mask = Image.open(mask_path).convert("L")
        image_t, mask_t = self.transform(image, mask)
        return image_t, mask_t, image_path.name


class PredictImageDataset(Dataset):
    def __init__(self, image_dir: str, transform: JointSegmentationTransform, image_mode: str = "rgb") -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_mode = image_mode.lower().strip()
        if self.image_mode not in IMAGE_MODE_CHOICES:
            raise ValueError(f"unsupported image-mode: {image_mode}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"predict image-dir not found: {self.image_dir}")
        self.files = [
            p
            for p in sorted(self.image_dir.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
        if not self.files:
            raise RuntimeError(f"no image files found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        pil_mode = "L" if self.image_mode == "gray" else "RGB"
        image = Image.open(path).convert(pil_mode)
        original_size = image.size  # (W,H)
        image_t = self.transform(image, None)
        return image_t, path.name, original_size


def _safe_torchvision_weights(model_fn, weights_enum, pretrained: bool):
    if not pretrained:
        return model_fn(weights=None)
    try:
        return model_fn(weights=weights_enum.DEFAULT)
    except Exception as exc:
        print(
            f"[WARN] failed to load pretrained weights for {model_fn.__name__}: {exc}. Using random init.",
            file=sys.stderr,
        )
        return model_fn(weights=None)


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, in_c: Sequence[int], out_c: int) -> None:
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        g_proj = self.Wg(g)
        s_proj = self.Ws(s)
        if g_proj.shape[-2:] != s_proj.shape[-2:]:
            s_proj = F.interpolate(s_proj, size=g_proj.shape[-2:], mode="bilinear", align_corners=True)
        x = self.relu(g_proj + s_proj)
        a = self.out(x)
        if a.shape[-2:] != s.shape[-2:]:
            s = F.interpolate(s, size=a.shape[-2:], mode="bilinear", align_corners=True)
        return a * s


class DecoderBlock(nn.Module):
    def __init__(self, in_c: Sequence[int], out_c: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.reduce_x = nn.Conv2d(in_c[0], out_c, kernel_size=1)
        self.reduce_s = nn.Conv2d(in_c[1], out_c, kernel_size=1)
        self.ag = AttentionGate([out_c, out_c], out_c)
        self.c1 = ConvBlock(out_c + out_c, out_c)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce_x(x)
        s = self.reduce_s(s)
        s = self.ag(x, s)
        if x.shape[-2:] != s.shape[-2:]:
            s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, s], dim=1)
        return self.c1(x)


class AttentionUNetViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_encoder_blocks: int = 0,
        model_name: str = "vit_large_patch16_224",
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for cnn_attention_unet_vit")
        try:
            self.encoder = timm.create_model(model_name, pretrained=pretrained)
        except Exception as exc:
            if pretrained:
                print(
                    f"[WARN] failed to load pretrained {model_name}: {exc}. Using random init.",
                    file=sys.stderr,
                )
                self.encoder = timm.create_model(model_name, pretrained=False)
            else:
                raise

        if not hasattr(self.encoder, "blocks"):
            raise RuntimeError(f"model {model_name} does not expose transformer blocks")

        self.encoder.head = nn.Identity()
        for i in range(max(0, freeze_encoder_blocks)):
            if i >= len(self.encoder.blocks):
                break
            for p in self.encoder.blocks[i].parameters():
                p.requires_grad = False

        embed_dim = getattr(self.encoder, "num_features", 1024)
        if embed_dim != 1024:
            print(
                f"[WARN] attention decoder expects 1024 channels, got {embed_dim}. Decoder may fail.",
                file=sys.stderr,
            )

        self.d1 = DecoderBlock([1024, 1024], 512)
        self.d2 = DecoderBlock([512, 1024], 256)
        self.d3 = DecoderBlock([256, 1024], 128)
        self.d4 = DecoderBlock([128, 1024], 64)
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

    def _pos_embed_for_tokens(self, token_count: int) -> torch.Tensor:
        pos_embed = self.encoder.pos_embed
        if token_count == pos_embed.shape[1]:
            return pos_embed
        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]
        gs_old = int(math.sqrt(patch_pos.shape[1]))
        gs_new = int(math.sqrt(token_count - 1))
        if gs_old * gs_old != patch_pos.shape[1] or gs_new * gs_new != token_count - 1:
            raise RuntimeError("non-square patch grid is not supported for AttentionUNetViT")
        patch_pos = patch_pos.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(gs_new, gs_new), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.encoder.patch_embed(x)  # [B,N,C]
        cls_token = self.encoder.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self._pos_embed_for_tokens(x.shape[1]).to(x.device)
        x = self.encoder.pos_drop(x)

        skips = []
        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)
            if i in [5, 11, 17, 23]:
                skips.append(x[:, 1:, :])
        if len(skips) < 4:
            raise RuntimeError("expected 4 transformer skip tensors from blocks [5,11,17,23]")

        patch_tokens = x[:, 1:, :]
        n = patch_tokens.shape[1]
        hw = int(math.sqrt(n))
        if hw * hw != n:
            raise RuntimeError("patch token count is not a square number")

        x = patch_tokens.transpose(1, 2).contiguous().view(b, 1024, hw, hw)
        skips = [s.transpose(1, 2).contiguous().view(b, 1024, hw, hw) for s in skips]
        z6, z12, z18, z24 = skips

        x = self.d1(x, z24)
        x = self.d2(x, z18)
        x = self.d3(x, z12)
        x = self.d4(x, z6)
        return self.output(x)


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        return torch.cat([b1, b3, b5, bp], dim=1)


class InceptionUNet(nn.Module):
    def __init__(self, num_classes: int = 2, input_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = InceptionModule(input_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = InceptionModule(32 * 4, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = InceptionModule(64 * 4, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = InceptionModule(128 * 4, 256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = InceptionModule(256 * 4, 512)

        self.up4 = nn.ConvTranspose2d(512 * 4, 256 * 4, kernel_size=2, stride=2)
        self.dec4 = InceptionModule((256 * 4) + (256 * 4), 256)

        self.up3 = nn.ConvTranspose2d(256 * 4, 128 * 4, kernel_size=2, stride=2)
        self.dec3 = InceptionModule((128 * 4) + (128 * 4), 128)

        self.up2 = nn.ConvTranspose2d(128 * 4, 64 * 4, kernel_size=2, stride=2)
        self.dec2 = InceptionModule((64 * 4) + (64 * 4), 64)

        self.up1 = nn.ConvTranspose2d(64 * 4, 32 * 4, kernel_size=2, stride=2)
        self.dec1 = InceptionModule((32 * 4) + (32 * 4), 32)

        self.final_conv = nn.Conv2d(32 * 4, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        bn = self.bottleneck(p4)

        up4 = self.up4(bn)
        d4 = self.dec4(torch.cat([up4, e4], dim=1))

        up3 = self.up3(d4)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))

        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        return self.final_conv(d1)


class _BaseUNetDecoder(nn.Module):
    def __init__(self, dropout: bool = False) -> None:
        super().__init__()
        self.dropout = dropout

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if self.dropout:
            layers.append(nn.Dropout(0.5))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        if self.dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    @staticmethod
    def up_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    @staticmethod
    def _resize_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=True)

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class VGGUNet(_BaseUNetDecoder):
    def __init__(self, variant: str, num_classes: int = 2, pretrained: bool = True, dropout: bool = False) -> None:
        super().__init__(dropout=dropout)
        variant = variant.lower().strip()
        if variant == "vgg16":
            vgg = _safe_torchvision_weights(models.vgg16, models.VGG16_Weights, pretrained)
            features = list(vgg.features.children())
            slices = (0, 5, 10, 17, 24, 31)
        elif variant == "vgg19":
            vgg = _safe_torchvision_weights(models.vgg19, models.VGG19_Weights, pretrained)
            features = list(vgg.features.children())
            slices = (0, 2, 7, 12, 21, 30)
        else:
            raise ValueError(f"unsupported VGG variant: {variant}")

        s0, s1, s2, s3, s4, s5 = slices
        self.encoder1 = nn.Sequential(*features[s0:s1])
        self.encoder2 = nn.Sequential(*features[s1:s2])
        self.encoder3 = nn.Sequential(*features[s2:s3])
        self.encoder4 = nn.Sequential(*features[s3:s4])
        self.encoder5 = nn.Sequential(*features[s4:s5])

        self.up5 = self.up_conv(512, 512)
        self.dec5 = self.conv_block(512 + 512, 512)

        self.up4 = self.up_conv(512, 256)
        self.dec4 = self.conv_block(256 + 256, 256)

        self.up3 = self.up_conv(256, 128)
        self.dec3 = self.conv_block(128 + 128, 128)

        self.up2 = self.up_conv(128, 64)
        self.dec2 = self.conv_block(64 + 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d5 = self.up5(e5)
        d5 = torch.cat((self._resize_to(d5, e4), e4), dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((self._resize_to(d4, e3), e3), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((self._resize_to(d3, e2), e2), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((self._resize_to(d2, e1), e1), dim=1)
        d2 = self.dec2(d2)

        out = self.final_conv(d2)
        if out.shape[-2:] != input_hw:
            out = F.interpolate(out, size=input_hw, mode="bilinear", align_corners=True)
        return out


class AlexNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        alexnet = _safe_torchvision_weights(models.alexnet, models.AlexNet_Weights, pretrained)
        features = alexnet.features
        self.enc1 = nn.Sequential(*features[0:3])
        self.enc2 = nn.Sequential(*features[3:6])
        self.enc3 = nn.Sequential(*features[6:8])
        self.enc4 = nn.Sequential(*features[8:10])
        self.enc5 = nn.Sequential(*features[10:12])

    def forward(self, x: torch.Tensor):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        return x1, x2, x3, x4, x5


class AlexNetUNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = AlexNetEncoder(pretrained=pretrained)
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = self.conv_block(256 + 256, 256)

        self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(256 + 64, 256)

        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    @staticmethod
    def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, _, _, _, x5 = self.encoder(x)
        x_bn = self.bottleneck(x5)
        y = self.upconv5(x_bn)
        if y.shape[-2:] != x5.shape[-2:]:
            x5 = F.interpolate(x5, size=y.shape[-2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, x5], dim=1)
        y = self.dec5(y)

        y = self.upconv4(y)
        if y.shape[-2:] != x1.shape[-2:]:
            x1 = F.interpolate(x1, size=y.shape[-2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, x1], dim=1)
        y = self.dec4(y)

        y = self.final_conv(y)
        return F.interpolate(y, size=(224, 224), mode="bilinear", align_corners=False)


class ResNetUNet(_BaseUNetDecoder):
    def __init__(
        self,
        backbone: str,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: bool = False,
    ) -> None:
        super().__init__(dropout=dropout)
        backbone = backbone.lower().strip()
        if backbone == "resnet50":
            resnet = _safe_torchvision_weights(models.resnet50, models.ResNet50_Weights, pretrained)
        elif backbone == "resnet101":
            resnet = _safe_torchvision_weights(models.resnet101, models.ResNet101_Weights, pretrained)
        else:
            raise ValueError(f"unsupported backbone: {backbone}")

        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        self.up5 = self.up_conv(2048, 1024)
        self.dec5 = self.conv_block(2048, 1024)

        self.up4 = self.up_conv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)

        self.up3 = self.up_conv(512, 256)
        self.dec3 = self.conv_block(512, 256)

        self.up2 = self.up_conv(256, 64)
        self.dec2 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d5 = self.up5(e5)
        d5 = torch.cat((self._resize_to(d5, e4), e4), dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((self._resize_to(d4, e3), e3), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((self._resize_to(d3, e2), e2), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((self._resize_to(d2, e1), e1), dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, size=input_hw, mode="bilinear", align_corners=True)
        return self.final_conv(d1)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: int = -100) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(outputs, targets)


class TverskyLossMultiClass(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, class_idx: int = 1, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_idx = class_idx
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(outputs, dim=1)[:, self.class_idx, :, :]
        target = (targets == self.class_idx).float()
        probs = probs.contiguous().view(-1)
        target = target.contiguous().view(-1)

        tp = (probs * target).sum()
        fn = ((1 - probs) * target).sum()
        fp = (probs * (1 - target)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky


class DiceLossClass1(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(outputs, dim=1)[:, 1, :, :]
        target = (targets == 1).float()
        probs = probs.reshape(-1)
        target = target.reshape(-1)
        intersection = (probs * target).sum()
        union = probs.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class TverskyLossBinary(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        if probs.shape[-2:] != targets.shape[-2:]:
            targets = F.interpolate(targets, size=probs.shape[-2:], mode="nearest")
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        tp = (probs * targets).sum()
        fp = ((1 - targets) * probs).sum()
        fn = (targets * (1 - probs)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky


def extract_logits(model_output, output_index: Optional[int] = None) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)):
        if not model_output:
            raise RuntimeError("model returned empty tuple/list")
        if output_index is not None:
            idx = output_index
            if idx < 0:
                idx = len(model_output) + idx
            if idx < 0 or idx >= len(model_output):
                raise IndexError(f"model-output-index {output_index} out of range for length {len(model_output)}")
            out = model_output[idx]
            if not torch.is_tensor(out):
                raise TypeError(f"selected model output at index {output_index} is not a tensor")
            return out
        if len(model_output) >= 2 and torch.is_tensor(model_output[1]):
            return model_output[1]
        if torch.is_tensor(model_output[0]):
            return model_output[0]
    raise TypeError(f"unsupported model output type: {type(model_output)}")


def resolve_model_output_index(model_id: str, output_index_arg: Optional[int]) -> Optional[int]:
    # Some SOTA implementations return tuples where the primary segmentation logits are at index 0.
    if output_index_arg is not None:
        return int(output_index_arg)
    if model_id == "sota_unet_bem":
        return 0
    return None


def align_targets_to_logits(logits: torch.Tensor, targets: torch.Tensor, binary_task: bool) -> torch.Tensor:
    if binary_task:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        if targets.shape[-2:] != logits.shape[-2:]:
            targets = F.interpolate(targets, size=logits.shape[-2:], mode="nearest")
        return targets
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    targets = targets.long()
    if targets.shape[-2:] != logits.shape[-2:]:
        targets = F.interpolate(targets.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()
    return targets


def dice_score_class_multiclass(outputs: torch.Tensor, targets: torch.Tensor, class_idx: int = 1, eps: float = 1e-6) -> float:
    probs = torch.softmax(outputs, dim=1)[:, class_idx, :, :]
    class_targets = (targets == class_idx).float()
    probs = probs.view(probs.size(0), -1)
    class_targets = class_targets.view(class_targets.size(0), -1)
    intersection = (probs * class_targets).sum(1)
    dice = (2.0 * intersection + eps) / (probs.sum(1) + class_targets.sum(1) + eps)
    return float(dice.mean().item())


def jaccard_score_class_multiclass(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_idx: int = 1,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    probs = torch.softmax(outputs, dim=1)[:, class_idx, :, :]
    class_targets = (targets == class_idx).float()
    preds = (probs > threshold).float()
    intersection = (preds * class_targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + class_targets.sum(dim=(1, 2)) - intersection
    jaccard = (intersection + eps) / (union + eps)
    return float(jaccard.mean().item())


def dice_score_binary(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> float:
    if pred_bin.dim() == 4:
        pred_bin = pred_bin.squeeze(1)
    if target_bin.dim() == 4:
        target_bin = target_bin.squeeze(1)
    pred_bin = pred_bin.float().reshape(pred_bin.size(0), -1)
    target_bin = target_bin.float().reshape(target_bin.size(0), -1)
    intersection = (pred_bin * target_bin).sum(dim=1)
    union = pred_bin.sum(dim=1) + target_bin.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.mean().item())


def jaccard_score_binary(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> float:
    if pred_bin.dim() == 4:
        pred_bin = pred_bin.squeeze(1)
    if target_bin.dim() == 4:
        target_bin = target_bin.squeeze(1)
    pred_bin = pred_bin.float().reshape(pred_bin.size(0), -1)
    target_bin = target_bin.float().reshape(target_bin.size(0), -1)
    intersection = (pred_bin * target_bin).sum(dim=1)
    union = pred_bin.sum(dim=1) + target_bin.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())


def build_column_envelope_mask(bin_2d: np.ndarray) -> np.ndarray:
    """
    Build a column-wise choroid envelope from upper/lower boundaries.
    For each column with positives, fill between upper/lower boundaries.
    """
    bin_2d = (bin_2d > 0).astype(np.uint8)
    roi = np.zeros_like(bin_2d, dtype=np.uint8)
    upper, lower = find_boundaries_columnwise(bin_2d)
    for col, (u, l) in enumerate(zip(upper, lower)):
        if u is None or l is None:
            continue
        roi[int(u) : int(l) + 1, col] = 1
    return roi


def apply_metric_area_binary(
    pred_bin: torch.Tensor,
    target_bin: torch.Tensor,
    metric_area: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if metric_area == "full":
        return pred_bin.float(), target_bin.float()
    if metric_area not in {"gt_choroid_columns", "union_choroid_columns"}:
        raise ValueError(f"unsupported metric-area: {metric_area}")

    pred_4d = pred_bin.dim() == 4
    target_4d = target_bin.dim() == 4
    pred_np = pred_bin.squeeze(1).detach().cpu().numpy().astype(np.uint8) if pred_4d else pred_bin.detach().cpu().numpy().astype(np.uint8)
    gt_np = target_bin.squeeze(1).detach().cpu().numpy().astype(np.uint8) if target_4d else target_bin.detach().cpu().numpy().astype(np.uint8)

    masked_pred = []
    masked_gt = []
    for i in range(pred_np.shape[0]):
        gt_roi = build_column_envelope_mask(gt_np[i])
        if metric_area == "gt_choroid_columns":
            roi = gt_roi
        else:
            pred_roi = build_column_envelope_mask(pred_np[i])
            roi = np.logical_or(gt_roi > 0, pred_roi > 0).astype(np.uint8)
        if not np.any(roi):
            # Fallback for empty GT masks: keep full image to avoid artificial 1.0 scores.
            roi = np.ones_like(gt_np[i], dtype=np.uint8)
        masked_pred.append((pred_np[i] * roi).astype(np.float32))
        masked_gt.append((gt_np[i] * roi).astype(np.float32))

    pred_out = torch.from_numpy(np.stack(masked_pred, axis=0)).to(pred_bin.device)
    gt_out = torch.from_numpy(np.stack(masked_gt, axis=0)).to(target_bin.device)
    if pred_4d:
        pred_out = pred_out.unsqueeze(1)
    if target_4d:
        gt_out = gt_out.unsqueeze(1)
    return pred_out, gt_out


def find_boundaries_columnwise(mask: np.ndarray) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    upper: List[Optional[int]] = []
    lower: List[Optional[int]] = []
    for col in range(mask.shape[1]):
        nz = np.where(mask[:, col] > 0)[0]
        if len(nz) > 0:
            upper.append(int(nz[0]))
            lower.append(int(nz[-1]))
        else:
            upper.append(None)
            lower.append(None)
    return upper, lower


def compute_errors_columnwise(
    pred_boundaries: Sequence[Optional[int]],
    gt_boundaries: Sequence[Optional[int]],
    pixel_size: float,
) -> Tuple[float, float]:
    signed_errors = []
    unsigned_errors = []
    for pred, gt in zip(pred_boundaries, gt_boundaries):
        if pred is None or gt is None:
            continue
        signed = (float(pred) - float(gt)) * float(pixel_size)
        signed_errors.append(signed)
        unsigned_errors.append(abs(signed))
    if not signed_errors:
        return 0.0, 0.0
    return float(np.mean(signed_errors)), float(np.mean(unsigned_errors))


def find_boundaries_global(mask: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    rows = np.any(mask > 0, axis=1)
    if not np.any(rows):
        return None, None
    upper = int(np.argmax(rows))
    lower = int(len(rows) - 1 - np.argmax(rows[::-1]))
    return upper, lower


def compute_errors_global(
    pred_boundary: Optional[int],
    gt_boundary: Optional[int],
    pixel_size: float,
) -> Tuple[float, float]:
    if pred_boundary is None or gt_boundary is None:
        return 0.0, 0.0
    signed = (float(pred_boundary) - float(gt_boundary)) * float(pixel_size)
    return signed, abs(signed)


def otsu_threshold(prob_map: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Numpy-only Otsu thresholding to avoid external dependency.
    """
    prob_map = np.asarray(prob_map, dtype=np.float32)
    if prob_map.size == 0:
        return np.zeros_like(prob_map, dtype=np.uint8), 0.5
    if np.allclose(prob_map.max(), prob_map.min()):
        threshold = float(prob_map.min())
        return (prob_map > threshold).astype(np.uint8), threshold

    hist, _ = np.histogram(prob_map.ravel(), bins=256, range=(0.0, 1.0))
    total = prob_map.size
    sum_total = float(np.dot(np.arange(256), hist))

    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold_bin = 127

    for t in range(256):
        w_b += hist[t]
        if w_b <= 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold_bin = t

    threshold = float(threshold_bin) / 255.0
    return (prob_map > threshold).astype(np.uint8), threshold


def infer_metric_profile(model_id: str, profile_arg: str) -> str:
    if profile_arg != "auto":
        return profile_arg
    return PROFILE_BY_MODEL_ID[model_id]


def infer_loss_name(model_id: str, loss_arg: str) -> str:
    if loss_arg != "auto":
        return loss_arg
    return LOSS_AUTO_BY_MODEL_ID[model_id]


def is_binary_model(model_id: str) -> bool:
    return model_id in {"sota_pgkd", "sota_deepgpet"}


def create_pgkd_model(image_size: int):
    try:
        from PGKD_Net.res2deform import ResDeTransDoubleUnet
    except Exception as exc:
        raise ImportError(
            "SOTA PGKD models require `PGKD_Net.res2deform.ResDeTransDoubleUnet` "
            "to be importable. Set PYTHONPATH to the PGKD repo before running."
        ) from exc
    return ResDeTransDoubleUnet(3, 1, image_size)


def create_deepgpet_model():
    """
    Expected notebook usage:
      from choseg import inference
      deepgpet = inference.DeepGPET(...)
      model = deepgpet.model
    """
    try:
        from choseg import inference  # type: ignore
    except Exception as exc:
        raise ImportError(
            "SOTA DeepGPET requires `choseg.inference.DeepGPET` "
            "to be importable. Install/package the DeepGPET runtime and set PYTHONPATH."
        ) from exc

    deepgpet = inference.DeepGPET()
    if not hasattr(deepgpet, "model"):
        raise RuntimeError("DeepGPET() does not expose `.model`")
    return deepgpet.model


def create_unet_bem_model(num_classes: int):
    try:
        from UNetBEM import UNetBEM  # type: ignore
    except Exception as exc:
        raise ImportError(
            "SOTA UNet_BEM requires `UNetBEM.UNetBEM` to be importable. "
            "Place UNetBEM.py in PYTHONPATH or install it as a package."
        ) from exc
    return UNetBEM(n_channels=3, n_classes=num_classes)


def create_model(
    model_id: str,
    num_classes: int,
    pretrained: bool,
    image_size: int,
    freeze_encoder_blocks: int,
) -> nn.Module:
    if model_id == "cnn_attention_unet_vit":
        return AttentionUNetViT(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_encoder_blocks=freeze_encoder_blocks,
            model_name="vit_large_patch16_224",
        )
    if model_id == "cnn_inception_unet":
        return InceptionUNet(num_classes=num_classes, input_channels=3)
    if model_id == "cnn_alexnet_unet":
        return AlexNetUNet(num_classes=num_classes, pretrained=pretrained)
    if model_id == "cnn_unet_vgg16":
        return VGGUNet("vgg16", num_classes=num_classes, pretrained=pretrained, dropout=False)
    if model_id == "cnn_unet_vgg19":
        return VGGUNet("vgg19", num_classes=num_classes, pretrained=pretrained, dropout=False)
    if model_id == "cnn_unet_resnet50":
        return ResNetUNet("resnet50", num_classes=num_classes, pretrained=pretrained, dropout=False)
    if model_id == "cnn_unet_resnet101":
        return ResNetUNet("resnet101", num_classes=num_classes, pretrained=pretrained, dropout=False)
    if model_id == "sota_pgkd":
        return create_pgkd_model(image_size=image_size)
    if model_id == "sota_deepgpet":
        return create_deepgpet_model()
    if model_id == "sota_unet_bem":
        return create_unet_bem_model(num_classes=num_classes)
    raise ValueError(f"unsupported model-id: {model_id}")


def build_criterion(
    loss_name: str,
    model_id: str,
    num_classes: int,
    class_weights: Optional[torch.Tensor],
) -> nn.Module:
    if loss_name == "weighted_ce":
        return WeightedCrossEntropyLoss(weight=class_weights, ignore_index=-100)
    if loss_name == "tversky_multiclass":
        return TverskyLossMultiClass(alpha=0.7, beta=0.3, class_idx=1)
    if loss_name == "dice_multiclass":
        return DiceLossClass1()
    if loss_name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    if loss_name == "tversky_binary":
        return TverskyLossBinary(alpha=0.7, beta=0.3, smooth=1e-6)
    raise ValueError(f"unsupported loss: {loss_name}")


def maybe_freeze_layers(model: nn.Module, model_id: str, freeze_layers: int) -> Tuple[int, int]:
    if freeze_layers <= 0:
        return 0, 0

    groups: List[nn.Module] = []
    if model_id == "cnn_alexnet_unet":
        encoder = getattr(model, "encoder", None)
        if encoder is not None:
            groups = [
                getattr(encoder, "enc1", None),
                getattr(encoder, "enc2", None),
                getattr(encoder, "enc3", None),
                getattr(encoder, "enc4", None),
                getattr(encoder, "enc5", None),
            ]
    elif model_id == "cnn_inception_unet":
        groups = [
            getattr(model, "enc1", None),
            getattr(model, "enc2", None),
            getattr(model, "enc3", None),
            getattr(model, "enc4", None),
            getattr(model, "bottleneck", None),
        ]
    elif model_id in {"cnn_unet_vgg16", "cnn_unet_vgg19", "cnn_unet_resnet50", "cnn_unet_resnet101"}:
        groups = [
            getattr(model, "encoder1", None),
            getattr(model, "encoder2", None),
            getattr(model, "encoder3", None),
            getattr(model, "encoder4", None),
            getattr(model, "encoder5", None),
        ]

    groups = [g for g in groups if isinstance(g, nn.Module)]
    total_groups = len(groups)
    if total_groups == 0:
        return 0, 0

    n = min(int(freeze_layers), total_groups)
    for module in groups[:n]:
        for p in module.parameters():
            p.requires_grad = False
    return n, total_groups


def scheduler_metric_from_result(result: EvalResult, key: str) -> float:
    if key == "loss":
        return float(result.loss)
    if key == "jaccard":
        return float(result.jaccard)
    if key == "dice":
        return float(result.dice)
    raise ValueError(f"unsupported scheduler-monitor: {key}")


def create_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.step_size),
            gamma=float(args.step_gamma),
        )
    if args.scheduler == "plateau":
        scheduler_monitor = args.monitor if args.scheduler_monitor == "auto" else args.scheduler_monitor
        mode = "min" if scheduler_monitor == "loss" else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
        )
    raise ValueError(f"unsupported scheduler: {args.scheduler}")


def make_prediction_binary(
    logits: torch.Tensor,
    profile: str,
    fixed_threshold: float,
) -> Tuple[torch.Tensor, List[float]]:
    probs = torch.sigmoid(logits)
    if profile == "sota_dynamic_mean":
        thr = float(probs.mean().item())
        return (probs > thr).float(), [thr] * probs.shape[0]
    if profile == "sota_dynamic_otsu":
        pred = []
        thrs = []
        probs_np = probs.detach().cpu().numpy()
        for i in range(probs_np.shape[0]):
            mask_np, thr = otsu_threshold(probs_np[i, 0])
            pred.append(mask_np.astype(np.float32))
            thrs.append(float(thr))
        pred_t = torch.from_numpy(np.stack(pred, axis=0)).unsqueeze(1).to(logits.device)
        return pred_t, thrs
    return (probs > fixed_threshold).float(), [float(fixed_threshold)] * probs.shape[0]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    binary_task: bool,
    metric_profile: str,
    threshold: float,
    pixel_size_micrometers: float,
    model_output_index: Optional[int],
    metric_area: str = "full",
    save_predictions_dir: Optional[Path] = None,
) -> EvalResult:
    model.eval()
    loss_meter = RunningAverage()
    dice_meter = RunningAverage()
    jaccard_meter = RunningAverage()
    up_signed_meter = RunningAverage()
    up_unsigned_meter = RunningAverage()
    low_signed_meter = RunningAverage()
    low_unsigned_meter = RunningAverage()

    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, masks, filenames in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            logits = extract_logits(outputs, output_index=model_output_index)
            aligned_targets = align_targets_to_logits(logits, masks, binary_task=binary_task)
            loss = criterion(logits, aligned_targets)

            bsz = int(images.size(0))
            loss_meter.update(float(loss.item()), n=bsz)

            if metric_profile == "cnn_class1":
                targets_long = aligned_targets.long()
                # Keep Dice/Jaccard on the same hard predictions for consistent reporting.
                pred_class_t = torch.argmax(logits, dim=1)
                pred_bin_t = (pred_class_t == 1).float()
                gt_bin_t = (targets_long == 1).float()
                pred_metric_t, gt_metric_t = apply_metric_area_binary(pred_bin_t, gt_bin_t, metric_area=metric_area)
                dice = dice_score_binary(pred_metric_t, gt_metric_t)
                jac = jaccard_score_binary(pred_metric_t, gt_metric_t)
                dice_meter.update(dice, n=bsz)
                jaccard_meter.update(jac, n=bsz)

                pred_class = pred_class_t.detach().cpu().numpy()
                gt_class = targets_long.detach().cpu().numpy()
                for i in range(pred_class.shape[0]):
                    pred_bin = (pred_class[i] == 1).astype(np.uint8)
                    gt_bin = (gt_class[i] == 1).astype(np.uint8)
                    pu, pl = find_boundaries_columnwise(pred_bin)
                    gu, gl = find_boundaries_columnwise(gt_bin)
                    us, uu = compute_errors_columnwise(pu, gu, pixel_size=pixel_size_micrometers)
                    ls, lu = compute_errors_columnwise(pl, gl, pixel_size=pixel_size_micrometers)
                    up_signed_meter.update(us)
                    up_unsigned_meter.update(uu)
                    low_signed_meter.update(ls)
                    low_unsigned_meter.update(lu)

                    if save_predictions_dir is not None:
                        out_path = save_predictions_dir / f"{Path(filenames[i]).stem}_mask.png"
                        Image.fromarray((pred_bin * 255).astype(np.uint8)).save(out_path)

            else:
                targets_bin = aligned_targets.float()
                pred_bin, _ = make_prediction_binary(
                    logits=logits,
                    profile=metric_profile,
                    fixed_threshold=threshold,
                )
                pred_metric_t, gt_metric_t = apply_metric_area_binary(pred_bin, targets_bin, metric_area=metric_area)
                dice = dice_score_binary(pred_metric_t, gt_metric_t)
                jac = jaccard_score_binary(pred_metric_t, gt_metric_t)
                dice_meter.update(dice, n=bsz)
                jaccard_meter.update(jac, n=bsz)

                pred_np = pred_bin.squeeze(1).detach().cpu().numpy().astype(np.uint8)
                gt_np = (targets_bin.squeeze(1).detach().cpu().numpy() > 0.5).astype(np.uint8)

                for i in range(pred_np.shape[0]):
                    if metric_profile == "sota_dynamic_otsu":
                        pu, pl = find_boundaries_columnwise(pred_np[i])
                        gu, gl = find_boundaries_columnwise(gt_np[i])
                        us, uu = compute_errors_columnwise(pu, gu, pixel_size=pixel_size_micrometers)
                        ls, lu = compute_errors_columnwise(pl, gl, pixel_size=pixel_size_micrometers)
                    else:
                        pu, pl = find_boundaries_global(pred_np[i])
                        gu, gl = find_boundaries_global(gt_np[i])
                        us, uu = compute_errors_global(pu, gu, pixel_size=pixel_size_micrometers)
                        ls, lu = compute_errors_global(pl, gl, pixel_size=pixel_size_micrometers)

                    up_signed_meter.update(us)
                    up_unsigned_meter.update(uu)
                    low_signed_meter.update(ls)
                    low_unsigned_meter.update(lu)

                    if save_predictions_dir is not None:
                        out_path = save_predictions_dir / f"{Path(filenames[i]).stem}_mask.png"
                        Image.fromarray((pred_np[i] * 255).astype(np.uint8)).save(out_path)

    return EvalResult(
        loss=loss_meter.avg,
        dice=dice_meter.avg,
        jaccard=jaccard_meter.avg,
        upper_signed_um=up_signed_meter.avg,
        upper_unsigned_um=up_unsigned_meter.avg,
        lower_signed_um=low_signed_meter.avg,
        lower_unsigned_um=low_unsigned_meter.avg,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_monitor_value: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_monitor_value": float(best_monitor_value),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "args": vars(args),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> Dict:
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    msg = model.load_state_dict(state_dict, strict=strict)
    if hasattr(msg, "missing_keys") and msg.missing_keys:
        print(f"[WARN] missing keys while loading checkpoint: {msg.missing_keys[:8]}", file=sys.stderr)
    if hasattr(msg, "unexpected_keys") and msg.unexpected_keys:
        print(f"[WARN] unexpected keys while loading checkpoint: {msg.unexpected_keys[:8]}", file=sys.stderr)

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def build_train_val_loaders(
    args: argparse.Namespace,
    binary_task: bool,
    preprocess_cfg: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    image_channels = 1 if preprocess_cfg["image_mode"] == "gray" else 3
    train_transform = JointSegmentationTransform(
        image_size=preprocess_cfg["image_size"],
        augment=args.augment,
        normalize=preprocess_cfg["normalize"],
        binary_mask=binary_task,
        image_channels=image_channels,
        mask_threshold=preprocess_cfg["mask_threshold"],
        random_resized_crop=preprocess_cfg["random_resized_crop"],
        horizontal_flip_prob=preprocess_cfg["horizontal_flip_prob"],
        vertical_flip_prob=preprocess_cfg["vertical_flip_prob"],
        rotation_degrees=preprocess_cfg["rotation_degrees"],
        color_jitter=preprocess_cfg["color_jitter"],
    )
    eval_transform = JointSegmentationTransform(
        image_size=preprocess_cfg["image_size"],
        augment=False,
        normalize=preprocess_cfg["normalize"],
        binary_mask=binary_task,
        image_channels=image_channels,
        mask_threshold=preprocess_cfg["mask_threshold"],
        random_resized_crop=False,
        horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0,
        rotation_degrees=0.0,
        color_jitter=False,
    )

    if args.val_image_dir and args.val_mask_dir:
        train_dataset = PairSegmentationDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            transform=train_transform,
            binary_mask=binary_task,
            image_mode=preprocess_cfg["image_mode"],
            allow_all_empty_masks=args.allow_empty_masks,
        )
        val_dataset = PairSegmentationDataset(
            image_dir=args.val_image_dir,
            mask_dir=args.val_mask_dir,
            transform=eval_transform,
            binary_mask=binary_task,
            image_mode=preprocess_cfg["image_mode"],
            allow_all_empty_masks=args.allow_empty_masks,
        )
    else:
        full_dataset_train = PairSegmentationDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            transform=train_transform,
            binary_mask=binary_task,
            image_mode=preprocess_cfg["image_mode"],
            allow_all_empty_masks=args.allow_empty_masks,
        )
        full_dataset_eval = PairSegmentationDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            transform=eval_transform,
            binary_mask=binary_task,
            image_mode=preprocess_cfg["image_mode"],
            allow_all_empty_masks=args.allow_empty_masks,
        )
        dataset_len = len(full_dataset_train)
        val_size = int(round(dataset_len * args.val_split))
        val_size = max(1, val_size)
        train_size = dataset_len - val_size
        if train_size < 1:
            raise RuntimeError(f"dataset too small for val-split={args.val_split}; train_size={train_size}")
        indices = torch.randperm(dataset_len, generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_dataset = Subset(full_dataset_train, train_indices)
        val_dataset = Subset(full_dataset_eval, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_eval_loader(
    args: argparse.Namespace,
    binary_task: bool,
    preprocess_cfg: Dict[str, Any],
) -> DataLoader:
    image_channels = 1 if preprocess_cfg["image_mode"] == "gray" else 3
    eval_transform = JointSegmentationTransform(
        image_size=preprocess_cfg["image_size"],
        augment=False,
        normalize=preprocess_cfg["normalize"],
        binary_mask=binary_task,
        image_channels=image_channels,
        mask_threshold=preprocess_cfg["mask_threshold"],
        random_resized_crop=False,
        horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0,
        rotation_degrees=0.0,
        color_jitter=False,
    )
    dataset = PairSegmentationDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        transform=eval_transform,
        binary_mask=binary_task,
        image_mode=preprocess_cfg["image_mode"],
        allow_all_empty_masks=args.allow_empty_masks,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    binary_task = is_binary_model(args.model_id)
    num_classes = 1 if binary_task else int(args.num_classes or 2)
    metric_profile = infer_metric_profile(args.model_id, args.metric_profile)
    loss_name = infer_loss_name(args.model_id, args.loss)
    preprocess_cfg = resolve_preprocess_config(args)
    model_output_index = resolve_model_output_index(args.model_id, args.model_output_index)

    model = create_model(
        model_id=args.model_id,
        num_classes=num_classes,
        pretrained=args.pretrained,
        image_size=preprocess_cfg["image_size"],
        freeze_encoder_blocks=args.freeze_encoder_blocks,
    ).to(device)
    frozen, total_groups = maybe_freeze_layers(model, args.model_id, args.freeze_layers)
    if total_groups > 0:
        print(f"[INFO] frozen encoder groups: {frozen}/{total_groups}")

    class_weights = None
    if loss_name == "weighted_ce":
        class_weights = parse_class_weights(
            raw=args.class_weights,
            num_classes=(2 if not binary_task else 1),
            device=device,
        )
    criterion = build_criterion(
        loss_name=loss_name,
        model_id=args.model_id,
        num_classes=num_classes,
        class_weights=class_weights,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(args=args, optimizer=optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    train_loader, val_loader = build_train_val_loaders(
        args=args,
        binary_task=binary_task,
        preprocess_cfg=preprocess_cfg,
    )

    print(
        "[INFO] preprocess | "
        f"image_size={preprocess_cfg['image_size']} "
        f"image_mode={preprocess_cfg['image_mode']} "
        f"normalize={preprocess_cfg['normalize']} "
        f"augment_style={preprocess_cfg['augment_style']} "
        f"mask_threshold={preprocess_cfg['mask_threshold']}"
    )
    if model_output_index is not None:
        print(f"[INFO] model_output_index={model_output_index}")

    start_epoch = 1
    best_value = -float("inf") if args.monitor in {"dice", "jaccard"} else float("inf")
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            ckpt = load_checkpoint(resume_path, model=model, optimizer=optimizer, device=device, strict=False)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_value = float(ckpt.get("best_monitor_value", best_value))
            print(f"[INFO] resumed from {resume_path} at epoch {start_epoch}")
        else:
            print(f"[WARN] resume checkpoint not found: {resume_path}", file=sys.stderr)

    save_best_path = Path(args.save_best_path) if args.save_best_path else Path(args.output_dir) / f"{args.model_id}_best.pth"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_meter = RunningAverage()

        for images, masks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                logits = extract_logits(outputs, output_index=model_output_index)
                aligned_targets = align_targets_to_logits(logits, masks, binary_task=binary_task)
                loss = criterion(logits, aligned_targets)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss_meter.update(float(loss.item()), n=int(images.size(0)))

        val_pred_dir = None
        if args.save_val_predictions_every > 0 and epoch % args.save_val_predictions_every == 0:
            val_pred_dir = output_dir / f"val_preds_epoch_{epoch:03d}"

        val_result = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            binary_task=binary_task,
            metric_profile=metric_profile,
            threshold=args.threshold,
            pixel_size_micrometers=args.pixel_size_micrometers,
            model_output_index=model_output_index,
            metric_area=args.metric_area,
            save_predictions_dir=val_pred_dir,
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler_monitor = args.monitor if args.scheduler_monitor == "auto" else args.scheduler_monitor
                scheduler_metric = scheduler_metric_from_result(val_result, scheduler_monitor)
                scheduler.step(scheduler_metric)
            else:
                scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        if args.monitor == "loss":
            monitor_value = val_result.loss
            improved = monitor_value < best_value - args.min_delta
        elif args.monitor == "jaccard":
            monitor_value = val_result.jaccard
            improved = monitor_value > best_value + args.min_delta
        else:
            monitor_value = val_result.dice
            improved = monitor_value > best_value + args.min_delta

        if improved:
            best_value = monitor_value
            patience_counter = 0
            save_checkpoint(
                path=save_best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_monitor_value=best_value,
                args=args,
            )
            best_mark = "*"
        else:
            patience_counter += 1
            best_mark = ""

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss_meter.avg:.6f} | "
            f"val_loss={val_result.loss:.6f} | "
            f"val_dice={val_result.dice:.6f} | "
            f"val_jaccard={val_result.jaccard:.6f} | "
            f"lr={current_lr:.2e} | "
            f"upper_signed={val_result.upper_signed_um:.2f}um | "
            f"lower_signed={val_result.lower_signed_um:.2f}um "
            f"{best_mark}"
        )

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"[INFO] early stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"[INFO] best checkpoint: {save_best_path}")
    print(f"[INFO] best {args.monitor}: {best_value:.6f}")


def run_eval(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    binary_task = is_binary_model(args.model_id)
    num_classes = 1 if binary_task else int(args.num_classes or 2)
    metric_profile = infer_metric_profile(args.model_id, args.metric_profile)
    loss_name = infer_loss_name(args.model_id, args.loss)
    preprocess_cfg = resolve_preprocess_config(args)
    model_output_index = resolve_model_output_index(args.model_id, args.model_output_index)
    print(
        "[INFO] preprocess | "
        f"image_size={preprocess_cfg['image_size']} "
        f"image_mode={preprocess_cfg['image_mode']} "
        f"normalize={preprocess_cfg['normalize']} "
        f"augment_style={preprocess_cfg['augment_style']} "
        f"mask_threshold={preprocess_cfg['mask_threshold']}"
    )
    if model_output_index is not None:
        print(f"[INFO] model_output_index={model_output_index}")
    print(f"[INFO] metric_area={args.metric_area}")

    model = create_model(
        model_id=args.model_id,
        num_classes=num_classes,
        pretrained=args.pretrained,
        image_size=preprocess_cfg["image_size"],
        freeze_encoder_blocks=args.freeze_encoder_blocks,
    ).to(device)

    class_weights = None
    if loss_name == "weighted_ce":
        class_weights = parse_class_weights(
            raw=args.class_weights,
            num_classes=(2 if not binary_task else 1),
            device=device,
        )
    criterion = build_criterion(
        loss_name=loss_name,
        model_id=args.model_id,
        num_classes=num_classes,
        class_weights=class_weights,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    load_checkpoint(ckpt_path, model=model, optimizer=None, device=device, strict=False)

    loader = build_eval_loader(args=args, binary_task=binary_task, preprocess_cfg=preprocess_cfg)
    pred_dir = Path(args.output_dir) / "eval_predictions" if args.save_predictions else None
    result = evaluate_model(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        binary_task=binary_task,
        metric_profile=metric_profile,
        threshold=args.threshold,
        pixel_size_micrometers=args.pixel_size_micrometers,
        model_output_index=model_output_index,
        metric_area=args.metric_area,
        save_predictions_dir=pred_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))

    if args.save_metrics_path:
        metrics_path = Path(args.save_metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"[INFO] metrics saved to {metrics_path}")


def run_predict(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    binary_task = is_binary_model(args.model_id)
    num_classes = 1 if binary_task else int(args.num_classes or 2)
    metric_profile = infer_metric_profile(args.model_id, args.metric_profile)
    preprocess_cfg = resolve_preprocess_config(args)
    model_output_index = resolve_model_output_index(args.model_id, args.model_output_index)
    print(
        "[INFO] preprocess | "
        f"image_size={preprocess_cfg['image_size']} "
        f"image_mode={preprocess_cfg['image_mode']} "
        f"normalize={preprocess_cfg['normalize']} "
        f"augment_style={preprocess_cfg['augment_style']} "
        f"mask_threshold={preprocess_cfg['mask_threshold']}"
    )
    if model_output_index is not None:
        print(f"[INFO] model_output_index={model_output_index}")

    model = create_model(
        model_id=args.model_id,
        num_classes=num_classes,
        pretrained=args.pretrained,
        image_size=preprocess_cfg["image_size"],
        freeze_encoder_blocks=args.freeze_encoder_blocks,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    load_checkpoint(ckpt_path, model=model, optimizer=None, device=device, strict=False)
    model.eval()

    image_channels = 1 if preprocess_cfg["image_mode"] == "gray" else 3
    transform = JointSegmentationTransform(
        image_size=preprocess_cfg["image_size"],
        augment=False,
        normalize=preprocess_cfg["normalize"],
        binary_mask=binary_task,
        image_channels=image_channels,
        mask_threshold=preprocess_cfg["mask_threshold"],
        random_resized_crop=False,
        horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0,
        rotation_degrees=0.0,
        color_jitter=False,
    )
    dataset = PredictImageDataset(
        image_dir=args.predict_image_dir,
        transform=transform,
        image_mode=preprocess_cfg["image_mode"],
    )

    def _predict_collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        filenames = [item[1] for item in batch]
        original_sizes = [item[2] for item in batch]  # keep (W,H) tuples
        return images, filenames, original_sizes

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_predict_collate,
    )

    out_mask_dir = Path(args.output_dir) / "masks"
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, filenames, original_sizes in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            logits = extract_logits(outputs, output_index=model_output_index)

            if metric_profile == "cnn_class1":
                pred_class = torch.argmax(logits, dim=1).detach().cpu().numpy()
                for i in range(pred_class.shape[0]):
                    pred_bin = (pred_class[i] == 1).astype(np.uint8)
                    mask_img = Image.fromarray((pred_bin * 255).astype(np.uint8))
                    ow, oh = tuple(int(x) for x in original_sizes[i])
                    if mask_img.size != (ow, oh):
                        mask_img = mask_img.resize((ow, oh), Image.NEAREST)
                    out_path = out_mask_dir / f"{Path(filenames[i]).stem}_mask.png"
                    mask_img.save(out_path)
            else:
                pred_bin, _ = make_prediction_binary(
                    logits=logits,
                    profile=metric_profile,
                    fixed_threshold=args.threshold,
                )
                pred_np = pred_bin.squeeze(1).detach().cpu().numpy().astype(np.uint8)
                for i in range(pred_np.shape[0]):
                    mask_img = Image.fromarray((pred_np[i] * 255).astype(np.uint8))
                    ow, oh = tuple(int(x) for x in original_sizes[i])
                    if mask_img.size != (ow, oh):
                        mask_img = mask_img.resize((ow, oh), Image.NEAREST)
                    out_path = out_mask_dir / f"{Path(filenames[i]).stem}_mask.png"
                    mask_img.save(out_path)

    print(f"[INFO] predictions saved to {out_mask_dir}")


def print_model_list() -> None:
    print("Available model IDs:")
    for model_id in ALL_MODEL_IDS:
        preset = NOTEBOOK_PREPROCESS_BY_MODEL_ID[model_id]
        print(f"- {model_id}")
        print(f"    {MODEL_DESCRIPTIONS[model_id]}")
        print(f"    default metric-profile: {PROFILE_BY_MODEL_ID[model_id]}")
        print(f"    default loss: {LOSS_AUTO_BY_MODEL_ID[model_id]}")
        print(
            "    notebook preprocess: "
            f"size={preset.image_size}, mode={preset.image_mode}, "
            f"normalize={preset.normalize}, augment={preset.augment_style}"
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.list_models:
        return
    if args.mode is None:
        raise ValueError("--mode is required unless --list-models is used")
    if args.model_id not in ALL_MODEL_IDS:
        raise ValueError(f"unsupported --model-id: {args.model_id}")

    if args.mode == "train":
        if not args.image_dir or not args.mask_dir:
            raise ValueError("--image-dir and --mask-dir are required for mode=train")
        if bool(args.val_image_dir) ^ bool(args.val_mask_dir):
            raise ValueError("--val-image-dir and --val-mask-dir must be provided together")
    elif args.mode == "eval":
        if not args.image_dir or not args.mask_dir:
            raise ValueError("--image-dir and --mask-dir are required for mode=eval")
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for mode=eval")
    elif args.mode == "predict":
        if not args.predict_image_dir:
            raise ValueError("--predict-image-dir is required for mode=predict")
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for mode=predict")
    else:
        raise ValueError(f"unsupported mode: {args.mode}")

    if args.metric_profile != "auto":
        allowed = {"cnn_class1", "sota_fixed", "sota_dynamic_mean", "sota_dynamic_otsu"}
        if args.metric_profile not in allowed:
            raise ValueError(f"unsupported metric-profile: {args.metric_profile}")
    if args.metric_area not in METRIC_AREA_CHOICES:
        raise ValueError(f"unsupported metric-area: {args.metric_area}")

    if args.loss != "auto":
        allowed = {"weighted_ce", "tversky_multiclass", "dice_multiclass", "bce_logits", "tversky_binary"}
        if args.loss not in allowed:
            raise ValueError(f"unsupported loss: {args.loss}")

    if args.scheduler not in {"none", "plateau", "step"}:
        raise ValueError(f"unsupported scheduler: {args.scheduler}")
    if args.scheduler_monitor not in {"auto", "dice", "jaccard", "loss"}:
        raise ValueError(f"unsupported scheduler-monitor: {args.scheduler_monitor}")
    if args.step_size <= 0:
        raise ValueError("--step-size must be > 0")
    if not 0.0 < float(args.step_gamma) <= 1.0:
        raise ValueError("--step-gamma must be in (0,1]")
    if not 0.0 < float(args.plateau_factor) <= 1.0:
        raise ValueError("--plateau-factor must be in (0,1]")
    if args.plateau_patience < 0:
        raise ValueError("--plateau-patience must be >= 0")
    if args.freeze_layers < 0:
        raise ValueError("--freeze-layers must be >= 0")

    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be > 0 when provided")
    if not 0.0 <= float(args.mask_threshold) <= 1.0:
        raise ValueError("--mask-threshold must be in [0,1]")

    # Validate resolved preprocessing + channel compatibility early.
    _ = resolve_preprocess_config(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified CNN/SOTA U-Net trainer/evaluator/predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--list-models", action="store_true", help="list available model IDs and exit")
    p.add_argument("--mode", choices=["train", "eval", "predict"], default=None, help="run mode")
    p.add_argument("--model-id", choices=ALL_MODEL_IDS, default="cnn_unet_resnet50", help="model registry entry")

    p.add_argument("--image-dir", type=str, default=None, help="input image directory for train/eval")
    p.add_argument("--mask-dir", type=str, default=None, help="mask directory for train/eval")
    p.add_argument("--val-image-dir", type=str, default=None, help="separate validation image directory")
    p.add_argument("--val-mask-dir", type=str, default=None, help="separate validation mask directory")
    p.add_argument("--predict-image-dir", type=str, default=None, help="input image directory for mode=predict")

    p.add_argument("--output-dir", type=str, default="rfa_outputs", help="output directory")
    p.add_argument("--checkpoint", type=str, default=None, help="checkpoint path for eval/predict")
    p.add_argument("--resume-from", type=str, default=None, help="resume training from checkpoint")
    p.add_argument("--save-best-path", type=str, default=None, help="path for best checkpoint during training")
    p.add_argument("--save-metrics-path", type=str, default=None, help="optional JSON file for eval metrics")
    p.add_argument("--save-predictions", action="store_true", help="save masks during eval")
    p.add_argument(
        "--save-val-predictions-every",
        type=int,
        default=0,
        help="if >0, save validation masks every N epochs during training",
    )

    p.add_argument("--image-size", type=int, default=None, help="input resize (auto from notebook profile if omitted)")
    p.add_argument("--num-classes", type=int, default=None, help="output classes for CNN models")
    p.add_argument("--batch-size", type=int, default=8, help="batch size")
    p.add_argument("--epochs", type=int, default=20, help="training epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "plateau", "step"], help="LR scheduler")
    p.add_argument(
        "--scheduler-monitor",
        type=str,
        default="auto",
        choices=["auto", "dice", "jaccard", "loss"],
        help="metric for ReduceLROnPlateau (auto follows --monitor)",
    )
    p.add_argument("--plateau-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    p.add_argument("--plateau-patience", type=int, default=5, help="ReduceLROnPlateau patience")
    p.add_argument("--step-size", type=int, default=5, help="StepLR step size")
    p.add_argument("--step-gamma", type=float, default=0.5, help="StepLR gamma")
    p.add_argument("--num-workers", type=int, default=2, help="data loader workers")
    p.add_argument("--val-split", type=float, default=0.2, help="random split ratio when val dirs are not provided")
    p.add_argument("--patience", type=int, default=10, help="early stopping patience, <=0 disables")
    p.add_argument("--min-delta", type=float, default=1e-4, help="minimum monitor improvement")
    p.add_argument("--monitor", choices=["dice", "jaccard", "loss"], default="dice", help="best-checkpoint monitor")
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad-norm, <=0 disables")
    p.add_argument("--freeze-layers", type=int, default=0, help="freeze first N encoder groups for CNN baselines")

    p.add_argument("--threshold", type=float, default=0.5, help="fixed threshold for binary metrics")
    p.add_argument("--mask-threshold", type=float, default=0.5, help="threshold used to binarize GT masks")
    p.add_argument("--pixel-size-micrometers", type=float, default=10.35, help="pixel size for boundary errors")
    p.add_argument("--class-weights", type=str, default="1.0,1.0", help="comma-separated class weights for weighted CE")
    p.add_argument(
        "--metric-profile",
        type=str,
        default="auto",
        choices=["auto", "cnn_class1", "sota_fixed", "sota_dynamic_mean", "sota_dynamic_otsu"],
        help="metric computation style",
    )
    p.add_argument(
        "--metric-area",
        type=str,
        default="full",
        choices=["full", "gt_choroid_columns", "union_choroid_columns"],
        help="pixel region used for Dice/Jaccard (full image, GT envelope, or union of pred+GT envelopes)",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="auto",
        choices=["auto", "weighted_ce", "tversky_multiclass", "dice_multiclass", "bce_logits", "tversky_binary"],
        help="loss function",
    )

    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="torch device")
    p.add_argument("--amp", action="store_true", help="enable automatic mixed precision")
    p.add_argument("--augment", action="store_true", help="enable train-time augmentation")
    p.add_argument(
        "--allow-empty-masks",
        action="store_true",
        help="allow training/eval when all masks are empty in a split (disabled by default)",
    )

    p.add_argument(
        "--notebook-defaults",
        dest="notebook_defaults",
        action="store_true",
        help="use notebook-aligned preprocessing defaults by model-id",
    )
    p.add_argument(
        "--no-notebook-defaults",
        dest="notebook_defaults",
        action="store_false",
        help="disable notebook defaults and use generic defaults",
    )
    p.set_defaults(notebook_defaults=True)

    p.add_argument(
        "--augment-style",
        type=str,
        default="auto",
        choices=["auto", "none", "cnn_joint", "pgkd_flip", "bem_joint"],
        help="augmentation recipe used when --augment is enabled",
    )
    p.add_argument(
        "--image-mode",
        type=str,
        default="auto",
        choices=["auto", "rgb", "gray"],
        help="image channel mode for loading inputs",
    )

    p.add_argument("--normalize", dest="normalize", action="store_true", help="normalize input with ImageNet stats")
    p.add_argument("--no-normalize", dest="normalize", action="store_false", help="disable ImageNet normalization")
    p.set_defaults(normalize=None)

    p.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pretrained encoders when available")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="disable pretrained encoders")
    p.set_defaults(pretrained=True)

    p.add_argument(
        "--freeze-encoder-blocks",
        type=int,
        default=0,
        help="for cnn_attention_unet_vit: freeze first N transformer blocks",
    )
    p.add_argument(
        "--model-output-index",
        type=int,
        default=None,
        help="select logits from tuple/list model outputs (auto uses 2nd output when present)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        print_model_list()
        return

    validate_args(args)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "predict":
        run_predict(args)
    else:
        raise RuntimeError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
