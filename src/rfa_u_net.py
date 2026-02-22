
# -*- coding: utf-8 -*-
"""RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

This script implements a Vision Transformer (ViT) encoder pre-trained with RETFound
weights and an Attention U-Net decoder for segmenting the choroid in OCT images.
"""

# # IMPROVEMENTS (2026-02)
# # 1) MULTI-RES INPUT SUPPORT:
# #    - ViT patch embedding input-size checks are now aligned to --image_size
# #      (must be a multiple of 16), enabling runs such as 320x320.
# # 2) STABLE DATA LOADING:
# #    - Added --num_workers to avoid environment-specific multiprocessing failures.
# # 3) TRAINING/ABLATION READINESS:
# #    - Existing flags for progressive unfreezing, token-pyramid skips,
# #      post-refine head, and edge-aware loss are preserved as first-class controls.
# # 4) PRE-ADAPTER INPUT ALIGNMENT (NEW):
# #    - Optional lightweight pre-adapter can remap inputs to a RETFound-friendly
# #      canonical size/statistics before ViT patch embedding.
# # 5) OCT-CHOROID SPECIALIZED PRE-ADAPTER (WHY + HOW):
# #    - WHY: external OCT data has scanner/domain shifts (intensity, speckle, contrast),
# #      and direct resizing can blur thin choroid boundaries.
# #    - HOW: pre-adapter uses anti-aliased resize + optional InstanceNorm + lightweight
# #      depthwise refinement; "gray_edge" mode emits [normalized gray, corrected gray, edge]
# #      channels to preserve boundary cues for choroid segmentation while staying RETFound-compatible.
# #    - EMPIRICAL NOTE: in our external-case review (A_5/B_5), pre-adapter qualitatively
# #      corrected contrast/brightness mismatch that previously caused segmentation failure.
# #      Failure-panel references (kept for reproducibility and future debugging):
# #      /home/alireza/Code/RETFound_GeneLab/rfa/RFA-U-Net/runs_compare_best_worst_vgg19_rfa_20260221_085710/best_worst_visuals/rfa_best_baseline/worst10/01_dice_0.4916_B (5)__panel.png
# #      /home/alireza/Code/RETFound_GeneLab/rfa/RFA-U-Net/runs_compare_best_worst_vgg19_rfa_20260221_085710/best_worst_visuals/rfa_best_baseline/worst10/02_dice_0.4967_A (5)__panel.png
# #      Decision anchor: keep defaults --use_pre_adapter --pre_adapter_mode gray_edge --pre_adapter_norm in.
# # 6) OCT-SAFE DEFAULTS (NEW):
# #    - Conservative geometry/intensity augmentation defaults are set for boundary-sensitive
# #      choroid masks (light jitter/noise, mild rotations/crops, lower adapter residual init).
# # 7) TRUE GRAYSCALE PIPELINE (NEW):
# #    - Supports 1-channel OCT end-to-end (--image_mode gray): loader convert('L'),
# #      ViT patch_embed in_chans=1, and pre-adapter/weights adaptation for RGB<->Gray checkpoints.
# # 8) BEST EXTERNAL REFERENCE SETUP (CURRENT):
# #    - Run: runs_baseline_prog_unfreeze_external_20260217_150145
# #    - External (N=439): Dice=0.8711, Jaccard=0.7781
# #    - Key settings: RETFound pretrained, token_pyramid, progressive unfreeze,
# #      normalize_imagenet=True, no post-refine, no shallow stem, no pre-adapter.
# # 9) ADAPTER PLACEMENT CONTROL (NEW):
# #    - Encoder adapter placement is configurable via --adapter_placement.
# #    - Default is pre-block (adapter -> block), which is usually better than post-block
# #      for domain/style shift since token correction happens before attention/MLP.

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision import models as tv_models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from timm.layers import drop_path, to_2tuple, trunc_normal_
import models_vit
from util.pos_embed import interpolate_pos_embed
import argparse
import sys
import csv
from datetime import datetime
try:
    import gdown
except Exception:
    gdown = None
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from dataset import OCTDataset


ABLATION_PRESETS = {
    "baseline": {"use_attention": False, "use_fusion": False, "use_upconvs": False},
    "attention_only": {"use_attention": True, "use_fusion": False, "use_upconvs": False},
    "fusion_only": {"use_attention": False, "use_fusion": True, "use_upconvs": False},
    "upconvs_only": {"use_attention": False, "use_fusion": False, "use_upconvs": True},
    "attention_fusion": {"use_attention": True, "use_fusion": True, "use_upconvs": False},
    "attention_upconvs": {"use_attention": True, "use_fusion": False, "use_upconvs": True},
    "fusion_upconvs": {"use_attention": False, "use_fusion": True, "use_upconvs": True},
    "full": {"use_attention": True, "use_fusion": True, "use_upconvs": True},
}

# Controlled A/B reference (external-data-full, 439 images, run date: 2026-02-14).
# Keep these as regression anchors when changing architecture/training defaults.
CONTROLLED_AB_NOTES = [
    "exp04_legacy_skip: best pretrained setting (dice=0.859447, jaccard=0.760354).",
    "exp02_v4_with_imagenet_norm: tied best (dice=0.859437, jaccard=0.760989).",
    "exp01_v4_baseline: reference baseline (dice=0.856934, jaccard=0.755490).",
    "exp06_no_pretrain_control: no RETFound pretrain control (dice=0.824202, jaccard=0.708974).",
    "Best external reference: runs_baseline_prog_unfreeze_external_20260217_150145 "
    "(dice=0.871067, jaccard=0.778074; token_pyramid + progressive unfreeze + normalize_imagenet; "
    "no post-refine/shallow-stem/pre-adapter).",
    "Conclusion: RETFound pretraining gives a clear gain; most decoder flag swaps are small deltas.",
    "Qualitative note: OCT pre-adapter improves robustness to brightness/contrast shift (observed in hard external cases such as A_5/B_5).",
]


def resolve_ablation_flags(preset: str):
    if preset not in ABLATION_PRESETS:
        raise ValueError(f"Unknown ablation preset '{preset}'. Valid: {list(ABLATION_PRESETS.keys())}")
    return ABLATION_PRESETS[preset]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_class_weights(raw: str, num_classes: int = 2):
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * num_classes
    if len(vals) != num_classes:
        raise ValueError(f"class_weights must provide {num_classes} values, got {len(vals)}")
    return vals


def parse_block_indices(spec: str, num_blocks: int):
    """
    Parse block index spec into sorted unique indices.
    Examples:
      all
      18-23
      0,1,4-7,23
    """
    text = (spec or "").strip().lower()
    if text in {"all", "*"}:
        return list(range(num_blocks))
    if not text:
        return []

    out = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            a = int(left.strip())
            b = int(right.strip())
            if a > b:
                a, b = b, a
            for i in range(a, b + 1):
                if i < 0 or i >= num_blocks:
                    raise ValueError(f"adapter block index {i} out of range [0, {num_blocks - 1}]")
                out.add(i)
        else:
            i = int(token)
            if i < 0 or i >= num_blocks:
                raise ValueError(f"adapter block index {i} out of range [0, {num_blocks - 1}]")
            out.add(i)
    return sorted(out)


def parse_unfreeze_schedule(spec: str):
    """
    Parse schedule string "epoch:freeze_blocks,epoch:freeze_blocks".
    Example: "1:24,10:21,20:18"
    """
    text = (spec or "").strip()
    if not text:
        return []
    schedule = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"invalid progressive_unfreeze_schedule token '{token}', expected epoch:freeze_blocks")
        ep_raw, fr_raw = token.split(":", 1)
        ep = int(ep_raw.strip())
        fr = int(fr_raw.strip())
        if ep < 1:
            raise ValueError(f"schedule epoch must be >=1, got {ep}")
        if fr < 0:
            raise ValueError(f"freeze block count must be >=0, got {fr}")
        schedule.append((ep, fr))
    schedule = sorted(schedule, key=lambda x: x[0])
    return schedule


def apply_encoder_freeze(model: nn.Module, freeze_blocks: int) -> None:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "blocks"):
        return
    freeze_blocks = max(0, int(freeze_blocks))
    for i, blk in enumerate(model.encoder.blocks):
        req_grad = i >= freeze_blocks
        for p in blk.parameters():
            p.requires_grad = req_grad


def adapt_patch_embed_in_channels(state_dict, target_in_channels: int):
    """
    Adapt patch embedding weights when switching between RGB(3) and Gray(1) inputs.
    """
    key = "patch_embed.proj.weight"
    if key not in state_dict:
        return state_dict
    w = state_dict[key]
    if not torch.is_tensor(w) or w.ndim != 4:
        return state_dict
    src_in = int(w.shape[1])
    tgt_in = int(target_in_channels)
    if src_in == tgt_in:
        return state_dict
    if src_in == 3 and tgt_in == 1:
        state_dict[key] = w.mean(dim=1, keepdim=True)
        print("🔁 Adapted patch_embed weights from 3-channel to 1-channel (mean over RGB).")
        return state_dict
    if src_in == 1 and tgt_in == 3:
        state_dict[key] = w.repeat(1, 3, 1, 1) / 3.0
        print("🔁 Adapted patch_embed weights from 1-channel to 3-channel (replicated).")
        return state_dict
    return state_dict


def load_state_dict_flexible(module: nn.Module, state_dict: dict):
    """
    Load state_dict while skipping shape-mismatched tensors.
    """
    own = module.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in own and torch.is_tensor(v) and torch.is_tensor(own[k]) and own[k].shape != v.shape:
            skipped.append((k, tuple(v.shape), tuple(own[k].shape)))
            continue
        filtered[k] = v
    msg = module.load_state_dict(filtered, strict=False)
    if skipped:
        preview = ", ".join([f"{k}: {s}->{t}" for k, s, t in skipped[:8]])
        print(f"[WARN] skipped {len(skipped)} mismatched tensors while loading: {preview}")
    return msg


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
    parser.add_argument('--val_image_dir', type=str, default=None, help='Optional validation-image directory for explicit fold-based training')
    parser.add_argument('--val_mask_dir', type=str, default=None, help='Optional validation-mask directory for explicit fold-based training')
    parser.add_argument('--weights_path', type=str, default='weights/best_rfa_unet.pth',
                        help='Path to the pre-trained weights file (used if weights_type is retfound or rfa-unet)')
    parser.add_argument('--weights_type', type=str, default='none', choices=['none', 'retfound', 'rfa-unet'],
                        help='Type of weights to load: "none" for random initialization, "retfound" for RETFound weights (training from scratch), "rfa-unet" for pre-trained RFA-U-Net weights (inference/fine-tuning)')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--image_mode', type=str, default='gray', choices=['rgb', 'gray'],
                        help='Input image channel mode: gray uses true 1-channel pipeline; rgb keeps legacy 3-channel')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader worker count')
    parser.add_argument('--test_only', action='store_true', help='Run inference on external data without training')
    parser.add_argument('--test_image_dir', type=str, default=None, help='Path to external test images (required if --test_only)')
    parser.add_argument('--test_mask_dir', type=str, default=None, help='Path to external test masks (required if --test_only)')
    parser.add_argument('--segment_dir', type=str, default=None, help='Path to directory containing images to segment (no masks needed)')
    parser.add_argument('--output_dir', type=str, default='segment_results', help='Directory to save segmentation results')
    parser.add_argument('--save_overlay', action='store_true', help='Save overlay images with segmentation boundaries')
    parser.add_argument('--pixel_size_micrometers', type=float, default=10.35, help='Pixel size in micrometers for boundary error computation')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binarizing predicted masks')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible data split')
    parser.add_argument('--ablation_preset', type=str, default='full', choices=list(ABLATION_PRESETS.keys()),
                        help='Ablation preset for decoder: baseline/attention/fusion/up-convs combinations')
    parser.add_argument('--list_ablation_presets', action='store_true',
                        help='List all ablation presets and exit')
    parser.add_argument('--save_best_path', type=str, default=None,
                        help='Best-checkpoint path (default: weights/best_rfa_unet_<ablation>.pth)')
    parser.add_argument('--ablation_results_csv', type=str, default=None,
                        help='Optional CSV path to append final ablation metrics')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max-norm (<=0 disables)')
    parser.add_argument('--freeze_encoder_blocks', type=int, default=21,
                        help='Freeze first N ViT encoder blocks when using pretrained weights')
    parser.add_argument('--progressive_unfreeze_schedule', type=str, default='',
                        help='Optional schedule to change frozen block count by epoch, e.g. "1:24,10:21,20:18"')
    parser.add_argument('--enable_encoder_adapters', action='store_true',
                        help='Enable bottleneck adapters on selected ViT encoder blocks')
    parser.add_argument('--adapter_rank', type=int, default=64,
                        help='Bottleneck rank for encoder adapters (used when --enable_encoder_adapters)')
    parser.add_argument('--adapter_blocks', type=str, default='all',
                        help='Encoder blocks to attach adapters to (e.g., all or 18-23)')
    parser.add_argument('--adapter_dropout', type=float, default=0.0,
                        help='Dropout in encoder adapters')
    parser.add_argument('--adapter_init_scale', type=float, default=1e-3,
                        help='Initial residual scale for adapter output')
    parser.add_argument('--adapter_placement', type=str, default='pre', choices=['pre', 'post'],
                        help='Adapter placement relative to each encoder block: pre (adapter->block) or post (block->adapter)')
    parser.add_argument('--early_stopping_patience', type=int, default=8,
                        help='Early stopping patience on validation choroid Dice (<=0 disables)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4,
                        help='Minimum choroid Dice improvement to reset early stopping')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['none', 'plateau'],
                        help='LR scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='ReduceLROnPlateau factor')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='ReduceLROnPlateau patience')
    parser.add_argument('--class_weights', type=str, default='1.0,2.0',
                        help='Comma-separated class weights used in Tversky loss (e.g. 1.0,2.0)')
    parser.add_argument('--loss_mode', type=str, default='both', choices=['both', 'choroid_only'],
                        help='Loss target: both classes (weighted) or choroid class only')
    parser.add_argument('--edge_loss_weight', type=float, default=0.0,
                        help='Optional edge-aware Dice loss weight on choroid boundaries (0 disables)')
    parser.add_argument('--multiscale_skip_mode', type=str, default='legacy', choices=['legacy', 'token_pyramid'],
                        help='Skip construction mode: legacy direct-token skips, or learned token pyramid skips')
    parser.add_argument('--decoder_arch', type=str, default='rfa', choices=['rfa', 'transunet', 'fundu'],
                        help='Decoder style: rfa, transunet (RETFound ViT + TransUNet-like), or fundu (Segmenter-like mask decoder + progressive post-adapter)')
    parser.add_argument('--transunet_hybrid_skips', dest='transunet_hybrid_skips', action='store_true',
                        help='For decoder_arch=transunet, use ResNet50 hybrid CNN skips (28/56/112) like original TransUNet')
    parser.add_argument('--no_transunet_hybrid_skips', dest='transunet_hybrid_skips', action='store_false',
                        help='For decoder_arch=transunet, use token-derived skips instead of ResNet50 hybrid skips')
    parser.set_defaults(transunet_hybrid_skips=True)
    parser.add_argument('--transunet_r50_pretrained', dest='transunet_r50_pretrained', action='store_true',
                        help='Initialize TransUNet hybrid ResNet50 skip backbone with ImageNet weights')
    parser.add_argument('--no_transunet_r50_pretrained', dest='transunet_r50_pretrained', action='store_false',
                        help='Initialize TransUNet hybrid ResNet50 skip backbone randomly')
    parser.set_defaults(transunet_r50_pretrained=True)
    parser.add_argument('--transunet_freeze_r50', action='store_true',
                        help='Freeze ResNet50 hybrid skip backbone when decoder_arch=transunet')
    parser.add_argument('--transunet_strict', dest='transunet_strict', action='store_true',
                        help='Enforce strict TransUNet behavior (disable skip attention, shallow stem fusion, post-refine, deep supervision)')
    parser.add_argument('--no_transunet_strict', dest='transunet_strict', action='store_false',
                        help='Allow skip attention / shallow stem fusion / post-refine / deep supervision with decoder_arch=transunet')
    parser.set_defaults(transunet_strict=True)
    parser.add_argument('--fundu_decoder_layers', type=int, default=2,
                        help='Number of transformer blocks in Fundu-style mask decoder (paper default: 2)')
    parser.add_argument('--skip_standardize_channels', type=int, default=0,
                        help='If >0 and token_pyramid is used, project all token skips to this common channel count (e.g., 64)')
    parser.add_argument('--skip_attention', type=str, default='none', choices=['none', 'se', 'cbam'],
                        help='Optional attention on projected skip features (none|se|cbam)')
    parser.add_argument('--skip_attention_reduction', type=int, default=16,
                        help='Channel reduction ratio for skip attention modules (SE/CBAM)')
    parser.add_argument('--force_bilinear_decoder', action='store_true',
                        help='Force bilinear+1x1 upsampling decoder path (disable transpose-conv upconvs)')
    parser.add_argument('--use_post_refine', action='store_true',
                        help='Enable residual post-refinement head on final 224x224 decoder features')
    parser.add_argument('--post_refine_depth', type=int, default=2,
                        help='Number of conv-BN-ReLU blocks in post-refinement head')
    parser.add_argument('--post_refine_channels', type=int, default=64,
                        help='Hidden channels in post-refinement head')
    parser.add_argument('--use_shallow_stem_fusion', dest='use_shallow_stem_fusion', action='store_true',
                        help='Fuse shallow high-resolution CNN stem features into late decoder stages')
    parser.add_argument('--no_shallow_stem_fusion', dest='use_shallow_stem_fusion', action='store_false',
                        help='Disable shallow high-resolution CNN stem fusion')
    parser.set_defaults(use_shallow_stem_fusion=True)
    parser.add_argument('--deep_supervision', action='store_true',
                        help='Enable auxiliary decoder heads (d2,d3) during training')
    parser.add_argument('--aux_weight_d2', type=float, default=0.20,
                        help='Auxiliary loss weight for d2 head when deep supervision is enabled')
    parser.add_argument('--aux_weight_d3', type=float, default=0.10,
                        help='Auxiliary loss weight for d3 head when deep supervision is enabled')
    parser.add_argument('--normalize_imagenet', action='store_true',
                        help='Apply ImageNet normalization to both train and eval transforms')
    parser.add_argument('--use_pre_adapter', dest='use_pre_adapter', action='store_true',
                        help='Enable lightweight pre-adapter before ViT patch embedding')
    parser.add_argument('--no_pre_adapter', dest='use_pre_adapter', action='store_false',
                        help='Disable lightweight pre-adapter before ViT patch embedding')
    parser.set_defaults(use_pre_adapter=True)
    parser.add_argument('--pre_adapter_target_size', type=int, default=224,
                        help='Canonical square size used by pre-adapter before patch embedding')
    parser.add_argument('--pre_adapter_hidden_channels', type=int, default=8,
                        help='Hidden channels in pre-adapter conv trunk')
    parser.add_argument('--pre_adapter_depth', type=int, default=1,
                        help='Number of depthwise-separable refinement blocks in pre-adapter trunk')
    parser.add_argument('--pre_adapter_norm', type=str, default='in', choices=['in', 'bn'],
                        help='Normalization inside pre-adapter: in=InstanceNorm2d, bn=BatchNorm2d')
    parser.add_argument('--pre_adapter_mode', type=str, default='gray_edge', choices=['rgb_residual', 'gray_edge'],
                        help='Pre-adapter style: rgb_residual (generic) or gray_edge (OCT-specialized)')
    parser.add_argument('--pre_adapter_residual_scale_init', type=float, default=0.1,
                        help='Initial residual scale for pre-adapter output blending (OCT-safe range ~0.05-0.15)')
    parser.add_argument('--pre_adapter_rgb_scalars', dest='pre_adapter_rgb_scalars', action='store_true',
                        help='Enable learnable per-channel RGB scalars in pre-adapter')
    parser.add_argument('--no_pre_adapter_rgb_scalars', dest='pre_adapter_rgb_scalars', action='store_false',
                        help='Disable learnable per-channel RGB scalars in pre-adapter')
    parser.set_defaults(pre_adapter_rgb_scalars=True)
    parser.add_argument('--augment_random_resized_crop', dest='augment_random_resized_crop', action='store_true',
                        help='Enable random resized crop in training augmentation')
    parser.add_argument('--no_augment_random_resized_crop', dest='augment_random_resized_crop', action='store_false',
                        help='Disable random resized crop in training augmentation')
    parser.set_defaults(augment_random_resized_crop=True)
    parser.add_argument('--augment_scale_min', type=float, default=0.9,
                        help='Minimum scale for random resized crop')
    parser.add_argument('--augment_hflip_prob', type=float, default=0.2,
                        help='Horizontal flip probability for training augmentation')
    parser.add_argument('--augment_rotation_deg', type=float, default=8.0,
                        help='Max absolute random rotation degrees for training augmentation')
    parser.add_argument('--augment_color_jitter', dest='augment_color_jitter', action='store_true',
                        help='Enable color jitter in training augmentation')
    parser.add_argument('--no_augment_color_jitter', dest='augment_color_jitter', action='store_false',
                        help='Disable color jitter in training augmentation')
    parser.set_defaults(augment_color_jitter=True)
    parser.add_argument('--oct_intensity_aug', action='store_true',
                        help='Enable OCT-style intensity augmentation (gamma + speckle + Gaussian noise + blur)')
    parser.add_argument('--oct_brightness_jitter', type=float, default=0.1,
                        help='Brightness jitter amplitude for OCT intensity aug')
    parser.add_argument('--oct_contrast_jitter', type=float, default=0.1,
                        help='Contrast jitter amplitude for OCT intensity aug')
    parser.add_argument('--oct_gamma_min', type=float, default=0.9,
                        help='Minimum gamma for OCT intensity aug')
    parser.add_argument('--oct_gamma_max', type=float, default=1.1,
                        help='Maximum gamma for OCT intensity aug')
    parser.add_argument('--oct_speckle_prob', type=float, default=0.25,
                        help='Probability of multiplicative speckle noise for OCT intensity aug')
    parser.add_argument('--oct_speckle_std_min', type=float, default=0.03,
                        help='Minimum sigma for multiplicative speckle noise')
    parser.add_argument('--oct_speckle_std_max', type=float, default=0.07,
                        help='Maximum sigma for multiplicative speckle noise')
    parser.add_argument('--oct_noise_prob', type=float, default=0.2,
                        help='Probability of additive Gaussian noise for OCT intensity aug')
    parser.add_argument('--oct_noise_std_min', type=float, default=0.005,
                        help='Minimum sigma for additive Gaussian noise')
    parser.add_argument('--oct_noise_std_max', type=float, default=0.015,
                        help='Maximum sigma for additive Gaussian noise')
    parser.add_argument('--oct_blur_prob', type=float, default=0.1,
                        help='Probability of Gaussian blur for OCT intensity aug')
    parser.add_argument('--oct_blur_sigma_max', type=float, default=0.6,
                        help='Maximum sigma for Gaussian blur in OCT intensity aug')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Held-out test fraction from full dataset (e.g., 0.2)')
    parser.add_argument('--val_split_in_trainval', type=float, default=0.2,
                        help='Validation fraction inside the remaining (1-test_split) subset')
    return parser.parse_args()
# Parse arguments
args = parse_args()

if not (0.0 < args.augment_scale_min <= 1.0):
    raise ValueError(f"--augment_scale_min must be in (0, 1], got {args.augment_scale_min}")
if not (0.0 <= args.augment_hflip_prob <= 1.0):
    raise ValueError(f"--augment_hflip_prob must be in [0, 1], got {args.augment_hflip_prob}")
if args.augment_rotation_deg < 0:
    raise ValueError(f"--augment_rotation_deg must be >= 0, got {args.augment_rotation_deg}")
if args.edge_loss_weight < 0.0:
    raise ValueError(f"--edge_loss_weight must be >= 0, got {args.edge_loss_weight}")
if args.skip_standardize_channels < 0:
    raise ValueError(f"--skip_standardize_channels must be >= 0, got {args.skip_standardize_channels}")
if args.skip_attention not in {"none", "se", "cbam"}:
    raise ValueError(f"--skip_attention must be one of ['none','se','cbam'], got {args.skip_attention}")
if args.skip_attention_reduction < 1:
    raise ValueError(f"--skip_attention_reduction must be >= 1, got {args.skip_attention_reduction}")
if args.post_refine_depth < 1:
    raise ValueError(f"--post_refine_depth must be >= 1, got {args.post_refine_depth}")
if args.post_refine_channels < 1:
    raise ValueError(f"--post_refine_channels must be >= 1, got {args.post_refine_channels}")
if args.pre_adapter_target_size <= 0:
    raise ValueError(f"--pre_adapter_target_size must be > 0, got {args.pre_adapter_target_size}")
if args.pre_adapter_target_size % 16 != 0:
    raise ValueError(
        f"--pre_adapter_target_size must be divisible by 16 for ViT patching, got {args.pre_adapter_target_size}"
    )
if args.pre_adapter_hidden_channels < 1:
    raise ValueError(f"--pre_adapter_hidden_channels must be >= 1, got {args.pre_adapter_hidden_channels}")
if args.pre_adapter_depth < 1:
    raise ValueError(f"--pre_adapter_depth must be >= 1, got {args.pre_adapter_depth}")
if args.pre_adapter_norm not in {"in", "bn"}:
    raise ValueError(f"--pre_adapter_norm must be one of ['in','bn'], got {args.pre_adapter_norm}")
if args.pre_adapter_mode not in {"rgb_residual", "gray_edge"}:
    raise ValueError(
        f"--pre_adapter_mode must be one of ['rgb_residual','gray_edge'], got {args.pre_adapter_mode}"
    )
if args.pre_adapter_residual_scale_init < 0.0:
    raise ValueError(
        f"--pre_adapter_residual_scale_init must be >= 0, got {args.pre_adapter_residual_scale_init}"
    )
if args.enable_encoder_adapters and args.adapter_rank <= 0:
    raise ValueError(f"--adapter_rank must be > 0, got {args.adapter_rank}")
if not (0.0 <= args.adapter_dropout < 1.0):
    raise ValueError(f"--adapter_dropout must be in [0, 1), got {args.adapter_dropout}")
if args.adapter_init_scale < 0.0:
    raise ValueError(f"--adapter_init_scale must be >= 0, got {args.adapter_init_scale}")
if args.adapter_placement not in {"pre", "post"}:
    raise ValueError(f"--adapter_placement must be one of ['pre','post'], got {args.adapter_placement}")
if args.decoder_arch == "transunet" and args.multiscale_skip_mode != "token_pyramid":
    print(
        "[WARN] decoder_arch=transunet requires token_pyramid skips; overriding --multiscale_skip_mode to token_pyramid.",
        file=sys.stderr,
    )
    args.multiscale_skip_mode = "token_pyramid"
if args.decoder_arch == "transunet" and args.transunet_strict and args.skip_attention != "none":
    print(
        "[WARN] decoder_arch=transunet + strict mode uses plain skip fusion; overriding --skip_attention to 'none'.",
        file=sys.stderr,
    )
    args.skip_attention = "none"
if args.decoder_arch == "transunet" and args.transunet_strict and args.use_shallow_stem_fusion:
    print(
        "[WARN] decoder_arch=transunet + strict mode disables shallow stem fusion.",
        file=sys.stderr,
    )
    args.use_shallow_stem_fusion = False
if args.decoder_arch == "transunet" and args.transunet_strict and args.use_post_refine:
    print(
        "[WARN] decoder_arch=transunet + strict mode disables post-refine head.",
        file=sys.stderr,
    )
    args.use_post_refine = False
if args.decoder_arch == "transunet" and args.transunet_strict and args.deep_supervision:
    print(
        "[WARN] decoder_arch=transunet + strict mode disables deep supervision.",
        file=sys.stderr,
    )
    args.deep_supervision = False
if args.decoder_arch == "transunet" and not args.transunet_strict:
    print(
        "[INFO] decoder_arch=transunet with --no_transunet_strict: keeping user-configured skip_attention / shallow-stem / post-refine / deep-supervision.",
        file=sys.stderr,
    )
if args.decoder_arch == "fundu" and args.multiscale_skip_mode != "token_pyramid":
    print(
        "[WARN] decoder_arch=fundu requires token_pyramid skips; overriding --multiscale_skip_mode to token_pyramid.",
        file=sys.stderr,
    )
    args.multiscale_skip_mode = "token_pyramid"
if args.decoder_arch == "fundu" and args.skip_attention != "cbam":
    print(
        "[WARN] decoder_arch=fundu uses CBAM skip attention by design; overriding --skip_attention to 'cbam'.",
        file=sys.stderr,
    )
    args.skip_attention = "cbam"
if args.decoder_arch == "fundu" and args.use_post_refine:
    print(
        "[WARN] decoder_arch=fundu already uses progressive post-adapter; disabling --use_post_refine.",
        file=sys.stderr,
    )
    args.use_post_refine = False
if args.decoder_arch == "fundu" and args.deep_supervision:
    print(
        "[WARN] decoder_arch=fundu disables deep supervision in current implementation.",
        file=sys.stderr,
    )
    args.deep_supervision = False
if args.decoder_arch == "fundu" and args.use_shallow_stem_fusion:
    print(
        "[WARN] decoder_arch=fundu disables shallow stem fusion to preserve Fundu-style decoder pathway.",
        file=sys.stderr,
    )
    args.use_shallow_stem_fusion = False
if args.decoder_arch == "fundu" and args.fundu_decoder_layers < 1:
    raise ValueError(f"--fundu_decoder_layers must be >= 1, got {args.fundu_decoder_layers}")
if not (0.0 <= args.oct_brightness_jitter <= 1.0):
    raise ValueError(f"--oct_brightness_jitter must be in [0,1], got {args.oct_brightness_jitter}")
if not (0.0 <= args.oct_contrast_jitter <= 1.0):
    raise ValueError(f"--oct_contrast_jitter must be in [0,1], got {args.oct_contrast_jitter}")
if not (0.0 < args.oct_gamma_min <= args.oct_gamma_max):
    raise ValueError(
        f"--oct_gamma_min/--oct_gamma_max invalid: min={args.oct_gamma_min}, max={args.oct_gamma_max}"
    )
if not (0.0 <= args.oct_speckle_prob <= 1.0):
    raise ValueError(f"--oct_speckle_prob must be in [0,1], got {args.oct_speckle_prob}")
if not (0.0 <= args.oct_noise_prob <= 1.0):
    raise ValueError(f"--oct_noise_prob must be in [0,1], got {args.oct_noise_prob}")
if not (0.0 <= args.oct_blur_prob <= 1.0):
    raise ValueError(f"--oct_blur_prob must be in [0,1], got {args.oct_blur_prob}")
if not (0.0 <= args.oct_speckle_std_min <= args.oct_speckle_std_max):
    raise ValueError(
        f"--oct_speckle_std_min/max invalid: min={args.oct_speckle_std_min}, max={args.oct_speckle_std_max}"
    )
if not (0.0 <= args.oct_noise_std_min <= args.oct_noise_std_max):
    raise ValueError(
        f"--oct_noise_std_min/max invalid: min={args.oct_noise_std_min}, max={args.oct_noise_std_max}"
    )
if args.oct_blur_sigma_max < 0.0:
    raise ValueError(f"--oct_blur_sigma_max must be >= 0, got {args.oct_blur_sigma_max}")
if args.use_pre_adapter and args.pre_adapter_residual_scale_init > 0.3:
    print(
        f"[WARN] pre-adapter residual init is high ({args.pre_adapter_residual_scale_init}). "
        "For OCT boundary preservation, 0.05-0.15 is typically safer.",
        file=sys.stderr,
    )
if args.use_pre_adapter and args.pre_adapter_norm == "in" and args.normalize_imagenet:
    print(
        "[WARN] Both input normalization and pre-adapter InstanceNorm are enabled; "
        "this can over-normalize OCT intensity statistics.",
        file=sys.stderr,
    )

if args.list_ablation_presets:
    print("Available ablation presets:")
    for preset_name, preset_flags in ABLATION_PRESETS.items():
        print(
            f"  - {preset_name}: "
            f"attention={preset_flags['use_attention']}, "
            f"fusion={preset_flags['use_fusion']}, "
            f"upconvs={preset_flags['use_upconvs']}"
        )
    print("\nControlled A/B reference notes:")
    for note in CONTROLLED_AB_NOTES:
        print(f"  - {note}")
    sys.exit(0)

if bool(args.val_image_dir) ^ bool(args.val_mask_dir):
    raise ValueError("--val_image_dir and --val_mask_dir must be provided together")

PROGRESSIVE_UNFREEZE_PLAN = parse_unfreeze_schedule(args.progressive_unfreeze_schedule)

# Configuration based on command-line arguments
ablation_flags = resolve_ablation_flags(args.ablation_preset)
config = {
    "image_size": args.image_size,
    "hidden_dim": 1024,
    "patch_size": 16,
    "num_channels": 1 if args.image_mode == "gray" else 3,
    "num_classes": 2,
    "retfound_weights_path": args.weights_path,
    "ablation_preset": args.ablation_preset,
    "use_attention": ablation_flags["use_attention"],
    "use_fusion": ablation_flags["use_fusion"],
    "use_upconvs": ablation_flags["use_upconvs"],
    "multiscale_skip_mode": args.multiscale_skip_mode,
    "decoder_arch": args.decoder_arch,
    "fundu_decoder_layers": args.fundu_decoder_layers,
    "transunet_hybrid_skips": args.transunet_hybrid_skips,
    "transunet_r50_pretrained": args.transunet_r50_pretrained,
    "transunet_freeze_r50": args.transunet_freeze_r50,
    "skip_standardize_channels": args.skip_standardize_channels,
    "skip_attention": args.skip_attention,
    "skip_attention_reduction": args.skip_attention_reduction,
    "force_bilinear_decoder": args.force_bilinear_decoder,
    "use_shallow_stem_fusion": args.use_shallow_stem_fusion,
    "deep_supervision": args.deep_supervision,
    "use_post_refine": args.use_post_refine,
    "post_refine_depth": args.post_refine_depth,
    "post_refine_channels": args.post_refine_channels,
    "progressive_unfreeze_schedule": args.progressive_unfreeze_schedule,
    "edge_loss_weight": args.edge_loss_weight,
    "use_pre_adapter": args.use_pre_adapter,
    "pre_adapter_target_size": args.pre_adapter_target_size,
    "pre_adapter_hidden_channels": args.pre_adapter_hidden_channels,
    "pre_adapter_depth": args.pre_adapter_depth,
    "pre_adapter_norm": args.pre_adapter_norm,
    "pre_adapter_mode": args.pre_adapter_mode,
    "pre_adapter_residual_scale_init": args.pre_adapter_residual_scale_init,
    "pre_adapter_rgb_scalars": args.pre_adapter_rgb_scalars,
    "enable_encoder_adapters": args.enable_encoder_adapters,
    "adapter_rank": args.adapter_rank,
    "adapter_blocks": args.adapter_blocks,
    "adapter_dropout": args.adapter_dropout,
    "adapter_init_scale": args.adapter_init_scale,
    "adapter_placement": args.adapter_placement,
}

# Weights file paths
RETFOUND_WEIGHTS_PATH = "weights/RETFound_oct_weights.pth"
RFA_UNET_WEIGHTS_PATH  = "weights/best_rfa_unet.pth"

# URL for downloading RFA-U-Net weights

RFA_UNET_WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=1zDEdAmNwNK8I-QEa6fqL5E3WjDn7Z-__"

# Function to download RFA-U-Net weights (unchanged)
def download_weights(weights_path, url):
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}. Downloading...")
        if gdown is None:
            raise ImportError("gdown is required to download RFA-U-Net weights from Google Drive. Install it with `pip install gdown`.")
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


class TransUNetHybridResNetSkips(nn.Module):
    """
    Hybrid CNN skip path used in original TransUNet (R50-ViT):
      s1: 512 channels @ 28x28
      s2: 256 channels @ 56x56
      s3:  64 channels @ 112x112
    """
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        try:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet50(weights=weights)
        except Exception:
            backbone = tv_models.resnet50(pretrained=bool(pretrained))

        # Adapt stem conv to grayscale (or arbitrary in_channels) if needed.
        if int(in_channels) != 3:
            old = backbone.conv1
            new_conv = nn.Conv2d(
                int(in_channels),
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            with torch.no_grad():
                if int(in_channels) == 1:
                    new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                else:
                    rep = (int(in_channels) + 2) // 3
                    w = old.weight.repeat(1, rep, 1, 1)[:, : int(in_channels), :, :] / float(rep)
                    new_conv.weight.copy_(w)
            backbone.conv1 = new_conv

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        s3 = x          # 64 @ 112x112
        x = self.maxpool(x)
        x = self.layer1(x)
        s2 = x          # 256 @ 56x56
        x = self.layer2(x)
        s1 = x          # 512 @ 28x28
        return s1, s2, s3

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

# Decoder block with ablation toggles
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, use_attention=True, use_fusion=True, use_upconvs=True, reduce_skip=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_fusion = use_fusion
        self.use_upconvs = use_upconvs

        if self.use_upconvs:
            self.up = nn.ConvTranspose2d(in_c[0], out_c, kernel_size=2, stride=2)
            self.reduce_channels_x = nn.Identity()
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.reduce_channels_x = nn.Conv2d(in_c[0], out_c, kernel_size=1)

        self.reduce_channels_s = nn.Conv2d(in_c[1], out_c, kernel_size=1) if reduce_skip else nn.Identity()
        self.ag = AttentionGate([out_c, out_c], out_c) if self.use_attention else nn.Identity()
        self.c_fusion = ConvBlock(out_c + out_c, out_c)
        self.c_plain = ConvBlock(out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        x = self.reduce_channels_x(x)
        # Optional no-skip path (used for final decoder stage refinement).
        if s is None:
            return self.c_plain(x)

        s = self.reduce_channels_s(s)

        if x.shape[-2:] != s.shape[-2:]:
            s = F.interpolate(s, size=x.shape[-2:], mode='bilinear', align_corners=True)

        if self.use_attention:
            s = self.ag(x, s)

        if self.use_fusion:
            x = torch.cat([x, s], dim=1)
            return self.c_fusion(x)

        if self.use_attention:
            x = x + s
        return self.c_plain(x)


class TransUNetDecoderBlock(nn.Module):
    """
    TransUNet-style decoder block:
      upsample (bilinear) -> concat skip (if provided) -> 2x(Conv3x3+BN+ReLU)
    """
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(int(in_channels) + int(skip_channels), int(out_channels))

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FunduBasicBlock(nn.Module):
    """
    Basic Conv-Norm-ReLU block used in Fundu-style pre/post adapters.
    """
    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn"):
        super().__init__()
        c_in = int(in_channels)
        c_out = int(out_channels)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        if str(norm).lower() == "in":
            self.norm = nn.InstanceNorm2d(c_out, affine=True)
        else:
            self.norm = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FunduMaskTransformerDecoder(nn.Module):
    """
    Segmenter-style mask decoder:
      - append class tokens
      - run a small transformer over [class + patch tokens]
      - obtain masks by class-token vs patch-token similarity.
    """
    def __init__(self, embed_dim: int = 1024, num_classes: int = 2, num_layers: int = 2, num_heads: int = 16):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        layers = max(1, int(num_layers))
        heads = max(1, int(num_heads))
        self.class_tokens = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=heads,
                    dim_feedforward=self.embed_dim * 4,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        trunc_normal_(self.class_tokens, std=0.02)

    def forward(self, patch_tokens):
        # patch_tokens: [B, N, C]
        b, n, c = patch_tokens.shape
        if int(c) != self.embed_dim:
            raise RuntimeError(f"FunduMaskTransformerDecoder expects embed_dim={self.embed_dim}, got {c}")
        gs = int(n ** 0.5)
        if gs * gs != n:
            raise RuntimeError(f"FunduMaskTransformerDecoder expects square token grid, got N={n}")

        cls = self.class_tokens.expand(b, -1, -1)
        x = torch.cat([cls, patch_tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, : self.num_classes, :]
        patches = x[:, self.num_classes :, :]

        cls = F.normalize(cls, dim=-1)
        patches = F.normalize(patches, dim=-1)
        logits = torch.einsum('bqc,bnc->bqn', cls, patches)  # [B, Q, N]
        return logits.view(b, self.num_classes, gs, gs)


class FunduSkipBranch(nn.Module):
    """
    RETFound token-map skip -> 64ch projection -> progressive bicubic upsampling -> optional attention.
    """
    def __init__(self, in_channels: int = 1024, out_channels: int = 64, up_steps: int = 0, norm: str = "bn", attn: nn.Module = None):
        super().__init__()
        self.proj = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, padding=0, bias=False)
        self.proj_norm = nn.BatchNorm2d(int(out_channels))
        self.proj_act = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([FunduBasicBlock(out_channels, out_channels, norm=norm) for _ in range(max(0, int(up_steps)))])
        self.attn = attn if attn is not None else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.proj_act(self.proj_norm(x))
        for blk in self.blocks:
            x = F.interpolate(x, scale_factor=2.0, mode='bicubic', align_corners=False)
            x = blk(x)
        return self.attn(x)


class FunduPostAdapter(nn.Module):
    """
    Progressive post-adapter from 14x14 mask logits to full-resolution segmentation.
    Stages: 14->28->56->112->224, with skip fusion (concat + Conv-Norm-ReLU) at each stage.
    """
    def __init__(self, num_classes: int = 2, base_channels: int = 64, norm: str = "bn"):
        super().__init__()
        c = int(base_channels)
        q = int(num_classes)
        self.stem = FunduBasicBlock(q, c, norm=norm)
        self.fuse_28 = FunduBasicBlock(c + c, c, norm=norm)
        self.fuse_56 = FunduBasicBlock(c + c, c, norm=norm)
        self.fuse_112 = FunduBasicBlock(c + c, c, norm=norm)
        self.fuse_224 = FunduBasicBlock(c + c, c, norm=norm)
        self.refine = FunduBasicBlock(c, c, norm=norm)
        self.out = nn.Conv2d(c, q, kernel_size=1, padding=0)

    @staticmethod
    def _cat_fuse(x, s, block):
        if s is not None:
            if s.shape[-2:] != x.shape[-2:]:
                s = F.interpolate(s, size=x.shape[-2:], mode='bicubic', align_corners=False)
            x = torch.cat([x, s], dim=1)
        return block(x)

    def forward(self, mask_14, skip_28, skip_56, skip_112, skip_224):
        x = self.stem(mask_14)
        x = F.interpolate(x, scale_factor=2.0, mode='bicubic', align_corners=False)   # 14 -> 28
        x = self._cat_fuse(x, skip_28, self.fuse_28)
        x = F.interpolate(x, scale_factor=2.0, mode='bicubic', align_corners=False)   # 28 -> 56
        x = self._cat_fuse(x, skip_56, self.fuse_56)
        x = F.interpolate(x, scale_factor=2.0, mode='bicubic', align_corners=False)   # 56 -> 112
        x = self._cat_fuse(x, skip_112, self.fuse_112)
        x = F.interpolate(x, scale_factor=2.0, mode='bicubic', align_corners=False)   # 112 -> 224
        x = self._cat_fuse(x, skip_224, self.fuse_224)
        x = self.refine(x)
        return self.out(x)


class TokenBottleneckAdapter(nn.Module):
    """
    Lightweight residual adapter for token sequences [B, N, C].
    """
    def __init__(self, hidden_dim, rank=64, dropout=0.0, init_scale=1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        z = self.norm(x)
        z = self.down(z)
        z = self.act(z)
        z = self.drop(z)
        z = self.up(z)
        return x + self.scale * z


class SkipSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        c = int(channels)
        r = max(1, int(reduction))
        hidden = max(1, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        a = self.fc(self.pool(x))
        return x * a


class SkipCBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        c = int(channels)
        r = max(1, int(reduction))
        hidden = max(1, c // r)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ca
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s = torch.amax(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_s, max_s], dim=1))
        return x * sa


def build_skip_attention(kind: str, channels: int, reduction: int):
    k = str(kind).strip().lower()
    if k == "none":
        return nn.Identity()
    if k == "se":
        return SkipSE(channels=channels, reduction=reduction)
    if k == "cbam":
        return SkipCBAM(channels=channels, reduction=reduction)
    raise ValueError(f"unsupported skip attention kind: {kind}")


class PostRefineHead(nn.Module):
    """
    Residual refinement head for sharper boundaries at full resolution.
    """
    def __init__(self, in_ch=64, hidden_ch=64, num_classes=2, depth=2):
        super().__init__()
        depth = max(1, int(depth))
        blocks = []
        c = int(in_ch)
        h = int(hidden_ch)
        for i in range(depth):
            out_c = h if i < depth - 1 else c
            blocks.extend(
                [
                    nn.Conv2d(c, out_c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
            )
            c = out_c
        self.refine = nn.Sequential(*blocks)
        self.head = nn.Conv2d(int(in_ch), int(num_classes), kernel_size=1, padding=0)

    def forward(self, x):
        x = x + self.refine(x)
        return self.head(x)


class DWConvBlock(nn.Module):
    """
    Depthwise-separable block with configurable norm for OCT style adaptation.
    """
    def __init__(self, channels=16, norm_type="in"):
        super().__init__()
        c = int(channels)
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)
        if norm_type == "bn":
            self.norm = nn.BatchNorm2d(c)
        else:
            self.norm = nn.InstanceNorm2d(c, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        return self.act(x)


class PreAdapter(nn.Module):
    """
    OCT-oriented pre-adapter:
      1) anti-aliased resize to canonical RETFound size,
      2) lightweight depthwise-separable refinement with InstanceNorm/BatchNorm,
      3) safe residual blend + optional RGB channel scalars.
    """
    def __init__(
        self,
        in_ch=3,
        hidden_ch=8,
        depth=1,
        target_size=224,
        residual_scale_init=0.1,
        norm_type="in",
        mode="gray_edge",
        use_rgb_scalars=True,
    ):
        super().__init__()
        self.target_size = int(target_size)
        depth = max(1, int(depth))
        hidden_ch = int(hidden_ch)
        if norm_type not in {"in", "bn"}:
            raise ValueError(f"norm_type must be 'in' or 'bn', got {norm_type}")
        if mode not in {"rgb_residual", "gray_edge"}:
            raise ValueError(f"mode must be 'rgb_residual' or 'gray_edge', got {mode}")
        self.norm_type = norm_type
        self.mode = mode

        c_in = int(in_ch)
        if self.mode == "gray_edge":
            if norm_type == "bn":
                gray_norm = nn.BatchNorm2d(1)
                stem_norm = nn.BatchNorm2d(hidden_ch)
            else:
                gray_norm = nn.InstanceNorm2d(1, affine=True)
                stem_norm = nn.InstanceNorm2d(hidden_ch, affine=True)
            self.gray_norm = gray_norm
            self.stem = nn.Sequential(
                nn.Conv2d(1, hidden_ch, kernel_size=3, padding=1, bias=False),
                stem_norm,
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(*[DWConvBlock(hidden_ch, norm_type=norm_type) for _ in range(depth)])
            self.to_gray = nn.Conv2d(hidden_ch, 1, kernel_size=1, padding=0, bias=True)
        else:
            if norm_type == "bn":
                stem_norm = nn.BatchNorm2d(hidden_ch)
            else:
                stem_norm = nn.InstanceNorm2d(hidden_ch, affine=True)
            self.stem = nn.Sequential(
                nn.Conv2d(c_in, hidden_ch, kernel_size=3, padding=1, bias=False),
                stem_norm,
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(*[DWConvBlock(hidden_ch, norm_type=norm_type) for _ in range(depth)])
            self.to_rgb = nn.Conv2d(hidden_ch, c_in, kernel_size=1, padding=0, bias=True)
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))
        self.use_rgb_scalars = bool(use_rgb_scalars)
        if self.use_rgb_scalars:
            self.rgb_scale = nn.Parameter(torch.ones(1, int(in_ch), 1, 1, dtype=torch.float32))
        else:
            self.register_parameter("rgb_scale", None)

    @staticmethod
    def _anti_alias_resize(x, target_size):
        # Fixed 3x3 box blur before bicubic resize to reduce aliasing.
        k = torch.ones((1, 1, 3, 3), device=x.device, dtype=x.dtype) / 9.0
        x_blur = F.conv2d(x, k.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
        return F.interpolate(
            x_blur,
            size=(int(target_size), int(target_size)),
            mode='bicubic',
            align_corners=False,
        )

    @staticmethod
    def _sobel_mag(x):
        kx = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0)
        ky = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        g = torch.sqrt(gx * gx + gy * gy + 1e-6)
        denom = g.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return g / denom

    def forward(self, x):
        base = self._anti_alias_resize(x, self.target_size)
        if self.mode == "gray_edge":
            if base.shape[1] == 1:
                gray = base
            else:
                gray = 0.2989 * base[:, 0:1] + 0.5870 * base[:, 1:2] + 0.1140 * base[:, 2:3]
            gray_norm = self.gray_norm(gray)
            z = self.stem(gray_norm)
            z = self.blocks(z)
            z = self.to_gray(z)
            corrected = gray + self.res_scale * z
            if base.shape[1] == 1:
                out = corrected
            else:
                edge = self._sobel_mag(corrected)
                out = torch.cat([gray_norm, corrected, edge], dim=1)
        else:
            z = self.stem(base)
            z = self.blocks(z)
            z = self.to_rgb(z)
            out = base + self.res_scale * z
        if self.rgb_scale is not None:
            out = out * self.rgb_scale
        return out


# Main RFA-U-Net model
class AttentionUNetViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.configured_image_size = int(config.get("image_size", 224))
        self.input_channels = int(config.get("num_channels", 3))
        if self.configured_image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.configured_image_size}")
        if self.configured_image_size % 16 != 0:
            raise ValueError(
                f"image_size={self.configured_image_size} is not divisible by patch size 16; "
                "use multiples of 16 for RETFound ViT."
            )
        if self.input_channels not in {1, 3}:
            raise ValueError(f"num_channels must be 1 or 3, got {self.input_channels}")
        self.use_pre_adapter = bool(config.get("use_pre_adapter", False))
        self.pre_adapter_target_size = int(config.get("pre_adapter_target_size", 224))
        if self.pre_adapter_target_size <= 0:
            raise ValueError(f"pre_adapter_target_size must be > 0, got {self.pre_adapter_target_size}")
        if self.pre_adapter_target_size % 16 != 0:
            raise ValueError(
                f"pre_adapter_target_size={self.pre_adapter_target_size} is not divisible by patch size 16."
            )
        # Initialize RETFound-based ViT encoder
        self.encoder = models_vit.RETFound_mae(
            num_classes=config["num_classes"],
            drop_path_rate=0.2,
            global_pool=True,
            in_chans=self.input_channels,
        )
        if self.use_pre_adapter:
            self.pre_adapter = PreAdapter(
                in_ch=self.input_channels,
                hidden_ch=int(config.get("pre_adapter_hidden_channels", 8)),
                depth=int(config.get("pre_adapter_depth", 1)),
                target_size=self.pre_adapter_target_size,
                residual_scale_init=float(config.get("pre_adapter_residual_scale_init", 0.1)),
                norm_type=str(config.get("pre_adapter_norm", "in")),
                mode=str(config.get("pre_adapter_mode", "gray_edge")),
                use_rgb_scalars=bool(config.get("pre_adapter_rgb_scalars", True)),
            )
            encoder_input_size = self.pre_adapter_target_size
            print(
                f"🧱 Pre-adapter enabled | target={self.pre_adapter_target_size} "
                f"hidden_ch={int(config.get('pre_adapter_hidden_channels', 8))} "
                f"depth={int(config.get('pre_adapter_depth', 1))} "
                f"norm={str(config.get('pre_adapter_norm', 'in'))} "
                f"mode={str(config.get('pre_adapter_mode', 'gray_edge'))} "
                f"rgb_scalars={bool(config.get('pre_adapter_rgb_scalars', True))}"
            )
            if self.pre_adapter_target_size != self.configured_image_size:
                print(
                    f"📐 Pre-adapter canonical size {self.pre_adapter_target_size} differs from input resize "
                    f"{self.configured_image_size}; decoder logits will be upsampled back to input size."
                )
        else:
            self.pre_adapter = None
            encoder_input_size = self.configured_image_size
        # Ensure patch embedding matches selected input channel count.
        self.encoder.patch_embed.proj = nn.Conv2d(self.input_channels, 1024, kernel_size=(16, 16), stride=(16, 16))
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
                state_dict = adapt_patch_embed_in_channels(state_dict, target_in_channels=self.input_channels)
                # remove incompatible keys, interpolate positional embeddings
                own = self.encoder.state_dict()
                for k in ['patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
                    if k in state_dict and state_dict[k].shape != own[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del state_dict[k]
                interpolate_pos_embed(self.encoder, state_dict)
                msg = load_state_dict_flexible(self.encoder, state_dict)
                print("Loaded RETFound weights, missing keys:", msg.missing_keys)
                trunc_normal_(self.encoder.head.weight, std=2e-5)
            else:
                # pure RFA-U-Net checkpoint
                state_dict = adapt_patch_embed_in_channels(state_dict, target_in_channels=self.input_channels)
                msg = load_state_dict_flexible(self, state_dict)
                print("Loaded RFA-U-Net weights, missing keys:", msg.missing_keys)

        # Allow training/inference at runtime sizes by resizing positional embeddings in forward().
        # If pre-adapter is enabled, ViT itself operates at pre_adapter_target_size.
        if hasattr(self.encoder, "patch_embed") and hasattr(self.encoder.patch_embed, "img_size"):
            self.encoder.patch_embed.img_size = (encoder_input_size, encoder_input_size)
            if encoder_input_size != 224:
                print(f"📐 Patch embed input size set to {encoder_input_size}x{encoder_input_size}")


        # Decoder blocks with ablation switches
        use_attention = config.get("use_attention", True)
        use_fusion = config.get("use_fusion", True)
        use_upconvs = config.get("use_upconvs", True)
        if bool(config.get("force_bilinear_decoder", False)):
            use_upconvs = False
        self.multiscale_skip_mode = config.get("multiscale_skip_mode", "legacy")
        self.decoder_arch = str(config.get("decoder_arch", "rfa")).strip().lower()
        self.skip_standardize_channels = int(config.get("skip_standardize_channels", 0))
        self.skip_attention = str(config.get("skip_attention", "none")).strip().lower()
        self.skip_attention_reduction = int(config.get("skip_attention_reduction", 16))
        self.transunet_hybrid_skips = bool(config.get("transunet_hybrid_skips", True))
        self.transunet_r50_pretrained = bool(config.get("transunet_r50_pretrained", True))
        self.transunet_freeze_r50 = bool(config.get("transunet_freeze_r50", False))
        self.fundu_decoder_layers = int(config.get("fundu_decoder_layers", 2))
        self.force_bilinear_decoder = bool(config.get("force_bilinear_decoder", False))
        if self.decoder_arch not in {"rfa", "transunet", "fundu"}:
            raise ValueError(f"decoder_arch must be one of ['rfa','transunet','fundu'], got {self.decoder_arch}")
        if self.skip_attention not in {"none", "se", "cbam"}:
            raise ValueError(f"skip_attention must be one of ['none','se','cbam'], got {self.skip_attention}")
        if self.skip_attention_reduction < 1:
            raise ValueError(f"skip_attention_reduction must be >=1, got {self.skip_attention_reduction}")
        self.use_shallow_stem_fusion = bool(config.get("use_shallow_stem_fusion", True))
        self.deep_supervision = bool(config.get("deep_supervision", False))
        self.use_post_refine = bool(config.get("use_post_refine", False))
        self.enable_encoder_adapters = bool(config.get("enable_encoder_adapters", False))
        self.adapter_block_indices = []
        self.encoder_adapters = nn.ModuleDict()
        self.transunet_conv_more = None
        self.transunet_hybrid_skip_net = None
        self.fundu_mask_decoder = None
        self.fundu_skip_28 = None
        self.fundu_skip_56 = None
        self.fundu_skip_112 = None
        self.fundu_skip_224 = None
        self.fundu_post_adapter = None
        self.d2_out_channels = 256
        self.d3_out_channels = 128
        self.decoder_out_channels = 64
        self.adapter_placement = str(config.get("adapter_placement", "pre")).strip().lower()
        if self.adapter_placement not in {"pre", "post"}:
            raise ValueError(f"adapter_placement must be 'pre' or 'post', got {self.adapter_placement}")
        if self.enable_encoder_adapters:
            hidden_dim = int(getattr(self.encoder, "embed_dim", config.get("hidden_dim", 1024)))
            num_blocks = len(self.encoder.blocks)
            self.adapter_block_indices = parse_block_indices(config.get("adapter_blocks", "all"), num_blocks)
            rank = int(config.get("adapter_rank", 64))
            dropout = float(config.get("adapter_dropout", 0.0))
            init_scale = float(config.get("adapter_init_scale", 1e-3))
            for i in self.adapter_block_indices:
                self.encoder_adapters[str(i)] = TokenBottleneckAdapter(
                    hidden_dim=hidden_dim,
                    rank=rank,
                    dropout=dropout,
                    init_scale=init_scale,
                )
            print(
                f"🧩 Encoder adapters enabled | rank={rank} blocks={self.adapter_block_indices} "
                f"dropout={dropout} init_scale={init_scale} placement={self.adapter_placement}"
            )
        if self.multiscale_skip_mode == "token_pyramid":
            # Learned token pyramid to inject distinct spatial scales before decoder fusion.
            if self.decoder_arch == "fundu":
                c1, c2, c3 = 64, 64, 64
            elif self.decoder_arch == "transunet" and self.transunet_hybrid_skips:
                # Original TransUNet hybrid skip channels from CNN (R50): [512, 256, 64].
                c1, c2, c3 = 512, 256, 64
            elif self.skip_standardize_channels > 0:
                c1 = c2 = c3 = int(self.skip_standardize_channels)
            else:
                if self.decoder_arch == "transunet":
                    # TransUNet-like decoder expects skip channels [512, 256, 64].
                    c1, c2, c3 = 512, 256, 64
                else:
                    c1, c2, c3 = 512, 256, 128
            if self.decoder_arch == "fundu":
                # Fundu-style fixed CBAM skip pipeline at scales 28/56/112/224.
                cbam = lambda: build_skip_attention("cbam", channels=64, reduction=self.skip_attention_reduction)
                self.fundu_skip_28 = FunduSkipBranch(in_channels=1024, out_channels=64, up_steps=1, norm="bn", attn=cbam())   # z24: 14->28
                self.fundu_skip_56 = FunduSkipBranch(in_channels=1024, out_channels=64, up_steps=2, norm="bn", attn=cbam())   # z18: 14->56
                self.fundu_skip_112 = FunduSkipBranch(in_channels=1024, out_channels=64, up_steps=3, norm="bn", attn=cbam())  # z12: 14->112
                self.fundu_skip_224 = FunduSkipBranch(in_channels=1024, out_channels=64, up_steps=4, norm="bn", attn=cbam())  # z6: 14->224
                self.fundu_mask_decoder = FunduMaskTransformerDecoder(
                    embed_dim=1024,
                    num_classes=config["num_classes"],
                    num_layers=self.fundu_decoder_layers,
                    num_heads=16,
                )
                self.fundu_post_adapter = FunduPostAdapter(
                    num_classes=config["num_classes"],
                    base_channels=64,
                    norm="bn",
                )
                self.d2_out_channels = 64
                self.d3_out_channels = 64
                self.decoder_out_channels = 64
                print(
                    f"🧱 Decoder arch=fundu | mask_decoder_layers={self.fundu_decoder_layers} "
                    f"with progressive post-adapter + CBAM skips (28/56/112/224)"
                )
            elif self.decoder_arch == "transunet" and self.transunet_hybrid_skips:
                self.skip_proj_d1 = None
                self.skip_proj_d2 = None
                self.skip_proj_d3 = None
                self.skip_attn_d1 = nn.Identity()
                self.skip_attn_d2 = nn.Identity()
                self.skip_attn_d3 = nn.Identity()
                self.transunet_hybrid_skip_net = TransUNetHybridResNetSkips(
                    in_channels=self.input_channels,
                    pretrained=self.transunet_r50_pretrained,
                )
                if self.transunet_freeze_r50:
                    for p in self.transunet_hybrid_skip_net.parameters():
                        p.requires_grad = False
            else:
                self.skip_proj_d1 = self._make_skip_pyramid(in_ch=1024, out_ch=c1, up_steps=1)  # 14 -> 28
                self.skip_proj_d2 = self._make_skip_pyramid(in_ch=1024, out_ch=c2, up_steps=2)  # 14 -> 56
                self.skip_proj_d3 = self._make_skip_pyramid(in_ch=1024, out_ch=c3, up_steps=3)  # 14 -> 112
                self.skip_attn_d1 = build_skip_attention(self.skip_attention, channels=c1, reduction=self.skip_attention_reduction)
                self.skip_attn_d2 = build_skip_attention(self.skip_attention, channels=c2, reduction=self.skip_attention_reduction)
                self.skip_attn_d3 = build_skip_attention(self.skip_attention, channels=c3, reduction=self.skip_attention_reduction)
            if self.decoder_arch == "fundu":
                # Fundu path handles decoding with mask-transformer + progressive post-adapter.
                pass
            elif self.decoder_arch == "transunet":
                # TransUNet-like decoder cup over RETFound tokens:
                # bottleneck conv_more (1024->512) then decoder channels (256,128,64,16).
                self.transunet_conv_more = nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
                self.d1 = TransUNetDecoderBlock(in_channels=512, out_channels=256, skip_channels=c1)
                self.d2 = TransUNetDecoderBlock(in_channels=256, out_channels=128, skip_channels=c2)
                self.d3 = TransUNetDecoderBlock(in_channels=128, out_channels=64, skip_channels=c3)
                self.d4 = TransUNetDecoderBlock(in_channels=64, out_channels=16, skip_channels=0)
                self.d2_out_channels = 128
                self.d3_out_channels = 64
                self.decoder_out_channels = 16
                print(
                    f"🧱 Decoder arch=transunet | conv_more=1024->512, decoder=(256,128,64,16), "
                    f"hybrid_skips={self.transunet_hybrid_skips} channels=({c1},{c2},{c3})"
                )
                if self.transunet_hybrid_skips:
                    print(
                        f"🧱 TransUNet hybrid skips: ResNet50(pretrained={self.transunet_r50_pretrained}, "
                        f"frozen={self.transunet_freeze_r50})"
                    )
            else:
                self.d1 = DecoderBlock([1024, c1], 512, use_attention=use_attention, use_fusion=use_fusion, use_upconvs=use_upconvs)
                self.d2 = DecoderBlock([512, c2], 256, use_attention=use_attention, use_fusion=use_fusion, use_upconvs=use_upconvs)
                self.d3 = DecoderBlock([256, c3], 128, use_attention=use_attention, use_fusion=use_fusion, use_upconvs=use_upconvs)
                self.d4 = DecoderBlock([128, 64], 64, use_attention=use_attention, use_fusion=use_fusion, use_upconvs=use_upconvs)
                self.d2_out_channels = 256
                self.d3_out_channels = 128
                self.decoder_out_channels = 64
            if self.decoder_arch != "fundu":
                print(
                    f"🧬 Token-pyramid skips | channels=({c1},{c2},{c3}) "
                    f"attention={self.skip_attention} reduction={self.skip_attention_reduction} "
                    f"force_bilinear={self.force_bilinear_decoder}"
                )
        else:
            self.d1 = DecoderBlock([1024, 1024], 512, use_attention=use_attention, use_fusion=use_fusion, use_upconvs=use_upconvs)
            self.d2 = DecoderBlock([512, 1024], 256, use_attention=use_attention, use_fusion=use_fusion, use_upconvs=use_upconvs)
            self.d3 = DecoderBlock([256, 1024], 128, use_attention=use_attention, use_fusion=use_fusion, use_upconvs=use_upconvs)
            self.d4 = DecoderBlock([128, 1024], 64, use_attention=use_attention, use_fusion=use_fusion, use_upconvs=use_upconvs)
            self.d2_out_channels = 256
            self.d3_out_channels = 128
            self.decoder_out_channels = 64

        if self.use_shallow_stem_fusion:
            shallow_d3_channels = self.d3_out_channels
            shallow_d4_channels = self.decoder_out_channels
            self.shallow_stem = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.shallow_to_d3 = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),  # 224 -> 112
                nn.Conv2d(64, shallow_d3_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(shallow_d3_channels),
                nn.ReLU(inplace=True),
            )
            self.shallow_to_d4 = nn.Sequential(
                nn.Conv2d(64, shallow_d4_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(shallow_d4_channels),
                nn.ReLU(inplace=True),
            )

        if self.deep_supervision:
            self.aux_head_d2 = nn.Conv2d(self.d2_out_channels, config["num_classes"], kernel_size=1)
            self.aux_head_d3 = nn.Conv2d(self.d3_out_channels, config["num_classes"], kernel_size=1)
        if self.use_post_refine:
            self.output = PostRefineHead(
                in_ch=self.decoder_out_channels,
                hidden_ch=int(config.get("post_refine_channels", 64)),
                num_classes=config["num_classes"],
                depth=int(config.get("post_refine_depth", 2)),
            )
            print(
                f"🔎 Post-refine head enabled | depth={int(config.get('post_refine_depth', 2))} "
                f"hidden_ch={int(config.get('post_refine_channels', 64))}"
            )
        else:
            self.output = nn.Conv2d(self.decoder_out_channels, config["num_classes"], kernel_size=1, padding=0)

    @staticmethod
    def _make_skip_pyramid(in_ch, out_ch, up_steps):
        layers = []
        c = in_ch
        for step in range(up_steps):
            is_last = (step == up_steps - 1)
            next_c = out_ch if is_last else max(out_ch, c // 2)
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(c, next_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(next_c),
                    nn.ReLU(inplace=True),
                ]
            )
            c = next_c
        return nn.Sequential(*layers)

    @staticmethod
    def _tokens_to_feature_map(tokens):
        # tokens: [B, N, C] where N = H*W
        b, n, c = tokens.shape
        hw = int(n ** 0.5)
        if hw * hw != n:
            raise RuntimeError(f"Non-square token grid: N={n}")
        return tokens.transpose(1, 2).reshape(b, c, hw, hw)

    @staticmethod
    def _resize_pos_embed(pos_embed, target_token_count):
        # pos_embed: [1, 1+N, C] (includes cls token)
        if target_token_count == pos_embed.shape[1]:
            return pos_embed
        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]
        old_n = patch_pos.shape[1]
        new_n = target_token_count - 1
        gs_old = int(old_n ** 0.5)
        gs_new = int(new_n ** 0.5)
        if gs_old * gs_old != old_n or gs_new * gs_new != new_n:
            raise RuntimeError(f"Cannot resize non-square pos_embed: old_n={old_n}, new_n={new_n}")
        patch_pos = patch_pos.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        x_in = x
        if self.pre_adapter is not None:
            x = self.pre_adapter(x_in)
        x_for_hybrid_skips = x
        batch_size = x.shape[0]
        x = self.encoder.patch_embed(x)  # [B, N, C]
        cls_token = self.encoder.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 1+N, C]
        pos_embed = self._resize_pos_embed(self.encoder.pos_embed, x.shape[1]).to(x.device)
        x = x + pos_embed
        x = self.encoder.pos_drop(x)
        skip_connections = []
        for i, blk in enumerate(self.encoder.blocks):
            key = str(i)
            if self.enable_encoder_adapters and self.adapter_placement == "pre" and key in self.encoder_adapters:
                x = self.encoder_adapters[key](x)
            x = blk(x)
            if self.enable_encoder_adapters and self.adapter_placement == "post" and key in self.encoder_adapters:
                x = self.encoder_adapters[key](x)
            if i in [5, 11, 17, 23]:
                skip_connections.append(x[:, 1:, :])  # drop cls token
        z6_tokens, z12_tokens, z18_tokens, z24_tokens = skip_connections
        z6 = self._tokens_to_feature_map(z6_tokens)
        z12 = self._tokens_to_feature_map(z12_tokens)
        z18 = self._tokens_to_feature_map(z18_tokens)
        z24 = self._tokens_to_feature_map(z24_tokens)

        if self.decoder_arch == "fundu":
            if self.fundu_mask_decoder is None or self.fundu_post_adapter is None:
                raise RuntimeError("decoder_arch=fundu is selected, but Fundu decoder modules are not initialized.")
            mask_14 = self.fundu_mask_decoder(z24_tokens)  # [B, C, 14, 14]
            s28 = self.fundu_skip_28(z24)
            s56 = self.fundu_skip_56(z18)
            s112 = self.fundu_skip_112(z12)
            s224 = self.fundu_skip_224(z6)
            main_output = self.fundu_post_adapter(mask_14, s28, s56, s112, s224)
            if main_output.shape[-2:] != x_in.shape[-2:]:
                main_output = F.interpolate(main_output, size=x_in.shape[-2:], mode='bilinear', align_corners=True)
            return main_output

        if self.multiscale_skip_mode == "token_pyramid":
            if self.decoder_arch == "transunet" and self.transunet_hybrid_skips and self.transunet_hybrid_skip_net is not None:
                # Strict TransUNet-style hybrid skips from CNN backbone.
                s1, s2, s3 = self.transunet_hybrid_skip_net(x_for_hybrid_skips)
            else:
                s1 = self.skip_proj_d1(z18)
                s2 = self.skip_proj_d2(z12)
                s3 = self.skip_proj_d3(z6)
                # Apply optional skip-attention (SE/CBAM) after projection and before decoder fusion.
                s1 = self.skip_attn_d1(s1)
                s2 = self.skip_attn_d2(s2)
                s3 = self.skip_attn_d3(s3)
            z24_in = self.transunet_conv_more(z24) if self.transunet_conv_more is not None else z24
            x = self.d1(z24_in, s1)
            d2 = self.d2(x, s2)
            d3 = self.d3(d2, s3)
        else:
            x = self.d1(z24, z18)
            d2 = self.d2(x, z12)
            d3 = self.d3(d2, z6)

        if self.use_shallow_stem_fusion:
            stem = self.shallow_stem(x_in)  # [B,64,224,224]
            s_d3 = self.shallow_to_d3(stem)  # [B,128,112,112]
            if s_d3.shape[-2:] != d3.shape[-2:]:
                s_d3 = F.interpolate(s_d3, size=d3.shape[-2:], mode='bilinear', align_corners=True)
            d3 = d3 + s_d3
        # Final upsampling stage as pure refinement (avoid reusing z6 twice).
        d4 = self.d4(d3, None)
        if self.use_shallow_stem_fusion:
            s_d4 = self.shallow_to_d4(stem)  # [B,64,224,224]
            if s_d4.shape[-2:] != d4.shape[-2:]:
                s_d4 = F.interpolate(s_d4, size=d4.shape[-2:], mode='bilinear', align_corners=True)
            d4 = d4 + s_d4

        main_output = self.output(d4)
        if main_output.shape[-2:] != x_in.shape[-2:]:
            main_output = F.interpolate(main_output, size=x_in.shape[-2:], mode='bilinear', align_corners=True)
        if self.deep_supervision:
            aux_d2 = self.aux_head_d2(d2)
            aux_d3 = self.aux_head_d3(d3)
            if aux_d2.shape[-2:] != main_output.shape[-2:]:
                aux_d2 = F.interpolate(aux_d2, size=main_output.shape[-2:], mode='bilinear', align_corners=False)
            if aux_d3.shape[-2:] != main_output.shape[-2:]:
                aux_d3 = F.interpolate(aux_d3, size=main_output.shape[-2:], mode='bilinear', align_corners=False)
            return {
                "main": main_output,
                "aux_d2": aux_d2,
                "aux_d3": aux_d3,
            }
        return main_output

# Tversky Loss (Aligned with root code)
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, class_weights=(1.0, 2.0), mode='both', choroid_class_idx=1):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth
        self.class_weights = class_weights
        self.mode = mode
        self.choroid_class_idx = int(choroid_class_idx)

    def forward(self, outputs, targets):
        # outputs: (B,2,H,W), targets: one-hot (B,2,H,W)
        probs = torch.softmax(outputs, dim=1)

        if self.mode == 'choroid_only':
            c = self.choroid_class_idx
            p = probs[:, c, :, :].contiguous().view(-1)
            t = targets[:, c, :, :].contiguous().view(-1)
            tp = (p * t).sum()
            fn = ((1 - p) * t).sum()
            fp = (p * (1 - t)).sum()
            tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
            return 1 - tversky

        per_class_losses = []
        weight_sum = 0.0
        for c, w in enumerate(self.class_weights):
            p = probs[:, c, :, :].contiguous().view(-1)
            t = targets[:, c, :, :].contiguous().view(-1)

            tp = (p * t).sum()
            fn = ((1 - p) * t).sum()
            fp = (p * (1 - t)).sum()
            tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
            per_class_losses.append(float(w) * (1 - tversky))
            weight_sum += float(w)

        return sum(per_class_losses) / max(weight_sum, 1e-8)


# Dice Loss (Defined but not used, for compatibility with root code)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = torch.softmax(outputs, dim=1).contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (outputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# Dice Score (Updated to compute both combined and choroid-specific Dice scores)
def dice_score(outputs, targets, smooth=1e-6):
    probs = torch.softmax(outputs, dim=1)
    # Compute combined Dice score (across both classes, matching root code)
    outputs_combined = probs.contiguous().view(-1)
    targets_combined = targets.contiguous().view(-1)
    intersection_combined = (outputs_combined * targets_combined).sum()
    dice_combined = (2. * intersection_combined + smooth) / (outputs_combined.sum() + targets_combined.sum() + smooth)

    # Compute choroid-specific Dice score (channel 1)
    outputs_choroid = probs[:, 1, :, :].contiguous().view(-1)
    targets_choroid = targets[:, 1, :, :].contiguous().view(-1)
    intersection_choroid = (outputs_choroid * targets_choroid).sum()
    dice_choroid = (2. * intersection_choroid + smooth) / (outputs_choroid.sum() + targets_choroid.sum() + smooth)

    return dice_combined.item(), dice_choroid.item()


def get_main_logits(model_outputs):
    if isinstance(model_outputs, dict):
        if "main" not in model_outputs:
            raise KeyError("Model output dict does not contain 'main'")
        return model_outputs["main"]
    return model_outputs


def edge_dice_loss_choroid(logits, targets, choroid_idx=1, eps=1e-6):
    """
    Edge-aware Dice loss on choroid boundaries using Sobel magnitude maps.
    """
    probs = torch.softmax(logits, dim=1)[:, choroid_idx:choroid_idx + 1, :, :]
    tgt = targets[:, choroid_idx:choroid_idx + 1, :, :].float()
    if probs.shape[-2:] != tgt.shape[-2:]:
        tgt = F.interpolate(tgt, size=probs.shape[-2:], mode='nearest')

    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=probs.device,
        dtype=probs.dtype,
    ).unsqueeze(0)
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=probs.device,
        dtype=probs.dtype,
    ).unsqueeze(0)

    pgx = F.conv2d(probs, kx, padding=1)
    pgy = F.conv2d(probs, ky, padding=1)
    tgx = F.conv2d(tgt, kx, padding=1)
    tgy = F.conv2d(tgt, ky, padding=1)
    p_edge = torch.sqrt(pgx * pgx + pgy * pgy + eps)
    t_edge = torch.sqrt(tgx * tgx + tgy * tgy + eps)

    p_edge = p_edge.reshape(p_edge.shape[0], -1)
    t_edge = t_edge.reshape(t_edge.shape[0], -1)
    inter = (p_edge * t_edge).sum(dim=1)
    den = p_edge.sum(dim=1) + t_edge.sum(dim=1)
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def compute_total_loss(
    model_outputs,
    targets,
    criterion,
    aux_weight_d2=0.2,
    aux_weight_d3=0.1,
    edge_loss_weight=0.0,
):
    logits = get_main_logits(model_outputs)
    loss = criterion(logits, targets)
    edge_w = float(edge_loss_weight)
    if edge_w > 0.0:
        loss = loss + edge_w * edge_dice_loss_choroid(logits, targets, choroid_idx=1)
    if isinstance(model_outputs, dict):
        if "aux_d2" in model_outputs:
            loss = loss + float(aux_weight_d2) * criterion(model_outputs["aux_d2"], targets)
        if "aux_d3" in model_outputs:
            loss = loss + float(aux_weight_d3) * criterion(model_outputs["aux_d3"], targets)
    return loss

# Boundary Detection and Error Computation
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
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        if image.shape[-1] == 1:
            # Grayscale pipeline: expand to pseudo-RGB so boundary colors can be drawn.
            image = np.repeat(image, 3, axis=2)
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
class JointSegmentationTransform:
    """Synchronized image/mask transform for segmentation."""
    def __init__(
        self,
        image_size=224,
        augment=False,
        normalize=False,
        num_classes=2,
        image_mode='rgb',
        random_resized_crop=True,
        crop_scale_min=0.9,
        hflip_prob=0.2,
        rotation_deg=8.0,
        color_jitter=True,
        oct_intensity_aug=False,
        oct_brightness_jitter=0.1,
        oct_contrast_jitter=0.1,
        oct_gamma_min=0.9,
        oct_gamma_max=1.1,
        oct_speckle_prob=0.25,
        oct_speckle_std_min=0.03,
        oct_speckle_std_max=0.07,
        oct_noise_prob=0.2,
        oct_noise_std_min=0.005,
        oct_noise_std_max=0.015,
        oct_blur_prob=0.1,
        oct_blur_sigma_max=0.6,
    ):
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.normalize = bool(normalize)
        self.num_classes = int(num_classes)
        self.image_mode = str(image_mode).strip().lower()
        if self.image_mode not in {'rgb', 'gray'}:
            raise ValueError(f"image_mode must be 'rgb' or 'gray', got {image_mode}")
        self.random_resized_crop = bool(random_resized_crop)
        self.crop_scale_min = float(crop_scale_min)
        self.hflip_prob = float(hflip_prob)
        self.rotation_deg = float(rotation_deg)
        self.color_jitter_enabled = bool(color_jitter)
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        self.oct_intensity_aug = bool(oct_intensity_aug)
        self.oct_brightness_jitter = float(oct_brightness_jitter)
        self.oct_contrast_jitter = float(oct_contrast_jitter)
        self.oct_gamma_min = float(oct_gamma_min)
        self.oct_gamma_max = float(oct_gamma_max)
        self.oct_speckle_prob = float(oct_speckle_prob)
        self.oct_speckle_std_min = float(oct_speckle_std_min)
        self.oct_speckle_std_max = float(oct_speckle_std_max)
        self.oct_noise_prob = float(oct_noise_prob)
        self.oct_noise_std_min = float(oct_noise_std_min)
        self.oct_noise_std_max = float(oct_noise_std_max)
        self.oct_blur_prob = float(oct_blur_prob)
        self.oct_blur_sigma_max = float(oct_blur_sigma_max)
        if self.image_mode == 'gray':
            self.mean = [0.5]
            self.std = [0.5]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def _apply_oct_intensity_aug(self, image_t):
        b = 1.0 + random.uniform(-self.oct_brightness_jitter, self.oct_brightness_jitter)
        c = 1.0 + random.uniform(-self.oct_contrast_jitter, self.oct_contrast_jitter)
        image_t = TF.adjust_brightness(image_t, b)
        image_t = TF.adjust_contrast(image_t, c)
        gamma = random.uniform(self.oct_gamma_min, self.oct_gamma_max)
        image_t = TF.adjust_gamma(image_t, gamma=gamma, gain=1.0)

        if random.random() < self.oct_speckle_prob:
            sigma = random.uniform(self.oct_speckle_std_min, self.oct_speckle_std_max)
            image_t = image_t * (1.0 + sigma * torch.randn_like(image_t))

        if random.random() < self.oct_noise_prob:
            sigma = random.uniform(self.oct_noise_std_min, self.oct_noise_std_max)
            image_t = image_t + sigma * torch.randn_like(image_t)

        if random.random() < self.oct_blur_prob and self.oct_blur_sigma_max > 0.0:
            sigma = random.uniform(0.1, self.oct_blur_sigma_max)
            k = max(3, int(2 * round(3 * sigma) + 1))
            if k % 2 == 0:
                k += 1
            image_t = TF.gaussian_blur(image_t, kernel_size=[k, k], sigma=[sigma, sigma])

        return image_t.clamp(0.0, 1.0)

    def __call__(self, image, mask=None):
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        if self.augment and mask is not None:
            if self.random_resized_crop:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image,
                    scale=(self.crop_scale_min, 1.0),
                    ratio=(1.0, 1.0),
                )
                image = TF.resized_crop(
                    image, i, j, h, w, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR
                )
                mask = TF.resized_crop(
                    mask, i, j, h, w, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST
                )

            if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if self.rotation_deg > 0.0:
                angle = random.uniform(-self.rotation_deg, self.rotation_deg)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            if self.color_jitter_enabled:
                image = self.color_jitter(image)

        image_t = TF.to_tensor(image)
        if self.augment and mask is not None and self.oct_intensity_aug:
            image_t = self._apply_oct_intensity_aug(image_t)
        if self.normalize:
            image_t = TF.normalize(image_t, mean=self.mean, std=self.std)

        if mask is None:
            return image_t

        mask_np = (np.array(mask) > 0).astype(np.int64)
        mask_t = torch.from_numpy(mask_np).long()
        mask_onehot = F.one_hot(mask_t, num_classes=self.num_classes).permute(2, 0, 1).float()
        return image_t, mask_onehot


_imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_img_eval_ops = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
if args.image_mode == 'gray':
    _imagenet_norm = transforms.Normalize(mean=[0.5], std=[0.5])
if args.normalize_imagenet:
    _img_eval_ops.append(_imagenet_norm)
image_only_transform = transforms.Compose(_img_eval_ops)

train_transform = JointSegmentationTransform(
    image_size=args.image_size,
    augment=True,
    normalize=args.normalize_imagenet,
    num_classes=2,
    image_mode=args.image_mode,
    random_resized_crop=args.augment_random_resized_crop,
    crop_scale_min=args.augment_scale_min,
    hflip_prob=args.augment_hflip_prob,
    rotation_deg=args.augment_rotation_deg,
    color_jitter=args.augment_color_jitter,
    oct_intensity_aug=args.oct_intensity_aug,
    oct_brightness_jitter=args.oct_brightness_jitter,
    oct_contrast_jitter=args.oct_contrast_jitter,
    oct_gamma_min=args.oct_gamma_min,
    oct_gamma_max=args.oct_gamma_max,
    oct_speckle_prob=args.oct_speckle_prob,
    oct_speckle_std_min=args.oct_speckle_std_min,
    oct_speckle_std_max=args.oct_speckle_std_max,
    oct_noise_prob=args.oct_noise_prob,
    oct_noise_std_min=args.oct_noise_std_min,
    oct_noise_std_max=args.oct_noise_std_max,
    oct_blur_prob=args.oct_blur_prob,
    oct_blur_sigma_max=args.oct_blur_sigma_max,
)
val_test_transform = JointSegmentationTransform(
    image_size=args.image_size,
    augment=False,
    normalize=args.normalize_imagenet,
    num_classes=2,
    image_mode=args.image_mode,
)
def save_segmentation_results(filenames, original_sizes, predicted_masks, output_dir, segment_dir, save_overlay=False, threshold=0.5):
    """
    Save segmentation results as masks and optionally as overlay images.
    FIXED: Loads original images directly from disk for overlays instead of using transformed tensors.
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    if save_overlay:
        overlay_dir = os.path.join(output_dir, 'overlays')
        os.makedirs(overlay_dir, exist_ok=True)

    for i in range(len(filenames)):
        filename = filenames[i]
        base_name = os.path.splitext(filename)[0]
        original_size = original_sizes[i]  # (width, height)

        # Get predicted mask for choroid class (channel 1)
        pred_mask = predicted_masks[i, 1].cpu().numpy()
        pred_mask_binary = (pred_mask > threshold).astype(np.uint8) * 255

        # Ensure original_size is in correct format (width, height)
        if isinstance(original_size, (list, tuple)) and len(original_size) == 2:
            target_size = (int(original_size[0]), int(original_size[1]))  # (width, height)
        else:
            print(f"⚠️ Warning: Unexpected original_size format: {original_size}. Using fallback 224x224.")
            target_size = (224, 224)

        # Resize predicted mask to original image dimensions
        pred_mask_pil = Image.fromarray(pred_mask_binary.astype(np.uint8))
        pred_mask_resized = pred_mask_pil.resize(target_size, Image.NEAREST)
        mask_for_overlay = np.array(pred_mask_resized)

        # Save final binary mask
        mask_path = os.path.join(mask_dir, f'{base_name}_mask.png')
        pred_mask_resized.save(mask_path)
        print(f"💾 Saved segmentation mask: {mask_path}")

        if save_overlay:
            try:
                # CRITICAL FIX: Load ORIGINAL image directly from disk
                img_path = os.path.join(segment_dir, filename)
                original_image = Image.open(img_path).convert('RGB')
                
                # Handle size mismatch between recorded size and actual image
                if original_image.size != target_size:
                    print(f"⚠️ Size mismatch for {filename}: recorded {target_size} vs actual {original_image.size}. Resizing image.")
                    original_image = original_image.resize(target_size, Image.LANCZOS)
                
                # Create overlay using ACTUAL original pixels
                overlay = create_overlay_image(np.array(original_image), mask_for_overlay)
                overlay_path = os.path.join(overlay_dir, f'{base_name}_overlay.png')
                overlay.save(overlay_path)
                print(f"✅ Saved overlay: {overlay_path}")
                
            except Exception as e:
                print(f"❌ Error creating overlay for {filename}: {str(e)}")
                continue

def create_overlay_image(image_np, mask_np):
    """
    Create an overlay image with segmentation boundaries using original pixel data.
    FIXED: Proper boundary detection and drawing with bounds checking.
    """
    # Ensure we're working with clean copies in correct format
    overlay_image = image_np.copy().astype(np.uint8)
    binary_mask = (mask_np > 0).astype(np.uint8)  # Ensure binary mask
    
    # Find boundaries (column-wise processing)
    upper_boundaries, lower_boundaries = find_boundaries(binary_mask)
    
    # Draw boundaries with bounds checking
    H, W = overlay_image.shape[:2]
    for col in range(W):
        if col >= len(upper_boundaries) or col >= len(lower_boundaries):
            continue
            
        # Upper boundary (red) - check bounds
        if upper_boundaries[col] is not None:
            row = int(upper_boundaries[col])
            if 0 <= row < H and 0 <= col < W:
                overlay_image[row, col] = [255, 0, 0]  # Red
        
        # Lower boundary (green) - check bounds
        if lower_boundaries[col] is not None:
            row = int(lower_boundaries[col])
            if 0 <= row < H and 0 <= col < W:
                overlay_image[row, col] = [0, 255, 0]  # Green
    
    return Image.fromarray(overlay_image)

def find_boundaries(mask):
    """
    Find upper and lower boundaries for each column in a binary mask.
    FIXED: Proper column-wise processing with robust handling of empty columns.
    """
    upper_boundaries = []
    lower_boundaries = []
    
    # Process each column (x-coordinate)
    for col in range(mask.shape[1]):
        col_data = mask[:, col]
        non_zero_indices = np.where(col_data > 0)[0]
        
        if len(non_zero_indices) > 0:
            upper_boundaries.append(non_zero_indices[0])    # First non-zero from top
            lower_boundaries.append(non_zero_indices[-1])   # Last non-zero from top
        else:
            upper_boundaries.append(None)
            lower_boundaries.append(None)
            
    return upper_boundaries, lower_boundaries

# Training and Evaluation Functions
def train_fold(
    train_loader,
    valid_loader,
    test_loader,
    model,
    criterion,
    optimizer,
    device,
    num_epochs,
    scaler,
    threshold,
    save_best_path,
    scheduler=None,
    early_stopping_patience=0,
    early_stopping_min_delta=0.0,
    aux_weight_d2=0.2,
    aux_weight_d3=0.1,
    edge_loss_weight=0.0,
    progressive_unfreeze_schedule=None,
    initial_freeze_blocks=0,
):
    # track best validation choroid-dice
    best_choroid = -1.0
    patience_counter = 0
    progressive_unfreeze_schedule = progressive_unfreeze_schedule or []
    current_freeze = int(initial_freeze_blocks)

    for epoch in range(num_epochs):
        # Optional progressive unfreezing (applies at epoch start, 1-based epochs).
        if progressive_unfreeze_schedule and hasattr(model, "encoder"):
            epoch_idx = epoch + 1
            target_freeze = current_freeze
            for start_ep, freeze_blocks in progressive_unfreeze_schedule:
                if epoch_idx >= start_ep:
                    target_freeze = int(freeze_blocks)
            if target_freeze != current_freeze:
                apply_encoder_freeze(model, target_freeze)
                current_freeze = target_freeze
                print(f"🔓 Progressive unfreeze update at epoch {epoch_idx}: freeze_encoder_blocks={current_freeze}")

        # — training pass —
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                logits = get_main_logits(outputs)
                if torch.isnan(logits).any():
                    print("Warning: Model outputs contain nan values, skipping batch")
                    continue
                loss = compute_total_loss(
                    outputs,
                    masks,
                    criterion,
                    aux_weight_d2=aux_weight_d2,
                    aux_weight_d3=aux_weight_d3,
                    edge_loss_weight=edge_loss_weight,
                )
                if torch.isnan(loss):
                    print("Warning: Loss is nan, skipping batch")
                    continue
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, LR: {current_lr:.2e}")

        # — validation pass —
        model.eval()
        dice_choroid_scores = []
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                logits = get_main_logits(outputs)
                _, dch = dice_score(logits, masks)
                dice_choroid_scores.append(dch)

        avg_choroid = np.mean(dice_choroid_scores)
        print(f"  → Validation Choroid Dice: {avg_choroid:.4f}")
        if scheduler is not None:
            scheduler.step(avg_choroid)

        # — save best checkpoint —
        improved = avg_choroid > (best_choroid + early_stopping_min_delta)
        if improved:
            best_choroid = avg_choroid
            patience_counter = 0
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_choroid_dice': best_choroid
            }
            os.makedirs(os.path.dirname(save_best_path) or ".", exist_ok=True)
            torch.save(ckpt, save_best_path)
            print(f"💾 New best checkpoint saved: {save_best_path}")
        else:
            patience_counter += 1
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                print(f"⏹ Early stopping at epoch {epoch + 1} (patience={early_stopping_patience})")
                break

    # Load the best checkpoint before final metrics so reported results match saved model.
    if os.path.exists(save_best_path):
        ckpt = torch.load(save_best_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        state_dict = adapt_patch_embed_in_channels(state_dict, target_in_channels=config["num_channels"])
        load_state_dict_flexible(model, state_dict)
        print(f"🔁 Loaded best checkpoint for final evaluation: {save_best_path}")

    # — after all epochs, run full evaluation on validation & test sets as before —

    model.eval()
    dice_combined_scores, dice_choroid_scores = [], []
    upper_signed_errors, upper_unsigned_errors, lower_signed_errors, lower_unsigned_errors = [], [], [], []
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            logits = get_main_logits(outputs)
            dice_combined, dice_choroid = dice_score(logits, masks)
            dice_combined_scores.append(dice_combined)
            dice_choroid_scores.append(dice_choroid)
            predicted_masks = torch.softmax(logits, dim=1).cpu().numpy()
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
            logits = get_main_logits(model(images))
            outputs = torch.softmax(logits, dim=1)
            masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
            plot_boundaries(images, masks, outputs, threshold)
            break

    return avg_dice_combined, avg_dice_choroid, avg_upper_signed_error, avg_upper_unsigned_error, avg_lower_signed_error, avg_lower_unsigned_error


def append_ablation_results(csv_path, preset, save_best_path, metrics):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ablation_preset": preset,
        "use_attention": int(ABLATION_PRESETS[preset]["use_attention"]),
        "use_fusion": int(ABLATION_PRESETS[preset]["use_fusion"]),
        "use_upconvs": int(ABLATION_PRESETS[preset]["use_upconvs"]),
        "best_checkpoint": save_best_path,
        "dice_combined": float(metrics[0]),
        "dice_choroid": float(metrics[1]),
        "upper_signed_um": float(metrics[2]),
        "upper_unsigned_um": float(metrics[3]),
        "lower_signed_um": float(metrics[4]),
        "lower_unsigned_um": float(metrics[5]),
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"📝 Ablation metrics appended to: {csv_path}")


class OCTSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, image_size, transform=None, image_mode='rgb'):
        self.image_size = image_size
        self.transform = transform
        self.image_mode = str(image_mode).strip().lower()
        if self.image_mode not in {'rgb', 'gray'}:
            raise ValueError(f"image_mode must be 'rgb' or 'gray', got {image_mode}")
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
        pil_mode = 'L' if self.image_mode == 'gray' else 'RGB'
        image = Image.open(img_path).convert(pil_mode)
        original_size = image.size  # This returns (width, height)
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path), original_size

if __name__ == '__main__':
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print(
        f"🧪 Ablation preset: {config['ablation_preset']} | "
        f"attention={config['use_attention']} fusion={config['use_fusion']} upconvs={config['use_upconvs']}"
    )
    
    model = AttentionUNetViT(config).to(device)
    
    # Initial freeze policy for ViT blocks (supports progressive unfreezing later).
    apply_encoder_freeze(model, args.freeze_encoder_blocks)
    print(f"🔒 Frozen encoder blocks: first {args.freeze_encoder_blocks}")
    if args.enable_encoder_adapters:
        trainable_adapter_params = sum(p.numel() for p in model.encoder_adapters.parameters() if p.requires_grad)
        print(
            f"🧩 Adapter training enabled | rank={args.adapter_rank} "
            f"blocks={model.adapter_block_indices} trainable_params={trainable_adapter_params}"
        )

    # ===== SEGMENTATION MODE (No masks needed) =====
    if args.segment_dir:
        print(f"🔍 Running segmentation on images in: {args.segment_dir}")
        
        # Create segmentation dataset
        segment_dataset = OCTSegmentationDataset(
            args.segment_dir,
            args.image_size,
            transform=image_only_transform,
            image_mode=args.image_mode,
        )
        
        # Custom collate function to handle original sizes as tuples
        def custom_collate(batch):
            images = torch.stack([item[0] for item in batch])
            filenames = [item[1] for item in batch]
            # Keep original_sizes as list of tuples instead of converting to tensor
            original_sizes = [item[2] for item in batch]
            return images, filenames, original_sizes
        
        segment_loader = DataLoader(
            segment_dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True,
            collate_fn=custom_collate
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
                
            state_dict = adapt_patch_embed_in_channels(state_dict, target_in_channels=config["num_channels"])
            load_state_dict_flexible(model, state_dict)
            print(f"✅ Loaded weights from {config['retfound_weights_path']}")
        else:
            print(f"❌ Error: Weight file not found at {config['retfound_weights_path']}")
            sys.exit(1)
        
        model.eval()
        
        # CRITICAL FIX: Only collect necessary data (no image tensors)
        all_filenames = []
        all_original_sizes = []
        all_predicted_masks = []
        
        print(f"⚙️ Processing {len(segment_dataset)} images...")
        with torch.no_grad():
            for batch_idx, (images, filenames, original_sizes) in enumerate(segment_loader):
                images = images.to(device)
                outputs = model(images)
                logits = get_main_logits(outputs)
                predicted_masks = torch.softmax(logits, dim=1)
                
                # Collect only what we need for saving
                all_filenames.extend(filenames)
                all_original_sizes.extend(original_sizes)
                all_predicted_masks.append(predicted_masks.cpu())
                
                print(f"✅ Processed batch {batch_idx + 1}/{len(segment_loader)} ({len(filenames)} images)")
        
        # Concatenate all batches of masks
        all_predicted_masks = torch.cat(all_predicted_masks, dim=0)
        
        # Save results using ORIGINAL images from disk
        save_segmentation_results(
            all_filenames, 
            all_original_sizes, 
            all_predicted_masks, 
            args.output_dir, 
            args.segment_dir,  # Pass original image directory
            args.save_overlay, 
            args.threshold
        )
        
        print(f"\n🎉 Segmentation completed successfully!")
        print(f"   ► Results saved to: {args.output_dir}")
        print(f"   ► Masks directory: {os.path.join(args.output_dir, 'masks')}")
        if args.save_overlay:
            print(f"   ► Overlays directory: {os.path.join(args.output_dir, 'overlays')}")
        sys.exit(0)

    # ===== TRAINING MODE =====
    assert args.image_dir and args.mask_dir, (
        '❌ --image_dir and --mask_dir are required for training. '
        'Use --segment_dir for inference without masks.'
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
            
        state_dict = adapt_patch_embed_in_channels(state_dict, target_in_channels=config["num_channels"])
        load_state_dict_flexible(model, state_dict)
        print(f"✅ Loaded weights from {config['retfound_weights_path']}")
    
    # Setup training components
    class_weights = parse_class_weights(args.class_weights, num_classes=2)
    criterion = TverskyLoss(
        alpha=0.7,
        beta=0.3,
        smooth=1e-6,
        class_weights=class_weights,
        mode=args.loss_mode,
        choroid_class_idx=1,
    ).to(device)
    # Keep all params in optimizer so progressively-unfrozen blocks start updating immediately.
    trainable_params = model.parameters()
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )
    save_best_path = args.save_best_path or os.path.join("weights", f"best_rfa_unet_{args.ablation_preset}.pth")
    
    if args.val_image_dir and args.val_mask_dir:
        # Explicit fold-based split: train on image_dir/mask_dir, validate on val_* dirs.
        train_dataset = OCTDataset(
            args.image_dir,
            args.mask_dir,
            args.image_size,
            transform=train_transform,
            num_classes=2,
            image_mode=args.image_mode,
        )
        valid_dataset = OCTDataset(
            args.val_image_dir,
            args.val_mask_dir,
            args.image_size,
            transform=val_test_transform,
            num_classes=2,
            image_mode=args.image_mode,
        )
        # Keep a test loader object for visualization code-path compatibility.
        test_dataset = valid_dataset
        split_policy_msg = "explicit-fold (train dirs + val dirs)"
    else:
        # Prepare datasets (split by indices to keep train/eval transforms isolated).
        full_dataset_eval = OCTDataset(
            args.image_dir,
            args.mask_dir,
            args.image_size,
            transform=val_test_transform,
            num_classes=2,
            image_mode=args.image_mode,
        )
        if not (0.0 < args.test_split < 1.0):
            raise ValueError(f"--test_split must be in (0,1), got {args.test_split}")
        if not (0.0 < args.val_split_in_trainval < 1.0):
            raise ValueError(f"--val_split_in_trainval must be in (0,1), got {args.val_split_in_trainval}")

        total_size = len(full_dataset_eval)
        test_size = int(round(total_size * args.test_split))
        test_size = max(1, min(test_size, total_size - 2))
        trainval_size = total_size - test_size
        valid_size = int(round(trainval_size * args.val_split_in_trainval))
        valid_size = max(1, min(valid_size, trainval_size - 1))
        train_size = trainval_size - valid_size
        split_generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(total_size, generator=split_generator).tolist()
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]

        full_dataset_train = OCTDataset(
            args.image_dir,
            args.mask_dir,
            args.image_size,
            transform=train_transform,
            num_classes=2,
            image_mode=args.image_mode,
        )
        train_dataset = Subset(full_dataset_train, train_indices)
        valid_dataset = Subset(full_dataset_eval, valid_indices)
        test_dataset = Subset(full_dataset_eval, test_indices)
        split_policy_msg = f"random split: test_split={args.test_split}, val_split_in_trainval={args.val_split_in_trainval}"
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"\n📊 Dataset Information:")
    print(f"   • Training samples: {len(train_dataset)}")
    print(f"   • Validation samples: {len(valid_dataset)}")
    print(f"   • Test samples: {len(test_dataset)}")
    print(f"   • Split policy: {split_policy_msg}")
    print(f"   • Image mode/channels: {args.image_mode}/{config['num_channels']}")
    print(f"   • LR: {args.lr}")
    print(f"   • Optimizer: {args.optimizer}")
    print(f"   • Weight decay: {args.weight_decay}")
    print(f"   • Grad clip: {args.grad_clip}")
    print(f"   • Class weights: {class_weights}")
    print(f"   • Loss mode: {args.loss_mode}")
    print(f"   • Edge loss weight: {args.edge_loss_weight}")
    print(f"   • Decoder architecture: {args.decoder_arch}")
    if args.decoder_arch == "transunet":
        print(
            f"   • TransUNet hybrid skips (R50): {args.transunet_hybrid_skips} "
            f"(pretrained={args.transunet_r50_pretrained}, freeze={args.transunet_freeze_r50})"
        )
    if args.decoder_arch == "fundu":
        print(
            f"   • Fundu mask-decoder layers: {args.fundu_decoder_layers} "
            f"(Segmenter-style mask transformer + progressive post-adapter)"
        )
    print(f"   • Multiscale skip mode: {args.multiscale_skip_mode}")
    print(
        f"   • Post-refine head: {args.use_post_refine} "
        f"(depth={args.post_refine_depth}, ch={args.post_refine_channels})"
    )
    print(f"   • Shallow stem fusion: {args.use_shallow_stem_fusion}")
    print(f"   • Deep supervision: {args.deep_supervision} (w_d2={args.aux_weight_d2}, w_d3={args.aux_weight_d3})")
    print(f"   • ImageNet normalization: {args.normalize_imagenet}")
    print(
        f"   • Pre-adapter: {args.use_pre_adapter} "
        f"(target={args.pre_adapter_target_size}, hidden={args.pre_adapter_hidden_channels}, "
        f"depth={args.pre_adapter_depth}, norm={args.pre_adapter_norm}, mode={args.pre_adapter_mode}, "
        f"res_scale_init={args.pre_adapter_residual_scale_init}, "
        f"rgb_scalars={args.pre_adapter_rgb_scalars})"
    )
    print(
        f"   • OCT intensity aug: {args.oct_intensity_aug} "
        f"(b={args.oct_brightness_jitter}, c={args.oct_contrast_jitter}, "
        f"gamma=[{args.oct_gamma_min},{args.oct_gamma_max}], "
        f"speckle_p/std=[{args.oct_speckle_prob},{args.oct_speckle_std_min}-{args.oct_speckle_std_max}], "
        f"noise_p/std=[{args.oct_noise_prob},{args.oct_noise_std_min}-{args.oct_noise_std_max}], "
        f"blur_p/smax=[{args.oct_blur_prob},{args.oct_blur_sigma_max}])"
    )
    print(
        f"   • Augment: crop={args.augment_random_resized_crop}(scale_min={args.augment_scale_min}), "
        f"hflip_p={args.augment_hflip_prob}, rot={args.augment_rotation_deg}, jitter={args.augment_color_jitter}"
    )
    print(f"   • Scheduler: {args.scheduler}")
    if args.scheduler == 'plateau':
        print(f"   • Scheduler factor/patience: {args.scheduler_factor}/{args.scheduler_patience}")
    print(f"   • Freeze encoder blocks: {args.freeze_encoder_blocks}")
    print(f"   • Progressive unfreeze schedule: '{args.progressive_unfreeze_schedule}'")
    print(
        f"   • Encoder adapters: {args.enable_encoder_adapters} "
        f"(rank={args.adapter_rank}, blocks='{args.adapter_blocks}', "
        f"dropout={args.adapter_dropout}, init_scale={args.adapter_init_scale}, "
        f"placement={args.adapter_placement})"
    )
    print(f"   • Early stopping patience/min_delta: {args.early_stopping_patience}/{args.early_stopping_min_delta}")
    print(f"   • Best checkpoint path: {save_best_path}")
    print(f"   • Starting training for {args.num_epochs} epochs...")
    
    # Train the model
    dice_combined, dice_choroid, upper_signed, upper_unsigned, lower_signed, lower_unsigned = train_fold(
        train_loader,
        valid_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        device,
        args.num_epochs,
        scaler,
        args.threshold,
        save_best_path,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        aux_weight_d2=args.aux_weight_d2,
        aux_weight_d3=args.aux_weight_d3,
        edge_loss_weight=args.edge_loss_weight,
        progressive_unfreeze_schedule=PROGRESSIVE_UNFREEZE_PLAN,
        initial_freeze_blocks=args.freeze_encoder_blocks,
    )
    
    # Print final results
    print(f"\n🏁 Final Validation Results:")
    print(f"   • Dice (Combined): {dice_combined:.4f}")
    print(f"   • Dice (Choroid): {dice_choroid:.4f}")
    print(f"   • Upper Boundary Signed Error: {upper_signed:.2f} μm")
    print(f"   • Upper Boundary Unsigned Error: {upper_unsigned:.2f} μm")
    print(f"   • Lower Boundary Signed Error: {lower_signed:.2f} μm")
    print(f"   • Lower Boundary Unsigned Error: {lower_unsigned:.2f} μm")

    if args.ablation_results_csv:
        append_ablation_results(
            csv_path=args.ablation_results_csv,
            preset=args.ablation_preset,
            save_best_path=save_best_path,
            metrics=(dice_combined, dice_choroid, upper_signed, upper_unsigned, lower_signed, lower_unsigned),
        )
