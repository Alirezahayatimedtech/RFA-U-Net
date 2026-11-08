"""Compact OCT segmentation with an optional frozen Choroidalyzer prior."""

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_model


GRID = (160, 224)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(inplace=True),
    )


class ChoroidRefiner(nn.Module):
    def __init__(self, with_prior):
        super().__init__()
        self.with_prior = with_prior
        self.enc1 = conv_block(2 if with_prior else 1, 12)
        self.enc2 = conv_block(12, 24)
        self.enc3 = conv_block(24, 36)
        self.center = conv_block(36, 48)
        self.dec3 = conv_block(48 + 36, 36)
        self.dec2 = conv_block(36 + 24, 24)
        self.dec1 = conv_block(24 + 12, 12)
        self.head = nn.Conv2d(12, 2, 1)
        if with_prior:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, image, prior=None):
        if self.with_prior:
            if prior is None:
                raise ValueError("This refiner requires a Choroidalyzer logit prior.")
            image = torch.cat((image, prior.clamp(-6, 6) / 6), dim=1)

        first = self.enc1(image)
        second = self.enc2(F.avg_pool2d(first, 2))
        third = self.enc3(F.avg_pool2d(second, 2))
        center = self.center(F.avg_pool2d(third, 2))
        third = self.dec3(torch.cat((resize(center, third.shape[-2:]), third), dim=1))
        second = self.dec2(torch.cat((resize(third, second.shape[-2:]), second), dim=1))
        first = self.dec1(torch.cat((resize(second, first.shape[-2:]), first), dim=1))
        return self.head(first)


def resize(tensor, shape):
    return F.interpolate(tensor, size=shape, mode="bilinear", align_corners=False, antialias=True)


def normalized_grid(pixels):
    return resize(pixels.float() / 127.5 - 1, GRID)


def make_prior(repository, weights, device):
    if repository is not None:
        sys.path.insert(0, str(Path(repository).resolve()))
    try:
        from choroidalyze import Choroidalyzer
    except ImportError as exc:
        raise ImportError("Install Choroidalyzer or provide its checkout with --prior_repo.") from exc
    return Choroidalyzer(device=str(device), local_weights_path=str(weights) if weights else None)


@torch.inference_mode()
def prior_logits(prior, pixels, device):
    # The upstream depth-seven U-Net expects dimensions divisible by 128.
    height, width = pixels.shape
    pad_h, pad_w = (-height) % 128, (-width) % 128
    image = pixels.to(device).float().view(1, 1, height, width) / 127.5 - 1
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else image
    return prior.model(padded)[0, 0, :height, :width].float()


def checkpoint_settings(path):
    with safe_open(path, framework="pt") as checkpoint:
        metadata = checkpoint.metadata() or {}
    prior = metadata.get("with_prior")
    if prior not in {"true", "false"}:
        raise ValueError("Refiner checkpoint requires with_prior metadata.")
    threshold = float(metadata.get("threshold", "0.5"))
    flip = metadata.get("flip", "false")
    postprocess = metadata.get("postprocess", "none")
    if not 0 < threshold < 1 or flip not in {"true", "false"} or postprocess not in {"none", "layer"}:
        raise ValueError("Invalid refiner checkpoint inference settings.")
    return prior == "true", threshold, flip == "true", postprocess


class NativeRefiner:
    def __init__(self, weights, with_prior, device, prior_repository=None, prior_weights=None):
        self.device = device
        self.model = ChoroidRefiner(with_prior).to(device)
        load_model(self.model, weights, device=str(device))
        self.prior = make_prior(prior_repository, prior_weights, device) if with_prior else None
        self.eval()

    def eval(self):
        self.model.eval()

    @torch.inference_mode()
    def predict(self, scan, flip=False, existing_prior=None):
        pixels = torch.from_numpy(np.array(scan.convert("L")))
        shape = pixels.shape
        image = normalized_grid(pixels.to(self.device)[None, None])
        prior = existing_prior
        if prior is None and self.prior is not None:
            prior = prior_logits(self.prior, pixels, self.device)
        grid_prior = resize(prior[None, None], GRID) if prior is not None else None
        correction = self.model(image, grid_prior)[:, :1]
        if flip:
            flipped_prior = grid_prior.flip(-1) if grid_prior is not None else None
            correction = (correction + self.model(image.flip(-1), flipped_prior)[:, :1].flip(-1)) / 2
        result = resize(correction, shape)[0, 0]
        if prior is not None:
            result = result + prior
        return result.sigmoid().cpu()
