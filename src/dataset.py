import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def image_paths(directory):
    return sorted(str(path) for path in Path(directory).iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def index_by_stem(paths):
    indexed = {}
    for path in paths:
        stem = Path(path).stem
        if stem in indexed:
            raise ValueError(f"Duplicate image or mask stem: {indexed[stem]} and {path}")
        indexed[stem] = path
    return indexed


class OCTDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, image_size, transform=None, num_classes=2,
                 image_mode='rgb', strict_pairs=False):
        self.image_size = image_size
        self.transform = transform
        self.num_classes = num_classes
        self.image_mode = str(image_mode).strip().lower()
        if self.image_mode not in {'rgb', 'gray'}:
            raise ValueError(f"image_mode must be 'rgb' or 'gray', got {image_mode}")

        images = index_by_stem(image_paths(image_dir))
        masks = index_by_stem(image_paths(mask_dir))
        self.image_paths = []
        self.mask_paths = []
        unpaired = []
        for stem, image in sorted(images.items()):
            mask = masks.get(stem, masks.get(stem + '_mask'))
            if mask is None:
                unpaired.append(os.path.basename(image))
                continue
            self.image_paths.append(image)
            self.mask_paths.append(mask)
        if not self.image_paths or (strict_pairs and unpaired):
            raise ValueError(
                f"Expected paired OCT images and masks in {image_dir} and {mask_dir}. "
                f"Found {len(self.image_paths)} pairs and {len(unpaired)} unpaired images "
                f"({', '.join(unpaired[:5])})."
            )
        if unpaired:
            print(f"Warning: skipping {len(unpaired)} training images without a matching mask.")

    def __len__(self):
        return len(self.image_paths)

    def load_pair(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = load_scan(img_path, self.image_mode)
        with Image.open(mask_path) as source:
            mask = source.convert('L')
        if image.size != mask.size:
            raise ValueError(f"Image and mask dimensions differ for {img_path}: {image.size} vs {mask.size}.")
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.load_pair(idx)
        if self.transform:
            try:
                return self.transform(image, mask)
            except TypeError:
                # Support legacy image-only transforms.
                image = self.transform(image)

        mask = mask.resize((self.image_size, self.image_size), resample=Image.Resampling.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask) > 0).long()
        mask_onehot = F.one_hot(mask_tensor, num_classes=self.num_classes).permute(2, 0, 1).float()
        return image, mask_onehot


def load_scan(path, image_mode):
    with Image.open(path) as source:
        if source.mode.startswith("I;16") or source.mode in {"I", "F"}:
            pixels = np.asarray(source, dtype=np.float32)
            finite = np.isfinite(pixels)
            if not finite.any():
                raise ValueError(f"No finite intensity values in {path}.")
            low, high = np.percentile(pixels[finite], [0.5, 99.5])
            if high <= low:
                low, high = pixels[finite].min(), pixels[finite].max()
            scaled = np.zeros(pixels.shape, dtype=np.uint8)
            if high > low:
                scaled[finite] = (np.clip((pixels[finite] - low) / (high - low), 0, 1) * 255).round()
            return Image.fromarray(scaled).convert('L' if image_mode == 'gray' else 'RGB')
        return source.convert('L' if image_mode == 'gray' else 'RGB')
