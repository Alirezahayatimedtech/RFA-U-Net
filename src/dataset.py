import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

class OCTDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, image_size, transform=None, num_classes=2, image_mode='rgb'):
        self.image_size  = image_size
        self.transform   = transform
        self.num_classes = num_classes
        self.image_mode = str(image_mode).strip().lower()
        if self.image_mode not in {'rgb', 'gray'}:
            raise ValueError(f"image_mode must be 'rgb' or 'gray', got {image_mode}")

        # Supported extensions
        exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

        # Build mapping from basename → filename for images
        image_map = {}
        for fname in os.listdir(image_dir):
            base, ext = os.path.splitext(fname)
            if ext.lower() in exts:
                image_map[base] = fname

        # Build mapping from basename → filename for masks.
        # Supports both exact stem matches and "<stem>_mask" naming.
        mask_map = {}
        for fname in os.listdir(mask_dir):
            base, ext = os.path.splitext(fname)
            if ext.lower() in exts:
                # Exact key
                if base not in mask_map:
                    mask_map[base] = fname
                # Common pattern: image stem + "_mask"
                if base.endswith('_mask'):
                    base_stripped = base[:-5]
                    if base_stripped and base_stripped not in mask_map:
                        mask_map[base_stripped] = fname

        # Keep only basenames present in both
        common_bases = sorted(set(image_map.keys()) & set(mask_map.keys()))

        # Prepare full paths
        self.image_paths = [os.path.join(image_dir, image_map[b]) for b in common_bases]
        self.mask_paths  = [os.path.join(mask_dir,  mask_map[b]) for b in common_bases]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # paths
        img_path  = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # load
        pil_mode = 'L' if self.image_mode == 'gray' else 'RGB'
        image    = Image.open(img_path).convert(pil_mode)
        mask_pil = Image.open(mask_path).convert('L')
        if self.transform:
            # Preferred path: joint image/mask transform.
            try:
                image_t, mask_t = self.transform(image, mask_pil)
                return image_t, mask_t
            except TypeError:
                # Backward-compatible path: image-only transform.
                image = self.transform(image)

        # Fallback path (no transform or image-only transform): resize + one-hot mask here.
        mask_np = (np.array(mask_pil) > 0).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np).resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_np = np.array(mask_pil)
        mask_tensor = torch.from_numpy(mask_np).long()
        mask_onehot = F.one_hot(mask_tensor, num_classes=self.num_classes).permute(2, 0, 1).float()
        return image, mask_onehot
