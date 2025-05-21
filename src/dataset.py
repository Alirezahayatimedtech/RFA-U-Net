import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

class OCTDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, image_size, transform=None, num_classes=2):
        self.image_size  = image_size
        self.transform   = transform
        self.num_classes = num_classes

        # Supported extensions
        exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

        # Build mapping from basename → filename for images
        image_map = {}
        for fname in os.listdir(image_dir):
            base, ext = os.path.splitext(fname)
            if ext.lower() in exts:
                image_map[base] = fname

        # Build mapping from basename → filename for masks
        mask_map = {}
        for fname in os.listdir(mask_dir):
            base, ext = os.path.splitext(fname)
            if ext.lower() in exts:
                mask_map[base] = fname

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
        image    = Image.open(img_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')

        # binarize mask on any non-zero pixel
        mask_np = np.array(mask_pil)
        mask_np = (mask_np > 0).astype(np.uint8)

        # apply image transform if given
        if self.transform:
            image = self.transform(image)

        # resize mask
        mask_pil = Image.fromarray(mask_np)
        mask_pil = mask_pil.resize(
            (self.image_size, self.image_size),
            resample=Image.NEAREST
        )
        mask_np = np.array(mask_pil)

        # one-hot encode
        mask_tensor = torch.from_numpy(mask_np).long()
        mask_onehot = F.one_hot(mask_tensor, num_classes=self.num_classes)
        mask_onehot = mask_onehot.permute(2, 0, 1).float()

        return image, mask_onehot


