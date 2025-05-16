# dataset.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class OCTDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size, transform=None, num_classes=2):
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.image_size  = image_size
        self.transform   = transform      # only for images
        self.num_classes = num_classes

        # only keep supported image files
        self.images = [
            fname for fname in os.listdir(self.image_dir)
            if fname.lower().endswith(('.jpg', '.JPG', '.tif'))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Handle different image extensions when looking for corresponding mask files
        if img_name.lower().endswith('.jpg'):
            mask_name = img_name[:-4] + '_mask.png'
        elif img_name.lower().endswith('.tif'):
            mask_name = img_name[:-4] + '_mask.png'
        else:
            raise ValueError(f"Unsupported image extension: {img_name}")

        img_path  = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir,  mask_name)

        # Load
        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')   # grayscale mask

        # Apply transform to image only
        if self.transform:
            image = self.transform(image)

        # Resize mask _after_ mapping with nearest-neighbor
        mask = mask.resize(
            (self.image_size, self.image_size),
            resample=Image.NEAREST
        )
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np > 0).astype(np.uint8)     # ensure binary

        # To one-hot tensor
        mask_tensor = torch.from_numpy(mask_np).long()                       # (H, W)
        mask_onehot = F.one_hot(mask_tensor, num_classes=self.num_classes)   # (H, W, C)
        mask_onehot = mask_onehot.permute(2, 0, 1).float()                   # (C, H, W)

        return image, mask_onehot
