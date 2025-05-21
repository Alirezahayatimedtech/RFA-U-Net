import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

class OCTDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, image_size, transform=None, num_classes=2):
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.image_size  = image_size
        self.transform   = transform
        self.num_classes = num_classes

        # Accept .jpg, .jpeg, .png, .tif, .tiff
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        self.images = [
            fname for fname in os.listdir(self.image_dir)
            if fname.lower().endswith(exts)
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base, _  = os.path.splitext(img_name)

        # Find a matching mask file with same base and one of the exts
        mask_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        mask_name = None
        for me in mask_exts:
            candidate = base + me
            if os.path.exists(os.path.join(self.mask_dir, candidate)):
                mask_name = candidate
                break
        if mask_name is None:
            raise FileNotFoundError(f"No mask found for base '{base}' in {self.mask_dir}")

        # Load image and mask
        img_path  = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        image     = Image.open(img_path).convert('RGB')
        mask_pil  = Image.open(mask_path).convert('L')

        # Map unwanted labels → {0,1}
        mask_np = np.array(mask_pil)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_np = np.where(mask_np == 249, 1, 0).astype(np.uint8)

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)

        # Resize mask with nearest-neighbor
        mask_pil = Image.fromarray(mask_np)
        mask_pil = mask_pil.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_np  = np.array(mask_pil)

        # One-hot encode mask
        mask_tensor = torch.from_numpy(mask_np).long()
        mask_onehot = F.one_hot(mask_tensor, num_classes=self.num_classes)
        mask_onehot = mask_onehot.permute(2, 0, 1).float()

        return image, mask_onehot
