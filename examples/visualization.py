# %% [markdown]
# # RFA-U-Net 4-Panel Boundary Visualization
#
# Plots (left→right):
# 1. Original image  
# 2. Ground-truth mask  
# 3. Predicted mask  
# 4. Overlayed boundaries (GT=red, Pred=yellow)

# %%
import os, sys
# ensure src/ is on Python path
sys.path.insert(0, os.path.abspath("src"))

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from rfa_u_net import AttentionUNetViT, OCTDataset, val_test_transform

# %%
# 1) Model config & load
config = {
    "image_size": 224,
    "hidden_dim": 1024,
    "patch_size": 16,
    "num_channels": 3,
    "num_classes": 2,
    "retfound_weights_path": "weights/best_rfa_unet.pth",
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionUNetViT(config).to(device)

# load checkpoint (allow full dict or just state_dict)
ckpt = torch.load(config["retfound_weights_path"], map_location=device, weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict, strict=False)
model.eval()
print(f"✅ Loaded weights from {config['retfound_weights_path']}")

# %%
def viz_four_panel(image, true_mask, pred_prob, thresh=0.5):
    """
    image:    torch.Tensor [1,3,H,W]
    true_mask/numpy [H,W] {0,1}
    pred_prob/numpy [H,W] probabilities
    """
    H, W = true_mask.shape
    # compute boundaries
    def boundaries(m):
        up, low = [], []
        for c in range(W):
            idxs = np.where(m[:, c] > 0)[0]
            if len(idxs):
                up.append(idxs[0])
                low.append(idxs[-1])
            else:
                up.append(np.nan)
                low.append(np.nan)
        return up, low

    img = image.cpu().squeeze(0).permute(1,2,0).numpy()
    pred_bin = (pred_prob > thresh).astype(np.uint8)
    tu, tl = boundaries(true_mask)
    pu, pl = boundaries(pred_bin)

    fig, ax = plt.subplots(1,4,figsize=(20,5))
    ax[0].imshow(img);                    ax[0].set_title("Original");  ax[0].axis("off")
    ax[1].imshow(true_mask, cmap="gray"); ax[1].set_title("GT Mask");   ax[1].axis("off")
    ax[2].imshow(pred_bin,  cmap="gray"); ax[2].set_title("Pred Mask"); ax[2].axis("off")
    ax[3].imshow(img)
    ax[3].plot(range(W), tu, ".",  color="red",    markersize=1)
    ax[3].plot(range(W), tl, ".",  color="red",    markersize=1)
    ax[3].plot(range(W), pu, ".",  color="yellow", markersize=1)
    ax[3].plot(range(W), pl, ".",  color="yellow", markersize=1)
    ax[3].set_title("Boundaries");       ax[3].axis("off")
    plt.tight_layout()
    plt.show()

# %%
# 2) Run inference & visualize first 20 samples
test_image_dir = "path/to/data"
test_mask_dir  = "path/to/mask"
ds = OCTDataset(test_image_dir, test_mask_dir, config["image_size"], transform=val_test_transform)

for i in range(min(20, len(ds))):
    img, gt_oh = ds[i]
    true_mask = gt_oh[1].numpy()  # only choroid channel

    with torch.no_grad():
        out  = model(img.unsqueeze(0).to(device))
        prob = torch.sigmoid(out)[0,1].cpu().numpy()

    viz_four_panel(img, true_mask, prob, thresh=0.5)
