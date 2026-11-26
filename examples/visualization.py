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
from rfa_u_net import JointSegmentationTransform, OCTDataset, checkpoint_model, parse_args, predict_choroid
from choroid import native_mask

# %%
args = parse_args(["--weights_type", "rfa-unet"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, config = checkpoint_model(args, device)
model.eval()
image_mode = "gray" if config["num_channels"] == 1 else "rgb"
transform = JointSegmentationTransform(
    config["image_size"], normalize=config["normalize_imagenet"], image_mode=image_mode,
)

# %%
def viz_four_panel(image, true_mask, pred_bin):
    """
    image: PIL image
    true_mask and pred_bin: native-resolution binary NumPy arrays
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

    img = np.asarray(image.convert("RGB"))
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
ds = OCTDataset(test_image_dir, test_mask_dir, config["image_size"], image_mode=image_mode)

for i in range(min(20, len(ds))):
    image, reference = ds.load_pair(i)
    with torch.inference_mode():
        probability = predict_choroid(model, transform(image).unsqueeze(0).to(device))[0].cpu()
    mask = native_mask(probability, image.size, 0.5)
    viz_four_panel(image, np.asarray(reference) > 0, mask)
