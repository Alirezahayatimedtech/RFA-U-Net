
## 📦 Pre-trained Weights

The script (`src/rfa-u-net.py`) will automatically download or use the appropriate pre-trained weights based on the `--weights_type` argument.

### 🎯 Available Weights

* **`RETFound_oct_weights.pth`**
  Pre-trained RETFound weights used to initialize the Vision Transformer encoder for **training from scratch**.
  📥 **Manual download required** from the [RETFound\_MAE repository](https://github.com/rmaphoh/RETFound_MAE).

* **`rfa_unet_best.pth`**
  Best pre-trained weights for RFA-U-Net, trained on an OCT dataset.
  ✅ Automatically downloaded on first use when running with `--weights_type rfa-unet`.
  Suitable for **inference** or **fine-tuning**.

---

## 🔍 Weight Details

### `RETFound_oct_weights.pth`

* **Source**: [RETFound\_MAE GitHub Repository](https://github.com/rmaphoh/RETFound_MAE)
* **Usage**: Required when using `--weights_type retfound` to train RFA-U-Net from scratch
* **Setup**: Must be manually downloaded and placed in the `weights/` directory

### `rfa_unet_best.pth`

* **Source**: Hosted on Google Drive
* **Usage**: Used when `--weights_type rfa-unet` is specified for inference or continued training
* **Setup**: Automatically downloaded if not already present (uses `gdown`)

---

## 📝 Notes

* To use **RETFound** weights:
  You must manually download `RETFound_oct_weights.pth` from the [RETFound\_MAE repository](https://github.com/rmaphoh/RETFound_MAE) and place it in this directory **before** running the script with `--weights_type retfound`.

* To use **RFA-U-Net** pre-trained weights (`rfa_unet_best.pth`):
  They are **automatically downloaded** if missing when running the script with `--weights_type rfa-unet`.

* To **force re-download** of `rfa_unet_best.pth`:
  Simply delete the file from the `weights/` directory and rerun the script.

* Ensure the `gdown` package is installed (already included in `requirements.txt`) for automatic downloading.

* When **evaluating boundary errors**, you can specify the pixel size in micrometers using:

  ```bash
  --pixel_size_micrometers 10.35  # default is 10.35 μm
  ```

  Adjust this based on your dataset’s imaging specs.

