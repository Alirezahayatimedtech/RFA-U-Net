
---

## 📦 Pre-trained Weights

Pre-trained weights are **optional**. You can:

* Train the model **from scratch** using `--weights_type none`
* Use **pre-trained RETFound weights** for ViT initialization (`--weights_type retfound`)
* Use **RFA-U-Net pre-trained weights** for fine-tuning or inference (`--weights_type rfa-unet`)

---

### 🔗 Available Weights

* **`RETFound_oct_weights.pth`**
  Pre-trained RETFound Vision Transformer weights, used to initialize the encoder when training RFA-U-Net from scratch.
  📥 *Must be downloaded manually* from the [RETFound\_MAE repository](https://github.com/rmaphoh/RETFound_MAE) and placed in the `weights/` directory.

* **`rfa_unet_best.pth`**
  Best-performing RFA-U-Net model weights trained on an OCT dataset.
  ✅ Automatically downloaded to the `weights/` directory when `--weights_type rfa-unet` is used (unless a custom path is specified via `--weights_path`).

---

### 📁 Weight Details

#### `RETFound_oct_weights.pth`

* **Source**: [RETFound\_MAE GitHub Repo](https://github.com/rmaphoh/RETFound_MAE)
* **Usage**: Required when running the script with `--weights_type retfound`
* **Setup**: Must be manually downloaded and saved in the `weights/` directory

#### `rfa_unet_best.pth`

* **Source**: Hosted on Google Drive
* **Usage**: Automatically downloaded on first run with `--weights_type rfa-unet`
* **Setup**: You can override the default location using `--weights_path`.
  To re-download, delete the existing file and rerun the script.

---

### 📝 Notes

* **To use RETFound weights**, manually place `RETFound_oct_weights.pth` in the `weights/` directory before running:

  ```bash
  python src/rfa-u-net.py --image_dir path/to/data/images --mask_dir path/to/data/masks --weights_type retfound
  ```

* **To use RFA-U-Net weights**, run with:

  ```bash
  python src/rfa-u-net.py --image_dir path/to/data/images --mask_dir path/to/data/masks --weights_type rfa-unet
  ```

* **To use a custom path for weights**, specify it explicitly:

  ```bash
  python src/rfa-u-net.py --image_dir path/to/images --mask_dir path/to/masks --weights_type rfa-unet --weights_path /custom/path/rfa_unet_best.pth
  ```

* **To train from scratch**, use:

  ```bash
  python src/rfa-u-net.py --image_dir path/to/images --mask_dir path/to/masks --weights_type none
  ```

* **Boundary error evaluation**:
  Pixel size (in micrometers) can be adjusted with:

  ```bash
  --pixel_size_micrometers 10.35
  ```

  *(Default is 10.35 μm. Adjust this based on your OCT image resolution.)*

---

