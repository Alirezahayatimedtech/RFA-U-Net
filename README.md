# RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

**RFA-U-Net** is a deep learning model to segment the **choroid** in Optical Coherence Tomography (OCT) images. It uses a **RETFound**-pretrained Vision Transformer (ViT) encoder and an **Attention U-Net** decoder, trained with Tversky/Dice losses and evaluated via Dice scores and micrometer-scale boundary errors.

**Last updated: 2025-11-13**
🆕 New: **Segmentation-only mode** for unlabeled OCT images via `--segment_dir` (no masks needed).

---

## 🚀 Key Features

* **Encoder**: RETFound MAE ViT backbone
* **Decoder**: Attention U-Net with gated skip connections
* **Losses**: Tversky + Dice to handle class imbalance
* **Metrics**:

  * Dice score (overall & choroid-only)
  * Signed/Unsigned boundary errors (μm)
* **Visualization**: Four-panel view & overlay boundary plots
* **🆕 Segmentation-only inference**:

  * Run on **unlabeled OCT images** using `--segment_dir`
  * Saves **binary masks** and optional **boundary overlays** to disk

---

## 📁 Repo Structure

```text
RFA-U-Net/
├── src/
│   ├── rfa_u_net.py          # Main train / infer / segment-only script
│   ├── models_vit.py         # RETFound ViT implementation
│   └── util/
│       └── pos_embed.py      # Positional embedding utils
├── examples/
│   └── visualization.ipynb   # Qualitative demo notebook
├── weights/
│   └── README.md             # How to download pretrained weights
├── requirements.txt
└── README.md                 # You are here
```

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch 1.9+
* NVIDIA GPU recommended

See `requirements.txt` for full list:

```text
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
timm>=0.4.12
huggingface-hub>=0.14.1
gdown>=4.7.1
Pillow>=8.0.0
rarfile>=4.0
```

---

## 🛠 Installation

```bash
git clone https://github.com/Alirezahayatimedtech/RFA-U-Net.git
cd RFA-U-Net
pip install -r requirements.txt
```

### 🔐 Hugging Face Authentication

If you plan to load **RETFound** weights from HF Hub, set your token:

```bash
export HUGGINGFACE_HUB_TOKEN="hf_yourTokenHere"
# or
huggingface-cli login
```

---

## 📂 Dataset Format (for Training / Test with Masks)

```text
data/
├── images/
│   ├── sample1.jpg
│   └── sample2.jpg
└── masks/
    ├── sample1.png
    └── sample2.png
```

* **Images**: RGB OCT scans (`.jpg`, `.png`, `.tif`, …)
* **Masks**: Binary masks (`.png`) matching the image filenames (background/choroid as 2-channel one-hot internally)

---

## 🎯 External-Data-Only Testing (with Masks)

Evaluate pre-trained RFA-U-Net on your own **labeled** data (no further training):

```bash
python src/rfa_u_net.py \
  --test_only \
  --test_image_dir path/to/images \
  --test_mask_dir  path/to/masks \
  --weights_type rfa-unet \
  --threshold 0.5 \
  --pixel_size_micrometers 12.5
```

Key arguments:

* `--test_only`: skip training, run evaluation only
* `--threshold`: binarization cutoff (default `0.5`)
* `--pixel_size_micrometers`: μm/pixel for boundary error computation (default `10.35`)

**Sample output**:

```text
Choroid Dice on external data: 0.9523
Upper signed/unsigned error: -0.85/5.90 μm
Lower signed/unsigned error:  1.12/20.50 μm
```

---

## 🧠 Training & Inference (with Masks)

### 1️⃣ Train from scratch (no pre-training)

```bash
python src/rfa_u_net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type none \
  --num_epochs 30 \
  --batch_size 4
```

### 2️⃣ Fine-tune with RETFound backbone

```bash
python src/rfa_u_net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type retfound \
  --num_epochs 20 \
  --batch_size 8
```

This will:

* Download **RETFound** weights from HF Hub if missing
* Initialize the ViT encoder from RETFound
* Train the Attention U-Net decoder on your dataset

### 3️⃣ Fine-tune with pre-trained RFA-U-Net weights

(Default downloads via `gdown` if not present)

```bash
python src/rfa_u_net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type rfa-unet \
  --num_epochs 15 \
  --batch_size 8
```

---

## 🆕 Segmentation-Only Mode (No Masks Required)

This mode runs **pure inference** on *unlabeled* OCT images: it loads a trained model and saves segmentation masks (and optional overlays) to disk.

### CLI Example

```bash
python src/rfa_u_net.py \
  --segment_dir path/to/unlabeled_images \
  --weights_type rfa-unet \
  --weights_path weights/best_rfa_unet.pth \
  --output_dir segment_results \
  --batch_size 4 \
  --threshold 0.5 \
  --save_overlay
```

**Arguments specific to segmentation-only:**

* `--segment_dir` **(required for this mode)**

  * Path to a folder with OCT images (no masks needed)
  * Supported extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`
* `--weights_type`

  * Recommended: `rfa-unet` (use a trained RFA-U-Net checkpoint)
  * Can also use `retfound` if you trained a custom checkpoint
* `--weights_path`

  * Path to the `.pth` checkpoint (defaults to `weights/best_rfa_unet.pth`)
* `--output_dir`

  * Base directory where results are saved (default: `segment_results`)
* `--threshold`

  * Probability cutoff for binarizing choroid predictions (default: `0.5`)
* `--save_overlay`

  * If set, saves RGB overlays with choroid boundaries drawn on the original image

**What gets saved:**

* `output_dir/masks/`

  * `_mask.png` files: binary choroid masks resized to each image’s **original size**
* `output_dir/overlays/` (only if `--save_overlay` is used)

  * `_overlay.png` files: original image with choroid **boundaries** drawn in color

The script prints per-image debug info (prediction stats, resize shapes) and warns if a mask is all black (no pixels above threshold).

---

## 📥 Download Link for RFA-U-Net Pretrained Weights

### URL for pretrained weights (Google Drive link)

```text
RFA_UNET_WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=1zDEdAmNwNK8I-QEa6fqL5E3WjDn7Z-__"
```

The script will automatically download to:

```text
weights/best_rfa_unet.pth
```

if missing and `--weights_type rfa-unet` is used.

---

## 📊 Results Snapshot

| Metric                    | Value |
| ------------------------- | ----- |
| Dice Score (choroid)      | ~0.95 |
| Upper Signed Error (μm)   | ~–0.9 |
| Upper Unsigned Error (μm) | ~6.0  |
| Lower Signed Error (μm)   | ~1.1  |
| Lower Unsigned Error (μm) | ~21.4 |

---

## 🖼 Example Outputs

* **Four-panel visualization**:

  * Original OCT
  * Ground-truth mask
  * Predicted mask
  * Boundary overlay (true vs predicted; upper/lower boundaries)

![Boundary overlay example](examples/sample_output.png)

Segmentation-only mode also creates overlay PNGs directly from your unlabeled OCTs (if `--save_overlay` is enabled).

---

## 📝 Changelog

* **2025-11-13**

  * Added **segmentation-only mode** via `--segment_dir`
  * New output utilities:

    * Save per-image masks and overlays
    * Preserve original image resolution for saved masks
  * Updated README to document segmentation-only usage

---

## 📬 Contact & Citation

Hayati *et al.* “RFA-U-Net: Choroid Segmentation in OCT with RETFound Attention U-Net,” *medRxiv* (2025).
DOI: `10.1101/2025.05.03.25326923`

For issues or questions, open a GitHub Issue or email **[alirezahayati17@yahoo.com](mailto:alirezahayati17@yahoo.com)**.

---

## 👨‍💻 Contributors

* **Alireza Hayati** – Lead developer
* **Roya Arian** – Mentor
* **Narges Sa** – Mentor

MIT License. See [LICENSE](LICENSE).
