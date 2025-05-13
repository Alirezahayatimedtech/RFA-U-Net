
# RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

**RFA-U-Net** is a deep learning model designed to segment the **choroid** in Optical Coherence Tomography (OCT) images. It combines a **Vision Transformer (ViT)** encoder pre-trained with **RETFound** weights and an **Attention U-Net** decoder. The model is optimized using **Tversky** and **Dice** losses and evaluated via **Dice scores** and **boundary error metrics** (in micrometers).

---

## 🚀 Key Features

- **Encoder**: RETFound Vision Transformer for advanced feature extraction  
- **Decoder**: Attention U-Net with skip connections for precise segmentation  
- **Losses**: Tversky and Dice loss to handle class imbalance  
- **Metrics**: Dice score, boundary errors (signed/unsigned, in μm)  
- **Visualization**: Boundary overlays with true vs. predicted masks  

---

## 📰 Related Research

Hayati *et al.* “RFA-U-Net: Choroid Segmentation in OCT with RETFound Attention U-Net,” *medRxiv* (2025).  
[DOI: 10.1101/2025.05.03.25326923](https://www.medrxiv.org/content/10.1101/2025.05.03.25326923v1)

---

## 📁 Repository Structure

```

RFA-U-Net/
├── src/
│   ├── rfa-u-net.py          # Main model script (train, infer, external test)
│   ├── models\_vit.py         # RETFound-based ViT implementation
│   └── util/
│       └── pos\_embed.py      # Positional-embedding interpolation
├── examples/
│   ├── sample\_image.jpg
│   ├── sample\_mask.png
│   ├── sample\_output.png
│   └── visualization.ipynb   # Notebook for qualitative results
├── weights/
│   └── README.md             # Info on downloading pre‐trained weights
├── README.md                 # This file
├── LICENSE
└── requirements.txt

```

---

## ⚙️ Requirements

- Python 3.8+  
- PyTorch 1.9.0+  
- NVIDIA GPU (strongly recommended)  
- Dependencies (from `requirements.txt`):
```

torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
optuna>=2.10.0
timm>=0.4.12
tqdm>=4.61.0
Pillow>=8.0.0
gdown>=4.7.1

````

---

## 🛠 Installation

```bash
git clone https://github.com/Alirezahayatimedtech/RFA-U-Net.git
cd RFA-U-Net
pip install -r requirements.txt
````

> **Note:** If you choose `--weights_type rfa-unet`, the script will auto-download `rfa_unet_best.pth` via `gdown`.

---

## 📂 Dataset Format

```
data/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── masks/
    ├── image1_mask.png
    └── image2_mask.png
```

* **Images**: RGB OCT images
* **Masks**: Corresponding binary masks (use suffix `_mask.png`)
* **Note**: You must supply your own data; no dataset is included here.

---

## 🎯 External-Data-Only Testing

Evaluate a pre-trained model on your own images/masks without any training:

```bash
python src/rfa-u-net.py \
  --test_only \
  --test_image_dir path/to/external/images \
  --test_mask_dir  path/to/external/masks \
  --weights_type rfa-unet \
  --threshold 0.5 \
  --pixel_size_micrometers 12.5
```

* `--test_only`
  Run in inference-only mode, skipping all training.
* `--test_image_dir`
  Directory of OCT images to evaluate.
* `--test_mask_dir`
  Directory of ground-truth masks.
* `--weights_type {none|retfound|rfa-unet}`
  Choose which pretrained weights to load.
* `--threshold`
  Binarization threshold for predicted masks (default `0.5`).
* `--pixel_size_micrometers`
  Pixel size in μm for computing boundary errors (default `10.35`).

**Output**:

```
Choroid Dice on external data: 0.9523
Upper signed/unsigned error: -0.85/5.90 μm
Lower signed/unsigned error:  1.12/20.50 μm
```

---

## 🧠 Training & Inference

### Train from scratch (no pre-trained)

```bash
python src/rfa-u-net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type none \
  --num_epochs 30 \
  --batch_size 4
```

### Fine-tune using RETFound weights

```bash
python src/rfa-u-net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type retfound \
  --num_epochs 20 \
  --batch_size 8
```

### Fine-tune using RFA-U-Net weights

```bash
python src/rfa-u-net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type rfa-unet \
  --num_epochs 15 \
  --batch_size 8
```

---

## 📊 Results Snapshot

| Metric                    | Value   |
| ------------------------- | ------- |
| Dice Score (Test)         | \~0.95  |
| Upper Signed Error (μm)   | \~-0.89 |
| Upper Unsigned Error (μm) | \~6.04  |
| Lower Signed Error (μm)   | \~1.05  |
| Lower Unsigned Error (μm) | \~21.4  |

---

## 🖼 Example Outputs

![Sample segmentation overlay](examples/sample_output.png)

---

## 🔧 Advanced Settings

Customize image size, epochs, batch, etc.:

```bash
python src/rfa-u-net.py \
  --image_dir my_data/images \
  --mask_dir my_data/masks \
  --weights_type rfa-unet \
  --weights_path /path/to/rfa_unet_best.pth \
  --image_size 256 \
  --num_epochs 30 \
  --batch_size 4 \
  --pixel_size_micrometers 12.5 \
  --threshold 0.6
```

---

## 🙏 Acknowledgments

* ViT & RETFound code from [RETFound\_MAE](https://github.com/rmaphoh/RETFound_MAE)
* Inspired by U-Net and Transformer architectures

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

---

## 📬 Contact

For issues or questions, please open a GitHub Issue or email: [alirezahayati17@yahoo.com](mailto:alirezahayati17@yahoo.com)

---


---

## 🤝 Contributors

* **Alireza Hayati** – Lead developer
* **Roya Arian** – Training support & Mentor
* **Narges Sa** – Training support & Mentor

---

## 🔖 Tags

\#DeepLearning #MedicalImaging #OCT #ChoroidSegmentation #VisionTransformer #UNet #RETFound #PyTorch #BiomedicalImaging #MedicalAI #ImageSegmentation
