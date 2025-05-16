# RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

**RFA-U-Net** is a deep learning model to segment the **choroid** in Optical Coherence Tomography (OCT) images. It uses a **RETFound**-pretrained Vision Transformer (ViT) encoder and an **Attention U-Net** decoder, trained with Tversky/Dice losses and evaluated via Dice scores and micrometer-scale boundary errors.

---

## 🚀 Key Features

- **Encoder**: RETFound MAE ViT backbone  
- **Decoder**: Attention U-Net with gated skip connections  
- **Losses**: Tversky + Dice to handle class imbalance  
- **Metrics**:  
  - Dice score (overall & choroid-only)  
  - Signed/Unsigned boundary errors (μm)  
- **Visualization**: Four-panel view & overlay boundary plots  

---

## 📁 Repo Structure

```

RFA-U-Net/
├── src/
│   ├── rfa-u-net.py          # Main train/infer script
│   ├── models\_vit.py         # RETFound ViT implementation
│   └── util/
│       └── pos\_embed.py      # Positional embedding utils
├── examples/
│   └── visualization.ipynb   # Qualitative demo notebook
├── weights/
│   └── README.md             # How to download pretrained weights
├── requirements.txt
└── README.md                 # You are here

````

---

## ⚙️ Requirements

- Python 3.8+  
- PyTorch 1.9+  
- NVIDIA GPU recommended  
- See `requirements.txt` for full list:
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

## 📂 Dataset Format

```
data/
├── images/
│   ├── sample1.jpg
│   └── sample2.jpg
└── masks/
    ├── sample1_mask.png
    └── sample2_mask.png
```

* **Images**: RGB OCT scans (`.jpg`)
* **Masks**: Binary masks (`.png`) matching `*_mask.png` suffix

---

## 🎯 External-Data-Only Testing

Evaluate pre-trained RFA-U-Net on your own data (no training):

```bash
python src/rfa-u-net.py \
  --test_only \
  --test_image_dir path/to/images \
  --test_mask_dir  path/to/masks \
  --weights_type rfa-unet \
  --threshold 0.5 \
  --pixel_size_micrometers 12.5
```

* `--test_only`: skip training
* `--threshold`: binarization cutoff (default 0.5)
* `--pixel_size_micrometers`: μm/pixel (default 10.35)

**Sample output**:

```
Choroid Dice on external data: 0.9523
Upper signed/unsigned error: -0.85/5.90 μm
Lower signed/unsigned error:  1.12/20.50 μm
```

---

## 🧠 Training & Inference

### 1. From scratch (no pre-training)

```bash
python src/rfa-u-net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type none \
  --num_epochs 30 \
  --batch_size 4
```

### 2. Fine-tune with RETFound

```bash
python src/rfa-u-net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type retfound \
  --num_epochs 20 \
  --batch_size 8
```

### 3. Fine-tune with RFA-U-Net

(Default downloads via gdown if not present)

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

| Metric                    | Value  |
| ------------------------- | ------ |
| Dice Score (choroid)      | \~0.95 |
| Upper Signed Error (μm)   | \~–0.9 |
| Upper Unsigned Error (μm) | \~6.0  |
| Lower Signed Error (μm)   | \~1.1  |
| Lower Unsigned Error (μm) | \~21.4 |

---

## 🖼 Example Outputs

![Boundary overlay example](examples/sample_output.png)

---

## 📬 Contact & Citation

Hayati *et al.* “RFA-U-Net: Choroid Segmentation in OCT with RETFound Attention U-Net,” *medRxiv* (2025).
DOI: 10.1101/2025.05.03.25326923

For issues or questions, open a GitHub Issue or email **[alirezahayati17@yahoo.com](mailto:alirezahayati17@yahoo.com)**.

---

##  Contributors

* **Alireza Hayati** – Lead developer
* **Roya Arian** – Mentor
* **Narges Sa** – Mentor

MIT License. See [LICENSE](LICENSE).


