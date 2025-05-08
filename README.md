# RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

**RFA-U-Net** is a deep learning model designed to segment the **choroid** in Optical Coherence Tomography (OCT) images. It combines a **Vision Transformer (ViT)** encoder pre-trained with **RETFound** weights and an **Attention U-Net** decoder. The model is optimized using **Tversky** and **Dice** loss functions and evaluated through **Dice scores** and **boundary error metrics** (in micrometers).

---

## 🚀 Key Features

* **Encoder**: RETFound Vision Transformer for advanced feature extraction
* **Decoder**: Attention U-Net with skip connections for precise segmentation
* **Loss Functions**: Tversky and Dice loss for handling class imbalance
* **Evaluation Metrics**: Dice score, boundary errors (signed/unsigned, in micrometers)
* **Visualization**: Boundary overlays with true and predicted masks

---

## 📰 Related Research

We recommend reading the article ["RFA-U-Net: Choroid Segmentation in OCT with RETFound Attention U-Net"](https://www.medrxiv.org/content/10.1101/2025.05.03.25326923v1), which provides in-depth validation and benchmarks for the RFA-U-Net model. The study presents:

* Comprehensive evaluation on multiple OCT datasets.
* Improved boundary delineation of the choroid, especially in diseased eyes.
* A robust comparison with state-of-the-art segmentation networks.
* Insights into the impact of ViT-based encoders in biomedical imaging.

> **Citation:**
> Alireza Hayati, Roya Arian, Narges Sa. "RFA-U-Net: Choroid Segmentation in OCT with RETFound Attention U-Net." *medRxiv* (2025). DOI: 10.1101/2025.05.03.25326923.

---

## 📁 Repository Structure

```
RFA-U-Net/
├── src/
│   ├── rfa-u-net.py          # Main model training/inference script
│   ├── models_vit.py         # ViT implementation (adapted from RETFound_MAE)
│   └── util/
│       └── pos_embed.py      # Positional embedding utility
├── examples/
│   ├── sample_image.jpg
│   ├── sample_mask.png
│   ├── sample_output.png
│   └── visualization.ipynb   # Notebook for visualization
├── weights/
│   └── README.md             # Pre-trained weights info
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚙️ Requirements

* Python 3.8 or higher
* PyTorch 1.9.0 or higher
* NVIDIA GPU (recommended)
* Dependencies listed in `requirements.txt`

---

## 🛠 Installation

```bash
git clone https://github.com/Alirezahayatimedtech/RFA-U-Net.git
cd RFA-U-Net
pip install -r requirements.txt
```

> **Note:** Pre-trained weights (`rfa_unet_best.pth`) are auto-downloaded via `gdown` if you specify `--weights_type rfa-unet`.

---

## 📦 Pre-trained Weights

* `RETFound_oct_weights.pth`: Download from [RETFound\_MAE](https://github.com/rmaphoh/RETFound_MAE) and place in `weights/`.
* `rfa_unet_best.pth`: Auto-downloaded or provided manually using `--weights_path`.

See `weights/README.md` for details.

---

## 📂 Dataset Format

```
data/
├── images/
│   ├── image1.jpg
└── masks/
    ├── image1_mask.png
```

* **Images**: RGB OCT images
* **Masks**: Binary masks

*Note: Dataset is not provided; users must supply their own data.*

---

## 🧠 Usage Examples

### Train from scratch (no pre-trained weights)

```bash
python src/rfa-u-net.py --image_dir data/images --mask_dir data/masks --weights_type none
```

### Train using RETFound weights

```bash
python src/rfa-u-net.py --image_dir data/images --mask_dir data/masks --weights_type retfound
```

### Fine-tune using RFA-U-Net weights

```bash
python src/rfa-u-net.py --image_dir data/images --mask_dir data/masks --weights_type rfa-unet
```

---

## 🔍 Inference & Visualization

* Run `src/rfa-u-net.py` with appropriate flags.
* Open `examples/visualization.ipynb` to visualize the segmentation results with boundary overlays.

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

**

---

## 🔧 Advanced Settings

Example with custom parameters:

```bash
python src/rfa-u-net.py \
  --image_dir my_data/images \
  --mask_dir my_data/masks \
  --weights_type rfa-unet \
  --weights_path /custom/path/rfa_unet_best.pth \
  --image_size 256 \
  --num_epochs 30 \
  --batch_size 4 \
  --pixel_size_micrometers 12.5
```

---

## 🙏 Acknowledgments

* ViT and RETFound code from [RETFound\_MAE](https://github.com/rmaphoh/RETFound_MAE)
* Inspired by the U-Net and Vision Transformer architectures

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

---

## 📬 Contact

For issues or questions:

* GitHub Issues
* Email: 📧 [alirezahayati17@yahoo.com](mailto:alirezahayati17@yahoo.com)

---

## 🤝 Contributors

* **Alireza Hayati** – Lead developer
* **Roya Arian** – Training support
* **Narges Sa** – Training support

---

## 🔖 Tags

\#DeepLearning #MedicalImaging #OCT #ChoroidSegmentation #VisionTransformer #UNet #RETFound #PyTorch #BiomedicalImaging #MedicalAI #ImageSegmentation
