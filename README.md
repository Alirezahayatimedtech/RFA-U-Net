

# RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

**RFA-U-Net** is a deep learning model designed to segment the **choroid** in Optical Coherence Tomography (OCT) images. It integrates a **Vision Transformer (ViT)** encoder pre-trained with **RETFound** weights and an **Attention U-Net** decoder. The model is trained using **Tversky** and **Dice** losses and evaluated with **Dice scores** and **boundary error metrics** (in micrometers).

---

## 🚀 Features

* **Encoder**: RETFound Vision Transformer for robust feature extraction
* **Decoder**: Attention U-Net with skip connections for precise choroid segmentation
* **Loss Functions**: Tversky and Dice loss for class imbalance
* **Evaluation Metrics**: Dice score, boundary errors (signed/unsigned, in micrometers)
* **Visualization**: Boundary overlays with true and predicted masks

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

## ⚙️ Prerequisites

* Python 3.8 or higher
* PyTorch 1.9.0 or higher
* NVIDIA GPU (recommended)
* All dependencies listed in `requirements.txt`

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/Alirezahayatimedtech/RFA-U-Net.git
cd RFA-U-Net
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note**: The script uses `gdown` to automatically download pre-trained weights (`rfa_unet_best.pth`) from Google Drive when `--weights_type rfa-unet` is specified.

---

## 📦 Pre-trained Weights

Pre-trained weights are optional. You may:

* Train from scratch
* Use **RETFound** weights (`RETFound_oct_weights.pth`) for ViT initialization
* Use pre-trained **RFA-U-Net** weights (`rfa_unet_best.pth`) for fine-tuning or inference

### Available Weights

* `RETFound_oct_weights.pth`: Must be downloaded manually from the [RETFound\_MAE repo](https://github.com/rmaphoh/RETFound_MAE) and placed in the `weights/` directory.
* `rfa_unet_best.pth`: Automatically downloaded the first time you run with `--weights_type rfa-unet`. Can also be manually provided via `--weights_path`.

See `weights/README.md` for details.

---

## 📂 Dataset Structure

The model expects data in the following format:

```
data/
├── images/
│   ├── image1.jpg
│   └── ...
└── masks/
    ├── image1_mask.png
    └── ...
```

* **Images**: RGB OCT images (3 channels)
* **Masks**: Binary masks (1 channel)

> ⚠️ Due to privacy concerns, the dataset is not included. You must provide your own data in the above format.

---

## 🧠 Usage

### 🔧 Training the Model

Ensure your dataset is in the correct structure, then run one of the following:

**Train from scratch (random initialization):**

```bash
python src/rfa-u-net.py --image_dir data/images --mask_dir data/masks --weights_type none
```

**Train from scratch using RETFound weights:**

```bash
python src/rfa-u-net.py --image_dir data/images --mask_dir data/masks --weights_type retfound
```

**Fine-tune or infer using pre-trained RFA-U-Net (auto-downloads weights):**

```bash
python src/rfa-u-net.py --image_dir data/images --mask_dir data/masks --weights_type rfa-unet
```

**Using a custom weights path:**

```bash
python src/rfa-u-net.py --image_dir data/images --mask_dir data/masks --weights_type rfa-unet --weights_path /path/to/rfa_unet_best.pth
```

---

### 🛠 Command-Line Arguments

```
--image_dir             Path to input images (default: data/images)
--mask_dir              Path to masks (default: data/masks)
--weights_type          [none | retfound | rfa-unet] (default: none)
--weights_path          Custom path to weights (optional)
--image_size            Input image size (default: 224)
--num_epochs            Number of training epochs (default: 50)
--batch_size            Batch size (default: 8)
--pixel_size_micrometers  Pixel size for error calc (default: 10.35)
```

**Example with custom settings:**

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

## 🔍 Inference with Pre-trained Weights

1. Run the script with `--weights_type rfa-unet` or provide a custom path
2. Open and run `examples/visualization.ipynb`
3. Modify the notebook to load your image, mask, and weight file
4. Visualize:

* Original image
* Ground truth mask
* Predicted mask
* Boundary overlays (red, green, blue, yellow)

---

## 📊 Results

* **Test Dice Score**: \~0.95
* **Boundary Errors**:

  * Upper Signed Error: \~-0.89 μm
  * Upper Unsigned Error: \~6.04 μm
  * Lower Signed Error: \~1.05 μm
  * Lower Unsigned Error: \~21.4 μm

---

## 🖼 Example Output

*(Insert images here if available)*

---

## 🧪 Notes

* **Mixed Precision Training**: Uses `torch.cuda.amp` for faster GPU training
* **Numerical Stability**: Uses `GradScaler` to avoid NaNs. Check data integrity and learning rate (`1e-4` default) if you encounter issues.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

* ViT and RETFound implementation (`models_vit.py`) adapted from [RETFound\_MAE](https://github.com/rmaphoh/RETFound_MAE)
* Inspired by Vision Transformer and U-Net architectures

---

## 📬 Contact

For questions or issues, open an issue on GitHub or email
📧 **[alirezahayati17@yahoo.com](mailto:alirezahayati17@yahoo.com)**

---

## 🤝 Contributors

Thanks to the following people for their contributions:

* **Alireza Hayati** – Initial development
* **Roya Arian** – Training support
* **Narges Sa** – Training support

---

## 📌 Tags

\#DeepLearning #MedicalImaging #OCT #ChoroidSegmentation #VisionTransformer
\#UNet #RETFound #ImageSegmentation #SemanticSegmentation #MedicalAI
\#PyTorch #BiomedicalImaging #AIinHealthcare #ComputerVision

---
