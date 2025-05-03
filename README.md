
# RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

**RFA-U-Net** is a deep learning model designed to segment the **choroid** in Optical Coherence Tomography (OCT) images. It combines a **Vision Transformer (ViT)** encoder pre-trained with **RETFound weights** and an **Attention U-Net** decoder. The model is optimized using **Tversky** and **Dice** loss functions and evaluated using metrics such as **Dice scores** and **boundary errors** (in micrometers).

---

## 🚀 Features

* **Encoder**: RETFound Vision Transformer for robust feature extraction
* **Decoder**: Attention U-Net with skip connections for precise segmentation
* **Loss Functions**: Tversky and Dice loss to handle class imbalance
* **Evaluation Metrics**: Dice score, signed/unsigned boundary errors (μm)
* **Visualization**: Boundary overlays of true and predicted masks

---

## 📁 Repository Structure

```
RFA-U-Net/
├── src/
│   ├── rfa-u-net.py          # Main model, training, and evaluation script
│   └── util/
│       └── pos_embed.py      # Positional embedding interpolation utility
├── examples/
│   ├── sample_image.jpg      # Example OCT image
│   ├── sample_mask.png       # Corresponding mask
│   ├── sample_output.png     # Sample segmentation result
│   └── visualization.ipynb   # Notebook for visualizing predictions
├── weights/
│   ├── rfa_unet_best.pth     # Best pre-trained model
│   └── README.md             # Info about pre-trained weights
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

---

## 📦 Pre-trained Weights

The pre-trained model weights (`rfa_unet_best.pth`) will be **automatically downloaded** to the `weights/` directory the first time you run the script (`src/rfa-u-net.py`). These are the best weights saved after training on an OCT dataset.

See [`weights/README.md`](weights/README.md) for more details.

---

## 📂 Dataset Structure

The model expects data in the following format:

```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.tif
│   └── ...
└── masks/
    ├── image1_mask.png
    ├── image2_mask.png
    └── ...
```

* **Images**: RGB OCT images (3 channels)
* **Masks**: Binary masks (1 channel) representing the choroid

> **Note**: Due to privacy concerns, the dataset is not included. Users must provide their own OCT dataset.

---

## 🧠 Training the Model

Prepare your dataset in the `data/` directory as shown above.

Then run the script with:

```bash
python src/rfa-u-net.py --image_dir path/to/your/data/images \
                        --mask_dir path/to/your/data/masks \
                        --weights_path weights/rfa_unet_best.pth
```

The script will:

* Automatically download weights (if not already present)
* Train the model for 50 epochs
* Evaluate on the validation set
* Visualize the first batch of predictions using `plot_boundaries`

---

### 🔧 Available Command-Line Arguments

```
--image_dir       Path to image folder (default: data/images)
--mask_dir        Path to mask folder (default: data/masks)
--weights_path    Path to weights file (default: weights/rfa_unet_best.pth)
--image_size      Input size (default: 224)
--num_epochs      Number of training epochs (default: 50)
--batch_size      Batch size (default: 8)
```

Example with custom settings:

```bash
python src/rfa-u-net.py --image_dir my_data/images \
                        --mask_dir my_data/masks \
                        --weights_path weights/rfa_unet_best.pth \
                        --image_size 256 \
                        --num_epochs 30 \
                        --batch_size 4
```

---

## 🧪 Inference with Pre-trained Weights

To use the pre-trained RFA-U-Net model for inference:

1. Ensure `rfa_unet_best.pth` is downloaded to `weights/` (it will download automatically if missing).
2. Open `examples/visualization.ipynb`.
3. Update the notebook to load your own image/mask and the weights file.
4. Run the notebook to see visual results.

---

## 📊 Visualizing Results

The notebook will generate:

* ✅ Original image
* ✅ Ground truth mask
* ✅ Predicted mask
* ✅ Boundary overlays:

  * Red = True upper
  * Green = True lower
  * Blue = Predicted upper
  * Yellow = Predicted lower

---

## 📈 Results

* **Test Dice Score**: \~0.95
* **Boundary Errors**:

  * Upper Signed Error: \~-0.89 μm
  * Upper Unsigned Error: \~6.04 μm
  * Lower Signed Error: \~1.05 μm
  * Lower Unsigned Error: \~21.4 μm

---

## 🖼 Example Output

*(Insert your image here if available)*

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* RETFound weights and Vision Transformer implementation from [RETFound\_MAE](https://github.com/rmaphoh/RETFound_MAE)
* Inspired by Vision Transformer and U-Net architectures

---

## 📬 Contact

For questions or issues, open an issue on GitHub or contact
📧 **[alirezahayati17@yahoo.com](mailto:alirezahayati17@yahoo.com)**

---

## 🤝 Contributors

Thanks to the following contributors:

* **Alireza Hayati** – Initial development
* **Roya Arian** – Helped with training experiments
* **Narges Sa** – Helped with training experiments

---

## 📌 Tags / Hashtags

\#DeepLearning #MedicalImaging #OCT #ChoroidSegmentation #VisionTransformer
\#UNet #RETFound #ComputerVision #MedicalAI #SemanticSegmentation
\#PyTorch #AIinHealthcare #ImageSegmentation #BiomedicalImaging

---

Let me know if you'd like this in a `.md` file or want to embed a Colab badge!
