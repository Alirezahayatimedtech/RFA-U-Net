# RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

RFA-U-Net is a deep learning model designed for segmenting the choroid in Optical Coherence Tomography (OCT) images. It combines a Vision Transformer (ViT) encoder pre-trained with RETFound weights and an Attention U-Net decoder. The model is optimized using Tversky and Dice loss functions and evaluated with metrics such as Dice scores and boundary errors (in micrometers).

## Features

- **Encoder**: Pre-trained RETFound Vision Transformer for robust feature extraction  
- **Decoder**: Attention U-Net with skip connections for precise choroid segmentation  
- **Loss Functions**: Tversky and Dice loss to handle class imbalance  
- **Evaluation Metrics**: Dice score, signed/unsigned boundary errors (in micrometers)  
- **Visualization**: Boundary visualization with true and predicted masks  

## Repository Structure
```
RFA-U-Net/
├── src/
│   ├── retfound.py           # Main model, training, and evaluation script
│   └── util/
│       └── pos_embed.py      # Utility for positional embedding interpolation
├── examples/
│   ├── sample_image.jpg      # Placeholder for sample OCT image
│   ├── sample_mask.png       # Placeholder for sample mask
│   ├── visualization.ipynb   # Jupyter notebook for visualizing segmentation results
│   └── sample_output.png     # Placeholder for sample segmentation output
├── weights/
│   └── README.md             # Instructions for downloading RETFound weights
├── .gitignore                # Git ignore file
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## Prerequisites

- Python 3.8 or higher  
- PyTorch 1.9.0 or higher  
- NVIDIA GPU (recommended for training)  
- Dependencies listed in `requirements.txt`  

## Installation

Clone the repository:
```bash
git clone https://github.com/Alirezahayatimedtech/RFA-U-Net.git
cd RFA-U-Net
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the pre-trained RETFound weights (`RETFound_oct_weights.pth`) from this Google Drive link and place them in the `weights/` directory. See `weights/README.md` for details.

Clone the RETFound_MAE repository for model definitions:
```bash
git clone https://github.com/rmaphoh/RETFound_MAE.git
```

## Dataset

The model expects OCT images and corresponding choroid masks in the following structure:

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

- **Images**: RGB OCT images (3 channels)  
- **Masks**: Binary masks (single channel) for the choroid region  

**Note**: Due to privacy concerns, the dataset is not included. Users must provide their own OCT dataset with the above structure.

## Usage

### Training the Model

1. Prepare the dataset: Ensure your dataset follows the structure above and is placed in a `data/` directory.

2. Update the configuration in `src/retfound.py`:
```python
image_dir = 'path/to/your/data/images'
mask_dir = 'path/to/your/data/masks'
config = {
    "image_size": 224,
    "hidden_dim": 1024,
    "patch_size": 16,
    "num_channels": 3,
    "num_classes": 2,
    "retfound_weights_path": "weights/RETFound_oct_weights.pth"
}
```

3. Train the model:
```bash
python src/retfound.py
```

This will train the model for 50 epochs, evaluate on the validation set, and visualize the first batch of the test set using `plot_boundaries`.

### Visualizing Results

To visualize segmentation results on a specific image:

- Open `examples/visualization.ipynb` in Jupyter Notebook or Google Colab.
- Update the notebook to load your image and mask.
- Run the notebook to generate visualizations, including:
  - Original image  
  - True mask  
  - Predicted mask  
  - Boundaries overlay (true upper: red, true lower: green, predicted upper: blue, predicted lower: yellow)  

## Results

- **Validation Dice Score**: ~0.92 (average across folds)  
- **Test Dice Score**: ~0.91  
- **Boundary Errors**:  
  - Upper Signed Error: ~5.2 μm  
  - Upper Unsigned Error: ~6.8 μm  
  - Lower Signed Error: ~4.9 μm  
  - Lower Unsigned Error: ~7.1 μm  

## Example Output
![download](https://github.com/user-attachments/assets/f86b0c96-b683-47f1-9cf5-d159c40cc59a)

![download (3)](https://github.com/user-attachments/assets/b971b051-7c57-46e9-b1eb-a08bed3edcb6)



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RETFound weights and Vision Transformer implementation from [RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE)  
- Inspired by Vision Transformer and U-Net architectures  

## Contact

For questions or issues, please open an issue on GitHub or contact [alirezahayati17@yahoo.com].
```

---

