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

The `weights/` directory contains the best pre-trained weights for RFA-U-Net:

* `rfa_unet_weights.pth`: Best pre-trained weights for RFA-U-Net, which can be used directly for inference or further fine-tuning.

See `weights/README.md` for instructions on downloading these weights.



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

Prepare the dataset: Ensure your dataset follows the structure above and is placed in a `data/` directory.

Train the model by specifying the dataset paths and other configurations via command-line arguments:


```
python src/retfound.py --image_dir path/to/your/data/images --mask_dir path/to/your/data/masks --weights_path weights/rfa_unet_best.pth
```
This will train the model for 50 epochs, evaluate on the validation set, and visualize the first batch of the test set using plot_boundaries.

### Available Command-Line Arguments:





--image_dir: Path to the directory containing OCT images (default: data/images)



--mask_dir: Path to the directory containing mask images (default: data/masks)



--weights_path: Path to the pre-trained weights file (default: weights/rfa_unet_best.pth)



--image_size: Input image size (default: 224)



--num_epochs: Number of training epochs (default: 50)



--batch_size: Batch size for training (default: 8)

Example with custom settings:
```
python src/retfound.py --image_dir my_data/images --mask_dir my_data/masks --weights_path weights/rfa_unet_best.pth --image_size 256 --num_epochs 30 --batch_size 4
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


- **Test Dice Score**: ~0.95  
- **Boundary Errors**:  
  - Upper Signed Error: ~-0.89 μm 
  - Upper Unsigned Error: ~6.04 μm 
  - Lower Signed Error: ~1.05 μm  
  - Lower Unsigned Error: ~21.4 μm 

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
---
## 🤝 Contributors

Thanks to the following people for their contributions:

- (https://github.com/alirezahayatimedtech) – Initial development
- (https://github.com/royaarian101) – Helped with training experiments
- (https://github.com/narges-sa) – Helped with training experiments

## 📌 Tags / Hashtags

#DeepLearning #MedicalImaging #OCT #ChoroidSegmentation #VisionTransformer  
#UNet #RETFound #ComputerVision #MedicalAI #SemanticSegmentation  
#PyTorch #AIinHealthcare #ImageSegmentation #BiomedicalImaging

