RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation
RFA-U-Net is a deep learning model designed for segmenting the choroid in Optical Coherence Tomography (OCT) images. It combines a Vision Transformer (ViT) encoder pre-trained with RETFound weights and an Attention U-Net decoder. The model is optimized using Tversky and Dice loss functions and evaluated with metrics such as Dice scores and boundary errors (in micrometers).
Features

Encoder: Pre-trained RETFound Vision Transformer for robust feature extraction.
Decoder: Attention U-Net with skip connections for precise choroid segmentation.
Loss Functions: Tversky and Dice loss to handle class imbalance.
Evaluation Metrics: Dice score, signed/unsigned boundary errors (in micrometers).
Visualization: Boundary visualization with true and predicted masks.

Repository Structure
RFA-U-Net/
├── src/
│   ├── retfound.py              # Main model, training, and evaluation script
│   └── util/
│       └── pos_embed.py         # Utility for positional embedding interpolation
├── examples/
│   ├── sample_image.jpg         # Placeholder for sample OCT image
│   ├── sample_mask.png          # Placeholder for sample mask
│   ├── visualization.ipynb      # Jupyter notebook for visualizing segmentation results
│   └── sample_output.png        # Placeholder for sample segmentation output
├── weights/
│   └── README.md                # Instructions for downloading RETFound weights
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt

Prerequisites

Python 3.8+
PyTorch 1.9.0+
NVIDIA GPU (recommended for training)
Dependencies listed in requirements.txt

Installation

Clone the repository:git clone https://github.com/Alirezahayatimedtech/RFA-U-Net.git
cd RFA-U-Net


Install dependencies:pip install -r requirements.txt


Download the pre-trained RETFound weights (RETFound_oct_weights.pth) from this Google Drive link and place them in the weights/ directory. See weights/README.md for details.
Clone the RETFound_MAE repository for model definitions:git clone https://github.com/rmaphoh/RETFound_MAE.git



Dataset
The model expects OCT images and corresponding choroid masks in the following structure:
data/
├── images/
│   ├── image1.jpg
│   ├── image2.tif
│   └── ...
└── masks/
    ├── image1_mask.png
    ├── image2_mask.png
    └── ...


Images: RGB OCT images (3 channels).
Masks: Binary masks (single channel) for the choroid region.
Note: Due to privacy concerns, the dataset is not included. Users must provide their own OCT dataset with the above structure.

Usage
Training the Model

Prepare the dataset: Ensure your dataset follows the structure above and is placed in a data/ directory.
Update the configuration: Modify the paths in src/retfound.py to point to your dataset:image_dir = 'path/to/your/data/images'
mask_dir = 'path/to/your/data/masks'

Verify the weights path:config = {
    "image_size": 224,
    "hidden_dim": 1024,
    "patch_size": 16,
    "num_channels": 3,
    "num_classes": 2,
    "retfound_weights_path": "weights/RETFound_oct_weights.pth"
}


Train the model:python src/retfound.py

This will train the model for 50 epochs, evaluate on the validation set, and visualize the first batch of the test set using plot_boundaries.

Visualizing Results
To visualize segmentation results on a specific image:

Open examples/visualization.ipynb in Jupyter Notebook or Google Colab.
Update the notebook to load your image and mask (replace the dummy data).
Run the notebook to generate visualizations, which include:
Original image
True mask
Predicted mask
Boundaries overlay (true upper: red, true lower: green, predicted upper: blue, predicted lower: yellow)



Results

Validation Dice Score: ~0.92 (average across folds)
Test Dice Score: ~0.91
Boundary Errors:
Upper Signed Error: ~5.2 μm
Upper Unsigned Error: ~6.8 μm
Lower Signed Error: ~4.9 μm
Lower Unsigned Error: ~7.1 μm



Example Output

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

RETFound weights and Vision Transformer implementation from RETFound_MAE.
Inspired by Vision Transformer and U-Net architectures.

Contact
For questions or issues, please open an issue on GitHub or contact [alirezahayati17@yahoo.com].
