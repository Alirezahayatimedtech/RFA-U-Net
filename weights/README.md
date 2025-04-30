Pre-trained Weights
This directory is a placeholder for the pre-trained RETFound weights (RETFound_oct_weights.pth) required to run the RFA-U-Net model.
Instructions

Download the weights from this link (replace with actual link).
Place the RETFound_oct_weights.pth file in this weights/ directory.
Update the retfound_weights_path in src/retfound.py if necessary:config = {
    ...
    "retfound_weights_path": "weights/RETFound_oct_weights.pth"
}



Notes

The weights are sourced from the RETFound_MAE repository or a fine-tuned version for OCT segmentation.
Ensure the weights file is compatible with the RETFound_mae model (patch size 16, embed_dim 1024, depth 24, num_heads 16).


