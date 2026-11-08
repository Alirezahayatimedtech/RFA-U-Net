# Model weights

- `--weights_type none` initializes a new model.
- `--weights_type retfound` initializes the encoder from the RETFound OCT checkpoint. Set `--weights_path` to a local copy, place it at `weights/RETFound_oct_weights.pth`, or use the configured Hugging Face download.
- `--weights_type rfa-unet` loads a trained segmentation checkpoint. Set `--weights_path` to a local copy or let the runner download the released checkpoint to `weights/best_rfa_unet.pth`.

The released RFA-U-Net checkpoint uses RGB, ImageNet normalization and the original attention decoder. The runner restores these automatically. Checkpoints produced by the current trainer include architecture and preprocessing settings. For an older custom checkpoint without settings, pass the flags used during training.

```bash
python src/rfa_u_net.py --segment_dir data/images --weights_type rfa-unet --save_overlay
```

[RETFound weights and access instructions](https://github.com/rmaphoh/RETFound_MAE) · [Released RFA-U-Net weights](https://drive.google.com/uc?export=download&id=1zDEdAmNwNK8I-QEa6fqL5E3WjDn7Z-__)
