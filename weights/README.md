# Model weights

- `--weights_type none` initializes a new model.
- `--weights_type retfound` initializes the encoder from the RETFound OCT checkpoint. Set `--weights_path` to a local copy, place it at `weights/RETFound_oct_weights.pth`, or use the configured Hugging Face download.
- `--weights_type rfa-unet` loads a trained segmentation checkpoint. Set `--weights_path` to a local copy or let the runner download the released checkpoint to `weights/best_rfa_unet.pth`.

The released RFA-U-Net checkpoint uses RGB, ImageNet normalization and the original attention decoder. The runner restores these automatically. Checkpoints produced by the current trainer include architecture and preprocessing settings. For an older custom checkpoint without settings, pass the flags used during training.

```bash
python src/rfa_u_net.py --segment_dir data/images --weights_type rfa-unet --save_overlay
```

[RETFound weights and access instructions](https://github.com/rmaphoh/RETFound_MAE) · [Released RFA-U-Net weights](https://drive.google.com/uc?export=download&id=1zDEdAmNwNK8I-QEa6fqL5E3WjDn7Z-__)

## OIMHS choroid refiner

- [oimhs_refiner.safetensors](oimhs_refiner.safetensors) is the standalone model loaded by `--weights_type refiner`.
- [oimhs_refiner_hybrid.safetensors](oimhs_refiner_hybrid.safetensors) adds a learned correction to Choroidalyzer v1.0. Use `--weights_type refiner --weights_path weights/oimhs_refiner_hybrid.safetensors --prior_repo /path/to/Choroidalyzer`. A local v1.0 checkpoint can be supplied with `--prior_weights`.

Both checkpoints carry the settings selected on OIMHS validation patients. See [the benchmark](../benchmarks/REFINER.md) for the evaluation scope.

| Checkpoint | SHA-256 |
| --- | --- |
| Standalone | `bd4f06e45bd8d2acdfd6267d72649e3c36ce5c762884081af0d16c2217ccb320` |
| Hybrid | `d6a941e3503eeb8fb1946da8f0d60326b8b2d6e951f039e139c0a0d93958ef9d` |
