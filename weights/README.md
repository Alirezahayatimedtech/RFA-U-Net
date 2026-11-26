# RFA-U-Net A06 General v1

[Download RFA-U-Net-A06-General-v1.pth](https://drive.google.com/file/d/16LOpbw0lVi7JyehSJcQr9M7Q_uZjrlXP/view) · 1.36 GB

The single-model checkpoint includes the trained RETFound encoder, OCT pre-adapter, attention decoder, and settings for 224 × 224 RGB inference. The default CLI uses threshold 0.50. It downloads and verifies the file on first use.

```bash
python src/rfa_u_net.py --segment_dir path/to/oct/images --weights_type rfa-unet
```

The CLI writes native-resolution masks to `segment_results/masks/`. For an existing local copy, pass `--weights_path path/to/RFA-U-Net-A06-General-v1.pth`.

SHA-256

```text
4465ea4183e4a78c0f7eae518f6d41a00e046c14f30d2fa384c513ef983ef039
```

The download scored 83.64% mean choroid Dice on 439 Optovue OCT scans. The [Optovue architecture report](../reports/optovue-architecture.md) covers that evaluation and the target-adaptation experiments.

Use `--weights_type retfound` for encoder initialization from [RETFound weights](https://github.com/rmaphoh/RETFound_MAE). Use `--weights_type none` to train from scratch.
