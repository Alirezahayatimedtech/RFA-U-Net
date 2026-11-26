# RFA-U-Net

RFA-U-Net uses a RETFound encoder and an attention U-Net decoder to segment the choroid in OCT B-scans. It produces binary masks at the original image resolution.

[Download RFA-U-Net Weights](https://drive.google.com/file/d/16LOpbw0lVi7JyehSJcQr9M7Q_uZjrlXP/view) (1.36 GB) · [Checksum and manual setup](weights/README.md)

## Quick start

Use Python 3.10 or newer. An NVIDIA GPU is recommended.

```bash
git clone https://github.com/Alirezahayatimedtech/RFA-U-Net.git
cd RFA-U-Net
python -m pip install -r requirements.txt
python src/rfa_u_net.py --weights_type rfa-unet --segment_dir path/to/oct_scans
```

The first run downloads and verifies the weights. Masks are saved in `segment_results/masks/`. Add `--save_overlay` to also save boundary overlays.

## Benchmark

The downloadable checkpoint achieved **83.64% mean Dice** on 439 external Optovue scans at threshold 0.50, without Optovue fine-tuning. [Evaluation and architecture experiments](reports/optovue-architecture.md) · [Optovue dataset](https://doi.org/10.5281/zenodo.7618624)

## More information

- [Training, fine-tuning, and evaluation](docs/usage.md)
- [Research paper](https://doi.org/10.1101/2025.05.03.25326923) · [MIT license](LICENSE)

## Contact and contributors

Project contributors include Alireza Hayati, Roya Arian, and Narges Saeedizadeh.

For project questions, email [alirezahayati17@yahoo.com](mailto:alirezahayati17@yahoo.com).
