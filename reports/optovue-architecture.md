# RFA-U-Net architecture and inference on Optovue OCT

The five-model RFA-U-Net ensemble reached **93.29% mean choroid Dice** in the Optovue adaptation experiment. The general-purpose RFA-U-Net checkpoint scored **83.64%** on the complete 439-scan cohort before any Optovue fine-tuning. The two evaluations use different scan sets and answer different questions. The general checkpoint was designated using the highest development-validation Dice among the five original RFA-U-Net runs.

## External dataset and evaluation

The cohort consists of 439 labeled Optovue OCT B-scans from the diabetic-retinopathy archive. [Kafieh et al., Zenodo, `diabetic.rar`](https://doi.org/10.5281/zenodo.7618624)

For adaptation, the stored split assigned 274 scans to training, 65 to validation, and 100 to testing. Numeric filename suffixes provided 69, 17, and 25 disjoint groups, respectively. Patient identity within those groups remains unverified. This is a retrospective rerun of a previously evaluated test split. Each model used threshold **0.50** with no horizontal-flip averaging. Metrics were averaged over scans. Boundary MAE uses columns present in both the reference and prediction and the cohort's axial calibration of 1500 ÷ 278 µm/pixel.

## Architecture changes

We kept the RETFound encoder, attention-gated decoder, and gray-edge instance-normalized pre-adapter. The rectangular recipe changes the input and pre-adapter grid from **224 × 224** to **224 × 592**, closer to the native Optovue aspect ratio. It disables ImageNet normalization and shallow feature fusion. Each run initializes from an original RFA-U-Net checkpoint and fine-tunes the full network using the Optovue training partition with validation-based early stopping.

| Recipe on the same 100 test scans | Dice | Upper-boundary MAE | Lower-boundary MAE |
|---|---:|---:|---:|
| Square, ImageNet normalization, five-model mean | 92.57% | 7.32 µm | 30.14 µm |
| Square, no ImageNet normalization, five-model mean | 92.31% | 7.75 µm | 31.61 µm |
| Rectangular, no ImageNet normalization, five-model mean | 93.09% | 7.18 µm | 27.75 µm |
| Rectangular, no ImageNet normalization, with shallow feature fusion, five-model mean | 92.69% | 7.62 µm | 29.40 µm |
| Rectangular, no ImageNet normalization, five-model probability ensemble | **93.29%** | **7.06 µm** | **26.79 µm** |

With ImageNet normalization disabled, the rectangular grid increased Dice by **0.78 percentage points** and reduced lower-boundary MAE by **3.86 µm** versus the square grid. Adding shallow fusion reduced Dice by **0.40 points** in the rectangular recipe. Averaging the five rectangular models' probabilities before applying the 0.50 threshold increased Dice by **0.20 points** over their individual-score mean.

## General-purpose RFA-U-Net checkpoint

The downloadable model uses 224 × 224 RGB input, ImageNet normalization, the gray-edge pre-adapter, and the attention decoder with token-pyramid skips. It uses a single checkpoint and can run on other OCT datasets. Its accuracy on additional scanners and clinical populations requires evaluation. On all 439 Optovue scans, the five original RFA-U-Net checkpoints averaged **84.74% Dice** and the downloadable checkpoint scored **83.64% Dice** with the same native-resolution scoring and threshold. [Download and inference instructions](../weights/README.md)
