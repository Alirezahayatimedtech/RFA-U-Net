# RFA-U-Net: RETFound Attention U-Net for OCT Choroid Segmentation

**RFA-U-Net** is a deep learning model to segment the **choroid** in Optical Coherence Tomography (OCT) images. It uses a **RETFound**-pretrained Vision Transformer (ViT) encoder and an **Attention U-Net** decoder, trained with Tversky/Dice losses and evaluated via Dice scores and micrometer-scale boundary errors.

**Last updated: 2026-11-08**
🆕 New:
* **Segmentation-only mode** for unlabeled OCT images via `--segment_dir` (no masks needed)
* **Unified CNN/SOTA training and evaluation runner** via `src/cnn_sota_unet_models.py`
* **RFA-U-Net ablation-study presets** in `src/rfa_u_net.py` via `--ablation_preset`
* **OCT pre-adapter + true grayscale pipeline** in `src/rfa_u_net.py` / `src/dataset.py`
  * Helps stabilize **multi-device / multi-scanner OCT** variation (especially **contrast / brightness** shift) before RETFound patch embedding

---

## 🚀 Key Features

* **Encoder**: RETFound MAE ViT backbone
* **Decoder**: Attention U-Net with gated skip connections
* **🆕 OCT pre-adapter (optional, recommended)**:
  * `--use_pre_adapter --pre_adapter_mode gray_edge --pre_adapter_norm in`
  * Adds anti-aliased resize + OCT-oriented intensity correction + edge-preserving input alignment before RETFound
  * Designed to address scanner-dependent contrast and brightness shifts
* **Losses**: Tversky + Dice to handle class imbalance
* **Metrics**:

  * Dice score (overall & choroid-only)
  * Signed/Unsigned boundary errors (μm)
* **Visualization**: Four-panel view & overlay boundary plots
* **🆕 Segmentation-only inference**:

  * Run on **unlabeled OCT images** using `--segment_dir`
  * Saves **binary masks** and optional **boundary overlays** to disk
* **🆕 Ablation protocol support**:

  * Toggle decoder components with predefined presets
  * Save per-ablation best checkpoints automatically
  * Optionally append metrics to a CSV for study reporting

---

## 📁 Repo Structure

```text
RFA-U-Net/
├── src/
│   ├── rfa_u_net.py          # Main train / infer / segment-only script
│   ├── cnn_sota_unet_models.py # Unified runner for CNN + SOTA U-Net baselines
│   ├── models_vit.py         # RETFound ViT implementation
│   └── util/
│       └── pos_embed.py      # Positional embedding utils
├── examples/
│   └── visualization.py      # Qualitative demo script
├── weights/
│   └── README.md             # How to download pretrained weights
├── requirements.txt
└── README.md                 # You are here
```

---

## ⚙️ Requirements

* Python 3.10+
* PyTorch 2.7+
* NVIDIA GPU recommended

See [requirements.txt](requirements.txt) for the full dependency list.

---

## 🛠 Installation

```bash
git clone https://github.com/Alirezahayatimedtech/RFA-U-Net.git
cd RFA-U-Net
pip install -r requirements.txt
```

### 🔐 Hugging Face Authentication

If you plan to load **RETFound** weights from HF Hub, set your token:

```bash
export HUGGINGFACE_HUB_TOKEN="hf_yourTokenHere"
# or
huggingface-cli login
```

---

## 📂 Dataset Format (for Training / Test with Masks)

```text
data/
├── images/
│   ├── sample1.jpg
│   └── sample2.jpg
└── masks/
    ├── sample1.png
    └── sample2.png
```

* **Images**: RGB OCT scans (`.jpg`, `.png`, `.tif`, …)
* **Masks**: Binary masks (`.png`) matching the image filenames (background/choroid as 2-channel one-hot internally)

---

## External evaluation

Evaluate a trained checkpoint against native-resolution binary masks. Image and mask dimensions must match. The CLI writes one row per scan and reports Dice, Jaccard, Bruch’s membrane (BM) and choroid–sclera interface (CSI) errors, axial thickness error, and reference-column coverage.

```bash
python src/rfa_u_net.py \
  --test_only \
  --test_image_dir data/external/images \
  --test_mask_dir data/external/masks \
  --weights_type rfa-unet \
  --metadata_csv data/external/scans.csv \
  --metrics_csv runs/external.csv
```

The CSV uses exact image filenames and stable, de-identified patient IDs. The scanner and native axial pixel size are optional. Use a separate manifest for each external cohort.

```csv
image,patient_id,scanner,axial_um_per_pixel
scan01.png,p001,Spectralis,3.87
scan02.png,p001,Spectralis,3.87
scan03.png,p002,Triton,2.60
```

The physical values above are examples. Use each exported scan’s actual axial calibration. When all scans share one known calibration, `--pixel_size_micrometers` supplies a fallback. Scans without calibration are reported in pixels. Boundary errors use columns present in both prediction and reference. A missed scan contributes to Dice and coverage, and its boundary errors are blank. With patient IDs, the report averages scans within each patient and then averages patients. Scanner summaries use the same procedure within each scanner.

The released checkpoint restores its historical RGB preprocessing and decoder automatically. Newly trained checkpoints carry their own architecture and input preprocessing. For older custom checkpoints without metadata, pass the architecture and preprocessing flags used during training.

---

## 🧠 Training & Inference (with Masks)

Supply `--metadata_csv` to keep every patient’s scans in one partition. Training selects the best checkpoint on validation Dice and then evaluates the held-out partition. Explicit validation folders are checked for shared patient IDs when metadata is provided. With multiple scans per patient, use patient grouping or explicit patient-disjoint folders.

The training loss can add `--thickness_loss_weight 0.5` to penalize differences between predicted and reference axial thickness. This is an experimental starting weight. The existing `--edge_loss_weight` can be tested separately. Compare candidates using the same patient splits, then freeze the settings before evaluating the external cohort.

### 1️⃣ Train from scratch (no pre-training)

```bash
python src/rfa_u_net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type none \
  --num_epochs 30 \
  --batch_size 4
```

### 2️⃣ Fine-tune with RETFound backbone

```bash
python src/rfa_u_net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type retfound \
  --image_mode gray \
  --use_pre_adapter \
  --pre_adapter_mode gray_edge \
  --pre_adapter_norm in \
  --num_epochs 20 \
  --batch_size 8
```

This will:

* Download **RETFound** weights from HF Hub if missing
* Initialize the ViT encoder from RETFound
* Train the Attention U-Net decoder on your dataset

### 2b️⃣ Recommended OCT scanner-robust RETFound setup (contrast/brightness shift)

```bash
python src/rfa_u_net.py \
  --image_dir data/images \
  --mask_dir data/masks \
  --weights_type retfound \
  --image_mode gray \
  --use_pre_adapter \
  --pre_adapter_mode gray_edge \
  --pre_adapter_norm in \
  --pre_adapter_hidden_channels 8 \
  --pre_adapter_depth 1 \
  --pre_adapter_residual_scale_init 0.10 \
  --num_epochs 50 \
  --batch_size 8
```

Why this helps:
* The **pre-adapter** reduces scanner-dependent **contrast/brightness** mismatch before tokenization.
* `gray_edge` mode preserves boundary cues (important for thin choroid interfaces).
* The **true grayscale pipeline** (`--image_mode gray`) avoids redundant replicated RGB channels for OCT.

### 3️⃣ Fine-tune with pre-trained RFA-U-Net weights

(Default downloads via `gdown` if not present)

```bash
python src/rfa_u_net.py \
  --image_dir data/images \
  --mask_dir  data/masks \
  --weights_type rfa-unet \
  --num_epochs 15 \
  --batch_size 8
```

---

## 🧪 RFA-U-Net Ablation Study Protocol

`src/rfa_u_net.py` now supports fixed ablation presets that isolate decoder components:

* `baseline` (plain decoder)
* `attention_only`
* `fusion_only`
* `upconvs_only`
* `attention_fusion`
* `attention_upconvs`
* `fusion_upconvs`
* `full` (attention + fusion + up-convs)

List available presets:

```bash
python src/rfa_u_net.py --list_ablation_presets
```

Run one ablation:

```bash
python src/rfa_u_net.py \
  --image_dir data/images \
  --mask_dir data/masks \
  --weights_type retfound \
  --ablation_preset attention_fusion \
  --num_epochs 150 \
  --batch_size 8 \
  --seed 42 \
  --save_best_path weights/best_rfa_unet_attention_fusion.pth \
  --ablation_results_csv runs/ablation_results.csv
```

Run the full 8-preset sweep:

```bash
for p in baseline attention_only fusion_only upconvs_only attention_fusion attention_upconvs fusion_upconvs full; do
  python src/rfa_u_net.py \
    --image_dir data/images \
    --mask_dir data/masks \
    --weights_type retfound \
    --ablation_preset "$p" \
    --num_epochs 150 \
    --batch_size 8 \
    --seed 42 \
    --save_best_path "weights/best_rfa_unet_${p}.pth" \
    --ablation_results_csv runs/ablation_results.csv
done
```

Useful ablation flags:

* `--ablation_preset`: choose one of the 8 presets
* `--save_best_path`: set preset-specific checkpoint path
* `--ablation_results_csv`: append final metrics row per run
* `--seed`: fixed split seed for reproducible comparisons

---

## 🆕 Unified CNN + SOTA U-Net Runner

This repository now includes `src/cnn_sota_unet_models.py`, a single CLI for:

* training (`--mode train`)
* evaluation (`--mode eval`)
* prediction/export masks (`--mode predict`)

across **7 CNN-U-Net baselines** + **3 SOTA models**:

* `cnn_attention_unet_vit`
* `cnn_inception_unet`
* `cnn_alexnet_unet`
* `cnn_unet_vgg16`
* `cnn_unet_vgg19`
* `cnn_unet_resnet50`
* `cnn_unet_resnet101`
* `sota_pgkd`
* `sota_deepgpet`
* `sota_unet_bem`

### Key additions in this script

* notebook-aligned preprocessing defaults by model (`--notebook-defaults`)
* model-specific loss/metric defaults (`--loss auto`, `--metric-profile auto`)
* checkpointing + resume
* optional layer freezing for transfer learning
* scheduler support (`none`, `plateau`, `step`)
* multiple Dice/IoU evaluation areas:
  * `--metric-area full`
  * `--metric-area gt_choroid_columns`
  * `--metric-area union_choroid_columns`

### Important dependency note for SOTA models

For `sota_pgkd`, `sota_deepgpet`, and `sota_unet_bem`, ensure required external repos/modules are importable (e.g., via `PYTHONPATH`) before running.

### Quick usage

List model IDs and defaults:

```bash
python src/cnn_sota_unet_models.py --list-models
```

Train a CNN baseline (example: ResNet50 U-Net):

```bash
python src/cnn_sota_unet_models.py \
  --mode train \
  --model-id cnn_unet_resnet50 \
  --image-dir data/train/training/images \
  --mask-dir data/train/training/labels \
  --output-dir runs/cnn_unet_resnet50 \
  --epochs 150 \
  --batch-size 8 \
  --lr 1e-4 \
  --loss weighted_ce \
  --class-weights 1.0,2.0 \
  --scheduler plateau \
  --monitor dice \
  --augment \
  --notebook-defaults
```

Train DeepGPET:

```bash
python src/cnn_sota_unet_models.py \
  --mode train \
  --model-id sota_deepgpet \
  --image-dir data/train/training/images \
  --mask-dir data/train/training/labels \
  --output-dir runs/sota_deepgpet \
  --epochs 150 \
  --batch-size 4 \
  --lr 1e-3 \
  --loss bce_logits \
  --scheduler plateau \
  --monitor dice \
  --augment \
  --notebook-defaults
```

Evaluate on a labeled test set with strict union metric area:

```bash
python src/cnn_sota_unet_models.py \
  --mode eval \
  --model-id sota_pgkd \
  --image-dir data/train/test/images \
  --mask-dir data/train/test/labels \
  --checkpoint runs/sota_pgkd/sota_pgkd_best.pth \
  --save-metrics-path runs/sota_pgkd/test_metrics.json \
  --metric-area union_choroid_columns \
  --notebook-defaults
```

Predict masks for unlabeled images:

```bash
python src/cnn_sota_unet_models.py \
  --mode predict \
  --model-id cnn_inception_unet \
  --predict-image-dir path/to/images \
  --checkpoint runs/cnn_inception_unet/cnn_inception_unet_best.pth \
  --output-dir runs/cnn_inception_unet/predict_out \
  --notebook-defaults
```

---

## 🆕 Segmentation-Only Mode (No Masks Required)

This mode runs **pure inference** on *unlabeled* OCT images: it loads a trained model and saves segmentation masks (and optional overlays) to disk.

### CLI Example

```bash
python src/rfa_u_net.py \
  --segment_dir path/to/unlabeled_images \
  --weights_type rfa-unet \
  --weights_path weights/best_rfa_unet.pth \
  --output_dir segment_results \
  --batch_size 4 \
  --threshold 0.5 \
  --save_overlay
```

**Arguments specific to segmentation-only:**

* `--segment_dir` **(required for this mode)**

  * Path to a folder with OCT images (no masks needed)
  * Supported extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`
* `--weights_type`

  * Recommended: `rfa-unet` (use a trained RFA-U-Net checkpoint)
  * Can also use `retfound` if you trained a custom checkpoint
* `--weights_path`

  * Path to the `.pth` checkpoint (defaults to `weights/best_rfa_unet.pth`)
* `--output_dir`

  * Base directory where results are saved (default: `segment_results`)
* `--threshold`

  * Probability cutoff for binarizing choroid predictions (default: `0.5`)
* `--save_overlay`

  * If set, saves RGB overlays with choroid boundaries drawn on the original image

**What gets saved:**

* `output_dir/masks/`

  * `_mask.png` files: binary choroid masks resized to each image’s **original size**
* `output_dir/overlays/` (only if `--save_overlay` is used)

  * `_overlay.png` files: original image with choroid **boundaries** drawn in color

The script prints per-image debug info (prediction stats, resize shapes) and warns if a mask is all black (no pixels above threshold).

---

## 📥 Download Link for RFA-U-Net Pretrained Weights

### URL for pretrained weights (Google Drive link)

```text
RFA_UNET_WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=1zDEdAmNwNK8I-QEa6fqL5E3WjDn7Z-__"
```

The script will automatically download to:

```text
weights/best_rfa_unet.pth
```

if missing and `--weights_type rfa-unet` is used.

---

## Boundary refinement and thickness export

`--tta_hflip` averages the original and horizontally flipped predictions after restoring their orientation. It runs two model passes. `--postprocess layer` keeps the largest connected region and fills holes within its occupied columns. Both options are available for labeled evaluation and batch segmentation. Choose thresholds and refinement on a validation cohort because interrupted annotations and low-signal scans can behave differently.

```bash
python src/rfa_u_net.py \
  --segment_dir data/images \
  --weights_type rfa-unet \
  --save_overlay \
  --output_dir runs/segmentation \
  --metadata_csv data/scans.csv \
  --measurements_csv runs/segmentation/thickness.csv
```

The thickness CSV contains mean and median axial thickness across occupied columns whose boundaries stay within the image. It also records missing and axially clipped columns through scan width, predicted columns and measured columns. The measurements describe the segmented B-scan. They are not fovea-centered or perpendicular-to-layer measurements.

## Public benchmark

On 95 held-out OIMHS patients, horizontal flip averaging with `--threshold 0.35` improved mean Dice from 0.660 to 0.691 using the released weights. Choroidalyzer v1.0 scored 0.842 on the same scans. See [the reproducible OIMHS comparison](benchmarks/README.md) for boundary errors, the tuning split and scope.

## Original study results

The historical values below come from the original evaluation setup. The public OIMHS benchmark above uses a separate cohort and protocol. [RFA-U-Net preprint](https://doi.org/10.1101/2025.05.03.25326923)

| Metric                    | Value |
| ------------------------- | ----- |
| Dice Score (choroid)      | ~0.95 |
| Upper Signed Error (μm)   | ~–0.9 |
| Upper Unsigned Error (μm) | ~6.0  |
| Lower Signed Error (μm)   | ~1.1  |
| Lower Unsigned Error (μm) | ~21.4 |

---

## 🖼 Example Outputs

* **Four-panel visualization**:

  * Original OCT
  * Ground-truth mask
  * Predicted mask
  * Boundary overlay (true vs predicted; upper/lower boundaries)

See [the visualization script](examples/visualization.py) for the four-panel example.

Segmentation-only mode also creates overlay PNGs directly from your unlabeled OCTs (if `--save_overlay` is enabled).

---

## 📝 Changelog

* **2026-02-22**

  * Added **OCT pre-adapter** documentation (`--use_pre_adapter`, `--pre_adapter_mode gray_edge`, `--pre_adapter_norm in`)
  * Documented pre-adapter rationale for **multi-device / multi-scanner OCT** robustness (contrast / brightness shift)
  * Added **true grayscale pipeline** usage (`--image_mode gray`) for OCT training/inference
  * Updated README examples to reflect current RETFound + OCT-pre-adapter usage

* **2026-02-11**

  * Added `src/cnn_sota_unet_models.py` for unified CNN + SOTA training/evaluation/prediction
  * Added RFA-U-Net ablation protocol in `src/rfa_u_net.py`:
    * `--ablation_preset` with 8 predefined component combinations
    * `--list_ablation_presets` for quick preset discovery
    * `--save_best_path` for per-ablation checkpoint management
    * `--ablation_results_csv` to append comparable ablation metrics
  * Added model registry for 7 CNN-U-Net baselines + 3 SOTA models
  * Added notebook-aligned preprocessing profiles per model
  * Added configurable metric area support:
    * `full`
    * `gt_choroid_columns`
    * `union_choroid_columns`
  * Added scheduler, layer-freezing, checkpoint resume, and CLI documentation in README

* **2025-11-13**

  * Added **segmentation-only mode** via `--segment_dir`
  * New output utilities:

    * Save per-image masks and overlays
    * Preserve original image resolution for saved masks
  * Updated README to document segmentation-only usage

---

## 📬 Contact & Citation

Hayati *et al.* “RFA-U-Net: Choroid Segmentation in OCT with RETFound Attention U-Net,” *medRxiv* (2025).
DOI: `10.1101/2025.05.03.25326923`

For issues or questions, open a GitHub Issue or email **[alirezahayati17@yahoo.com](mailto:alirezahayati17@yahoo.com)**.

---

## 👨‍💻 Contributors

* **Alireza Hayati** – Lead developer
* **Roya Arian** – Mentor
* **Narges Sa** – Mentor

MIT License. See [LICENSE](LICENSE).
