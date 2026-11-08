# Trained choroid refiner

The refiner is a compact two-head U-Net trained on OIMHS. A choroid head learns the region and axial thickness, with extra weight near boundaries. An auxiliary retina head supplies anatomical context during training. A second variant learns a logit correction on top of frozen Choroidalyzer v1.0 region logits. The hybrid restores the correction to the native grid before adding it to the prior.

## Patient split

The [OIMHS archive](https://doi.org/10.6084/m9.figshare.23508453.v1) contains Heidelberg Spectralis scans from patients with macular holes. The comparison reuses the 95 held-out patients from [the original branch benchmark](README.md). The other 24 patients are split into 18 for training and 6 for validation using the same seed-42 shuffle. Each patient contributes one eye, selected by the lower eye ID for bilateral cases.

| Partition | Patients | B-scans | Role |
| --- | ---: | ---: | --- |
| Training | 18 | 324 | Fit weights on up to 18 equally spaced slices per patient |
| Validation | 6 | 42 | Select epoch, flip averaging, threshold and optional mask cleanup |
| Primary test | 95 | 95 | Compare the central B-scan with Choroidalyzer |
| Secondary test | 95 | 380 | Compare four noncentral B-scans per patient |

The secondary scans use slice-index fractions 0.15, 0.30, 0.70 and 0.85 of the selected eye's scan list. All reported means weight patients equally. Paired 95% confidence intervals resample patients 10,000 times. Boundary errors use matched columns and are reported in native pixels. Coverage and missed scans are recorded separately.

The refiner was adapted using labeled patients from the same OIMHS cohort. The comparator used the published Choroidalyzer weights and its default 0.50 threshold. The findings establish held-out performance on this cohort. Cross-device and external-cohort performance require separate testing. The comparison covers choroid-region segmentation and thickness. Choroidalyzer also provides vessel and fovea outputs. [OIMHS paper](https://doi.org/10.1038/s41597-023-02675-1) · [Choroidalyzer paper](https://pubmed.ncbi.nlm.nih.gov/38833259/)

## Results

| Held-out scans | Method | Dice | BM MAE (px) | CSI MAE (px) | Thickness MAE (px) | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Central | Choroidalyzer v1.0 | 0.84159 | 4.88 | 8.79 | 7.82 | 94.98% |
| Central | Hybrid refiner | 0.93568 | 0.83 | 4.82 | 4.95 | 98.33% |
| Central | **Standalone refiner** | **0.96497** | **0.59** | **2.82** | **3.00** | **99.64%** |
| Noncentral | Choroidalyzer v1.0 | 0.86956 | 3.92 | 7.57 | 5.85 | 97.25% |
| Noncentral | Hybrid refiner | 0.95288 | 0.86 | 3.57 | 3.60 | 99.56% |
| Noncentral | **Standalone refiner** | **0.96345** | **0.57** | **2.88** | **3.05** | **99.59%** |

Standalone Dice improved by 12.34 percentage points on central scans (95% CI 10.26 to 14.54) and 9.39 points on noncentral scans (95% CI 8.35 to 10.50). The paired thickness MAE differences were −4.82 px (95% CI −6.45 to −3.37) and −2.80 px (95% CI −4.10 to −1.88), respectively. All methods detected at least one reference column on every tested scan.

Validation selected epoch 32 and threshold 0.45 for the standalone model, and epoch 22 and threshold 0.65 for the hybrid. Both used horizontal flip averaging and connected-layer cleanup. Native-grid validation Dice was 0.96947 and 0.96528, respectively. The standalone model is the default and runs without Choroidalyzer.

## Reproduce

The reported runs used CPU PyTorch 2.15.0 with four OpenMP and MKL threads. Use the released OIMHS archive with MD5 `a5637889d2c6ad7ba3573a1eb7f7f3f2`. The training and comparison scripts use the choroid pixels marked yellow in that archive. For the comparator, use Choroidalyzer checkout `28f49c718b374e8512a8625f9cb4d2c64a7db66f` and v1.0 weights with SHA-256 `4b9b0c0548e0524d0ad57700c7372cdf3853452c905d1ebf85bfc48748623697`.

```bash
pip install -r requirements.txt openpyxl pandas scikit-image
git clone https://github.com/justinengelmann/Choroidalyzer.git
git -C Choroidalyzer checkout 28f49c718b374e8512a8625f9cb4d2c64a7db66f
```

Train the standalone model and select its inference settings on the validation patients.

```bash
python benchmarks/train_oimhs_refiner.py train "OIMHS dataset.zip" runs/unsealed.safetensors --epochs 32
python benchmarks/train_oimhs_refiner.py validate "OIMHS dataset.zip" runs/unsealed.safetensors \\
  --output runs/standalone.safetensors
python benchmarks/train_oimhs_refiner.py compare "OIMHS dataset.zip" runs/standalone.safetensors \\
  --prior-repo Choroidalyzer --noncentral
```

For the hybrid, train with `--with-prior --prior-repo Choroidalyzer --epochs 24`. Pass `--prior-repo Choroidalyzer` to its validation and comparison runs. The scripts can also use a local copy of the v1.0 weights through `--prior-weights`. [Choroidalyzer v1.0 weights](https://github.com/justinengelmann/Choroidalyzer/releases/tag/v1.0)
