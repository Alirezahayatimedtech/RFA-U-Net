# OIMHS choroid benchmark

This comparison measures changes to inference with the released RFA-U-Net checkpoint. It uses one central B-scan from each of the 119 OIMHS participants with macular holes. The demographics workbook links 125 eyes to participants. For bilateral cases, the benchmark takes the lower eye ID, then the middle image after numeric sorting. This cohort contains Heidelberg Spectralis scans. [OIMHS study](https://doi.org/10.1038/s41597-023-02675-1) · [Released archive](https://doi.org/10.6084/m9.figshare.23508453.v1)

The PNGs contain the scan in the left half and the annotation in the right half. The choroid band below the retina is yellow in the released archive (RGB 255, 255, 0), although the paper illustrates a different annotation palette. The archive MD5 is `a5637889d2c6ad7ba3573a1eb7f7f3f2`. The released checkpoint SHA-256 is `0cf05c6ebf551ab3db2ffef69a1ac5cd1e088a0d0bcc2ba518f8ddab4276586e`.

A seeded shuffle of sorted patient IDs assigns 24 participants to tuning and 95 to held-out evaluation. The tuning set selects the best Dice across seven thresholds from 0.35 to 0.65, with and without horizontal flip averaging and connected-layer cleanup. Both RFA variants use identical weights and native-grid scoring. The selected variant averages horizontal flips at a threshold of 0.35, without connected-layer cleanup.

## Held-out results

| Method | Dice | Jaccard | BM MAE (px) | CSI MAE (px) | Thickness MAE (px) | Column coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Released RFA-U-Net, threshold 0.50 | 0.65975 | 0.51054 | 7.34 | 17.91 | 20.97 | 88.60% |
| Released RFA-U-Net, flip averaging, threshold 0.35 | 0.69132 | 0.54676 | 7.50 | 17.21 | 19.98 | 91.75% |
| Choroidalyzer v1.0, threshold 0.50 | 0.84159 | 0.74022 | 4.88 | 8.79 | 7.82 | 94.98% |

The paired Dice increase was 3.16 percentage points (95% participant bootstrap CI 2.66 to 3.68). The selected settings improved CSI error, thickness error and coverage, while BM MAE increased by 0.17 pixels. Boundary errors use matched reference and prediction columns. Coverage tracks the fraction of reference columns with predictions. All three methods detected at least one reference column on every held-out scan. Choroidalyzer had the highest Dice and the lowest errors on this cohort. Errors are reported in pixels because per-scan axial calibration was not used.

This result is specific to one central B-scan per patient in a macular-hole cohort from one device. The released weights were reused without retraining. The thickness loss, performance on full volumes, and generalization to the original study cohorts or other devices require separate evaluation.

## Reproduce

```bash
pip install -r requirements.txt openpyxl
python benchmarks/oimhs.py "/path/to/OIMHS dataset.zip" weights/best_rfa_unet.pth
```

To compare Choroidalyzer v1.0 on the same 95 held-out scans, install its dependencies and pass a local clone. The comparator runs its default published weights and region threshold of 0.5.

```bash
pip install pandas scikit-image
git clone https://github.com/justinengelmann/Choroidalyzer.git
git -C Choroidalyzer checkout 28f49c718b374e8512a8625f9cb4d2c64a7db66f
python benchmarks/oimhs.py "/path/to/OIMHS dataset.zip" weights/best_rfa_unet.pth \\
  --choroidalyzer-repo Choroidalyzer
```

## Related work

- Lee et al. used graph-based surface extraction and measured BM and CSI separately. This motivates reporting both errors alongside overlap and thickness. [Lee et al., 2021](https://doi.org/10.1016/j.jneumeth.2021.109267)
- Bio-Net used choroidal thickness as an anatomical prior. The optional RFA-U-Net loss uses a differentiable column-thickness objective with the existing region head. [Zhang et al., 2020](https://doi.org/10.1109/JBHI.2020.3023144)
- Choroidalyzer split at the participant level and evaluated derived measurements. It also provides vessel and fovea predictions. The optional comparator here uses its choroid region output. [Engelmann et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38833259/)
