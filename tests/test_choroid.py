import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from choroid import (
    ScanMetadata, contiguous_layer, native_mask, read_metadata, scan_metrics,
    split_patients, summarize_scans, thickness_loss_choroid, thickness_measurements,
)
from dataset import OCTDataset, load_scan
from rfa_u_net import predict_choroid


class ChoroidTests(unittest.TestCase):
    def test_patient_splits_are_disjoint_and_deterministic(self):
        images = [f"{patient}_{scan}.png" for patient in range(5) for scan in range(patient + 1)]
        metadata = {image: ScanMetadata(image.split("_")[0]) for image in images}
        splits = split_patients(images, metadata, 0.2, 0.25, 42)
        self.assertEqual(splits, split_patients(images, metadata, 0.2, 0.25, 42))
        self.assertEqual(sorted(index for split in splits for index in split), list(range(len(images))))
        groups = [{metadata[images[index]].patient_id for index in split} for split in splits]
        self.assertEqual(sum(map(len, groups)), len(set.union(*groups)))

    def test_native_spacing_and_missing_boundaries(self):
        reference = np.zeros((12, 5), dtype=bool)
        reference[3:7, :4] = True
        prediction = np.zeros_like(reference)
        prediction[4:9, :2] = True
        result = scan_metrics(prediction, reference, 2.5)
        self.assertEqual(result["bm_mae_um"], 2.5)
        self.assertEqual(result["csi_mae_um"], 5.0)
        self.assertEqual(result["thickness_mae_um"], 2.5)
        self.assertEqual(result["column_coverage"], 0.5)
        self.assertIsNone(result["bm_mae_px"])
        miss = scan_metrics(np.zeros_like(reference), reference)
        self.assertEqual(miss["dice"], 0)
        self.assertEqual(miss["column_coverage"], 0)
        self.assertIsNone(miss["csi_mae_px"])
        self.assertEqual(miss["missed_scan"], 1)

    def test_patient_macro_weights_patients_equally(self):
        reference = np.ones((4, 3), dtype=bool)
        rows = [
            {"image": "a", "patient_id": "one", **scan_metrics(reference, reference)},
            {"image": "b", "patient_id": "one", **scan_metrics(reference, reference)},
            {"image": "c", "patient_id": "two", **scan_metrics(~reference, reference)},
        ]
        self.assertAlmostEqual(summarize_scans(rows)["dice"], 2 / 3)
        self.assertEqual(summarize_scans(rows, patient_macro=True)["dice"], 0.5)

    def test_layer_cleanup_and_native_dimensions(self):
        mask = np.zeros((9, 10), dtype=bool)
        mask[2:6, 1:7] = True
        mask[4, 4] = False
        mask[0, 9] = True
        cleaned = contiguous_layer(mask)
        self.assertEqual(cleaned.sum(), 24)
        self.assertTrue(cleaned[4, 4])
        self.assertFalse(cleaned[:, 7:].any())
        probability = torch.from_numpy(mask).float()
        self.assertEqual(native_mask(probability, (20, 18), 0.5).shape, (18, 20))
        self.assertTrue(np.array_equal(cleaned, native_mask(probability, (10, 9), 0.5, "layer")))

    def test_thickness_excludes_axially_clipped_columns(self):
        mask = np.zeros((10, 4), dtype=bool)
        mask[:3, 0] = mask[3:8, 1] = mask[8:, 2] = True
        result = thickness_measurements(mask, 2.5)
        self.assertEqual(result["axially_clipped_columns"], 2)
        self.assertEqual(result["measured_columns"], 1)
        self.assertEqual(result["mean_axial_thickness_um"], 12.5)
        self.assertIsNone(thickness_measurements(mask, None)["median_axial_thickness_um"])

    def test_thickness_loss_is_finite_and_differentiable(self):
        probability = torch.zeros((1, 12, 6), requires_grad=True)
        target = torch.zeros((1, 2, 12, 6))
        target[:, 1, 4:8] = 1
        loss = thickness_loss_choroid(probability, target)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(probability.grad).all())
        self.assertEqual(thickness_loss_choroid(target[:, 1], target).item(), 0)
        self.assertEqual(thickness_loss_choroid(probability, torch.zeros_like(target)).item(), 0)

    def test_horizontal_flip_is_restored_before_averaging(self):
        class Model(torch.nn.Module):
            released_layout = True

            def forward(self, images):
                return torch.cat([torch.zeros_like(images), images], dim=1)

        image = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        self.assertTrue(torch.allclose(predict_choroid(Model(), image, tta_hflip=True), image[:, 0].sigmoid()))

    def test_strict_pairing_and_native_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks = root / "images", root / "masks"
            images.mkdir()
            masks.mkdir()
            Image.new("L", (12, 9)).save(images / "a.png")
            Image.new("L", (12, 9)).save(images / "b.png")
            Image.new("L", (11, 9)).save(masks / "a_mask.png")
            with self.assertRaisesRegex(ValueError, "unpaired"):
                OCTDataset(images, masks, 16, strict_pairs=True)
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                OCTDataset(images, masks, 16).load_pair(0)

    def test_high_bit_depth_images_keep_contrast(self):
        with tempfile.TemporaryDirectory() as directory:
            scan = Path(directory) / "image.png"
            image = np.arange(400, 1000, dtype=np.uint16).reshape(20, 30)
            Image.fromarray(image).save(scan)
            self.assertGreater(np.unique(np.asarray(load_scan(scan, "gray"))).size, 100)

    def test_metadata_rejects_invalid_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            with path.open("w", newline="") as handle:
                csv.writer(handle).writerows([["image", "patient_id", "axial_um_per_pixel"], ["a.png", "p1", "nan"]])
            with self.assertRaisesRegex(ValueError, "Invalid axial"):
                read_metadata(path)


if __name__ == "__main__":
    unittest.main()
