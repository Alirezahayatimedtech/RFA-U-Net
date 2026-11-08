"""Patient grouping, native-resolution measurement, and optional choroid refinement."""

import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ScanMetadata:
    patient_id: str = ""
    scanner: str = ""
    axial_um_per_pixel: float = None


def read_metadata(path):
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image", "patient_id"} <= set(reader.fieldnames):
            raise ValueError("Metadata CSV requires image and patient_id columns.")
        metadata = {}
        for row in reader:
            image = (row["image"] or "").strip()
            patient_id = (row["patient_id"] or "").strip()
            if not image or image != os.path.basename(image) or not patient_id or image in metadata:
                raise ValueError(f"Invalid or duplicate image/patient_id at metadata line {reader.line_num}.")
            spacing = (row.get("axial_um_per_pixel") or "").strip()
            axial_um = float(spacing) if spacing else None
            if axial_um is not None and (not math.isfinite(axial_um) or axial_um <= 0):
                raise ValueError(f"Invalid axial_um_per_pixel for {image}.")
            metadata[image] = ScanMetadata(patient_id, (row.get("scanner") or "").strip(), axial_um)
    if not metadata:
        raise ValueError("Metadata CSV has no scans.")
    return metadata


def metadata_for(image, metadata):
    if metadata and image not in metadata:
        raise ValueError(f"No metadata row for {image}.")
    return metadata.get(image, ScanMetadata())


def split_patients(image_paths, metadata, test_fraction, val_fraction, seed):
    by_patient = defaultdict(list)
    for idx, path in enumerate(image_paths):
        patient = metadata_for(os.path.basename(path), metadata).patient_id
        by_patient[patient].append(idx)
    patients = sorted(by_patient)
    if len(patients) < 3:
        raise ValueError("A patient-grouped train/validation/test split needs at least three patients.")
    random.Random(seed).shuffle(patients)
    n_test = max(1, min(round(len(patients) * test_fraction), len(patients) - 2))
    n_trainval = len(patients) - n_test
    n_val = max(1, min(round(n_trainval * val_fraction), n_trainval - 1))
    groups = patients[:n_trainval - n_val], patients[n_trainval - n_val:n_trainval], patients[n_trainval:]
    return tuple(sorted(idx for patient in group for idx in by_patient[patient]) for group in groups)


def contiguous_layer(mask):
    """Keep the largest 8-connected region and fill holes within its occupied columns."""
    components, count = label(mask, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return mask
    sizes = np.bincount(components.ravel())
    region = components == (1 + sizes[1:].argmax())
    top, bottom, present = column_bounds(region)
    rows = np.arange(mask.shape[0])[:, None]
    return present & (rows >= top) & (rows <= bottom)


def column_bounds(mask):
    present = mask.any(axis=0)
    return mask.argmax(axis=0), mask.shape[0] - 1 - mask[::-1].argmax(axis=0), present


def native_mask(probability, size, threshold, postprocess="none"):
    """Resize continuous probabilities to the source grid before thresholding."""
    width, height = size
    resized = F.interpolate(probability[None, None], size=(height, width), mode="bilinear", align_corners=False)
    mask = resized[0, 0].numpy() > threshold
    return contiguous_layer(mask) if postprocess == "layer" else mask


def thickness_loss_choroid(probability, targets):
    """Penalize soft axial thickness error on annotated columns, scaled by image height."""
    reference = targets[:, 1].float()
    predicted_fraction = probability.mean(dim=-2)
    reference_fraction = reference.mean(dim=-2)
    valid = reference_fraction > 0
    error = F.smooth_l1_loss(predicted_fraction, reference_fraction, reduction="none", beta=0.05)
    return (error * valid).sum() / valid.sum().clamp_min(1)


def thickness_measurements(mask, axial_um):
    top, bottom, present = column_bounds(mask)
    complete = present & (top > 0) & (bottom < mask.shape[0] - 1)
    thickness = bottom[complete] - top[complete] + 1
    mean_px = float(thickness.mean()) if thickness.size else None
    median_px = float(np.median(thickness)) if thickness.size else None
    return {
        "predicted_columns": int(present.sum()),
        "measured_columns": int(complete.sum()),
        "axially_clipped_columns": int((present & ~complete).sum()),
        "scan_width_px": mask.shape[1],
        "mean_axial_thickness_px": mean_px,
        "median_axial_thickness_px": median_px,
        "mean_axial_thickness_um": mean_px * axial_um if mean_px is not None and axial_um is not None else None,
        "median_axial_thickness_um": median_px * axial_um if median_px is not None and axial_um is not None else None,
    }


def scan_metrics(pred, reference, axial_um=None):
    """Measure overlap and axial errors on matched columns, with explicit coverage."""
    pred_area, reference_area = int(pred.sum()), int(reference.sum())
    intersection = int((pred & reference).sum())
    union = pred_area + reference_area - intersection
    top_p, bottom_p, present_p = column_bounds(pred)
    top_r, bottom_r, present_r = column_bounds(reference)
    matched = present_p & present_r
    n_matched = int(matched.sum())
    n_reference = int(present_r.sum())
    scale = axial_um if axial_um is not None else 1.0
    result = {
        "dice": 2 * intersection / (pred_area + reference_area) if union else 1.0,
        "jaccard": intersection / union if union else 1.0,
        "pixel_accuracy": float((pred == reference).mean()),
        "reference_columns": n_reference,
        "matched_columns": n_matched,
        "column_coverage": n_matched / n_reference if n_reference else None,
        "missed_scan": int(n_reference > 0 and n_matched == 0),
    }
    bm = (top_p[matched] - top_r[matched]) * scale
    csi = (bottom_p[matched] - bottom_r[matched]) * scale
    unit = "um" if axial_um is not None else "px"
    for name, errors in (("bm", bm), ("csi", csi), ("thickness", csi - bm)):
        for error_unit in ("px", "um"):
            valid = n_matched and unit == error_unit
            result[f"{name}_bias_{error_unit}"] = float(errors.mean()) if valid else None
            result[f"{name}_mae_{error_unit}"] = float(np.abs(errors).mean()) if valid else None
    return result


def summarize_scans(rows, patient_macro=False):
    fields = (
        "dice", "jaccard", "pixel_accuracy", "column_coverage", "missed_scan",
        *(f"{name}_{metric}_{unit}" for name in ("bm", "csi", "thickness")
          for metric in ("bias", "mae") for unit in ("px", "um")),
    )
    groups = defaultdict(list)
    for row in rows:
        key = row["patient_id"] if patient_macro else row["image"]
        groups[key].append(row)

    summary = {}
    for field in fields:
        means = []
        for scans in groups.values():
            values = [scan[field] for scan in scans if scan[field] is not None]
            if values:
                means.append(float(np.mean(values)))
        summary[field] = float(np.mean(means)) if means else None
    return summary
