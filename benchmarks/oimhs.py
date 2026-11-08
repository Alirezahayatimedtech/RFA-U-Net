"""Compare inference options on one central B-scan per OIMHS participant."""

import argparse
import io
from itertools import product
from pathlib import Path
import random
import re
import sys
from zipfile import ZipFile

import numpy as np
from openpyxl import load_workbook
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from choroid import native_mask, scan_metrics
from rfa_u_net import JointSegmentationTransform, checkpoint_model, parse_args, predict_choroid


THRESHOLDS = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
BASELINE = (False, "none", 0.5)
FIELDS = ("dice", "jaccard", "bm_mae_px", "csi_mae_px", "thickness_mae_px", "column_coverage")


def patient_scans(archive):
    eyes = {}
    for name in archive.namelist():
        match = re.fullmatch(r"Images/(\d+)/(\d+)\.png", name)
        if match:
            eye, scan = map(int, match.groups())
            eyes.setdefault(eye, []).append((scan, name))

    workbook = load_workbook(io.BytesIO(archive.read("Demographics of the participants.xlsx")),
                             read_only=True, data_only=True)
    try:
        patient_eyes = {}
        for patient, eye, *_ in workbook.active.iter_rows(min_row=2, values_only=True):
            patient_eyes.setdefault(patient, []).append(eye)
    finally:
        workbook.close()
    if len(patient_eyes) != 119 or {eye for ids in patient_eyes.values() for eye in ids} != set(eyes):
        raise ValueError("OIMHS demographics do not match the archive's eye folders.")

    return {patient: [name for _, name in sorted(eyes[min(eye_ids)])]
            for patient, eye_ids in patient_eyes.items()}


def central_scans(archive):
    return {patient: scans[len(scans) // 2] for patient, scans in patient_scans(archive).items()}


def load_scan_and_labels(archive, member):
    with Image.open(io.BytesIO(archive.read(member))) as source:
        image = source.convert("RGB")
    width, height = image.size
    if width % 2:
        raise ValueError(f"Expected side-by-side scan and annotation in {member}.")
    middle = width // 2
    scan = image.crop((0, 0, middle, height))
    labels = np.asarray(image.crop((middle, 0, width, height)))
    # The released PNGs use yellow for the band below the green retina.
    # The article shows a different palette for the annotation workflow.
    return scan, labels


def load_scan_and_mask(archive, member):
    scan, labels = load_scan_and_labels(archive, member)
    return scan, (labels == (255, 255, 0)).all(axis=2)


@torch.inference_mode()
def evaluate(archive, members, patients, model, transform, device, settings, split):
    results = {setting: [] for setting in settings}
    for index, patient in enumerate(patients, 1):
        scan, reference = load_scan_and_mask(archive, members[patient])
        image = transform(scan).unsqueeze(0).to(device)
        original = predict_choroid(model, image)[0].cpu()
        probabilities = {False: original}
        if any(flip for flip, _, _ in settings):
            flipped = predict_choroid(model, image.flip(-1))[0].cpu().flip(-1)
            probabilities[True] = (original + flipped) / 2
        for flip, postprocess, threshold in settings:
            mask = native_mask(probabilities[flip], scan.size, threshold, postprocess)
            results[flip, postprocess, threshold].append(scan_metrics(mask, reference))
        if index % 10 == 0 or index == len(patients):
            print(f"{split}: {index}/{len(patients)} participants", flush=True)
    return results


def mean(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return float(np.mean(values)) if values else float("nan")


def print_result(setting, rows):
    print(setting, " ".join(f"{field}={mean(rows, field):.5f}" for field in FIELDS),
          f"missed={sum(row['missed_scan'] for row in rows)}", flush=True)


def compare_choroidalyzer(archive, members, patients, repository, device):
    sys.path.insert(0, str(Path(repository).resolve()))
    from choroidalyze import Choroidalyzer

    model = Choroidalyzer(device=str(device))
    results = []
    for patient in patients:
        scan, reference = load_scan_and_mask(archive, members[patient])
        image = torch.from_numpy(np.array(scan.convert("L"))).unsqueeze(0)
        mask = model.predict(image)[0].ge(0.5).numpy()
        results.append(scan_metrics(mask, reference))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="Unmodified Figshare OIMHS dataset.zip")
    parser.add_argument("checkpoint", help="Released RFA-U-Net checkpoint")
    parser.add_argument("--choroidalyzer-repo", help="Optional local clone for the default Choroidalyzer comparison")
    options = parser.parse_args()
    args = parse_args(["--weights_type", "rfa-unet", "--weights_path", options.checkpoint])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = checkpoint_model(args, device)
    if not model.released_layout:
        raise ValueError("This benchmark requires the released RFA-U-Net checkpoint.")
    model.eval()
    transform = JointSegmentationTransform(config["image_size"], normalize=True, image_mode="rgb")
    with ZipFile(options.archive) as archive:
        members = central_scans(archive)
        patients = sorted(members)
        random.Random(42).shuffle(patients)
        candidates = list(product((False, True), ("none", "layer"), THRESHOLDS))
        tuning = evaluate(archive, members, patients[:24], model, transform, device, candidates, "Tuning")
        selected = max([BASELINE, *candidates], key=lambda setting: mean(tuning[setting], "dice"))
        print("Tuning baseline and selected:", flush=True)
        print_result(BASELINE, tuning[BASELINE])
        print_result(selected, tuning[selected])
        heldout = evaluate(archive, members, patients[24:], model, transform, device,
                           list(dict.fromkeys((BASELINE, selected))), "Held-out")
        if options.choroidalyzer_repo:
            del model
            comparison = compare_choroidalyzer(archive, members, patients[24:], options.choroidalyzer_repo, device)
            print_result("Choroidalyzer v1.0, default threshold", comparison)
    print("Held-out baseline and selected:", flush=True)
    print_result(BASELINE, heldout[BASELINE])
    print_result(selected, heldout[selected])
    delta = np.array([right["dice"] - left["dice"]
                      for left, right in zip(heldout[BASELINE], heldout[selected])])
    rng = np.random.default_rng(42)
    boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    low, high = np.percentile(boot, (2.5, 97.5))
    print(f"Paired Dice change: {delta.mean():+.5f} (95% participant bootstrap CI {low:+.5f} to {high:+.5f})")


if __name__ == "__main__":
    main()
