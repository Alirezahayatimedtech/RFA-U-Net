"""Train and compare compact choroid models on the established OIMHS patient split."""

import argparse
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from oimhs import FIELDS, load_scan_and_labels, load_scan_and_mask, mean, patient_scans, print_result
from choroid import native_mask, scan_metrics
from choroid_refiner import (
    GRID, ChoroidRefiner, NativeRefiner, checkpoint_settings, make_prior, normalized_grid, prior_logits, resize,
)


def split_scans(archive):
    scans = patient_scans(archive)
    patients = sorted(scans)
    random.Random(42).shuffle(patients)
    return tuple([(patient, scans[patient]) for patient in part]
                 for part in (patients[:18], patients[18:24], patients[24:]))


def spaced_scans(members, maximum):
    return [members[index] for index in np.linspace(0, len(members) - 1, min(maximum, len(members))).round().astype(int)]


def development_scans(archive, patients, limit, prior, device):
    images, targets, priors = [], [], []
    for patient, members in patients:
        for member in spaced_scans(members, limit):
            scan, labels = load_scan_and_labels(archive, member)
            pixels = torch.from_numpy(np.array(scan.convert("L")))
            choroid = (labels == (255, 255, 0)).all(axis=2)
            retina = (labels == (0, 255, 0)).all(axis=2)
            mask = torch.from_numpy(np.stack((choroid, retina)))
            images.append(normalized_grid(pixels[None, None])[0])
            targets.append(F.interpolate(mask[None].float(), size=GRID, mode="nearest-exact")[0])
            if prior is not None:
                raw = prior_logits(prior, pixels, device)
                priors.append(resize(raw[None, None], GRID)[0].cpu())
        print(f"Prepared development patient {patient}", flush=True)
    tensors = torch.stack(images), torch.stack(targets)
    return (*tensors, torch.stack(priors) if priors else None)


def loss_fn(output, masks, prior):
    choroid = output[:, :1] + prior if prior is not None else output[:, :1]
    target = masks[:, :1]
    probability = choroid.sigmoid()
    # Assign more weight near the choroid surfaces while retaining global region supervision.
    rim = F.max_pool2d(target, 5, 1, 2) + F.max_pool2d(-target, 5, 1, 2)
    region_loss = F.binary_cross_entropy_with_logits(choroid, target, weight=1 + 2 * rim)
    dims = (1, 2, 3)
    dice_loss = 1 - ((2 * (probability * target).sum(dims) + 1)
                     / (probability.sum(dims) + target.sum(dims) + 1)).mean()
    thickness_loss = F.smooth_l1_loss(probability.mean(-2), target.mean(-2), beta=0.03)
    retina_loss = F.binary_cross_entropy_with_logits(output[:, 1:2], masks[:, 1:2])
    residual_penalty = 0.004 * output[:, :1].square().mean() if prior is not None else 0
    return region_loss + dice_loss + 0.5 * thickness_loss + 0.2 * retina_loss + residual_penalty


@torch.inference_mode()
def validation_score(model, validation, batch_size, device):
    model.eval()
    images, targets, priors = validation
    scores = []
    for start in range(0, len(images), batch_size):
        stop = start + batch_size
        image = images[start:stop].to(device)
        truth = targets[start:stop, :1].to(device).bool()
        prior = priors[start:stop].to(device) if priors is not None else None
        pred = model(image, prior)[:, :1]
        if prior is not None:
            pred = pred + prior
        pred = pred > 0
        dims = (1, 2, 3)
        overlap = (pred & truth).sum(dims)
        size = pred.sum(dims) + truth.sum(dims)
        scores.extend(torch.where(size > 0, 2 * overlap / size, 1.0).tolist())
    return float(np.mean(scores))


def write_checkpoint(path, state, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".safetensors", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_file(state, temporary, metadata=metadata)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args, device):
    torch.manual_seed(42)
    model = ChoroidRefiner(args.with_prior).to(device)
    prior = make_prior(args.prior_repo, args.prior_weights, device) if args.with_prior else None
    with ZipFile(args.archive) as archive:
        train_patients, validation_patients, _ = split_scans(archive)
        print(f"Preparing {len(train_patients)} train and {len(validation_patients)} validation patients.", flush=True)
        images, targets, priors = development_scans(archive, train_patients, 18, prior, device)
        validation = development_scans(archive, validation_patients, 7, prior, device)
    print(f"{len(images)} training and {len(validation[0])} validation B-scans.", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    best, best_epoch = -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for indices in torch.randperm(len(images)).split(args.batch_size):
            image, mask = images[indices].to(device), targets[indices].to(device)
            logits = priors[indices].to(device) if priors is not None else None
            if torch.rand(()).item() < 0.5:
                image, mask = image.flip(-1), mask.flip(-1)
                if logits is not None:
                    logits = logits.flip(-1)
            shape = (len(indices), 1, 1, 1)
            scale = 0.9 + torch.rand(shape, device=device) * 0.2
            offset = torch.rand(shape, device=device) * 0.16 - 0.08
            image = (image * scale + offset).clamp(-1, 1)
            optimizer.zero_grad()
            loss = loss_fn(model(image, logits), mask, logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        score = validation_score(model, validation, args.batch_size, device)
        scheduler.step(score)
        print(f"Epoch {epoch:02d}: training loss {loss_sum / n_batches:.5f}, validation Dice {score:.5f}", flush=True)
        if score > best:
            best, best_epoch = score, epoch
            state = {key: value.detach().cpu().half().contiguous() for key, value in model.state_dict().items()}
            write_checkpoint(args.checkpoint, state, {"with_prior": str(args.with_prior).lower()})
        if epoch - best_epoch >= 8:
            break
    print(f"Selected epoch {best_epoch}, validation Dice {best:.5f}.", flush=True)


def validate(args, device):
    with_prior, _, _, _ = checkpoint_settings(args.checkpoint)
    model = NativeRefiner(args.checkpoint, with_prior, device, args.prior_repo, args.prior_weights)
    settings = [(flip, threshold, cleanup) for flip in (False, True)
                for threshold in (0.35, 0.45, 0.50, 0.55, 0.65) for cleanup in ("none", "layer")]
    candidates = {setting: [] for setting in settings}
    with ZipFile(args.archive) as archive:
        _, validation, _ = split_scans(archive)
        for _, members in validation:
            for member in spaced_scans(members, 7):
                scan, reference = load_scan_and_mask(archive, member)
                raw = prior_logits(model.prior, torch.from_numpy(np.array(scan.convert("L"))), device) if with_prior else None
                for flip in (False, True):
                    probability = model.predict(scan, flip, raw)
                    for setting in settings:
                        if setting[0] == flip:
                            mask = native_mask(probability, scan.size, setting[1], setting[2])
                            candidates[setting].append(scan_metrics(mask, reference))
    selected = max(candidates, key=lambda setting: mean(candidates[setting], "dice"))
    print_result(f"Validation selected {selected}", candidates[selected])
    if args.output:
        flip, threshold, cleanup = selected
        write_checkpoint(args.output, load_file(args.checkpoint), {
            "with_prior": str(with_prior).lower(), "threshold": str(threshold),
            "flip": str(flip).lower(), "postprocess": cleanup, "validation": "oimhs-patient-seed42",
        })


def summarize_group(rows):
    result = {}
    for field in FIELDS:
        values = [row[field] for row in rows if row[field] is not None]
        result[field] = float(np.mean(values)) if values else None
    result["missed_scan"] = sum(row["missed_scan"] for row in rows)
    return result


def report_comparison(label, baseline_groups, candidate_groups):
    baseline = [summarize_group(group) for group in baseline_groups]
    candidate = [summarize_group(group) for group in candidate_groups]
    print(f"{label}: {len(baseline)} patients, {sum(map(len, baseline_groups))} B-scans")
    print_result("Choroidalyzer v1.0", baseline)
    print_result("Challenger", candidate)
    rng = np.random.default_rng(42)
    for field in FIELDS:
        deltas = np.array([new[field] - old[field] for old, new in zip(baseline, candidate)
                           if old[field] is not None and new[field] is not None])
        if len(deltas):
            low, high = np.percentile(rng.choice(deltas, (10000, len(deltas))).mean(1), (2.5, 97.5))
            print(f"{field} paired change {deltas.mean():+.5f}, 95% CI [{low:+.5f}, {high:+.5f}], n={len(deltas)}")


def compare(args, device):
    with_prior, threshold, flip, cleanup = checkpoint_settings(args.checkpoint)
    model = NativeRefiner(args.checkpoint, with_prior, device, args.prior_repo, args.prior_weights)
    prior = model.prior if with_prior else make_prior(args.prior_repo, args.prior_weights, device)
    central, noncentral = [], []
    with ZipFile(args.archive) as archive:
        _, _, heldout = split_scans(archive)
        for index, (_, members) in enumerate(heldout, 1):
            center = len(members) // 2
            indices = [center]
            if args.noncentral:
                off_center = sorted({round((len(members) - 1) * part) for part in (0.15, 0.30, 0.70, 0.85)} - {center})
                indices.extend(off_center)
            baseline_rows, candidate_rows = [], []
            for scan_index in indices:
                scan, reference = load_scan_and_mask(archive, members[scan_index])
                pixels = torch.from_numpy(np.array(scan.convert("L")))
                raw = prior_logits(prior, pixels, device)
                baseline_rows.append(scan_metrics((raw >= 0).cpu().numpy(), reference))
                probability = model.predict(scan, flip, raw if with_prior else None)
                candidate_rows.append(scan_metrics(native_mask(probability, scan.size, threshold, cleanup), reference))
            central.append((baseline_rows[:1], candidate_rows[:1]))
            if len(indices) > 1:
                noncentral.append((baseline_rows[1:], candidate_rows[1:]))
            if index % 25 == 0:
                print(f"Held-out {index}/{len(heldout)}", flush=True)
    report_comparison("Central B-scans", *zip(*central))
    if noncentral:
        report_comparison("Noncentral B-scans", *zip(*noncentral))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "validate", "compare"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--with-prior", action="store_true")
    parser.add_argument("--prior-repo", type=Path)
    parser.add_argument("--prior-weights", type=Path)
    parser.add_argument("--output", type=Path, help="Export a checkpoint with settings selected by validate")
    parser.add_argument("--noncentral", action="store_true", help="Also compare four noncentral B-scans per held-out patient")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    {"train": train, "validate": validate, "compare": compare}[args.mode](args, device)


if __name__ == "__main__":
    main()
