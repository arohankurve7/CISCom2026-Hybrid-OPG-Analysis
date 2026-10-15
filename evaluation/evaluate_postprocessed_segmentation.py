from pathlib import Path
import sys
import csv

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fdi.fdi_assigner import (
    extract_raw_predictions,
    physical_instance_cleanup,
    PHYSICAL_DUPLICATE_IOU_THRESHOLD,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
)

DATASET = ROOT / "datasets/fdi_500/test"
MODEL_PATH = ROOT / "models/fdi_seg_v1/best.pt"
OUTPUT = ROOT / "evaluation/postprocessed_segmentation_50.csv"
MATCH_THRESHOLD = 0.50


def load_gt(label_path, height, width):
    masks = []
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    for line in label_path.read_text().splitlines():
        values = line.strip().split()
        if len(values) < 7:
            continue

        coordinates = np.array(
            [float(v) for v in values[1:]],
            dtype=np.float32
        ).reshape(-1, 2)

        coordinates[:, 0] *= width
        coordinates[:, 1] *= height

        polygon = np.round(coordinates).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        masks.append(mask.astype(bool))

    return masks


def evaluate(gt_masks, pred_masks):
    ng, npred = len(gt_masks), len(pred_masks)

    if ng == 0 or npred == 0:
        return {
            "gt": ng,
            "pred": npred,
            "matched": 0,
            "fp": npred,
            "fn": ng,
            "mean_iou_matched": 0.0,
            "mean_dice_matched": 0.0,
            "mean_iou_gt_adjusted": 0.0,
            "mean_dice_gt_adjusted": 0.0,
        }

    gt_areas = [int(m.sum()) for m in gt_masks]
    pred_areas = [int(m.sum()) for m in pred_masks]

    ious = np.zeros((ng, npred), dtype=np.float32)
    dices = np.zeros((ng, npred), dtype=np.float32)

    for i, gt in enumerate(gt_masks):
        for j, pred in enumerate(pred_masks):
            intersection = np.logical_and(gt, pred).sum()
            union = gt_areas[i] + pred_areas[j] - intersection

            if union:
                ious[i, j] = intersection / union

            total = gt_areas[i] + pred_areas[j]
            if total:
                dices[i, j] = 2 * intersection / total

    # Maximize the number of qualifying matches first,
    # then their IoU, under one-to-one assignment.
    valid = ious >= MATCH_THRESHOLD
    weights = valid.astype(np.float64) * 2 + ious.astype(np.float64)
    rows, cols = linear_sum_assignment(-weights)

    pairs = [
        (i, j) for i, j in zip(rows, cols)
        if valid[i, j]
    ]

    matched = len(pairs)
    matched_ious = [float(ious[i, j]) for i, j in pairs]
    matched_dices = [float(dices[i, j]) for i, j in pairs]

    return {
        "gt": ng,
        "pred": npred,
        "matched": matched,
        "fp": npred - matched,
        "fn": ng - matched,
        "mean_iou_matched": (
            float(np.mean(matched_ious)) if matched else 0.0
        ),
        "mean_dice_matched": (
            float(np.mean(matched_dices)) if matched else 0.0
        ),
        "mean_iou_gt_adjusted": (
            float(np.sum(matched_ious) / ng)
        ),
        "mean_dice_gt_adjusted": (
            float(np.sum(matched_dices) / ng)
        ),
        "_ious": matched_ious,
        "_dices": matched_dices,
    }


def main():
    model = YOLO(str(MODEL_PATH))
    images = sorted((DATASET / "images").glob("*.png"))
    assert len(images) == 50, f"Expected 50 images, got {len(images)}"

    totals = {
        "raw": {"gt": 0, "pred": 0, "matched": 0,
                "fp": 0, "fn": 0, "ious": [], "dices": []},
        "cleaned": {"gt": 0, "pred": 0, "matched": 0,
                    "fp": 0, "fn": 0, "ious": [], "dices": []},
    }

    rows = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read {image_path}")

        height, width = image.shape[:2]
        gt_masks = load_gt(
            DATASET / "labels" / f"{image_path.stem}.txt",
            height, width
        )

        raw, _ = extract_raw_predictions(model, image_path)
        cleaned = physical_instance_cleanup(raw)

        record = {"image": image_path.name}

        for name, detections in (
            ("raw", raw),
            ("cleaned", cleaned),
        ):
            result = evaluate(
                gt_masks,
                [item["mask"] for item in detections]
            )

            for key in (
                "gt", "pred", "matched", "fp", "fn",
                "mean_iou_matched", "mean_dice_matched",
                "mean_iou_gt_adjusted", "mean_dice_gt_adjusted",
            ):
                record[f"{name}_{key}"] = result[key]

            t = totals[name]
            for key in ("gt", "pred", "matched", "fp", "fn"):
                t[key] += result[key]

            t["ious"].extend(result.get("_ious", []))
            t["dices"].extend(result.get("_dices", []))

        rows.append(record)
        print(
            f"{image_path.name}: "
            f"GT={len(gt_masks)} | "
            f"Raw={len(raw)} | "
            f"Cleaned={len(cleaned)} | "
            f"Raw matched={record['raw_matched']} | "
            f"Cleaned matched={record['cleaned_matched']}"
        )

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== POST-PROCESSING COMPARISON ==========")
    print(f"Images: {len(images)}")
    print(f"YOLO image size: {IMAGE_SIZE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Duplicate IoU threshold: {PHYSICAL_DUPLICATE_IOU_THRESHOLD}")
    print(f"GT matching IoU threshold: {MATCH_THRESHOLD}")

    for name in ("raw", "cleaned"):
        t = totals[name]
        precision = t["matched"] / t["pred"] if t["pred"] else 0
        recall = t["matched"] / t["gt"] if t["gt"] else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        print(f"\n{name.upper()}")
        print(f"Ground truth: {t['gt']}")
        print(f"Predictions: {t['pred']}")
        print(f"Matched: {t['matched']}")
        print(f"False positives: {t['fp']}")
        print(f"False negatives: {t['fn']}")
        print(f"Precision: {precision:.4%}")
        print(f"Recall: {recall:.4%}")
        print(f"F1: {f1:.4%}")
        print(
            f"Mean IoU (matched): "
            f"{np.mean(t['ious']) if t['ious'] else 0:.4%}"
        )
        print(
            f"Mean Dice (matched): "
            f"{np.mean(t['dices']) if t['dices'] else 0:.4%}"
        )
        print(
            f"Mean IoU (GT-adjusted): "
            f"{sum(t['ious']) / t['gt'] if t['gt'] else 0:.4%}"
        )
        print(
            f"Mean Dice (GT-adjusted): "
            f"{sum(t['dices']) / t['gt'] if t['gt'] else 0:.4%}"
        )

    print(f"\nCSV saved: {OUTPUT}")


if __name__ == "__main__":
    main()
