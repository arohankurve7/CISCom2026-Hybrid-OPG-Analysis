
from pathlib import Path
import csv

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models/fdi_seg_v1/best.pt")
DATASET = Path(__file__).resolve().parent.parent / "datasets/fdi_500/test"
OUTPUT = Path(__file__).resolve().parent / "segmentation_metrics_50.csv"

CONF = 0.25
IMGSZ = 1024
MATCH_IOU = 0.50


def ground_truth_masks(label_path, height, width):
    masks = []

    for line in label_path.read_text().splitlines():
        values = list(map(float, line.split()))
        if len(values) < 7 or (len(values) - 1) % 2:
            continue

        polygon = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
        polygon[:, 0] *= width
        polygon[:, 1] *= height

        points = np.round(polygon).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)

        if mask.any():
            masks.append(mask.astype(bool))

    return masks


def predicted_masks(result, height, width):
    if result.masks is None:
        return []

    masks = []

    for polygon in result.masks.xy:
        points = np.round(polygon).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)

        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)

        if mask.any():
            masks.append(mask.astype(bool))

    return masks


def calculate_scores(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    iou = intersection / union if union else 0.0
    denominator = gt.sum() + pred.sum()
    dice = 2 * intersection / denominator if denominator else 0.0

    return float(iou), float(dice)


def evaluate_image(gt_masks, pred_masks):
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    matrix = np.zeros((n_gt, n_pred))
    dice_matrix = np.zeros((n_gt, n_pred))

    for i, gt in enumerate(gt_masks):
        for j, pred in enumerate(pred_masks):
            matrix[i, j], dice_matrix[i, j] = calculate_scores(gt, pred)

    matches = []

    if n_gt and n_pred:
        rows, cols = linear_sum_assignment(-matrix)

        for i, j in zip(rows, cols):
            if matrix[i, j] >= MATCH_IOU:
                matches.append((i, j))

    ious = [matrix[i, j] for i, j in matches]
    dices = [dice_matrix[i, j] for i, j in matches]

    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp

    return {
        "gt_count": n_gt,
        "pred_count": n_pred,
        "matched": tp,
        "false_positive": fp,
        "false_negative": fn,
        "iou_sum": sum(ious),
        "dice_sum": sum(dices),
    }


def main():
    model = YOLO(MODEL_PATH)
    images = sorted(
        p for p in (DATASET / "images").iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    rows = []
    totals = {
        "gt_count": 0,
        "pred_count": 0,
        "matched": 0,
        "false_positive": 0,
        "false_negative": 0,
        "iou_sum": 0.0,
        "dice_sum": 0.0,
    }

    for image_path in images:
        label_path = DATASET / "labels" / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing ground truth: {label_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        height, width = image.shape[:2]
        gt = ground_truth_masks(label_path, height, width)

        result = model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False,
        )[0]

        pred = predicted_masks(result, height, width)
        metrics = evaluate_image(gt, pred)

        rows.append({"image": image_path.name, **metrics})

        for key in totals:
            totals[key] += metrics[key]

        print(
            f"{image_path.name}: "
            f"GT={len(gt)}, Pred={len(pred)}, "
            f"Matched={metrics['matched']}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", *totals.keys()])
        writer.writeheader()
        writer.writerows(rows)

    matched = totals["matched"]
    gt_count = totals["gt_count"]

    mean_iou = totals["iou_sum"] / matched if matched else 0
    mean_dice = totals["dice_sum"] / matched if matched else 0

    # Missed GT instances contribute zero to these averages.
    gt_mean_iou = totals["iou_sum"] / gt_count if gt_count else 0
    gt_mean_dice = totals["dice_sum"] / gt_count if gt_count else 0

    print("\n========== FINAL RESULTS ==========")
    print(f"Test images: {len(images)}")
    print(f"Ground-truth teeth: {gt_count}")
    print(f"Predicted teeth: {totals['pred_count']}")
    print(f"Matched teeth (IoU >= {MATCH_IOU}): {matched}")
    print(f"False positives: {totals['false_positive']}")
    print(f"False negatives: {totals['false_negative']}")
    print(f"Mean IoU (matched): {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"Mean Dice (matched): {mean_dice:.4f} ({mean_dice*100:.2f}%)")
    print(f"Mean IoU (GT-adjusted): {gt_mean_iou:.4f} ({gt_mean_iou*100:.2f}%)")
    print(f"Mean Dice (GT-adjusted): {gt_mean_dice:.4f} ({gt_mean_dice*100:.2f}%)")
    print(f"Results saved: {OUTPUT}")


if __name__ == "__main__":
    main()
