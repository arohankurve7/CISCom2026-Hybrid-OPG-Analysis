from pathlib import Path
from collections import Counter

from ultralytics import YOLO

from fdi.fdi_assigner import (
    MODEL_PATH,
    extract_raw_predictions,
    physical_instance_cleanup,
    split_arches,
    assign_missing_aware_fdi,
)


ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGES = (
    ROOT
    / "datasets"
    / "fdi_500"
    / "test"
    / "images"
)

TEST_LABELS = (
    ROOT
    / "datasets"
    / "fdi_500"
    / "test"
    / "labels"
)

FDI_NAMES = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]


def load_gt_fdis(stem):
    path = TEST_LABELS / f"{stem}.txt"

    if not path.exists():
        return []

    result = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        cls = int(line.split()[0])

        if 0 <= cls < len(FDI_NAMES):
            result.append(FDI_NAMES[cls])

    return result


def overlap_count(gt, prediction):
    gt_counter = Counter(gt)
    pred_counter = Counter(prediction)

    return sum(
        min(
            gt_counter[fdi],
            pred_counter[fdi],
        )
        for fdi in set(gt_counter) | set(pred_counter)
    )


def main():
    model = YOLO(str(MODEL_PATH))

    images = sorted(
        list(TEST_IMAGES.glob("*.png"))
        + list(TEST_IMAGES.glob("*.jpg"))
        + list(TEST_IMAGES.glob("*.jpeg"))
    )

    total_gt = 0
    total_overlap = 0
    exact_cases = 0

    case_results = []

    print()
    print("STABLE RESOLVED FDI EVALUATION")
    print("=" * 90)

    for image_path in images:
        raw, image = extract_raw_predictions(
            model,
            image_path,
        )

        physical = physical_instance_cleanup(
            raw
        )

        upper, lower = split_arches(
            physical,
            image.shape[0],
        )

        assignments = assign_missing_aware_fdi(
            upper,
            lower,
        )

        if assignments is None:
            resolved_prediction = []
        else:
            resolved_prediction = [
                str(item["resolved_fdi"])
                for item in assignments
            ]

        gt = load_gt_fdis(
            image_path.stem
        )

        overlap = overlap_count(
            gt,
            resolved_prediction,
        )

        gt_counter = Counter(gt)
        pred_counter = Counter(
            resolved_prediction
        )

        is_exact = (
            gt_counter == pred_counter
        )

        if is_exact:
            exact_cases += 1

        total_gt += len(gt)
        total_overlap += overlap

        pct = (
            100.0 * overlap / len(gt)
            if gt else 0.0
        )

        case_results.append({
            "image": image_path.name,
            "gt_count": len(gt),
            "physical_count": len(physical),
            "resolved_count": len(resolved_prediction),
            "overlap": overlap,
            "overlap_pct": pct,
            "exact": is_exact,
        })

        print(
            f"{image_path.name:20s} "
            f"GT={len(gt):2d} "
            f"Physical={len(physical):2d} "
            f"Resolved={len(resolved_prediction):2d} "
            f"Overlap={overlap:2d}/{len(gt):2d} "
            f"({pct:6.2f}%) "
            f"Exact={'YES' if is_exact else 'NO'}"
        )

    overall_overlap_pct = (
        100.0 * total_overlap / total_gt
        if total_gt else 0.0
    )

    exact_case_pct = (
        100.0 * exact_cases / len(images)
        if images else 0.0
    )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print("Test images        :", len(images))
    print("GT teeth           :", total_gt)
    print("Resolved overlap   :", f"{total_overlap}/{total_gt}")
    print(
        "Resolved overlap % :",
        f"{overall_overlap_pct:.2f}%"
    )
    print(
        "Exact cases        :",
        f"{exact_cases}/{len(images)} "
        f"({exact_case_pct:.2f}%)"
    )


if __name__ == "__main__":
    main()
