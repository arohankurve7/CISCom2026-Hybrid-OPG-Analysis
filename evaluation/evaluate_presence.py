from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO

from fdi.fdi_assigner import (
    MODEL_PATH,
    extract_raw_predictions,
    physical_instance_cleanup,
    split_arches,
    assign_missing_aware_fdi,
)
from fdi.tooth_registry import FDI_ORDER


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


def load_gt_present(stem):
    path = TEST_LABELS / f"{stem}.txt"

    if not path.exists():
        return set()

    present = set()

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        cls = int(line.split()[0])

        if 0 <= cls < len(FDI_NAMES):
            present.add(FDI_NAMES[cls])

    return present


def main():
    model = YOLO(str(MODEL_PATH))

    images = sorted(
        list(TEST_IMAGES.glob("*.png"))
        + list(TEST_IMAGES.glob("*.jpg"))
        + list(TEST_IMAGES.glob("*.jpeg"))
    )

    TP_missing = 0
    FP_missing = 0
    FN_missing = 0
    TN_present = 0

    exact_cases = 0
    total_cases = 0

    per_fdi = defaultdict(
        lambda: {
            "correct": 0,
            "wrong": 0,
        }
    )

    case_errors = []

    for image_path in images:
        raw, image = extract_raw_predictions(
            model,
            image_path,
        )

        physical = physical_instance_cleanup(raw)

        upper, lower = split_arches(
            physical,
            image.shape[0],
        )

        assignments = assign_missing_aware_fdi(
            upper,
            lower,
        )

        if assignments is None:
            pred_present = set()
        else:
            pred_present = {
                str(item["resolved_fdi"])
                for item in assignments
                if str(item["resolved_fdi"]) in FDI_ORDER
            }

        gt_present = load_gt_present(
            image_path.stem
        )

        gt_missing = set(FDI_ORDER) - gt_present
        pred_missing = set(FDI_ORDER) - pred_present

        total_cases += 1

        if gt_present == pred_present:
            exact_cases += 1

        false_present = sorted(
            pred_present - gt_present,
            key=FDI_ORDER.index,
        )

        missed_present = sorted(
            gt_present - pred_present,
            key=FDI_ORDER.index,
        )

        if false_present or missed_present:
            case_errors.append({
                "image": image_path.name,
                "false_present": false_present,
                "missed_present": missed_present,
            })

        for tooth in FDI_ORDER:
            gt_is_missing = tooth in gt_missing
            pred_is_missing = tooth in pred_missing

            if gt_is_missing and pred_is_missing:
                TP_missing += 1

            elif not gt_is_missing and pred_is_missing:
                FP_missing += 1

            elif gt_is_missing and not pred_is_missing:
                FN_missing += 1

            else:
                TN_present += 1

            if gt_is_missing == pred_is_missing:
                per_fdi[tooth]["correct"] += 1
            else:
                per_fdi[tooth]["wrong"] += 1

    total_slots = (
        TP_missing
        + FP_missing
        + FN_missing
        + TN_present
    )

    accuracy = (
        (TP_missing + TN_present) / total_slots
        if total_slots else 0
    )

    precision = (
        TP_missing / (TP_missing + FP_missing)
        if TP_missing + FP_missing else 0
    )

    recall = (
        TP_missing / (TP_missing + FN_missing)
        if TP_missing + FN_missing else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print()
    print("======================================")
    print("RESOLVED FDI MISSING/PRESENCE EVALUATION")
    print("======================================")

    print("Test images          :", total_cases)
    print("FDI slots evaluated  :", total_slots)

    print()
    print("Exact case matches   :", exact_cases)

    if total_cases:
        print(
            "Exact case accuracy  :",
            f"{100 * exact_cases / total_cases:.2f}%"
        )

    print()
    print(
        "Tooth-slot accuracy  :",
        f"{accuracy * 100:.2f}%"
    )

    print()
    print(
        "Missing Precision    :",
        f"{precision * 100:.2f}%"
    )
    print(
        "Missing Recall       :",
        f"{recall * 100:.2f}%"
    )
    print(
        "Missing F1           :",
        f"{f1 * 100:.2f}%"
    )

    print()
    print("TP missing :", TP_missing)
    print("FP missing :", FP_missing)
    print("FN missing :", FN_missing)
    print("TN present :", TN_present)

    print("\n--- PER-FDI ACCURACY ---")

    for tooth in FDI_ORDER:
        c = per_fdi[tooth]["correct"]
        w = per_fdi[tooth]["wrong"]

        acc = c / (c + w) if c + w else 0

        print(
            f"{tooth}: "
            f"{acc * 100:6.2f}% "
            f"(errors={w})"
        )

    print("\n--- CASES WITH ERRORS ---")

    for item in case_errors:
        print(
            f"{item['image']}: "
            f"false_present={item['false_present']} "
            f"missed_present={item['missed_present']}"
        )


if __name__ == "__main__":
    main()
