from ultralytics import YOLO

from evaluation.evaluate_presence import (
    TEST_IMAGES,
    TEST_LABELS,
    load_gt_present,
)
from fdi.fdi_assigner import (
    MODEL_PATH,
    extract_raw_predictions,
    physical_instance_cleanup,
    split_arches,
    assign_missing_aware_fdi,
)
from fdi.tooth_registry import FDI_ORDER


BILATERAL_PAIRS = [
    ("18", "28"), ("17", "27"), ("16", "26"), ("15", "25"),
    ("14", "24"), ("13", "23"), ("12", "22"), ("11", "21"),
    ("48", "38"), ("47", "37"), ("46", "36"), ("45", "35"),
    ("44", "34"), ("43", "33"), ("42", "32"), ("41", "31"),
]


def exact_state(present, right, left):
    return (right in present, left in present)


def category_state(present, right, left):
    right_present, left_present = exact_state(present, right, left)

    if right_present and left_present:
        return "both_present"
    if not right_present and not left_present:
        return "both_absent"
    return "asymmetric"


def main():
    model = YOLO(str(MODEL_PATH))

    images = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    exact_correct = 0
    category_correct = 0
    total_pairs = 0

    for image_path in images:
        label_path = TEST_LABELS / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)

        gt_present = load_gt_present(image_path.stem)

        raw, image = extract_raw_predictions(model, image_path)
        physical = physical_instance_cleanup(raw)
        upper, lower = split_arches(physical, image.shape[0])
        assignments = assign_missing_aware_fdi(upper, lower) or []

        pred_present = {
            str(item["resolved_fdi"])
            for item in assignments
            if str(item["resolved_fdi"]) in FDI_ORDER
        }

        for right, left in BILATERAL_PAIRS:
            total_pairs += 1

            if exact_state(pred_present, right, left) == exact_state(
                gt_present, right, left
            ):
                exact_correct += 1

            if category_state(pred_present, right, left) == category_state(
                gt_present, right, left
            ):
                category_correct += 1

    print("\n=== BILATERAL FDI EVALUATION ===")
    print("Test images            :", len(images))
    print("Bilateral pairs        :", total_pairs)
    print(
        "Exact left/right state :",
        f"{exact_correct}/{total_pairs}",
        f"({100 * exact_correct / total_pairs:.2f}%)",
    )
    print(
        "Three-category state   :",
        f"{category_correct}/{total_pairs}",
        f"({100 * category_correct / total_pairs:.2f}%)",
    )


if __name__ == "__main__":
    main()
