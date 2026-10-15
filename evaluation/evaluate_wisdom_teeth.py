from pathlib import Path

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

WISDOM_FDIS = ["18", "28", "48", "38"]


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

    correct_slots = 0
    total_slots = 0

    exact_cases = 0
    total_cases = 0

    tp_present = 0
    fp_present = 0
    fn_present = 0
    tn_absent = 0

    per_fdi = {
        fdi: {
            "correct": 0,
            "total": 0,
        }
        for fdi in WISDOM_FDIS
    }

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
            }

        gt_present = load_gt_present(
            image_path.stem
        )

        gt_wisdom = {
            fdi
            for fdi in WISDOM_FDIS
            if fdi in gt_present
        }

        pred_wisdom = {
            fdi
            for fdi in WISDOM_FDIS
            if fdi in pred_present
        }

        total_cases += 1

        if gt_wisdom == pred_wisdom:
            exact_cases += 1

        false_present = sorted(
            pred_wisdom - gt_wisdom,
            key=WISDOM_FDIS.index,
        )

        missed_present = sorted(
            gt_wisdom - pred_wisdom,
            key=WISDOM_FDIS.index,
        )

        if false_present or missed_present:
            case_errors.append({
                "image": image_path.name,
                "false_present": false_present,
                "missed_present": missed_present,
            })

        for fdi in WISDOM_FDIS:
            gt_is_present = fdi in gt_wisdom
            pred_is_present = fdi in pred_wisdom

            total_slots += 1
            per_fdi[fdi]["total"] += 1

            if gt_is_present == pred_is_present:
                correct_slots += 1
                per_fdi[fdi]["correct"] += 1

            if gt_is_present and pred_is_present:
                tp_present += 1

            elif not gt_is_present and pred_is_present:
                fp_present += 1

            elif gt_is_present and not pred_is_present:
                fn_present += 1

            else:
                tn_absent += 1

    accuracy = (
        correct_slots / total_slots
        if total_slots else 0.0
    )

    precision = (
        tp_present / (tp_present + fp_present)
        if tp_present + fp_present else 0.0
    )

    recall = (
        tp_present / (tp_present + fn_present)
        if tp_present + fn_present else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    exact_accuracy = (
        exact_cases / total_cases
        if total_cases else 0.0
    )

    print()
    print("======================================")
    print("RESOLVED FDI WISDOM-TEETH EVALUATION")
    print("======================================")

    print("Test images          :", total_cases)
    print("Wisdom slots         :", total_slots)

    print()
    print(
        "Wisdom-slot accuracy:",
        f"{accuracy * 100:.2f}%"
    )

    print(
        "Exact case accuracy :",
        f"{exact_accuracy * 100:.2f}%"
    )

    print()
    print(
        "Presence Precision  :",
        f"{precision * 100:.2f}%"
    )
    print(
        "Presence Recall     :",
        f"{recall * 100:.2f}%"
    )
    print(
        "Presence F1         :",
        f"{f1 * 100:.2f}%"
    )

    print()
    print("TP present :", tp_present)
    print("FP present :", fp_present)
    print("FN present :", fn_present)
    print("TN absent  :", tn_absent)

    print("\n--- PER-WISDOM-FDI ACCURACY ---")

    for fdi in WISDOM_FDIS:
        correct = per_fdi[fdi]["correct"]
        total = per_fdi[fdi]["total"]

        acc = correct / total if total else 0.0

        print(
            f"{fdi}: "
            f"{acc * 100:6.2f}% "
            f"({correct}/{total})"
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
