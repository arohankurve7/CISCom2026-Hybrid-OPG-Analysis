from typing import Dict, List, Set

from fdi.tooth_registry import FDI_ORDER


WISDOM_TEETH = {"18", "28", "38", "48"}


def normalize_fdi_set(fdis) -> Set[str]:
    """
    Normalize an iterable of FDI values to strings and retain only
    valid permanent-dentition FDI positions.
    """
    valid = set(FDI_ORDER)

    return {
        str(fdi)
        for fdi in fdis
        if str(fdi) in valid
    }


def present_fdis_from_registry(registry) -> Set[str]:
    """
    Extract FDI positions marked present in the 32-position registry.
    """
    return {
        str(item["fdi"])
        for item in registry
        if item.get("status") == "present"
    }


def not_detected_fdis_from_registry(registry) -> Set[str]:
    """
    Extract FDI positions not detected by the inference pipeline.

    This is an inference-state description and should not, by itself,
    be treated as clinical proof that a tooth is missing.
    """
    return {
        str(item["fdi"])
        for item in registry
        if item.get("status") == "not_detected"
    }


def derive_presence_summary(registry) -> Dict[str, object]:
    """
    Summarize tooth presence from the fixed 32-position FDI registry.
    """
    present = present_fdis_from_registry(registry)
    not_detected = not_detected_fdis_from_registry(registry)

    return {
        "present_fdis": sorted(
            present,
            key=FDI_ORDER.index,
        ),
        "not_detected_fdis": sorted(
            not_detected,
            key=FDI_ORDER.index,
        ),
        "present_count": len(present),
        "not_detected_count": len(not_detected),
        "total_expected_positions": len(FDI_ORDER),
    }


def derive_wisdom_teeth_status(registry) -> Dict[str, object]:
    """
    Derive predicted third-molar presence from FDI registry positions:
    18, 28, 38 and 48.
    """
    present = present_fdis_from_registry(registry)

    per_tooth = {
        fdi: fdi in present
        for fdi in sorted(
            WISDOM_TEETH,
            key=FDI_ORDER.index,
        )
    }

    present_wisdom = [
        fdi
        for fdi, is_present in per_tooth.items()
        if is_present
    ]

    absent_or_not_detected = [
        fdi
        for fdi, is_present in per_tooth.items()
        if not is_present
    ]

    return {
        "per_tooth": per_tooth,
        "present_wisdom_teeth": present_wisdom,
        "absent_or_not_detected_wisdom_teeth": absent_or_not_detected,
        "present_count": len(present_wisdom),
        "all_four_present": len(present_wisdom) == 4,
    }


def derive_missing_from_ground_truth(gt_present_fdis) -> Set[str]:
    """
    Derive ground-truth missing FDI positions from annotated tooth presence.

    This mirrors the evaluation concept used in the existing
    dental-xray-analysis project:

        missing = expected FDI positions - present FDI positions
    """
    gt_present = normalize_fdi_set(gt_present_fdis)

    return set(FDI_ORDER) - gt_present


def derive_predicted_missing(pred_present_fdis) -> Set[str]:
    """
    Derive predicted missing positions from predicted FDI presence.

    This function is intended for evaluation against known FDI
    ground truth. Clinical interpretation should remain conservative
    because a model non-detection can also arise from inference error.
    """
    pred_present = normalize_fdi_set(pred_present_fdis)

    return set(FDI_ORDER) - pred_present


def compare_presence(
    gt_present_fdis,
    pred_present_fdis,
) -> Dict[str, object]:
    """
    Compare ground-truth and predicted tooth presence/missing sets.

    Returns set-level TP/FP/FN/TN counts for missing-tooth assessment.
    """
    expected = set(FDI_ORDER)

    gt_present = normalize_fdi_set(gt_present_fdis)
    pred_present = normalize_fdi_set(pred_present_fdis)

    gt_missing = expected - gt_present
    pred_missing = expected - pred_present

    tp = len(gt_missing & pred_missing)
    fp = len(pred_missing - gt_missing)
    fn = len(gt_missing - pred_missing)
    tn = len(gt_present & pred_present)

    return {
        "gt_present": sorted(gt_present, key=FDI_ORDER.index),
        "pred_present": sorted(pred_present, key=FDI_ORDER.index),
        "gt_missing": sorted(gt_missing, key=FDI_ORDER.index),
        "pred_missing": sorted(pred_missing, key=FDI_ORDER.index),
        "tp_missing": tp,
        "fp_missing": fp,
        "fn_missing": fn,
        "tn_present": tn,
    }
