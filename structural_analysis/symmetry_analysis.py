from typing import Dict

import numpy as np

from structural_analysis.quadrant_features import compute_quadrant_features


LEFT_RIGHT_BALANCE_THRESHOLD = 10.0
UPPER_LOWER_BALANCE_THRESHOLD = 15.0


def compute_regional_symmetry(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute regional dental-mask distribution using the original
    baseline quadrant convention:

        Q1 = upper-left
        Q2 = upper-right
        Q3 = lower-right
        Q4 = lower-left

    Values are descriptive image-level measurements and are not
    standalone clinical diagnoses.
    """
    quadrants = compute_quadrant_features(mask)

    left_percentage = quadrants["q1"] + quadrants["q4"]
    right_percentage = quadrants["q2"] + quadrants["q3"]

    upper_percentage = quadrants["q1"] + quadrants["q2"]
    lower_percentage = quadrants["q3"] + quadrants["q4"]

    left_right_diff = round(
        abs(left_percentage - right_percentage),
        4,
    )

    upper_lower_diff = round(
        abs(upper_percentage - lower_percentage),
        4,
    )

    return {
        "left_percentage": round(float(left_percentage), 4),
        "right_percentage": round(float(right_percentage), 4),
        "upper_percentage": round(float(upper_percentage), 4),
        "lower_percentage": round(float(lower_percentage), 4),
        "left_right_diff": float(left_right_diff),
        "upper_lower_diff": float(upper_lower_diff),
    }


def generate_structural_findings(mask: np.ndarray) -> Dict[str, object]:
    """
    Generate conservative structural-distribution findings using the
    thresholds retained from the original baseline dashboard.

    Balanced pattern:
        left-right difference < 10 percentage points
        AND
        upper-lower difference < 15 percentage points
    """
    binary_mask = (mask > 0).astype(np.uint8)

    if int(binary_mask.sum()) == 0:
        return {
            "measurements": compute_regional_symmetry(binary_mask),
            "balanced_pattern": None,
            "left_right_threshold_percentage_points":
                LEFT_RIGHT_BALANCE_THRESHOLD,
            "upper_lower_threshold_percentage_points":
                UPPER_LOWER_BALANCE_THRESHOLD,
            "summary": (
                "Structural symmetry assessment not applicable: "
                "no segmented dental region detected."
            ),
        }

    symmetry = compute_regional_symmetry(binary_mask)

    lr_diff = symmetry["left_right_diff"]
    ul_diff = symmetry["upper_lower_diff"]

    balanced = (
        lr_diff < LEFT_RIGHT_BALANCE_THRESHOLD
        and ul_diff < UPPER_LOWER_BALANCE_THRESHOLD
    )

    if balanced:
        summary = "Balanced regional dental-mask distribution pattern."
    else:
        summary = (
            "Noticeable regional difference in segmented "
            "dental-mask distribution."
        )

    return {
        "measurements": symmetry,
        "balanced_pattern": balanced,
        "left_right_threshold_percentage_points":
            LEFT_RIGHT_BALANCE_THRESHOLD,
        "upper_lower_threshold_percentage_points":
            UPPER_LOWER_BALANCE_THRESHOLD,
        "summary": summary,
    }
