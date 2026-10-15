from typing import Dict

import cv2
import numpy as np


def count_gaps(mask: np.ndarray) -> int:
    """
    Count horizontal gaps using the original baseline implementation.

    Only the central 10%-90% image width is examined.
    A background run longer than 10 columns is counted as one gap.
    """
    binary_mask = (mask > 0).astype(np.uint8)

    col_sum = np.sum(binary_mask, axis=0)
    tooth_cols = col_sum > 0

    gaps = 0
    in_gap = False
    gap_len = 0

    start = int(binary_mask.shape[1] * 0.10)
    end = int(binary_mask.shape[1] * 0.90)

    for has_tooth in tooth_cols[start:end]:
        if not has_tooth:
            in_gap = True
            gap_len += 1
        else:
            if in_gap and gap_len > 10:
                gaps += 1

            in_gap = False
            gap_len = 0

    if in_gap and gap_len > 10:
        gaps += 1

    return gaps


def compute_quadrant_features(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute Q1-Q4 regional features using the original baseline convention.

    Image convention:
        Q1 = upper-left
        Q2 = upper-right
        Q3 = lower-right
        Q4 = lower-left

    Values are percentages of total segmented pixels.
    """
    binary_mask = (mask > 0).astype(np.uint8)

    h, w = binary_mask.shape[:2]
    mid_x = w // 2
    mid_y = h // 2

    q1_raw = float(np.sum(binary_mask[:mid_y, :mid_x]))
    q2_raw = float(np.sum(binary_mask[:mid_y, mid_x:]))
    q3_raw = float(np.sum(binary_mask[mid_y:, mid_x:]))
    q4_raw = float(np.sum(binary_mask[mid_y:, :mid_x]))

    total = q1_raw + q2_raw + q3_raw + q4_raw

    if total == 0:
        q1 = q2 = q3 = q4 = 0.0
    else:
        q1 = round((q1_raw / total) * 100.0, 4)
        q2 = round((q2_raw / total) * 100.0, 4)
        q3 = round((q3_raw / total) * 100.0, 4)
        q4 = round((q4_raw / total) * 100.0, 4)

    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
    }


def compute_structural_features(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute the original baseline structural descriptors from a U-Net mask.

    In the hybrid pipeline these descriptors are retained for interpretable
    regional structural analysis; they are not Random Forest inputs.
    """
    binary_mask = (mask > 0).astype(np.uint8)

    h, w = binary_mask.shape[:2]

    quadrant_features = compute_quadrant_features(binary_mask)

    q1 = quadrant_features["q1"]
    q2 = quadrant_features["q2"]
    q3 = quadrant_features["q3"]
    q4 = quadrant_features["q4"]

    total = float(np.sum(binary_mask))

    total_area = round(
        (total / (h * w)) * 100.0,
        4,
    )

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    component_areas = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 20:
            component_areas.append(area)

    component_count = len(component_areas)

    largest_component_area = (
        max(component_areas)
        if component_areas
        else 0
    )

    largest_component_area = round(
        (largest_component_area / (h * w)) * 100.0,
        4,
    )

    left_density = q1 + q4
    right_density = q2 + q3

    upper_density = q1 + q2
    lower_density = q3 + q4

    left_right_diff = round(
        abs(left_density - right_density),
        4,
    )

    upper_lower_diff = round(
        abs(upper_density - lower_density),
        4,
    )

    gap_count = count_gaps(binary_mask)

    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "total_area": total_area,
        "component_count": component_count,
        "largest_component_area": largest_component_area,
        "left_right_diff": left_right_diff,
        "upper_lower_diff": upper_lower_diff,
        "gap_count": gap_count,
    }
