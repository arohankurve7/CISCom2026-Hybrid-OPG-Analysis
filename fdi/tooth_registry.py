import math

import cv2
import numpy as np


# =============================================================================
# FDI permanent dentition — fixed 32-position registry
# =============================================================================

FDI_ORDER = [
    "18", "17", "16", "15", "14", "13", "12", "11",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "48", "47", "46", "45", "44", "43", "42", "41",
    "31", "32", "33", "34", "35", "36", "37", "38",
]


def tooth_type_from_fdi(fdi):
    position = int(str(fdi)[1])

    mapping = {
        1: "Central Incisor",
        2: "Lateral Incisor",
        3: "Canine",
        4: "First Premolar",
        5: "Second Premolar",
        6: "First Molar",
        7: "Second Molar",
        8: "Third Molar",
    }

    return mapping[position]


def quadrant_metadata(fdi):
    quadrant = str(fdi)[0]

    mapping = {
        "1": {
            "quadrant": "Q1",
            "arch": "Maxillary",
            "side": "Right",
        },
        "2": {
            "quadrant": "Q2",
            "arch": "Maxillary",
            "side": "Left",
        },
        "3": {
            "quadrant": "Q3",
            "arch": "Mandibular",
            "side": "Left",
        },
        "4": {
            "quadrant": "Q4",
            "arch": "Mandibular",
            "side": "Right",
        },
    }

    return mapping[quadrant]


def anatomical_name(fdi):
    meta = quadrant_metadata(fdi)
    tooth_type = tooth_type_from_fdi(fdi)

    if tooth_type == "Third Molar":
        tooth_type = "Third Molar (Wisdom Tooth)"

    return (
        f'{meta["arch"]} '
        f'{meta["side"]} '
        f'{tooth_type}'
    )


def mask_to_polygon(binary_mask):
    mask = binary_mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return []

    largest = max(
        contours,
        key=cv2.contourArea,
    )

    epsilon = (
        0.003
        * cv2.arcLength(
            largest,
            True,
        )
    )

    approx = cv2.approxPolyDP(
        largest,
        epsilon,
        True,
    )

    return [
        {
            "x": int(point[0][0]),
            "y": int(point[0][1]),
        }
        for point in approx
    ]


def nearest_physical_instance(
    assignment,
    physical_instances,
):
    """
    Reconnect a resolved FDI assignment to the nearest physical
    tooth instance so segmentation geometry can be retained.
    """
    ax = float(assignment["centroid_x"])
    ay = float(assignment["centroid_y"])

    return min(
        physical_instances,
        key=lambda item: math.hypot(
            float(item["centroid_x"]) - ax,
            float(item["centroid_y"]) - ay,
        ),
    )


def build_detected_tooth(
    assignment,
    physical,
):
    """
    Build registry metadata for a detected FDI tooth.
    """
    fdi = str(assignment["resolved_fdi"])

    meta = quadrant_metadata(fdi)
    tooth_type = tooth_type_from_fdi(fdi)

    binary_mask = physical["mask"]

    ys, xs = np.where(binary_mask)

    if len(xs):
        bbox = {
            "x1": int(xs.min()),
            "y1": int(ys.min()),
            "x2": int(xs.max()),
            "y2": int(ys.max()),
        }
    else:
        bbox = None

    return {
        "fdi": fdi,
        "anatomical_name": anatomical_name(fdi),
        "tooth_type": tooth_type,
        "quadrant": meta["quadrant"],
        "arch": meta["arch"],
        "side": meta["side"],
        "status": "present",
        "selectable": True,
        "centroid": {
            "x": int(round(physical["centroid_x"])),
            "y": int(round(physical["centroid_y"])),
        },
        "bbox": bbox,
        "mask_polygon": mask_to_polygon(binary_mask),
        "pixel_area": int(physical["pixel_area"]),
    }


def build_not_detected_tooth(fdi):
    """
    Build an empty registry position for an FDI tooth that was not detected.
    """
    fdi = str(fdi)
    meta = quadrant_metadata(fdi)

    return {
        "fdi": fdi,
        "anatomical_name": anatomical_name(fdi),
        "tooth_type": tooth_type_from_fdi(fdi),
        "quadrant": meta["quadrant"],
        "arch": meta["arch"],
        "side": meta["side"],
        "status": "not_detected",
        "selectable": False,
        "centroid": None,
        "bbox": None,
        "mask_polygon": [],
    }


def build_tooth_registry(detected_teeth):
    """
    Build the fixed 32-position permanent-dentition FDI registry.

    Every expected FDI position is represented exactly once.
    Detected teeth retain their segmentation metadata; positions not
    detected by the pipeline are represented as 'not_detected'.
    """
    detected_by_fdi = {
        str(tooth["fdi"]): tooth
        for tooth in detected_teeth
    }

    registry = []

    for fdi in FDI_ORDER:
        if fdi in detected_by_fdi:
            registry.append(
                detected_by_fdi[fdi]
            )
        else:
            registry.append(
                build_not_detected_tooth(fdi)
            )

    return registry


def get_present_fdis(registry):
    """Return FDI numbers marked present in the registry."""
    return [
        item["fdi"]
        for item in registry
        if item["status"] == "present"
    ]


def get_not_detected_fdis(registry):
    """
    Return FDI positions not detected by the pipeline.

    Note:
    'not_detected' is intentionally retained here instead of
    automatically calling the tooth clinically 'missing'.
    Missing-tooth assessment is handled separately.
    """
    return [
        item["fdi"]
        for item in registry
        if item["status"] == "not_detected"
    ]
