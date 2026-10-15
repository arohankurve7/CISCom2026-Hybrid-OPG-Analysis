import cv2
import numpy as np


HIGH_RADIOPACITY_THRESHOLD = 240
MIN_COMPONENT_PIXELS = 450
ROI_EXPANSION_FRACTION = 0.25


def _expanded_bbox(bbox, image_shape):
    h, w = image_shape[:2]

    x1 = int(bbox["x1"])
    y1 = int(bbox["y1"])
    x2 = int(bbox["x2"])
    y2 = int(bbox["y2"])

    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)

    pad_x = int(round(bw * ROI_EXPANSION_FRACTION))
    pad_y = int(round(bh * ROI_EXPANSION_FRACTION))

    return {
        "x1": max(0, x1 - pad_x),
        "y1": max(0, y1 - pad_y),
        "x2": min(w - 1, x2 + pad_x),
        "y2": min(h - 1, y2 + pad_y),
    }


def build_dental_search_mask(image_shape, registry):
    search_mask = np.zeros(
        image_shape[:2],
        dtype=np.uint8,
    )

    for tooth in registry:
        if tooth.get("status") != "present":
            continue

        bbox = tooth.get("bbox")

        if not bbox:
            continue

        box = _expanded_bbox(
            bbox,
            image_shape,
        )

        search_mask[
            box["y1"]:box["y2"] + 1,
            box["x1"]:box["x2"] + 1,
        ] = 1

    return search_mask


def detect_high_radiopacity_components(gray_image, registry):
    """
    Conservative high-radiopacity candidate detector.

    Detects coherent very-high-density regions only within the
    FDI-localized dental search region.

    This does NOT classify crown, bridge, implant, filling, or RCT.
    """

    search_mask = build_dental_search_mask(
        gray_image.shape,
        registry,
    )

    bright_mask = (
        (search_mask > 0)
        & (gray_image > HIGH_RADIOPACITY_THRESHOLD)
    ).astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            bright_mask,
            connectivity=8,
        )
    )

    components = []

    for label in range(1, num_labels):
        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < MIN_COMPONENT_PIXELS:
            continue

        x = int(
            stats[label, cv2.CC_STAT_LEFT]
        )
        y = int(
            stats[label, cv2.CC_STAT_TOP]
        )
        w = int(
            stats[label, cv2.CC_STAT_WIDTH]
        )
        h = int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )

        components.append({
            "area_pixels": area,
            "centroid": {
                "x": round(float(centroids[label][0]), 2),
                "y": round(float(centroids[label][1]), 2),
            },
            "bbox": {
                "x1": x,
                "y1": y,
                "x2": x + w - 1,
                "y2": y + h - 1,
            },
            "interpretation":
                "High-radiopacity restorative-pattern candidate.",
        })

    components.sort(
        key=lambda item: item["area_pixels"],
        reverse=True,
    )

    return {
        "candidate_count": len(components),
        "components": components,
        "threshold_intensity":
            HIGH_RADIOPACITY_THRESHOLD,
        "minimum_component_pixels":
            MIN_COMPONENT_PIXELS,
        "method":
            "FDI-localized expanded dental region with conservative "
            "high-radiopacity connected-component analysis.",
        "disclaimer":
            "These findings are radiographic image-analysis candidates "
            "and are not confirmed restorative diagnoses. Low-contrast "
            "or less-radiopaque restorative hardware may not be detected.",
    }


def analyze_registry_radiopacity(
    gray_image,
    registry,
):
    return detect_high_radiopacity_components(
        gray_image,
        registry,
    )
