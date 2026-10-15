from pathlib import Path
from typing import Dict

import cv2
from ultralytics import YOLO

from segmentation.unet_dental_region import (
    load_unet_model,
    segment_opg_file,
)
from structural_analysis.quadrant_features import (
    compute_structural_features,
)
from structural_analysis.symmetry_analysis import (
    generate_structural_findings,
)
from fdi.fdi_assigner import (
    extract_raw_predictions,
    physical_instance_cleanup,
    split_arches,
    assign_missing_aware_fdi,
)
from fdi.tooth_registry import (
    build_tooth_registry,
    nearest_physical_instance,
    build_detected_tooth,
)
from fdi.presence_analysis import (
    derive_presence_summary,
    derive_wisdom_teeth_status,
)
from anatomy_insights.radiopacity_analysis import (
    analyze_registry_radiopacity,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_UNET_MODEL = (
    ROOT / "models" / "unet" / "dental_xray_seg.h5"
)

DEFAULT_YOLO_MODEL = (
    ROOT / "models" / "fdi_seg_v1" / "best.pt"
)


class HybridOPGInference:
    """
    Combined panoramic OPG inference pipeline.

    Branch 1:
        U-Net semantic dental-region segmentation
        -> regional structural descriptors

    Branch 2:
        YOLO tooth-instance segmentation
        -> physical cleanup
        -> anatomical FDI resolution
        -> fixed 32-position tooth registry
        -> presence and wisdom-tooth status
    """

    def __init__(
        self,
        unet_model_path=DEFAULT_UNET_MODEL,
        yolo_model_path=DEFAULT_YOLO_MODEL,
    ):
        self.unet_model_path = Path(unet_model_path)
        self.yolo_model_path = Path(yolo_model_path)

        if not self.unet_model_path.exists():
            raise FileNotFoundError(
                f"U-Net model not found: {self.unet_model_path}"
            )

        if not self.yolo_model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.yolo_model_path}"
            )

        self.unet_model = load_unet_model(
            self.unet_model_path
        )

        self.yolo_model = YOLO(
            str(self.yolo_model_path)
        )

    def run(self, image_path) -> Dict[str, object]:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"OPG image not found: {image_path}"
            )

        # -------------------------------------------------
        # Branch 1: U-Net regional structural analysis
        # -------------------------------------------------

        image, probability_map, dental_mask = segment_opg_file(
            image_path=image_path,
            model=self.unet_model,
        )

        structural_features = compute_structural_features(
            dental_mask
        )

        structural_findings = generate_structural_findings(
            dental_mask
        )

        # -------------------------------------------------
        # Branch 2: YOLO + anatomical FDI resolution
        # -------------------------------------------------

        raw_detections, yolo_image = extract_raw_predictions(
            self.yolo_model,
            image_path,
        )

        physical_instances = physical_instance_cleanup(
            raw_detections
        )

        # YOLO may return no rendered image when no tooth instances are detected
        # (e.g. edentulous radiographs). Fall back to the already-loaded input image.
        image_height = (
            yolo_image.shape[0]
            if yolo_image is not None
            else image.shape[0]
        )

        upper, lower = split_arches(
            physical_instances,
            image_height,
        )

        assignments = assign_missing_aware_fdi(
            upper,
            lower,
        )

        if assignments is None:
            assignments = []

        detected_teeth = []

        for assignment in assignments:
            physical = nearest_physical_instance(
                assignment,
                physical_instances,
            )

            detected_teeth.append(
                build_detected_tooth(
                    assignment,
                    physical,
                )
            )

        registry = build_tooth_registry(
            detected_teeth
        )

        presence = derive_presence_summary(
            registry
        )

        wisdom = derive_wisdom_teeth_status(
            registry
        )

        # -------------------------------------------------
        # Branch 3: conservative radiopacity analysis
        # -------------------------------------------------

        gray_image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if gray_image is None:
            raise ValueError(
                f"Unable to load grayscale OPG image: {image_path}"
            )

        radiopacity = analyze_registry_radiopacity(
            gray_image,
            registry,
        )

        return {
            "image": {
                "path": str(image_path),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            },

            "unet": {
                "segmented_pixels": int(dental_mask.sum()),
                "structural_features": structural_features,
                "structural_findings": structural_findings,
            },

            "fdi": {
                "raw_yolo_detections": len(raw_detections),
                "physical_instances": len(physical_instances),
                "upper_instances": len(upper),
                "lower_instances": len(lower),
                "resolved_instances": len(assignments),
                "assignments": assignments,
                "registry": registry,
            },

            "presence": presence,

            "wisdom_teeth": wisdom,

            "radiopacity": radiopacity,
        }
