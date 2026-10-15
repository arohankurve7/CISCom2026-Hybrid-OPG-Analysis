from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from pipeline.hybrid_inference import HybridOPGInference
from segmentation.unet_dental_region import segment_opg_file


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "dashboard" / "uploads"
GENERATED_DIR = ROOT / "dashboard" / "generated"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

app = Flask(__name__)
pipeline = HybridOPGInference()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/generated/<path:filename>")
def generated_file(filename):
    return send_from_directory(GENERATED_DIR, filename)


def create_segmented_view(image_path, output_path):
    """
    Create a visualization of the pretrained U-Net dental-region mask.

    The overlay represents semantic dental-region segmentation only.
    A vertical reference line is retained for visual left/right orientation.
    Anatomical Q1-Q4 statistics are calculated separately from the
    FDI-resolved tooth registry and are not defined by image geometry.
    """
    image, _, mask = segment_opg_file(
        str(image_path),
        pipeline.unet_model
    )

    if len(image.shape) == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    h, w = base.shape[:2]

    overlay = base.copy()

    # Uniform overlay: do not imply geometric anatomical quadrants.
    region_mask = mask > 0
    overlay[region_mask] = (80, 190, 80)

    blended = cv2.addWeighted(base, 0.68, overlay, 0.32, 0)

    # Visual left/right reference only.
    cv2.line(
        blended,
        (w // 2, 0),
        (w // 2, h),
        (230, 230, 230),
        2
    )

    cv2.imwrite(str(output_path), blended)



def create_analysis_overlay(image_path, registry, output_path):
    """
    Combined AI visualization:
    pretrained U-Net dental-region segmentation + resolved FDI labels.
    """
    image, _, mask = segment_opg_file(
        str(image_path),
        pipeline.unet_model
    )

    if len(image.shape) == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    h, w = base.shape[:2]

    # U-Net semantic dental-region overlay.
    overlay = base.copy()
    overlay[mask > 0] = (80, 190, 80)

    blended = cv2.addWeighted(
        base,
        0.68,
        overlay,
        0.32,
        0
    )

    # Vertical orientation reference only.
    cv2.line(
        blended,
        (w // 2, 0),
        (w // 2, h),
        (230, 230, 230),
        2
    )

    # Resolved FDI tooth numbers.
    font_scale = max(0.58, w / 2100)
    thickness = max(2, int(w / 1500))

    for tooth in registry:
        if tooth.get("status") != "present":
            continue

        centroid = tooth.get("centroid")
        fdi = str(tooth.get("fdi", ""))

        if not centroid or not fdi:
            continue

        x = int(centroid["x"])
        y = int(centroid["y"])

        (tw, th), _ = cv2.getTextSize(
            fdi,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        tx = max(2, min(w - tw - 2, x - tw // 2))
        ty = max(th + 2, min(h - 2, y))

        # Dark FDI label for clean contrast on the green overlay.
        cv2.putText(
            blended,
            fdi,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (25, 25, 25),
            thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), blended)

def create_numbered_view(image_path, registry, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to load OPG for numbered view.")

    h, w = image.shape[:2]

    font_scale = max(0.42, w / 2600)
    thickness = max(1, int(w / 1700))

    for tooth in registry:
        if tooth.get("status") != "present":
            continue

        centroid = tooth.get("centroid")
        fdi = str(tooth.get("fdi", ""))

        if not centroid or not fdi:
            continue

        x = int(centroid["x"])
        y = int(centroid["y"])

        (tw, th), _ = cv2.getTextSize(
            fdi,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        tx = max(2, min(w - tw - 2, x - tw // 2))
        ty = max(th + 2, min(h - 2, y))

        # Text halo for readability. No box, no polygon.
        cv2.putText(
            image,
            fdi,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 3,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            fdi,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No OPG image provided."}), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(uploaded.filename)
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Supported formats: PNG, JPG, JPEG."
        }), 400

    image_path = UPLOAD_DIR / filename
    uploaded.save(image_path)

    try:
        result = pipeline.run(image_path)

        registry = result["fdi"]["registry"]

        present = [
            str(t["fdi"])
            for t in registry
            if t["status"] == "present"
        ]

        not_detected = [
            str(t["fdi"])
            for t in registry
            if t["status"] != "present"
        ]

        stem = Path(filename).stem

        segmented_name = f"{stem}_segmented.png"
        numbered_name = f"{stem}_numbered.png"
        analysis_overlay_name = f"{stem}_analysis_overlay.png"

        create_segmented_view(
            image_path,
            GENERATED_DIR / segmented_name
        )

        create_numbered_view(
            image_path,
            registry,
            GENERATED_DIR / numbered_name
        )

        create_analysis_overlay(
            image_path,
            registry,
            GENERATED_DIR / analysis_overlay_name
        )

        # -------------------------------------------------
        # Bilateral FDI tooth-distribution insight
        # -------------------------------------------------
        bilateral_pairs = [
            ("18", "28"), ("17", "27"), ("16", "26"), ("15", "25"),
            ("14", "24"), ("13", "23"), ("12", "22"), ("11", "21"),
            ("48", "38"), ("47", "37"), ("46", "36"), ("45", "35"),
            ("44", "34"), ("43", "33"), ("42", "32"), ("41", "31"),
        ]

        present_set = set(present)

        concordant_pairs = 0
        asymmetric_pairs = []

        for right_fdi, left_fdi in bilateral_pairs:
            right_present = right_fdi in present_set
            left_present = left_fdi in present_set

            if right_present == left_present:
                concordant_pairs += 1
            else:
                asymmetric_pairs.append(
                    f"{right_fdi}-{left_fdi}"
                )

        bilateral_insight = {
            "total_pairs": len(bilateral_pairs),
            "concordant_pairs": concordant_pairs,
            "asymmetric_count": len(asymmetric_pairs),
            "asymmetric_pairs": asymmetric_pairs,
        }

        radiopacity_count = result["radiopacity"]["candidate_count"]

        # -------------------------------------------------
        # Dashboard regional tooth-area distribution
        # -------------------------------------------------
        # Anatomical quadrants are defined by resolved FDI positions,
        # not by geometric splitting of the panoramic image.
        quadrant_areas = {
            "Q1": 0,
            "Q2": 0,
            "Q3": 0,
            "Q4": 0,
        }

        quadrant_counts = {
            "Q1": 0,
            "Q2": 0,
            "Q3": 0,
            "Q4": 0,
        }

        for tooth in registry:
            if tooth.get("status") != "present":
                continue

            quadrant = tooth.get("quadrant")
            pixel_area = tooth.get("pixel_area") or 0

            if quadrant in quadrant_areas:
                quadrant_areas[quadrant] += pixel_area
                quadrant_counts[quadrant] += 1

        total_tooth_area = sum(quadrant_areas.values())

        if total_tooth_area > 0:
            quadrant_pct = {
                q: quadrant_areas[q] / total_tooth_area * 100.0
                for q in quadrant_areas
            }

            patient_right = quadrant_pct["Q1"] + quadrant_pct["Q4"]
            patient_left = quadrant_pct["Q2"] + quadrant_pct["Q3"]
            upper_arch = quadrant_pct["Q1"] + quadrant_pct["Q2"]
            lower_arch = quadrant_pct["Q3"] + quadrant_pct["Q4"]

            dashboard_quadrants = {
                "q1": round(quadrant_pct["Q1"], 4),
                "q2": round(quadrant_pct["Q2"], 4),
                "q3": round(quadrant_pct["Q3"], 4),
                "q4": round(quadrant_pct["Q4"], 4),

                "q1_count": quadrant_counts["Q1"],
                "q2_count": quadrant_counts["Q2"],
                "q3_count": quadrant_counts["Q3"],
                "q4_count": quadrant_counts["Q4"],

                "patient_right": round(patient_right, 4),
                "patient_left": round(patient_left, 4),
                "upper_arch": round(upper_arch, 4),
                "lower_arch": round(lower_arch, 4),

                "left_right_diff": round(
                    abs(patient_right - patient_left),
                    4
                ),
                "upper_lower_diff": round(
                    abs(upper_arch - lower_arch),
                    4
                ),

                "total_tooth_area": int(total_tooth_area),
                "applicable": True,
            }
        else:
            # Valid no-detection / edentulous output.
            dashboard_quadrants = {
                "q1": 0.0,
                "q2": 0.0,
                "q3": 0.0,
                "q4": 0.0,

                "q1_count": 0,
                "q2_count": 0,
                "q3_count": 0,
                "q4_count": 0,

                "patient_right": 0.0,
                "patient_left": 0.0,
                "upper_arch": 0.0,
                "lower_arch": 0.0,

                "left_right_diff": 0.0,
                "upper_lower_diff": 0.0,

                "total_tooth_area": 0,
                "applicable": False,
            }

        response = {
            "filename": filename,

            "images": {
                "original": f"/uploads/{filename}",
                "segmented": f"/generated/{segmented_name}",
                "numbered": f"/generated/{numbered_name}",
                "analysis_overlay": f"/generated/{analysis_overlay_name}",
            },

            "tooth_analysis": {
                "present_count": len(present),
                "not_detected_count": len(not_detected),
                "present_fdi": present,
                "not_detected_fdi": not_detected,
            },

            "wisdom_teeth": result["wisdom_teeth"],

            "bilateral_insight": bilateral_insight,

            "structural_analysis": {
                "method": "FDI-based regional tooth-area distribution",
                "features": dashboard_quadrants,
            },

            "radiopacity": {
                "candidate_present": radiopacity_count > 0,
                "interpretation": (
                    "High-radiopacity restorative-pattern material candidate detected."
                    if radiopacity_count > 0
                    else
                    "No high-radiopacity restorative-pattern material candidate detected."
                ),
                "disclaimer": result["radiopacity"]["disclaimer"],
            },
        }

        return jsonify(response)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
