from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf


DEFAULT_INPUT_SIZE = (512, 512)
DEFAULT_THRESHOLD = 0.5


class CompatibleConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    """
    Compatibility wrapper for legacy H5 models whose serialized
    Conv2DTranspose config contains groups=1.
    """

    def __init__(self, *args, groups=1, **kwargs):
        if groups != 1:
            raise ValueError(
                f"Unsupported Conv2DTranspose groups={groups}; "
                "compatibility wrapper supports groups=1 only."
            )
        super().__init__(*args, **kwargs)


def load_unet_model(model_path: str):
    """Load the pretrained U-Net segmentation model."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"U-Net model not found: {model_path}")

    return tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "Conv2DTranspose": CompatibleConv2DTranspose,
        },
        compile=False,
    )


def load_grayscale_opg(image_path: str) -> np.ndarray:
    """Load a panoramic OPG image in grayscale."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"OPG image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def prepare_unet_input(
    image: np.ndarray,
    input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> np.ndarray:
    """
    Prepare OPG for pretrained U-Net.

    Pipeline:
        grayscale
        -> resize 512 x 512
        -> normalize [0, 1]
        -> shape (1, 512, 512, 1)
    """
    resized = cv2.resize(
        image,
        input_size,
        interpolation=cv2.INTER_AREA,
    )

    normalized = resized.astype(np.float32) / 255.0

    tensor = np.expand_dims(normalized, axis=0)
    tensor = np.expand_dims(tensor, axis=-1)

    return tensor


def segment_dental_region(
    image: np.ndarray,
    model,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform semantic dental-region segmentation using pretrained U-Net.

    Returns:
        probability_map: U-Net probability map at original OPG resolution
        binary_mask: binary dental-region mask (0/1)
    """
    original_height, original_width = image.shape[:2]

    input_tensor = prepare_unet_input(image)

    prediction = model.predict(input_tensor, verbose=0)
    probability_map_512 = prediction[0, :, :, 0]

    probability_map = cv2.resize(
        probability_map_512,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )

    binary_mask = cv2.resize(
        (probability_map_512 > threshold).astype(np.uint8),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return probability_map, binary_mask


def segment_opg_file(
    image_path: str,
    model,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run U-Net semantic segmentation directly on an OPG file."""
    image = load_grayscale_opg(image_path)

    probability_map, binary_mask = segment_dental_region(
        image=image,
        model=model,
        threshold=threshold,
    )

    return image, probability_map, binary_mask
