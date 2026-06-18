"""Fundus image operations: border cropping, circular masking, sharpening.

All functions expect and return RGB arrays (the notebook version contained a
stray BGR/RGB conversion inside ``circle_crop``; here color handling is the
caller's responsibility).
"""

import cv2
import numpy as np


def crop_image(image: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """Crop away near-black borders around the fundus disc."""
    if not 0 <= tolerance <= 255:
        raise ValueError("The tolerance must be between 0 and 255.")

    if image.ndim == 2:
        mask = image > tolerance
        if not mask.any():
            return image
        return image[np.ix_(mask.any(1), mask.any(0))]

    if image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray_image > tolerance
        if not mask.any():
            return image
        cropped_channels = [
            image[:, :, i][np.ix_(mask.any(1), mask.any(0))] for i in range(3)
        ]
        return np.stack(cropped_channels, axis=-1)

    raise ValueError("Input image must be either a 2D grayscale or 3D color array.")


def circle_crop(
    image: np.ndarray,
    sigma_x: int,
    tolerance: int = 7,
    center: tuple[int, int] | None = None,
    radius: int | None = None,
) -> np.ndarray:
    """Mask the image to its inscribed circle and apply Ben-Graham sharpening."""
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a color image with three channels")

    image = crop_image(image, tolerance)
    height, width, _ = image.shape

    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        radius = min(center)

    circle_mask = np.zeros((height, width), np.uint8)
    cv2.circle(circle_mask, center, int(radius), 1, thickness=-1)
    image = cv2.bitwise_and(image, image, mask=circle_mask)

    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigma_x), -4, 128)

    return crop_image(image, tolerance)


def preprocess_image(
    image: np.ndarray, image_size: int, sigma_x: int, tolerance: int = 7
) -> np.ndarray:
    """Full preprocessing for one RGB image: circle crop, sharpen, resize."""
    image = circle_crop(image, sigma_x=sigma_x, tolerance=tolerance)
    return cv2.resize(image, (image_size, image_size))
