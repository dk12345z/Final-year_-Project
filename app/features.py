from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import gabor
from skimage.transform import resize

IMAGE_FEATURE_SIZE = 1764
FEATURE_SIZE = 9 + 6 + 26 + 16 + IMAGE_FEATURE_SIZE


def extract_handcrafted(image: np.ndarray) -> np.ndarray:
    """Compute colour moments, GLCM, LBP, Gabor, and HOG descriptors for an RGB image."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected an RGB image with three channels")
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        scale = 255.0 if rgb.size and float(np.nanmax(rgb)) <= 1.0 else 1.0
        rgb = np.clip(np.nan_to_num(rgb) * scale, 0, 255).astype(np.uint8)
    rgb_f = rgb.astype(np.float32) / 255.0
    means = rgb_f.mean(axis=(0, 1))
    colour = np.concatenate((means, rgb_f.std(axis=(0, 1)), ((rgb_f - means) ** 3).mean(axis=(0, 1))))

    gray = rgb2gray(rgb_f)
    quantized = (np.clip(gray * 255, 0, 255).astype(np.uint8) // 8).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=[1, 2, 4], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    texture = np.asarray([graycoprops(glcm, prop).mean() for prop in
                          ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")])

    lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(27), range=(0, 26), density=True)
    gabor_values: list[float] = []
    for frequency in (0.1, 0.2):
        for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            real, imag = gabor(gray, frequency=frequency, theta=theta)
            magnitude = np.hypot(real, imag)
            gabor_values.extend((float(magnitude.mean()), float(magnitude.std())))
    fixed_gray = resize(gray, (128, 128), anti_aliasing=True, preserve_range=True)
    hog_values = hog(fixed_gray, orientations=9, pixels_per_cell=(16, 16),
                     cells_per_block=(2, 2), feature_vector=True)
    vector = np.concatenate((colour, texture, lbp_hist, np.asarray(gabor_values), hog_values))
    if vector.size != FEATURE_SIZE:
        raise RuntimeError(f"Expected {FEATURE_SIZE} handcrafted values, got {vector.size}")
    return vector.astype(np.float32)
