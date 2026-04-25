"""Data loading, preprocessing, and augmentation for Medical MNIST.

Responsibilities:
    - Discover image paths and integer labels from an extracted dataset directory.
    - Build reproducible per-class tf.data pipelines (train and validation).
    - Provide a Gaussian noise injection utility used during denoising evaluation.

Typical usage::

    paths, labels = collect_image_paths("/content/medical_mnist", CLASS_NAMES)
    train_ds = make_class_dataset(paths, labels, class_idx=1, shuffle=True)
    noisy = add_gaussian_noise(images, noise_factor=0.3)
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES: List[str] = [
    "AbdomenCT",
    "BreastMRI",
    "ChestCT",
    "CXR",
    "Hand",
    "HeadCT",
]

IMAGE_SIZE: int = 64
CHANNELS: int = 1
BATCH_SIZE: int = 64
BUFFER_SIZE: int = 5_000
TRAIN_SPLIT: float = 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Path collection
# ─────────────────────────────────────────────────────────────────────────────


def collect_image_paths(
    root_dir: str | Path,
    class_names: List[str] = CLASS_NAMES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Walk *root_dir* and return parallel arrays of image paths and labels.

    The function handles arbitrary nesting depth: it matches any subdirectory
    whose basename appears in *class_names* and collects all .jpg / .jpeg / .png
    files found inside.

    Args:
        root_dir:    Root directory of the extracted Medical MNIST dataset.
        class_names: Ordered list of class folder names. The label integer for
                     each class equals its index in this list.

    Returns:
        A tuple (paths, labels) where:
            paths:  numpy array of str file paths, shape (N,).
            labels: numpy array of int32 class indices, shape (N,).
    """
    all_paths: List[str] = []
    all_labels: List[int] = []

    for dirpath, _, filenames in os.walk(root_dir):
        folder = os.path.basename(dirpath)
        if folder not in class_names:
            continue
        label_idx = class_names.index(folder)
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(dirpath, fname))
                all_labels.append(label_idx)

    return np.array(all_paths), np.array(all_labels, dtype=np.int32)


def train_val_split(
    paths: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = TRAIN_SPLIT,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split paths and labels into train and validation sets.

    Args:
        paths:          Array of image file paths, shape (N,).
        labels:         Array of integer labels, shape (N,).
        train_fraction: Fraction of data to use for training (default 0.9).
        seed:           Random seed for reproducibility.

    Returns:
        A tuple (train_paths, train_labels, val_paths, val_labels).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(paths))
    split = int(len(indices) * train_fraction)
    train_idx, val_idx = indices[:split], indices[split:]
    return paths[train_idx], labels[train_idx], paths[val_idx], labels[val_idx]


# ─────────────────────────────────────────────────────────────────────────────
# tf.data pipeline
# ─────────────────────────────────────────────────────────────────────────────


def _load_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load, decode, resize, and normalise a single image from disk.

    Args:
        path:  String tensor containing the file path.
        label: Integer label tensor (passed through unchanged).

    Returns:
        A tuple (image, label) where image is float32 in [0, 1],
        shape (IMAGE_SIZE, IMAGE_SIZE, CHANNELS).
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=CHANNELS, expand_animations=False)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_class_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    class_idx: int,
    shuffle: bool = True,
    batch_size: int = BATCH_SIZE,
    buffer_size: int = BUFFER_SIZE,
) -> tf.data.Dataset:
    """Build a batched, prefetched tf.data pipeline for a single class.

    Only images whose label equals *class_idx* are included, enabling
    per-region model training without loading the full dataset into memory.

    Args:
        paths:       Array of all image file paths (any class), shape (N,).
        labels:      Array of integer class labels, shape (N,).
        class_idx:   The class index to filter for.
        shuffle:     Whether to shuffle the dataset each epoch.
        batch_size:  Number of images per batch.
        buffer_size: Shuffle buffer size; capped to the class size automatically.

    Returns:
        A tf.data.Dataset yielding (image, label) batches.
    """
    mask = labels == class_idx
    cls_paths = paths[mask]
    cls_labels = labels[mask]

    ds = tf.data.Dataset.from_tensor_slices((cls_paths, cls_labels))
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        # Cap buffer to actual class size to avoid wasting memory.
        effective_buffer = min(buffer_size, int(mask.sum()))
        ds = ds.shuffle(effective_buffer, reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_all_class_datasets(
    train_paths: np.ndarray,
    train_labels: np.ndarray,
    val_paths: np.ndarray,
    val_labels: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Tuple[Dict[str, tf.data.Dataset], Dict[str, tf.data.Dataset]]:
    """Build train and validation datasets for every class at once.

    Args:
        train_paths:  Training image paths.
        train_labels: Training integer labels.
        val_paths:    Validation image paths.
        val_labels:   Validation integer labels.
        class_names:  Ordered list of class names (index = label integer).

    Returns:
        A tuple (train_ds_map, val_ds_map) where each is a dict mapping
        class name → tf.data.Dataset.
    """
    train_ds_map: Dict[str, tf.data.Dataset] = {}
    val_ds_map: Dict[str, tf.data.Dataset] = {}

    for class_idx, class_name in enumerate(class_names):
        train_ds_map[class_name] = make_class_dataset(
            train_paths, train_labels, class_idx, shuffle=True
        )
        val_ds_map[class_name] = make_class_dataset(
            val_paths, val_labels, class_idx, shuffle=False
        )

    return train_ds_map, val_ds_map


# ─────────────────────────────────────────────────────────────────────────────
# Noise injection
# ─────────────────────────────────────────────────────────────────────────────


def add_gaussian_noise(
    images: tf.Tensor,
    noise_factor: float = 0.3,
) -> tf.Tensor:
    """Add additive white Gaussian noise to a batch of images.

    Noise is sampled from N(0, noise_factor^2) and added pixel-wise.
    The result is clipped to [0, 1] to preserve valid pixel range.

    This is applied at evaluation time only (not during training), to test
    how well each model's bottleneck suppresses high-frequency noise.

    Args:
        images:       Float32 image tensor in [0, 1], shape (B, H, W, C).
        noise_factor: Standard deviation of the Gaussian noise (default 0.3).

    Returns:
        Noisy image tensor of the same shape, clipped to [0, 1].
    """
    noise = tf.random.normal(shape=tf.shape(images), stddev=noise_factor)
    return tf.clip_by_value(images + noise, 0.0, 1.0)
