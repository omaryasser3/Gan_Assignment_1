"""Unit tests for src/data_processing.py.

Tests cover:
    - collect_image_paths: correct counts, label alignment, file-extension filtering.
    - train_val_split: correct sizes, no overlap, reproducibility.
    - make_class_dataset: output shapes, dtype, value range, class filtering.
    - add_gaussian_noise: output shape, value range, noise is actually applied.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import tempfile
import unittest
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf

# ── Local ─────────────────────────────────────────────────────────────────────
from src.data_processing import (
    CLASS_NAMES,
    add_gaussian_noise,
    collect_image_paths,
    make_class_dataset,
    train_val_split,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_fake_dataset(root: Path, class_names: list, n_per_class: int = 5) -> None:
    """Create a minimal fake directory tree with blank PNG images.

    Args:
        root:         Root directory to create the fake dataset in.
        class_names:  List of class folder names to create.
        n_per_class:  Number of dummy images to write per class.
    """
    for name in class_names:
        class_dir = root / name
        class_dir.mkdir(parents=True)
        for i in range(n_per_class):
            # Write a tiny valid 1×1 grayscale PNG using TensorFlow.
            img_tensor = tf.zeros((1, 1, 1), dtype=tf.uint8)
            png_bytes = tf.image.encode_png(img_tensor).numpy()
            (class_dir / f"img_{i}.png").write_bytes(png_bytes)
        # Add a non-image file that should be ignored.
        (class_dir / "metadata.txt").write_text("ignore me")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: collect_image_paths
# ─────────────────────────────────────────────────────────────────────────────


class TestCollectImagePaths(unittest.TestCase):
    """Tests for the collect_image_paths function."""

    def setUp(self) -> None:
        """Create a temporary fake dataset used by all tests in this class."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.classes = CLASS_NAMES[:3]  # Use first 3 classes for speed.
        self.n_per_class = 4
        _make_fake_dataset(self.root, self.classes, self.n_per_class)

    def tearDown(self) -> None:
        """Remove the temporary directory after each test."""
        self.tmp_dir.cleanup()

    def test_total_count(self) -> None:
        """Total paths collected should equal n_per_class × n_classes."""
        paths, labels = collect_image_paths(self.root, self.classes)
        self.assertEqual(len(paths), self.n_per_class * len(self.classes))

    def test_label_alignment(self) -> None:
        """Each path should map to the correct integer label."""
        paths, labels = collect_image_paths(self.root, self.classes)
        for path, label in zip(paths, labels):
            folder = os.path.basename(os.path.dirname(path))
            expected_label = self.classes.index(folder)
            self.assertEqual(label, expected_label)

    def test_non_image_files_excluded(self) -> None:
        """Files with non-image extensions (.txt) must be ignored."""
        paths, _ = collect_image_paths(self.root, self.classes)
        for path in paths:
            self.assertTrue(
                path.lower().endswith((".jpg", ".jpeg", ".png")),
                msg=f"Non-image file included: {path}",
            )

    def test_empty_directory(self) -> None:
        """An empty root directory should return empty arrays."""
        empty_root = Path(self.tmp_dir.name) / "empty"
        empty_root.mkdir()
        paths, labels = collect_image_paths(empty_root, self.classes)
        self.assertEqual(len(paths), 0)
        self.assertEqual(len(labels), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: train_val_split
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainValSplit(unittest.TestCase):
    """Tests for the train_val_split function."""

    def _make_arrays(self, n: int = 100) -> tuple:
        paths = np.array([f"img_{i}.png" for i in range(n)])
        labels = np.arange(n, dtype=np.int32)
        return paths, labels

    def test_split_sizes(self) -> None:
        """Train and val sets together must equal the original size."""
        paths, labels = self._make_arrays(100)
        tp, tl, vp, vl = train_val_split(paths, labels, train_fraction=0.9)
        self.assertEqual(len(tp) + len(vp), 100)
        self.assertEqual(len(tl) + len(vl), 100)

    def test_no_overlap(self) -> None:
        """No path should appear in both train and val."""
        paths, labels = self._make_arrays(100)
        tp, _, vp, _ = train_val_split(paths, labels)
        self.assertEqual(len(set(tp) & set(vp)), 0)

    def test_reproducibility(self) -> None:
        """Same seed should produce identical splits."""
        paths, labels = self._make_arrays(100)
        tp1, _, vp1, _ = train_val_split(paths, labels, seed=42)
        tp2, _, vp2, _ = train_val_split(paths, labels, seed=42)
        np.testing.assert_array_equal(tp1, tp2)
        np.testing.assert_array_equal(vp1, vp2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds should (almost certainly) produce different splits."""
        paths, labels = self._make_arrays(100)
        tp1, _, _, _ = train_val_split(paths, labels, seed=0)
        tp2, _, _, _ = train_val_split(paths, labels, seed=99)
        self.assertFalse(np.array_equal(tp1, tp2))


# ─────────────────────────────────────────────────────────────────────────────
# Tests: make_class_dataset
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeClassDataset(unittest.TestCase):
    """Tests for the make_class_dataset function."""

    def setUp(self) -> None:
        """Build a small fake dataset and collect paths/labels."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        root = Path(self.tmp_dir.name)
        self.classes = CLASS_NAMES[:2]
        _make_fake_dataset(root, self.classes, n_per_class=6)
        self.paths, self.labels = collect_image_paths(root, self.classes)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_output_shape(self) -> None:
        """Each batch should have shape (B, 64, 64, 1)."""
        ds = make_class_dataset(self.paths, self.labels, class_idx=0, batch_size=3)
        for images, labels in ds.take(1):
            self.assertEqual(images.shape[1:], (64, 64, 1))

    def test_pixel_range(self) -> None:
        """All pixel values must be in [0, 1] after normalisation."""
        ds = make_class_dataset(self.paths, self.labels, class_idx=0, batch_size=3)
        for images, _ in ds.take(1):
            self.assertGreaterEqual(float(tf.reduce_min(images)), 0.0)
            self.assertLessEqual(float(tf.reduce_max(images)), 1.0)

    def test_dtype(self) -> None:
        """Images should be float32."""
        ds = make_class_dataset(self.paths, self.labels, class_idx=0, batch_size=3)
        for images, _ in ds.take(1):
            self.assertEqual(images.dtype, tf.float32)

    def test_class_filtering(self) -> None:
        """Dataset for class 0 should not contain images labelled as class 1."""
        ds = make_class_dataset(self.paths, self.labels, class_idx=0, batch_size=20)
        for _, labels in ds:
            unique = set(labels.numpy().tolist())
            self.assertTrue(unique.issubset({0}), msg=f"Unexpected labels: {unique}")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: add_gaussian_noise
# ─────────────────────────────────────────────────────────────────────────────


class TestAddGaussianNoise(unittest.TestCase):
    """Tests for the add_gaussian_noise function."""

    def _clean_batch(self, value: float = 0.5) -> tf.Tensor:
        """Return a constant image batch for deterministic testing."""
        return tf.constant(value, shape=(4, 64, 64, 1), dtype=tf.float32)

    def test_output_shape_preserved(self) -> None:
        """Shape must be identical to the input shape."""
        images = self._clean_batch()
        noisy = add_gaussian_noise(images, noise_factor=0.3)
        self.assertEqual(noisy.shape, images.shape)

    def test_pixel_range_clipped(self) -> None:
        """All output pixels must remain within [0, 1]."""
        images = self._clean_batch()
        noisy = add_gaussian_noise(images, noise_factor=0.5)
        self.assertGreaterEqual(float(tf.reduce_min(noisy)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(noisy)), 1.0)

    def test_noise_is_applied(self) -> None:
        """Output must differ from the clean input when noise_factor > 0."""
        images = self._clean_batch(value=0.5)
        noisy = add_gaussian_noise(images, noise_factor=0.3)
        self.assertFalse(
            tf.reduce_all(tf.equal(images, noisy)).numpy(),
            msg="Noisy output is identical to input — noise was not applied.",
        )

    def test_zero_noise_factor(self) -> None:
        """With noise_factor=0, output should be identical to input (no noise)."""
        images = self._clean_batch(value=0.5)
        noisy = add_gaussian_noise(images, noise_factor=0.0)
        np.testing.assert_allclose(images.numpy(), noisy.numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
