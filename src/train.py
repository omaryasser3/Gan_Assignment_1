"""Training orchestration for per-region AE and VAE models.

This module provides:
    - train_region:      Train one AE and one VAE on a single anatomical region.
    - train_all_regions: Loop over all regions and collect all trained models.
    - save_metadata:     Persist training configuration as JSON for reproducibility.

Typical usage::

    from src.data_processing import build_all_class_datasets
    from src.train import train_all_regions

    ae_models, vae_models, histories = train_all_regions(train_ds, val_ds)
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras

# ── Local ─────────────────────────────────────────────────────────────────────
from src.data_processing import CLASS_NAMES
from src.model import Autoencoder, VAE, build_models

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

# History dict returned per region after training.
RegionHistory = Dict[str, Dict[str, List[float]]]


# ─────────────────────────────────────────────────────────────────────────────
# Per-region training
# ─────────────────────────────────────────────────────────────────────────────


def train_region(
    class_name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    latent_dim: int = 16,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    kl_weight: float = 0.5,
    models_dir: Path = Path("models"),
    model_version: int = 1,
    channels: int = 1,
    image_size: int = 64,
    verbose: int = 1,
) -> Tuple[Autoencoder, VAE, RegionHistory]:
    """Train one AE and one VAE on a single anatomical region.

    Both models are compiled with Adam, trained for *epochs* epochs, then
    saved as versioned weight files inside *models_dir*.

    Args:
        class_name:    Name of the anatomical region (used in filenames).
        train_ds:      Training tf.data.Dataset for this region.
        val_ds:        Validation tf.data.Dataset for this region.
        latent_dim:    Dimensionality of the latent space.
        epochs:        Number of training epochs.
        learning_rate: Adam learning rate.
        kl_weight:     Beta weight for the VAE KL divergence term.
        models_dir:    Directory in which versioned weight files are saved.
        model_version: Integer version tag appended to saved filenames.
        channels:      Number of image channels (1 for grayscale).
        image_size:    Spatial size of the input images.
        verbose:       Keras fit verbosity (0 = silent, 1 = progress bar).

    Returns:
        A tuple (ae, vae, history) where history is a dict with keys:
            "ae":  {"train": [...], "val": [...]}
            "vae": {"train_total": [...], "train_recon": [...], "train_kl": [...]}
    """
    print(f"\n{'=' * 55}")
    print(f"  Region: {class_name}")
    print(f"{'=' * 55}")

    # Build a dummy input to trigger weight construction before summary/save.
    dummy = tf.zeros((1, image_size, image_size, channels))
    ae, vae = build_models(
        latent_dim=latent_dim,
        channels=channels,
        kl_weight=kl_weight,
        learning_rate=learning_rate,
        dummy_input=dummy,
    )

    # ── Train AE ──────────────────────────────────────────────────────────────
    print(f"  Training AE ({epochs} epochs) ...")
    ae_history = ae.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=verbose,
    )

    # ── Train VAE ─────────────────────────────────────────────────────────────
    print(f"  Training VAE ({epochs} epochs) ...")
    vae_history = vae.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=verbose,
    )

    # ── Save versioned weights ─────────────────────────────────────────────────
    region_dir = models_dir / class_name
    ae.save_versioned(region_dir, version=model_version)
    vae.save_versioned(region_dir, version=model_version)
    print(f"  Weights saved to {region_dir}/")

    history: RegionHistory = {
        "ae": {
            "train": ae_history.history["loss"],
            "val": ae_history.history["val_loss"],
        },
        "vae": {
            "train_total": vae_history.history["loss"],
            "train_recon": vae_history.history["recon_loss"],
            "train_kl": vae_history.history["kl_loss"],
        },
    }
    return ae, vae, history


# ─────────────────────────────────────────────────────────────────────────────
# Full training run across all regions
# ─────────────────────────────────────────────────────────────────────────────


def train_all_regions(
    train_ds_map: Dict[str, tf.data.Dataset],
    val_ds_map: Dict[str, tf.data.Dataset],
    class_names: List[str] = CLASS_NAMES,
    latent_dim: int = 16,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    kl_weight: float = 0.5,
    models_dir: Path = Path("models"),
    model_version: int = 1,
) -> Tuple[
    Dict[str, Autoencoder],
    Dict[str, VAE],
    Dict[str, RegionHistory],
]:
    """Train 12 models (1 AE + 1 VAE) × 6 anatomical regions.

    Each region is trained independently; no weights are shared across regions.
    Training metadata (hyperparameters + wall-clock time) is saved to
    ``models_dir/training_metadata.json`` for reproducibility.

    Args:
        train_ds_map:  Dict mapping class name → training tf.data.Dataset.
        val_ds_map:    Dict mapping class name → validation tf.data.Dataset.
        class_names:   List of class names to train on.
        latent_dim:    Latent space dimensionality (shared across all models).
        epochs:        Training epochs per model.
        learning_rate: Adam learning rate.
        kl_weight:     Beta weight for VAE KL term.
        models_dir:    Root directory for saving all model weights.
        model_version: Version tag for saved weight filenames.

    Returns:
        A tuple (ae_models, vae_models, all_histories) where each is a dict
        keyed by class name.
    """
    ae_models: Dict[str, Autoencoder] = {}
    vae_models: Dict[str, VAE] = {}
    all_histories: Dict[str, RegionHistory] = {}

    start_time = time.time()

    for class_name in class_names:
        ae, vae, history = train_region(
            class_name=class_name,
            train_ds=train_ds_map[class_name],
            val_ds=val_ds_map[class_name],
            latent_dim=latent_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            kl_weight=kl_weight,
            models_dir=models_dir,
            model_version=model_version,
        )
        ae_models[class_name] = ae
        vae_models[class_name] = vae
        all_histories[class_name] = history

    elapsed = time.time() - start_time
    print(f"\nAll {len(class_names)} regions trained in {elapsed:.1f}s.")

    save_metadata(
        models_dir=models_dir,
        class_names=class_names,
        latent_dim=latent_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        kl_weight=kl_weight,
        model_version=model_version,
        elapsed_seconds=elapsed,
    )

    return ae_models, vae_models, all_histories


# ─────────────────────────────────────────────────────────────────────────────
# Metadata persistence
# ─────────────────────────────────────────────────────────────────────────────


def save_metadata(
    models_dir: Path,
    class_names: List[str],
    latent_dim: int,
    epochs: int,
    learning_rate: float,
    kl_weight: float,
    model_version: int,
    elapsed_seconds: float,
) -> Path:
    """Write training configuration and timing to a JSON file.

    Storing this metadata alongside the weight files makes experiments
    fully reproducible: any future run can read back exactly what
    hyperparameters produced each saved model.

    Args:
        models_dir:      Directory where the JSON file is written.
        class_names:     Regions that were trained.
        latent_dim:      Latent space dimensionality used.
        epochs:          Number of training epochs used.
        learning_rate:   Adam learning rate used.
        kl_weight:       VAE KL weight used.
        model_version:   Version tag of the saved weights.
        elapsed_seconds: Total wall-clock training time in seconds.

    Returns:
        Path to the written JSON file.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_version": model_version,
        "class_names": class_names,
        "hyperparameters": {
            "latent_dim": latent_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "kl_weight": kl_weight,
        },
        "training_time_seconds": round(elapsed_seconds, 2),
    }

    out_path = models_dir / "training_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {out_path}")
    return out_path
