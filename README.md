# AE & VAE — Representation Learning on Medical MNIST

DSAI 490 · Assignment 1 · Unsupervised representation learning with Autoencoders and Variational Autoencoders on six anatomical imaging modalities.

---

## Project Structure

```
├── data/
│   ├── raw/            # Original Medical MNIST zip / extracted folders
│   └── processed/      # Cached processed splits (if any)
├── models/             # Saved model weights + training_metadata.json
├── notebooks/          # Exploratory Colab/Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # Dataset loading, tf.data pipelines, noise injection
│   ├── model.py             # AEEncoder, VAEEncoder, Decoder, Autoencoder, VAE
│   └── train.py             # Per-region and full training orchestration
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-directory>
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset

Download the Medical MNIST zip and extract it into `data/raw/`:

```
data/raw/
├── AbdomenCT/
├── BreastMRI/
├── ChestCT/
├── CXR/
├── Hand/
└── HeadCT/
```

---

## Usage

### Train all 12 models (1 AE + 1 VAE per region)

```python
from src.data_processing import collect_image_paths, train_val_split, build_all_class_datasets
from src.train import train_all_regions

paths, labels = collect_image_paths("data/raw")
train_paths, train_labels, val_paths, val_labels = train_val_split(paths, labels)
train_ds, val_ds = build_all_class_datasets(train_paths, train_labels, val_paths, val_labels)

ae_models, vae_models, histories = train_all_regions(
    train_ds_map=train_ds,
    val_ds_map=val_ds,
    latent_dim=16,
    epochs=20,
    learning_rate=1e-3,
    kl_weight=0.5,
    models_dir="models/",
)
```

Saved weights are versioned automatically:
```
models/
└── BreastMRI/
    ├── ae_v1.weights.h5
    └── vae_v1.weights.h5
```
A `training_metadata.json` file is written to `models/` with all hyperparameters and wall-clock time for reproducibility.

### Use a trained model

```python
from src.model import Autoencoder, VAE

ae = Autoencoder(latent_dim=16)
ae.load_weights("models/BreastMRI/ae_v1.weights.h5")

reconstruction = ae.reconstruct(images)
latent_codes   = ae.encode(images)
samples        = ae.sample(8)
```

### Inject noise for denoising evaluation

```python
from src.data_processing import add_gaussian_noise

noisy_images   = add_gaussian_noise(images, noise_factor=0.3)
ae_denoised    = ae.reconstruct(noisy_images)
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 16 | Bottleneck dimensionality |
| `epochs` | 20 | Training epochs per model |
| `learning_rate` | 1e-3 | Adam learning rate |
| `kl_weight` (β) | 0.5 | Weight on VAE KL divergence term |
| `noise_factor` (σ) | 0.3 | Gaussian noise std for denoising tests |
| `batch_size` | 64 | Images per training batch |

---

## Contributing

1. Branch off `main`: `git checkout -b feature/your-feature`
2. Follow the project's [Code Conventions](Code%20Conventions.pdf) — PEP 8, PEP 257 docstrings, type hints throughout.
3. Run `flake8 src/` and `black src/` before committing.
4. Add or update tests in `tests/` for any changed behaviour.
5. Open a pull request with a clear description of the change.
