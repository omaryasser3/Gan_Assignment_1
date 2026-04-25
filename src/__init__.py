"""src — AE/VAE representation learning package for Medical MNIST.

Public surface:
    - Autoencoder, VAE, build_models  (model.py)
    - make_class_dataset, add_gaussian_noise  (data_processing.py)
    - train_region  (train.py)
"""

from src.model import Autoencoder, VAE, build_models
from src.data_processing import make_class_dataset, add_gaussian_noise
from src.train import train_region

__all__ = [
    "Autoencoder",
    "VAE",
    "build_models",
    "make_class_dataset",
    "add_gaussian_noise",
    "train_region",
]
