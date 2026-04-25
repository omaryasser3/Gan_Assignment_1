"""Autoencoder and Variational Autoencoder models for Medical MNIST.

This module defines three Keras model classes:
    - AEEncoder:    Deterministic convolutional encoder.
    - VAEEncoder:   Probabilistic convolutional encoder that outputs (mu, log_var).
    - Decoder:      Shared transposed-convolution decoder used by both AE and VAE.
    - Autoencoder:  Full AE composed of AEEncoder + Decoder.
    - VAE:          Full VAE composed of VAEEncoder + Decoder with reparameterisation.

Intended to be imported by src/train.py and src/evaluate.py.
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Compression bottleneck: 64×64×1 → 16-d.
DEFAULT_LATENT_DIM: int = 16

# Weight applied to the KL term in the VAE ELBO loss.
# β = 0.5 balances reconstruction quality against latent-space regularity.
DEFAULT_KL_WEIGHT: float = 0.5

DEFAULT_CHANNELS: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────


class AEEncoder(keras.Model):
    """Deterministic convolutional encoder: image → latent vector z.

    Three strided Conv2D blocks progressively halve the spatial dimensions
    (64 → 32 → 16 → 8), followed by a flattening step and two Dense layers
    that project to a 256-d feature representation and then to the latent code.

    Args:
        latent_dim: Dimensionality of the output latent vector z.
        **kwargs:   Passed to keras.Model.__init__.
    """

    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM, **kwargs) -> None:
        """Initialise layers for the AE encoder."""
        super().__init__(**kwargs)

        self.conv_block = keras.Sequential(
            [
                layers.Conv2D(32, 4, strides=2, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(64, 4, strides=2, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(128, 4, strides=2, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ],
            name="conv_block",
        )
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(256, activation="relu", name="fc")
        self.z_out = layers.Dense(latent_dim, name="z")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass: map an image batch to latent codes.

        Args:
            x:        Input image tensor of shape (B, H, W, C).
            training: Whether the model is in training mode (affects BN).

        Returns:
            Latent code tensor of shape (B, latent_dim).
        """
        x = self.conv_block(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return self.z_out(x)


class VAEEncoder(keras.Model):
    """Probabilistic convolutional encoder: image → (mu, log_var).

    Identical convolutional backbone to AEEncoder, but the bottleneck splits
    into two independent Dense heads that predict the mean and log-variance of
    the approximate posterior q(z|x) = N(mu, diag(exp(log_var))).

    Args:
        latent_dim: Dimensionality of the latent distribution parameters.
        **kwargs:   Passed to keras.Model.__init__.
    """

    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM, **kwargs) -> None:
        """Initialise layers for the VAE encoder."""
        super().__init__(**kwargs)

        self.conv_block = keras.Sequential(
            [
                layers.Conv2D(32, 4, strides=2, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(64, 4, strides=2, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(128, 4, strides=2, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ],
            name="conv_block",
        )
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(256, activation="relu", name="fc")
        self.mu_out = layers.Dense(latent_dim, name="mu")
        self.lv_out = layers.Dense(latent_dim, name="log_var")

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass: map an image batch to Gaussian posterior parameters.

        Args:
            x:        Input image tensor of shape (B, H, W, C).
            training: Whether the model is in training mode (affects BN).

        Returns:
            A tuple (mu, log_var), each of shape (B, latent_dim).
        """
        x = self.conv_block(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return self.mu_out(x), self.lv_out(x)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder (shared)
# ─────────────────────────────────────────────────────────────────────────────


class Decoder(keras.Model):
    """Shared transposed-convolution decoder: latent vector z → image.

    Mirrors the encoder: a Dense projection expands z to 8×8×128, then three
    Conv2DTranspose blocks upsample back to the original 64×64 resolution.
    A final Conv2DTranspose with sigmoid activation constrains pixel values
    to [0, 1].

    Args:
        channels: Number of output image channels (1 for grayscale).
        **kwargs: Passed to keras.Model.__init__.
    """

    def __init__(self, channels: int = DEFAULT_CHANNELS, **kwargs) -> None:
        """Initialise layers for the shared decoder."""
        super().__init__(**kwargs)

        # Project and reshape to the spatial seed before upsampling.
        self.fc = layers.Dense(8 * 8 * 128, activation="relu", name="fc")
        self.reshape = layers.Reshape((8, 8, 128))

        self.deconv = keras.Sequential(
            [
                layers.Conv2DTranspose(128, 4, strides=2, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2DTranspose(64, 4, strides=2, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2DTranspose(32, 4, strides=2, padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                # Sigmoid ensures output is in [0, 1] to match normalised inputs.
                layers.Conv2DTranspose(channels, 3, padding="same", activation="sigmoid"),
            ],
            name="deconv_block",
        )

    def call(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass: decode a latent code to a reconstructed image.

        Args:
            z:        Latent code tensor of shape (B, latent_dim).
            training: Whether the model is in training mode (affects BN).

        Returns:
            Reconstructed image tensor of shape (B, 64, 64, channels).
        """
        x = self.fc(z)
        x = self.reshape(x)
        return self.deconv(x, training=training)


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder
# ─────────────────────────────────────────────────────────────────────────────


class Autoencoder(keras.Model):
    """Vanilla Autoencoder: deterministic encoder + shared decoder.

    Trained to minimise pixel-wise Mean Squared Error (MSE):
        L_AE = mean((x - x_hat) ** 2)

    The latent space is unconstrained; the model is optimised purely for
    reconstruction quality.  Random sampling (z ~ N(0,I)) is supported but
    may yield incoherent images if the encoded distribution differs from the
    prior.

    Args:
        latent_dim: Dimensionality of the latent code.
        channels:   Number of image channels.
        **kwargs:   Passed to keras.Model.__init__.
    """

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        channels: int = DEFAULT_CHANNELS,
        **kwargs,
    ) -> None:
        """Initialise the Autoencoder with its encoder, decoder, and metrics."""
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = AEEncoder(latent_dim, name="ae_encoder")
        self.decoder = Decoder(channels, name="ae_decoder")
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self) -> list:
        """Expose tracked metrics for Keras to reset between epochs."""
        return [self.loss_tracker]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Full forward pass: image → z → reconstructed image.

        Args:
            x:        Input image tensor of shape (B, H, W, C).
            training: Whether the model is in training mode.

        Returns:
            Reconstructed image tensor of shape (B, H, W, C).
        """
        z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def train_step(self, data: Tuple) -> dict:
        """Single gradient-update step on a batch.

        Args:
            data: A (x, label) tuple from tf.data; labels are ignored.

        Returns:
            Dictionary with the current mean loss value.
        """
        x, _ = data
        with tf.GradientTape() as tape:
            x_hat = self(x, training=True)
            loss = tf.reduce_mean(tf.square(x - x_hat))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data: Tuple) -> dict:
        """Evaluation step (no gradient update).

        Args:
            data: A (x, label) tuple from tf.data; labels are ignored.

        Returns:
            Dictionary with the current mean validation loss.
        """
        x, _ = data
        x_hat = self(x, training=False)
        loss = tf.reduce_mean(tf.square(x - x_hat))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Encode a batch of images to deterministic latent codes.

        Args:
            x: Input image tensor of shape (B, H, W, C).

        Returns:
            Latent code tensor of shape (B, latent_dim).
        """
        return self.encoder(x, training=False)

    def reconstruct(self, x: tf.Tensor) -> tf.Tensor:
        """Encode then decode a batch of images.

        Args:
            x: Input image tensor of shape (B, H, W, C).

        Returns:
            Reconstructed image tensor of shape (B, H, W, C).
        """
        return self(x, training=False)

    def sample(self, n: int) -> tf.Tensor:
        """Generate images by sampling the prior z ~ N(0, I).

        Note:
            Because the AE encoder is not regularised toward N(0,I), the
            decoded outputs may be incoherent.  Use VAE.sample for reliable
            generation.

        Args:
            n: Number of samples to generate.

        Returns:
            Generated image tensor of shape (n, 64, 64, channels).
        """
        z = tf.random.normal(shape=(n, self.latent_dim))
        return self.decoder(z, training=False)

    def save_versioned(self, directory: Path, version: int = 1) -> Path:
        """Save the model weights to a versioned file inside *directory*.

        Args:
            directory: Target directory (created if it does not exist).
            version:   Integer version tag appended to the filename.

        Returns:
            Path to the saved file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / f"ae_v{version}.weights.h5"
        self.save_weights(str(out_path))
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Variational Autoencoder
# ─────────────────────────────────────────────────────────────────────────────


class VAE(keras.Model):
    """Variational Autoencoder with reparameterisation trick.

    Trained to maximise the Evidence Lower BOund (ELBO):
        L_VAE = MSE(x, x_hat) + beta * KL[q(z|x) || N(0,I)]

    The KL term is the closed-form divergence between the learned diagonal
    Gaussian posterior and the standard normal prior:
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    The reparameterisation trick (z = mu + sigma * eps, eps ~ N(0,I))
    keeps the sampling operation differentiable, allowing gradients to flow
    through mu and log_var.

    Args:
        latent_dim: Dimensionality of the latent distribution.
        channels:   Number of image channels.
        kl_weight:  Weight beta applied to the KL term in the total loss.
        **kwargs:   Passed to keras.Model.__init__.
    """

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        channels: int = DEFAULT_CHANNELS,
        kl_weight: float = DEFAULT_KL_WEIGHT,
        **kwargs,
    ) -> None:
        """Initialise the VAE with its encoder, decoder, KL weight, and metrics."""
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.encoder = VAEEncoder(latent_dim, name="vae_encoder")
        self.decoder = Decoder(channels, name="vae_decoder")
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> list:
        """Expose all tracked metrics for Keras to reset between epochs."""
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    @staticmethod
    def reparameterize(mu: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        """Sample z using the reparameterisation trick.

        Instead of sampling z ~ N(mu, sigma^2) directly (non-differentiable),
        we draw eps ~ N(0,I) and compute z = mu + sigma * eps.  Gradients
        flow through mu and sigma; eps is treated as a fixed random constant.

        Args:
            mu:      Mean of the approximate posterior, shape (B, latent_dim).
            log_var: Log-variance of the posterior,     shape (B, latent_dim).

        Returns:
            Sampled latent code of shape (B, latent_dim).
        """
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Full forward pass: image → (x_hat, mu, log_var).

        Args:
            x:        Input image tensor of shape (B, H, W, C).
            training: Whether the model is in training mode.

        Returns:
            A tuple (x_hat, mu, log_var):
                x_hat:   Reconstructed image of shape (B, H, W, C).
                mu:      Posterior mean of shape (B, latent_dim).
                log_var: Posterior log-variance of shape (B, latent_dim).
        """
        mu, log_var = self.encoder(x, training=training)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z, training=training)
        return x_hat, mu, log_var

    def _compute_losses(
        self, x: tf.Tensor, x_hat: tf.Tensor, mu: tf.Tensor, log_var: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute the ELBO components: reconstruction loss and KL divergence.

        Args:
            x:       Clean input images.
            x_hat:   Reconstructed images.
            mu:      Posterior means.
            log_var: Posterior log-variances.

        Returns:
            A tuple (total_loss, recon_loss, kl_loss).
        """
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, x_hat), axis=(1, 2)
            )
        )
        # Sum over latent dims so KL stays proportionate to reconstruction magnitude.
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        )
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, data: Tuple) -> dict:
        """Single gradient-update step on a batch.

        Args:
            data: A (x, label) tuple from tf.data; labels are ignored.

        Returns:
            Dictionary with loss, recon_loss, and kl_loss values.
        """
        x, _ = data
        with tf.GradientTape() as tape:
            x_hat, mu, log_var = self(x, training=True)
            total, recon, kl = self._compute_losses(x, x_hat, mu, log_var)
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Tuple) -> dict:
        """Evaluation step (no gradient update).

        Args:
            data: A (x, label) tuple from tf.data; labels are ignored.

        Returns:
            Dictionary with validation loss, recon_loss, and kl_loss values.
        """
        x, _ = data
        x_hat, mu, log_var = self(x, training=False)
        total, recon, kl = self._compute_losses(x, x_hat, mu, log_var)
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Encode images to posterior means (used as deterministic codes at inference).

        Args:
            x: Input image tensor of shape (B, H, W, C).

        Returns:
            Posterior mean tensor of shape (B, latent_dim).
        """
        mu, _ = self.encoder(x, training=False)
        return mu

    def reconstruct(self, x: tf.Tensor) -> tf.Tensor:
        """Encode then decode a batch of images.

        Args:
            x: Input image tensor of shape (B, H, W, C).

        Returns:
            Reconstructed image tensor of shape (B, H, W, C).
        """
        x_hat, _, _ = self(x, training=False)
        return x_hat

    def sample(self, n: int) -> tf.Tensor:
        """Generate images by sampling the prior z ~ N(0, I).

        Because the VAE is trained to align q(z|x) with N(0,I) via the KL
        term, samples from the prior decode to anatomically plausible images.

        Args:
            n: Number of samples to generate.

        Returns:
            Generated image tensor of shape (n, 64, 64, channels).
        """
        z = tf.random.normal(shape=(n, self.latent_dim))
        return self.decoder(z, training=False)

    def interpolate(
        self, img_a: tf.Tensor, img_b: tf.Tensor, steps: int = 10
    ) -> tf.Tensor:
        """Linearly interpolate between two images in latent space.

        Encodes both images to their posterior means, then decodes points
        along the straight line between them.  The continuity guarantee only
        holds because the KL term ensures the latent manifold is gap-free.

        Args:
            img_a: First image tensor of shape (1, H, W, C).
            img_b: Second image tensor of shape (1, H, W, C).
            steps: Number of intermediate frames (endpoints not included).

        Returns:
            Interpolated image tensor of shape (steps + 2, H, W, C),
            where index 0 is the reconstruction of img_a and index -1
            is the reconstruction of img_b.
        """
        za = self.encode(img_a)
        zb = self.encode(img_b)
        alphas = tf.linspace(0.0, 1.0, steps + 2)
        frames = [self.decoder((1.0 - a) * za + a * zb, training=False) for a in alphas]
        return tf.concat(frames, axis=0)

    def save_versioned(self, directory: Path, version: int = 1) -> Path:
        """Save the model weights to a versioned file inside *directory*.

        Args:
            directory: Target directory (created if it does not exist).
            version:   Integer version tag appended to the filename.

        Returns:
            Path to the saved file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / f"vae_v{version}.weights.h5"
        self.save_weights(str(out_path))
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────


def build_models(
    latent_dim: int = DEFAULT_LATENT_DIM,
    channels: int = DEFAULT_CHANNELS,
    kl_weight: float = DEFAULT_KL_WEIGHT,
    learning_rate: float = 1e-3,
    dummy_input: Optional[tf.Tensor] = None,
) -> Tuple[Autoencoder, VAE]:
    """Instantiate, build, and compile one AE and one VAE with identical settings.

    Passing a *dummy_input* tensor triggers a forward pass on both models so
    that their weights are built before training begins, enabling summary() calls.

    Args:
        latent_dim:    Latent space dimensionality.
        channels:      Number of image channels.
        kl_weight:     Beta weight for the VAE KL term.
        learning_rate: Adam learning rate for both optimisers.
        dummy_input:   Optional tensor of shape (1, H, W, C) used to build weights.

    Returns:
        A tuple (ae, vae) of compiled model instances.
    """
    ae = Autoencoder(latent_dim=latent_dim, channels=channels, name="autoencoder")
    vae = VAE(
        latent_dim=latent_dim,
        channels=channels,
        kl_weight=kl_weight,
        name="vae",
    )

    if dummy_input is not None:
        ae(dummy_input)
        vae(dummy_input)

    ae.compile(optimizer=keras.optimizers.Adam(learning_rate))
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate))

    return ae, vae
