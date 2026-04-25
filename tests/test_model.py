"""Unit tests for src/model.py.

Tests cover:
    - AEEncoder:   output shape and dtype.
    - VAEEncoder:  output shapes for (mu, log_var) and dtype.
    - Decoder:     output shape and pixel range.
    - Autoencoder: end-to-end forward pass, encode, reconstruct, sample,
                   and loss value range.
    - VAE:         end-to-end forward pass, reparameterise, encode,
                   reconstruct, sample, interpolate, and loss decomposition.
    - build_models: returns compiled instances of the correct types.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import unittest

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf

# ── Local ─────────────────────────────────────────────────────────────────────
from src.model import (
    AEEncoder,
    Autoencoder,
    Decoder,
    VAE,
    VAEEncoder,
    build_models,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

BATCH = 4
IMAGE_SIZE = 64
CHANNELS = 1
LATENT_DIM = 16
DUMMY = tf.zeros((BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))


# ─────────────────────────────────────────────────────────────────────────────
# AEEncoder
# ─────────────────────────────────────────────────────────────────────────────


class TestAEEncoder(unittest.TestCase):
    """Tests for the AEEncoder class."""

    def setUp(self) -> None:
        """Instantiate a fresh AEEncoder before each test."""
        self.encoder = AEEncoder(latent_dim=LATENT_DIM)
        self.encoder(DUMMY)  # build weights

    def test_output_shape(self) -> None:
        """Output should be (batch, latent_dim)."""
        z = self.encoder(DUMMY, training=False)
        self.assertEqual(z.shape, (BATCH, LATENT_DIM))

    def test_output_dtype(self) -> None:
        """Output should be float32."""
        z = self.encoder(DUMMY, training=False)
        self.assertEqual(z.dtype, tf.float32)

    def test_deterministic(self) -> None:
        """Same input must produce the same output (no stochasticity in AE encoder)."""
        z1 = self.encoder(DUMMY, training=False)
        z2 = self.encoder(DUMMY, training=False)
        np.testing.assert_allclose(z1.numpy(), z2.numpy())


# ─────────────────────────────────────────────────────────────────────────────
# VAEEncoder
# ─────────────────────────────────────────────────────────────────────────────


class TestVAEEncoder(unittest.TestCase):
    """Tests for the VAEEncoder class."""

    def setUp(self) -> None:
        """Instantiate a fresh VAEEncoder before each test."""
        self.encoder = VAEEncoder(latent_dim=LATENT_DIM)
        self.encoder(DUMMY)  # build weights

    def test_output_shapes(self) -> None:
        """Both mu and log_var should be (batch, latent_dim)."""
        mu, log_var = self.encoder(DUMMY, training=False)
        self.assertEqual(mu.shape, (BATCH, LATENT_DIM))
        self.assertEqual(log_var.shape, (BATCH, LATENT_DIM))

    def test_output_dtypes(self) -> None:
        """Both outputs should be float32."""
        mu, log_var = self.encoder(DUMMY, training=False)
        self.assertEqual(mu.dtype, tf.float32)
        self.assertEqual(log_var.dtype, tf.float32)

    def test_mu_log_var_differ(self) -> None:
        """mu and log_var must be different tensors (independent heads)."""
        mu, log_var = self.encoder(DUMMY, training=False)
        self.assertFalse(
            tf.reduce_all(tf.equal(mu, log_var)).numpy(),
            msg="mu and log_var are identical — heads may share weights.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────


class TestDecoder(unittest.TestCase):
    """Tests for the shared Decoder class."""

    def setUp(self) -> None:
        """Instantiate and build a Decoder."""
        self.decoder = Decoder(channels=CHANNELS)
        dummy_z = tf.zeros((BATCH, LATENT_DIM))
        self.decoder(dummy_z)  # build weights

    def test_output_shape(self) -> None:
        """Output should be (batch, 64, 64, channels)."""
        z = tf.zeros((BATCH, LATENT_DIM))
        out = self.decoder(z, training=False)
        self.assertEqual(out.shape, (BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    def test_pixel_range(self) -> None:
        """Sigmoid activation must keep all pixels in [0, 1]."""
        # Use random latent codes to probe the full output range.
        z = tf.random.normal((BATCH, LATENT_DIM))
        out = self.decoder(z, training=False)
        self.assertGreaterEqual(float(tf.reduce_min(out)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(out)), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoencoder(unittest.TestCase):
    """Tests for the Autoencoder (AE) class."""

    def setUp(self) -> None:
        """Build and compile a fresh Autoencoder."""
        self.ae = Autoencoder(latent_dim=LATENT_DIM, channels=CHANNELS)
        self.ae(DUMMY)  # build weights
        self.ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    def test_call_output_shape(self) -> None:
        """Full forward pass output should match input shape."""
        out = self.ae(DUMMY, training=False)
        self.assertEqual(out.shape, DUMMY.shape)

    def test_encode_shape(self) -> None:
        """encode() should return (batch, latent_dim)."""
        z = self.ae.encode(DUMMY)
        self.assertEqual(z.shape, (BATCH, LATENT_DIM))

    def test_reconstruct_shape(self) -> None:
        """reconstruct() should return a tensor with the same shape as input."""
        out = self.ae.reconstruct(DUMMY)
        self.assertEqual(out.shape, DUMMY.shape)

    def test_sample_shape(self) -> None:
        """sample(n) should return (n, 64, 64, channels)."""
        samples = self.ae.sample(8)
        self.assertEqual(samples.shape, (8, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    def test_reconstruction_mse_non_negative(self) -> None:
        """MSE reconstruction loss on zero input must be >= 0."""
        out = self.ae(DUMMY, training=False)
        mse = float(tf.reduce_mean(tf.square(DUMMY - out)))
        self.assertGreaterEqual(mse, 0.0)

    def test_pixel_range_output(self) -> None:
        """Decoder sigmoid must keep outputs in [0, 1]."""
        out = self.ae(DUMMY, training=False)
        self.assertGreaterEqual(float(tf.reduce_min(out)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(out)), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────────────────────────────────────


class TestVAE(unittest.TestCase):
    """Tests for the VAE class."""

    def setUp(self) -> None:
        """Build and compile a fresh VAE."""
        self.vae = VAE(latent_dim=LATENT_DIM, channels=CHANNELS, kl_weight=0.5)
        self.vae(DUMMY)  # build weights
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    def test_call_output_shapes(self) -> None:
        """call() should return (x_hat, mu, log_var) with correct shapes."""
        x_hat, mu, log_var = self.vae(DUMMY, training=False)
        self.assertEqual(x_hat.shape, DUMMY.shape)
        self.assertEqual(mu.shape, (BATCH, LATENT_DIM))
        self.assertEqual(log_var.shape, (BATCH, LATENT_DIM))

    def test_reparameterize_shape(self) -> None:
        """reparameterize() output shape must match mu shape."""
        mu = tf.zeros((BATCH, LATENT_DIM))
        log_var = tf.zeros((BATCH, LATENT_DIM))
        z = VAE.reparameterize(mu, log_var)
        self.assertEqual(z.shape, (BATCH, LATENT_DIM))

    def test_reparameterize_stochastic(self) -> None:
        """Two calls to reparameterize with the same inputs should differ."""
        mu = tf.zeros((BATCH, LATENT_DIM))
        log_var = tf.zeros((BATCH, LATENT_DIM))
        z1 = VAE.reparameterize(mu, log_var)
        z2 = VAE.reparameterize(mu, log_var)
        self.assertFalse(
            tf.reduce_all(tf.equal(z1, z2)).numpy(),
            msg="reparameterize returned the same sample twice — check RNG.",
        )

    def test_encode_returns_mu(self) -> None:
        """encode() should return only mu (no log_var) for deterministic inference."""
        z = self.vae.encode(DUMMY)
        self.assertEqual(z.shape, (BATCH, LATENT_DIM))

    def test_reconstruct_shape(self) -> None:
        """reconstruct() must return a tensor with the same shape as input."""
        out = self.vae.reconstruct(DUMMY)
        self.assertEqual(out.shape, DUMMY.shape)

    def test_sample_shape(self) -> None:
        """sample(n) should return (n, 64, 64, channels)."""
        samples = self.vae.sample(6)
        self.assertEqual(samples.shape, (6, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    def test_pixel_range_output(self) -> None:
        """Decoder sigmoid must keep outputs in [0, 1]."""
        x_hat, _, _ = self.vae(DUMMY, training=False)
        self.assertGreaterEqual(float(tf.reduce_min(x_hat)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(x_hat)), 1.0)

    def test_kl_loss_non_negative(self) -> None:
        """KL divergence from N(0,I) to any distribution must be >= 0."""
        _, mu, log_var = self.vae(DUMMY, training=False)
        kl = -0.5 * tf.reduce_mean(1.0 + log_var - tf.square(mu) - tf.exp(log_var))
        self.assertGreaterEqual(float(kl), 0.0)

    def test_interpolate_shape(self) -> None:
        """interpolate() should return (steps+2) decoded frames."""
        img_a = DUMMY[0:1]
        img_b = DUMMY[1:2]
        frames = self.vae.interpolate(img_a, img_b, steps=8)
        self.assertEqual(frames.shape[0], 8 + 2)
        self.assertEqual(frames.shape[1:], (IMAGE_SIZE, IMAGE_SIZE, CHANNELS))


# ─────────────────────────────────────────────────────────────────────────────
# build_models
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildModels(unittest.TestCase):
    """Tests for the build_models factory function."""

    def test_returns_correct_types(self) -> None:
        """build_models should return an Autoencoder and a VAE."""
        ae, vae = build_models(dummy_input=DUMMY)
        self.assertIsInstance(ae, Autoencoder)
        self.assertIsInstance(vae, VAE)

    def test_models_are_compiled(self) -> None:
        """Both models should have an optimiser after build_models."""
        ae, vae = build_models(dummy_input=DUMMY)
        self.assertIsNotNone(ae.optimizer)
        self.assertIsNotNone(vae.optimizer)

    def test_custom_latent_dim(self) -> None:
        """build_models should honour a custom latent_dim."""
        ae, vae = build_models(latent_dim=8, dummy_input=DUMMY)
        z_ae = ae.encode(DUMMY)
        z_vae = vae.encode(DUMMY)
        self.assertEqual(z_ae.shape[-1], 8)
        self.assertEqual(z_vae.shape[-1], 8)


if __name__ == "__main__":
    unittest.main()
