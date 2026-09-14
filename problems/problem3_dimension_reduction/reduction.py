"""
reduction.py (Problem 3 — Dimension Reduction)

Two unsupervised dimension-reduction methods:

1. PCA (classical) - thin wrappers around scikit-learn's PCA, adding
   the reconstruction-MSE calculation the project spec asks for.
2. SimpleAutoencoder (deep) - a small feed-forward autoencoder trained
   with src/torch_utils.py's generic train_torch_model() (Phase 2),
   the same way every other neural network in this project has been.

NEITHER method ever sees a label (sky-condition, generation-regime, or
Output Power) while fitting - both only ever see the unified feature
matrix from problems/problem3_dimension_reduction/features.py. Labels
are used ONLY afterward, to color a visualization or evaluate a
downstream model - never to influence the representation itself.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def fit_pca(X_train, n_components: int, seed: int = 42) -> PCA:
    """
    Fit PCA on TRAINING features only.

    ML concept: PCA finds the directions (principal components) along
    which the data varies the most, and re-expresses each row as
    coordinates along those directions instead of the original
    columns. It's unsupervised - it never looks at any label, only at
    how the input features themselves vary (course_context/
    COURSE_CONTEXT.md, Week10).

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        TRAINING data only - never pass test or combined data here.
    n_components : int
        Target dimensionality (e.g. 2, 5, 10).
    seed : int
        PCA's `svd_solver` can have a randomized component for large
        data; passed for reproducibility even though PCA is
        deterministic for the exact/full solver used here by default.

    Returns
    -------
    sklearn.decomposition.PCA
        A fitted PCA object. Use `.transform(X)` to reduce any
        DataFrame (train or test) with it, and `.inverse_transform()`
        to reconstruct.
    """
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_train)
    return pca


def pca_reconstruction_mse(pca: PCA, X) -> float:
    """
    Reconstruction quality: reduce X to n_components, reconstruct it
    back to the original feature space, and measure the mean squared
    error against the real X. Lower = the bottleneck preserved more
    information.

    Parameters
    ----------
    pca : PCA
        A fitted PCA object.
    X : array-like
        Data to evaluate (can be train or test - this function doesn't
        fit anything, just measures reconstruction quality).

    Returns
    -------
    float
    """
    reduced = pca.transform(X)
    reconstructed = pca.inverse_transform(reduced)
    return float(np.mean((np.asarray(X) - reconstructed) ** 2))


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------


class SimpleAutoencoder(nn.Module):
    """
    A small feed-forward autoencoder: Input -> Dense -> ReLU ->
    Latent bottleneck -> Dense -> ReLU -> Output (reconstruction).

    `encode(x)` returns just the d-dimensional latent representation
    (what downstream models use). `forward(x)` returns the FULL
    reconstruction (input_dim-sized) - used only during training, to
    compute the reconstruction loss against the original input.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_size: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encode(x))


def encode_with_autoencoder(model: SimpleAutoencoder, X, device: str) -> np.ndarray:
    """
    Run a trained SimpleAutoencoder's encoder half on X and return the
    latent representation as a plain numpy array.

    Parameters
    ----------
    model : SimpleAutoencoder
    X : array-like
    device : str

    Returns
    -------
    numpy.ndarray, shape (n_samples, latent_dim)
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(device)
        latent = model.encode(X_tensor)
    return latent.cpu().numpy()


def autoencoder_reconstruction_mse(model: SimpleAutoencoder, X, device: str) -> float:
    """
    Reconstruction quality for a trained autoencoder - same idea as
    pca_reconstruction_mse(), just using the autoencoder's full
    forward pass (encode + decode) instead of PCA's transform/
    inverse_transform.

    Parameters
    ----------
    model : SimpleAutoencoder
    X : array-like
    device : str

    Returns
    -------
    float
    """
    model.eval()
    with torch.no_grad():
        X_array = np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(X_array).to(device)
        reconstruction = model(X_tensor).cpu().numpy()
    return float(np.mean((X_array - reconstruction) ** 2))
