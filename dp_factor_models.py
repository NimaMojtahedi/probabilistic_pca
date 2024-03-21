"""
dp_factor_models.py

Probabilistic PCA (PPCA) and Nonnegative Matrix Factorization (NMF) with:
- Gaussian noise
- Truncated stick-breaking Dirichlet Process prior over component scales
  to induce automatic, data-driven selection of the effective number of components.
Enhanced options:
- NMF Gamma priors (toggleable)
- NMF temporal smoothing on temporal weights (W) via a Gaussian Random Walk
  with positivity enforced by softplus, toggleable.

Author: Nima Mojtahedi
License: MIT

Why this design?
- PyMC4 is not a maintained product; modern, production-grade probabilistic
  programming in Python is served by PyMC v5+. This module uses PyMC v5.
- The Dirichlet Process (DP) via stick-breaking is applied as a *shrinkage prior*
  over columns of the factor-loading (PPCA) or basis (NMF) matrices. With a sufficiently
  large K_max, the posterior naturally "turns off" superfluous components.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from pymc.sampling.jax import sample_numpyro_nuts


# ---------------------------
# Utilities
# ---------------------------

def _stick_breaking_weights(v: pt.TensorVariable) -> pt.TensorVariable:
    """
    Compute truncated stick-breaking weights pi from Beta variables v.

    Given v_k ~ Beta(1, alpha), k=1..K-1
    pi_k = v_k * prod_{j<k} (1 - v_j)
    pi_K = prod_{j=1}^{K-1} (1 - v_j)

    Parameters
    ----------
    v : 1D PyTensor variable of shape (K-1,)

    Returns
    -------
    pi : 1D PyTensor variable of shape (K,)
    """
    # cumulative products of (1 - v)
    cumprod_1_minus_v = pt.cumprod(1 - v)
    # prepend a 1 for the empty product term
    head = pt.concatenate([pt.ones((1,), dtype=v.dtype), cumprod_1_minus_v[:-1]])
    pi_head = v * head  # shape (K-1,)
    pi_tail = cumprod_1_minus_v[-1:]  # shape (1,)
    pi = pt.concatenate([pi_head, pi_tail])  # shape (K,)
    return pi


def _maybe_sample(draws: int, tune: int, chains: int, target_accept: float = 0.9) -> az.InferenceData:
    """
    Try to sample with NumPyro/JAX if available; fallback to PyMC's default NUTS sampler.
    """
    try:
        idata = sample_numpyro_nuts(draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    except Exception:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, init="jitter+adapt_diag")
    return idata


@dataclass
class PPCAResult:
    idata: az.InferenceData
    X_mean_posterior: np.ndarray
    pi_mean: np.ndarray
    effective_K: int
    recon_rmse: float


@dataclass
class NMFResult:
    idata: az.InferenceData
    X_mean_posterior: np.ndarray
    pi_mean: np.ndarray
    effective_K: int
    recon_rmse: float


# ---------------------------
# Models
# ---------------------------

def fit_ppca_dp(
    X: np.ndarray,
    K_max: int = 10,
    alpha: float = 2.0,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 42,
    target_accept: float = 0.9,
    component_threshold: float = 0.02,
) -> PPCAResult:
    """
    Fit a Probabilistic PCA with Gaussian noise and a DP (stick-breaking) shrinkage prior
    over components.

    Model:
      Let N samples, D features, and K_max latent components (truncation).
      v_k ~ Beta(1, alpha) for k=1..K_max-1
      pi = stick_breaking(v) in R^K_max, sum_k pi_k = 1
      mu_d ~ Normal(0, 5)
      sigma_noise ~ HalfNormal(1)

      Column-wise shrinkage via pi:
        W_{:,k} ~ Normal(0, sigma_w * sqrt(pi_k))   with sigma_w ~ HalfNormal(1)
      Latent factors per sample:
        z_{n,k} ~ Normal(0, 1)

      Likelihood:
        X_{n,:} ~ Normal(mu + z_n @ W^T, sigma_noise)

    Parameters
    ----------
    X : array, shape (N, D)
        Data matrix (recommend centering columns beforehand; this function centers internally).
    K_max : int
        Truncation level for stick-breaking (upper bound on components).
    alpha : float
        DP concentration parameter; larger -> more/less shrinkage? Larger alpha spreads mass across
        more components; smaller alpha shrinks faster onto early components.
    draws, tune, chains, seed, target_accept : sampling controls
    component_threshold : float
        Threshold on E[pi_k] to count a component as "effective".

    Returns
    -------
    PPCAResult
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    N, D = X.shape

    # Center columns for PPCA (standard practice)
    X_mean_col = X.mean(axis=0, keepdims=True)
    X_centered = X - X_mean_col

    with pm.Model() as model:
        # Stick-breaking for component weights
        v = pm.Beta("v", alpha=1.0, beta=alpha, shape=K_max - 1)
        pi = pm.Deterministic("pi", _stick_breaking_weights(v))  # shape (K_max,)

        # Noise + loadings scale
        sigma_noise = pm.HalfNormal("sigma_noise", sigma=1.0)
        sigma_w = pm.HalfNormal("sigma_w", sigma=1.0)

        # Column-wise scales via sqrt(pi) (broadcast over D rows)
        # W_raw has shape (D, K_max), scaled deterministically
        W_raw = pm.Normal("W_raw", mu=0.0, sigma=sigma_w, shape=(D, K_max))
        W = pm.Deterministic("W", W_raw * pt.sqrt(pi)[None, :])

        # Latent factors per sample
        Z = pm.Normal("Z", mu=0.0, sigma=1.0, shape=(N, K_max))

        # Mean vector (we fit to centered data; set mean to 0)
        mu_vec = pt.zeros((D,), dtype=W.dtype)

        X_recon = pm.Deterministic("X_mean", Z @ W.T + mu_vec)

        # Gaussian likelihood on centered X
        pm.Normal("X_obs", mu=X_recon, sigma=sigma_noise, observed=X_centered)

        # Inference
        # pm.random.seed(seed)
        idata = _maybe_sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept)

    # Posterior expectations
    X_mean_posterior = idata.posterior["X_mean"].mean(dim=("chain", "draw")).to_numpy()
    pi_mean = idata.posterior["pi"].mean(dim=("chain", "draw")).to_numpy()
    effective_K = int((pi_mean > component_threshold).sum())

    # Reconstruction on original scale
    X_recon_centered = X_mean_posterior
    X_recon_full = X_recon_centered + X_mean_col  # add back column means
    rmse = float(np.sqrt(np.mean((X - X_recon_full) ** 2)))

    return PPCAResult(
        idata=idata,
        X_mean_posterior=X_recon_full,
        pi_mean=pi_mean,
        effective_K=effective_K,
        recon_rmse=rmse,
    )


def fit_nmf_dp(
    X: np.ndarray,
    K_max: int = 10,
    alpha: float = 2.0,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 123,
    target_accept: float = 0.9,
    component_threshold: float = 0.02,
    # New options:
    use_gamma_priors: bool = False,
    temporal_smoothing: bool = False,
    rw_sigma_scale: float = 0.3,
    gamma_shape_w: float = 2.0,
    gamma_rate_w: float = 2.0,
    gamma_shape_h: float = 2.0,
    gamma_rate_h: float = 2.0,
) -> NMFResult:
    """
    Nonnegative Matrix Factorization with Gaussian noise and DP shrinkage.
    Options:
    - Gamma priors for W/H (toggle via `use_gamma_priors`)
    - Temporal smoothing for W via Gaussian Random Walk with softplus positivity
      (toggle via `temporal_smoothing`).

    Model (base, no smoothing):
        v_k ~ Beta(1, alpha),  pi = SB(v)
        sigma_noise ~ HalfNormal(1)
        If use_gamma_priors:
            W_base ~ Gamma(shape=gamma_shape_w, rate=gamma_rate_w)   (N x K)
            H      ~ Gamma(shape=gamma_shape_h, rate=gamma_rate_h)   (K x D)
        else:
            W_base ~ HalfNormal(sigma_w)                             (N x K)
            H      ~ HalfNormal(sigma_h)                             (K x D)
        W = W_base * sqrt(pi)[None,:]
        X ~ Normal(W @ H, sigma_noise)

    Temporal smoothing (W):
        A_{:,k} ~ GaussianRandomWalk(sigma_rw[k])   (length N)
        W_base  = softplus(A)                       (ensure >= 0)
        W       = W_base * sqrt(pi)[None,:]
        (If both temporal_smoothing and use_gamma_priors=True, Gamma priors
         are applied to H only; W uses the smoothed construction.)

    Parameters
    ----------
    X : (N, D) array, nonnegative
        Data matrix.
    rw_sigma_scale : float
        Scale of HalfNormal prior for sigma_rw (each component k). Smaller -> smoother.
    """
    X = np.asarray(X, dtype=float)
    if (X < 0).any():
        raise ValueError("X contains negative entries; NMF requires nonnegative data.")
    N, D = X.shape

    with pm.Model() as model:
        # DP stick-breaking
        v = pm.Beta("v", alpha=1.0, beta=alpha, shape=K_max - 1)
        pi = pm.Deterministic("pi", _stick_breaking_weights(v))

        sigma_noise = pm.HalfNormal("sigma_noise", sigma=1.0)

        # H prior
        if use_gamma_priors:
            H = pm.Gamma("H", alpha=gamma_shape_h, beta=gamma_rate_h, shape=(K_max, D))
        else:
            sigma_h = pm.HalfNormal("sigma_h", sigma=1.0)
            H = pm.HalfNormal("H", sigma=sigma_h, shape=(K_max, D))

        # W prior (temporal dimension N == time)
        if temporal_smoothing:
            # Component-wise random walk standard deviations
            sigma_rw = pm.HalfNormal("sigma_rw", sigma=rw_sigma_scale, shape=(K_max,))
            # A: unconstrained latent temporal processes (N x K)
            A = pm.GaussianRandomWalk("A", sigma=sigma_rw, shape=(N, K_max))
            W_base = pm.Deterministic("W_base", pm.math.softplus(A))
        else:
            if use_gamma_priors:
                W_base = pm.Gamma("W_base", alpha=gamma_shape_w, beta=gamma_rate_w, shape=(N, K_max))
            else:
                sigma_w = pm.HalfNormal("sigma_w", sigma=1.0)
                W_base = pm.HalfNormal("W_base", sigma=sigma_w, shape=(N, K_max))

        # Apply DP shrinkage to columns
        W = pm.Deterministic("W", W_base * pt.sqrt(pi)[None, :])

        X_mean = pm.Deterministic("X_mean", W @ H)
        pm.Normal("X_obs", mu=X_mean, sigma=sigma_noise, observed=X)

        pm.random.seed(seed)
        idata = _maybe_sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept)

    X_mean_post = idata.posterior["X_mean"].mean(dim=("chain", "draw")).to_numpy()
    pi_mean = idata.posterior["pi"].mean(dim=("chain", "draw")).to_numpy()
    effective_K = int((pi_mean > component_threshold).sum())
    rmse = float(np.sqrt(np.mean((X - X_mean_post) ** 2)))

    return NMFResult(idata=idata, 
                     X_mean_posterior=X_mean_post, 
                     pi_mean=pi_mean, 
                     effective_K=effective_K, 
                     recon_rmse=rmse)

# ---------------------------
# Synthetic data helpers
# ---------------------------

def make_synthetic_ppca(
    N: int = 400, D: int = 20, K_true: int = 3, noise_sd: float = 0.2, seed: int = 7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create centered PPCA-like synthetic data.
    Returns (X, W_true, Z_true), with X centered already.
    """
    rng = np.random.default_rng(seed)
    W_true = rng.normal(0, 1.0, size=(D, K_true))
    Z_true = rng.normal(0, 1.0, size=(N, K_true))
    X = Z_true @ W_true.T + rng.normal(0, noise_sd, size=(N, D))
    # Center columns
    X -= X.mean(axis=0, keepdims=True)
    return X, W_true, Z_true


def make_synthetic_nmf(
    N: int = 300, D: int = 25, K_true: int = 4, noise_sd: float = 0.1, seed: int = 11
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create nonnegative synthetic data for NMF with Gaussian noise.
    Returns (X, W_true, H_true).
    """
    rng = np.random.default_rng(seed)
    W_true = rng.gamma(shape=2.0, scale=1.0, size=(N, K_true))
    H_true = rng.gamma(shape=2.0, scale=1.0, size=(K_true, D))
    X_mean = W_true @ H_true
    X = X_mean + rng.normal(0, noise_sd, size=X_mean.shape)
    X = np.clip(X, 0.0, None)  # ensure nonnegativity
    return X, W_true, H_true


# ---------------------------
# Demonstration
# ---------------------------

if __name__ == "__main__":
    # ============ PPCA demo ============
    X_ppca, W_t, Z_t = make_synthetic_ppca(N=400, D=20, K_true=3, noise_sd=0.25, seed=1)
    ppca_res = fit_ppca_dp(
        X_ppca,
        K_max=10,
        alpha=2.0,
        draws=800,
        tune=800,
        chains=2,
        seed=123,
        component_threshold=0.02,
    )

    print("\n[PPCA with DP shrinkage]")
    print(f"Effective K (threshold 0.02): {ppca_res.effective_K}")
    print(f"Posterior mean pi: {np.round(ppca_res.pi_mean, 3)}")
    print(f"Reconstruction RMSE: {ppca_res.recon_rmse:.4f}")

    # ============ NMF demo ============
    X_nmf, W_true, H_true = make_synthetic_nmf(N=300, D=25, K_true=4, noise_sd=0.1, seed=7)
    nmf_res = fit_nmf_dp(
        X_nmf,
        K_max=12,
        alpha=2.5,
        draws=800,
        tune=800,
        chains=2,
        seed=321,
        component_threshold=0.02,
    )

    print("\n[NMF with DP shrinkage]")
    print(f"Effective K (threshold 0.02): {nmf_res.effective_K}")
    print(f"Posterior mean pi: {np.round(nmf_res.pi_mean, 3)}")
    print(f"Reconstruction RMSE: {nmf_res.recon_rmse:.4f}")

    # Optional: Quick sanity checks on shapes
    assert X_ppca.shape == ppca_res.X_mean_posterior.shape
    assert X_nmf.shape == nmf_res.X_mean_posterior.shape
    print("\nAll shapes look good.")
