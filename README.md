
# Probabilistic Factor Models (Demo): PPCA & NMF with Dirichlet-Process Shrinkage

<div align="center">

[![Notebook: Video Demo](https://img.shields.io/badge/Notebook-Video%20(AVI)-blue?logo=jupyter)](dp_factor_video_demo.ipynb)
[![Notebook: Legacy (InferPy/TF)](https://img.shields.io/badge/Notebook-Legacy%20(InferPy%2FTF)-lightgrey?logo=jupyter)](Probabilistic_PCA_inferpy.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![PyMC](https://img.shields.io/badge/PyMC-%E2%89%A55-0b7a75)](#)

</div>

**TL;DR.** This demo repo provides **Probabilistic PCA (PPCA)** and **Nonnegative Matrix Factorization (NMF)** under a **Gaussian noise model**, with the number of components selected **automatically** via a **truncated Dirichlet-Process (DP) stick-breaking** prior.  
For NMF, you can optionally enable **Gamma priors** and **temporal smoothing** (Gaussian Random Walk on temporal weights with softplus).  
A hands-on notebook shows how to apply both models to **2D-in-time `.avi` video** data (shape `H×W×T`) and visualize **spatial components** and **temporal weights**.

> **Legacy preserved.** The original TensorFlow/InferPy notebook is kept as-is for historical comparison:  
> `Probabilistic_PCA_inferpy.ipynb`.

---

## Contents

- [Quickstart (5 min)](#quickstart-5-min)
- [Notebooks & Demos](#notebooks--demos)
- [API at a Glance](#api-at-a-glance)
- [Educational Guide](#educational-guide)
  - [📊 Classical PCA](#-classical-pca)
  - [📈 Probabilistic PCA (PPCA)](#-probabilistic-pca-ppca)
  - [🧩 Nonnegative Matrix Factorization (NMF)](#-nonnegative-matrix-factorization-nmf)
  - [🧭 Estimating Components with a Dirichlet Process](#-estimating-components-with-a-dirichlet-process)
  - [When to Use What](#when-to-use-what)
- [Diagnostics & Practical Tips](#diagnostics--practical-tips)
- [FAQ & Exercises](#faq--exercises)
- [References](#references)

---

## Quickstart (5 min)

```bash
pip install "pymc>=5.10" arviz numpy matplotlib
# optional speedups:
pip install "jax[cpu]" numpyro
# for the video demo:
pip install opencv-python
````

Minimal usage:

```python
import numpy as np
from dp_factor_models import fit_ppca_dp, fit_nmf_dp

# Example data (N samples × D features)
N, D = 300, 50
X = np.abs(np.random.randn(N, D))  # NMF needs nonnegativity; PPCA will center internally

# PPCA with DP shrinkage
ppca = fit_ppca_dp(X, K_max=12, alpha=2.0, draws=600, tune=600, chains=2, target_accept=0.9)
print("PPCA effective K:", ppca.effective_K, "RMSE:", round(ppca.recon_rmse, 4))

# NMF with DP shrinkage + options
nmf = fit_nmf_dp(
    X,
    K_max=12, alpha=2.0, draws=600, tune=600, chains=2, target_accept=0.9,
    use_gamma_priors=True,        # toggleable
    temporal_smoothing=False      # toggleable; True assumes rows=time (W is temporal)
)
print("NMF effective K:", nmf.effective_K, "RMSE:", round(nmf.recon_rmse, 4))
```

---

## Notebooks & Demos

* **Video (AVI) demo:** [`dp_factor_video_demo.ipynb`](dp_factor_video_demo.ipynb)
  Loads `(H, W, T)`, reshapes to `(features=H×W, samples=T)`, fits **DP-PPCA** & **DP-NMF**, then plots:

  * **Spatial components** (2D images) and **temporal weights** (time series), ranked by posterior mean DP weight.
* **Legacy comparison:** [`Probabilistic_PCA_inferpy.ipynb`](Probabilistic_PCA_inferpy.ipynb)
  PPCA (InferPy/TF). Kept unchanged for historical context.

---

## API at a Glance

### `fit_ppca_dp(X, K_max=10, alpha=2.0, draws=1000, tune=1000, chains=2, seed=42, target_accept=0.9, component_threshold=0.02, ...)`

* **Input:** `X` is `(N, D)` (columns are centered internally).
* **Noise:** Gaussian. **DP stick-breaking** scales component columns, shrinking redundant factors.
* **Returns:** `PPCAResult` with:

  * `idata` (ArviZ `InferenceData`),
  * `X_mean_posterior` (N×D),
  * `pi_mean` (stick weights),
  * `effective_K`,
  * `recon_rmse`.

### `fit_nmf_dp(X, K_max=10, alpha=2.0, draws=1000, tune=1000, chains=2, seed=123, target_accept=0.9, component_threshold=0.02, *, use_gamma_priors=False, temporal_smoothing=False, rw_sigma_scale=0.3, gamma_shape_w=2.0, gamma_rate_w=2.0, gamma_shape_h=2.0, gamma_rate_h=2.0)`

* **Input:** `X` is `(N, D)` and **nonnegative**.
* **Options:**

  * `use_gamma_priors`: use Gamma(shape, rate) for **H** and (if no smoothing) **W**; otherwise HalfNormal.
  * `temporal_smoothing`: model **W** (rows=time) with a **GaussianRandomWalk** per component → softplus → ≥0 → DP shrink.
  * `rw_sigma_scale`: HalfNormal prior scale on RW σ; smaller ⇒ smoother time courses.
* **Returns:** `NMFResult` with `idata`, `X_mean_posterior` (N×D), `pi_mean`, `effective_K`, `recon_rmse`.

> **Spatial vs temporal mapping**
>
> * **PPCA:** `W` (D×K) ≈ spatial maps; `Z` (N×K) ≈ temporal weights (available in `idata.posterior`).
> * **NMF:** `H` (K×D) → spatial (reshape to H×W); `W` (N×K) → temporal.

---

## Educational Guide

This integrates and extends your original theoretical notes to be concise and practice-oriented.

### 📊 Classical PCA

**Formulation**

* Algebraic method that finds orthogonal directions (principal components) maximizing variance.
* Solved via eigenvalue decomposition (covariance matrix) or singular value decomposition (SVD).

**✅ Pros**

* Computationally efficient (closed-form).
* Deterministic and stable.
* Simple geometric intuition, widely available.

**⚠️ Cons**

* No explicit probabilistic model or uncertainty.
* Sensitive to outliers/noise.
* Cannot handle missing data directly.
* Hard to extend into Bayesian/generative frameworks.

---

### 📈 Probabilistic PCA (PPCA)

**Model (Gaussian latent variable)**
[
x = W z + \mu + \epsilon,\quad z \sim \mathcal{N}(0,I),\quad \epsilon \sim \mathcal{N}(0,\sigma^2 I).
]

**✅ Pros**

* Explicit Gaussian noise model → robustness to noise.
* Likelihood-based → model selection & statistical testing.
* Naturally handles missing data.
* Posterior over latent variables (uncertainty).
* Foundation for Bayesian PCA, mixtures of PPCA, factor analyzers, VAEs.

**⚠️ Cons**

* More compute than PCA.
* Assumes Gaussian noise.
* Slightly more complex to implement/interpret.

---

### 🧩 Nonnegative Matrix Factorization (NMF)

**Model (Gaussian likelihood in this repo)**
Approximate (X \approx W H) with (W \ge 0), (H \ge 0). Encourages **parts-based**, additive structure.

**Options provided**

* **Gamma priors** (shape-rate) for stronger positivity/sparsity bias.
* **Temporal smoothing** on **W** (rows=time) via a **Gaussian Random Walk** with **softplus** to enforce nonnegativity → smooth, interpretable time courses for videos/neural data.

---

### 🧭 Estimating Components with a Dirichlet Process

This demo infers effective dimensionality rather than fixing (k):

* Place a **Dirichlet Process** prior using a **truncated stick-breaking** construction over up to (K_{\max}) components.
* Draw (v_k \sim \mathrm{Beta}(1,\alpha)) for (k=1,\dots,K_{\max}-1) and construct:
  [
  \pi_1 = v_1,\quad \pi_k = v_k \prod_{j<k}(1-v_j),\quad \pi_{K_{\max}} = \prod_{j=1}^{K_{\max}-1}(1-v_j).
  ]
* Scale each component column by (\sqrt{\pi_k}). Inference shrinks redundant components by driving their weights toward zero.
* **Selected dimensionality** is the count of components with non-negligible posterior weight (e.g., (\mathbb{E}[\pi_k]>\tau), (\tau\approx 0.02)).
* **(\alpha)** controls parsimony: smaller → stronger shrinkage to early components; larger → more active components.

**Benefits**

* Data-driven complexity control.
* Uncertainty over effective (k).
* Compatible with PPCA’s missing-data handling.

> For strictly single-subspace structure, ARD in Bayesian PCA can similarly prune components; DP shrinkage generalizes naturally to heterogeneous latent structure.

---

### When to Use What

| Goal / Data Need                        | Recommended                       |
| --------------------------------------- | --------------------------------- |
| Fast, deterministic reduction           | **Classical PCA**                 |
| Probabilistic model, missing data       | **PPCA (Gaussian)**               |
| Parts-based spatial maps + time courses | **NMF (Gaussian)**                |
| Smooth temporal weights (rows=time)     | **NMF + temporal_smoothing=True** |
| Stronger positivity/sparsity            | **NMF + use_gamma_priors=True**   |
| Avoid hand-tuning (k)                   | **PPCA/NMF + DP stick-breaking**  |

---

## Diagnostics & Practical Tips

* **Effective K:** sort `pi_mean` and count weights above (\tau\in[0.01,0.05]).
* **Sampler health:** if divergences occur, raise `target_accept` to `0.9–0.95`, reduce `K_max`, or enable JAX/NumPyro.
* **Large videos (e.g., `250×250×1000`):** ROI crop, patchify, or frame-subsample; `temporal_smoothing=True` stabilizes W.
* **Identifiability:** factor models have scale/rotation ambiguities—prioritize subspaces, recon error, and replicate stability.

---

## FAQ & Exercises

**Q:** Why Gaussian noise for NMF here?
**A:** This demo targets real-valued intensities. For counts, Poisson or Negative Binomial are natural extensions.

**Q:** How should I set (\alpha) (DP concentration)?
**A:** Start with `1–3`. Smaller → fewer active components; larger → more diffuse usage.

<details><summary><b>Exercise 1 (solution hidden): Recover rank on synthetic data</b></summary>

Generate rank-4 data, run `fit_ppca_dp(X, K_max=12)`, and verify `effective_K ≈ 4` via `pi_mean`.

</details>

<details><summary><b>Exercise 2: Smooth vs non-smooth time courses</b></summary>

On video data, toggle `temporal_smoothing=True` and sweep `rw_sigma_scale` from `0.5 → 0.2`. Compare temporal plots and reconstruction RMSE.

</details>

---

## References

* Tipping & Bishop (1999). *Probabilistic PCA*. **JRSS-B**.
* Blei & Jordan (2006). *Variational Inference for the Dirichlet Process*.
* Lee & Seung (1999). *Learning the parts of objects by non-negative matrix factorization*.

