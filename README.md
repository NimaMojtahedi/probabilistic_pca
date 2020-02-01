# PPCA: Overview and Theoretical Comparison to PCA ✨

<div align="center">

[![Notebook](https://img.shields.io/badge/Notebook-Open-blue?logo=jupyter)](Probabilistic_PCA_inferpy.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](#)
[![Edward](https://img.shields.io/badge/Edward-1.3%2B-4c9aff)](#)

</div>

_This repo accompanies the notebook `Probabilistic_PCA_inferpy.ipynb`. It compares Classical PCA and Probabilistic PCA (PPCA) at a theoretical level and explains how we estimate the number of components using a Dirichlet Process (DP) prior._

**Quick links:** [Open the notebook](Probabilistic_PCA_inferpy.ipynb)

---

## Contents

- [Classical PCA](#-classical-pca)
- [Probabilistic PCA (PPCA)](#-probabilistic-pca-ppca)
- [Estimating the number of components (Dirichlet Process)](#-estimating-the-number-of-components-with-a-dirichlet-process)
- [Summary](#-summary)

---

## 📊 Classical PCA

**Formulation:**

* Algebraic method that finds orthogonal directions (principal components) maximizing variance.
* Solved via eigenvalue decomposition (covariance matrix) or singular value decomposition (SVD).

**✅ Pros:**

* Computationally efficient (closed-form solution).
* Deterministic and stable.
* Simple geometric intuition, easy to interpret.
* Ubiquitous in libraries and workflows.

**⚠️ Cons:**

* No explicit probabilistic model or uncertainty estimates.
* Sensitive to outliers and noise.
* Cannot handle missing data directly.
* Hard to extend into Bayesian or generative frameworks.

---

## 📈 Probabilistic PCA (PPCA)

**Formulation:**

* Latent variable Gaussian model:

  $$
  x = Wz + \mu + \epsilon, \quad z \\sim \mathcal{N}(0,I), \quad \epsilon \\sim \mathcal{N}(0, \sigma^2 I)
  $$
* Parameters estimated via maximum likelihood, often with Expectation–Maximization (EM).

**✅ Pros:**

* Explicit Gaussian noise model → more robust to noise.
* Likelihood-based → enables model selection and statistical testing.
* Naturally handles missing data via EM inference.
* Provides posterior distributions over latent variables (uncertainty quantification).
* Forms the foundation for Bayesian PCA, mixtures of PPCA, factor analyzers, and modern deep latent variable models (e.g., VAEs).

**⚠️ Cons:**

* More computationally expensive than classical PCA.
* Assumes Gaussian noise, limiting robustness for non-Gaussian data.
* Slightly more complex to implement and interpret.

---

## 🧭 Estimating the number of components with a Dirichlet Process

In the notebook, the effective dimensionality is inferred rather than fixed:

- A Dirichlet Process prior with a truncated stick‑breaking construction places weights over up to K_max components. Inference (variational/EM) shrinks redundant components by driving their posterior weights toward zero.
- The selected dimensionality is the count of components with non‑negligible posterior weight and loadings, yielding automatic model order selection without manual k tuning.
- The concentration parameter (α) governs how readily new components are used; putting a hyperprior on α lets the data balance parsimony and flexibility.
- Benefits: data‑driven complexity control, uncertainty over the effective k, and compatibility with PPCA’s handling of missing data.
- Diagnostics: posterior stick lengths/weights, stabilization of σ², and improved held‑out predictive likelihood.

For strictly single‑subspace structure, Bayesian PCA with ARD can similarly prune unused components; the DP approach extends naturally to multi‑modal or heterogeneous latent structure.

## ⚖️ Summary

* **Use Classical PCA** when you want **speed, simplicity, and interpretability** (e.g., preprocessing, visualization, exploratory data analysis).
* **Use PPCA** when you need a **generative probabilistic model**, want to **handle missing data**, or plan to **extend to Bayesian or mixture models**.


