# Probabilistic PCA vs Classical PCA

This document provides a concise comparison between **Classical PCA** and **Probabilistic PCA (PPCA)**, highlighting their respective strengths, weaknesses, and use cases.

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
  x = Wz + \mu + \epsilon, \quad z \sim \mathcal{N}(0,I), \quad \epsilon \sim \mathcal{N}(0, \sigma^2I)
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

## ⚖️ Summary

* **Use Classical PCA** when you want **speed, simplicity, and interpretability** (e.g., preprocessing, visualization, exploratory data analysis).
* **Use PPCA** when you need a **generative probabilistic model**, want to **handle missing data**, or plan to **extend to Bayesian or mixture models**.


