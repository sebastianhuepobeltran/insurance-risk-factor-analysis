# Insurance Risk Factor Analysis
### Multivariate Analysis of Common and Unique Factors via Exploratory Factor Analysis (EFA)

---

## Overview

This repository contains a statistical analysis of **common and unique factors** using multivariate statistical methods in R. The project was initially developed as part of my work in the **Master's in Statistics at Universidad Nacional de Colombia**. I am now organizing and extending the analysis so that it can be reproduced and presented as part of my data science and statistical modeling portfolio.

---

## Theoretical Framework & Mathematical Model

The classical exploratory factor model assumes an observed continuous random vector $\mathbf{X} \in \mathbb{R}^p$ is driven by a lower-dimensional vector of $k$ unobserved common factors $\mathbf{F} \in \mathbb{R}^k$ (where $k \ll p$) plus unique errors $\mathbf{U} \in \mathbb{R}^p$:

$$\mathbf{X} = \boldsymbol{\mu} + \mathbf{\Lambda} \mathbf{F} + \mathbf{U}$$

Where:
* $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$ is the mean vector of observed variables.
* $\mathbf{\Lambda} \in \mathbb{R}^{p \times k}$ represents the matrix of **factor loadings** (covariance structure).
* $\mathbf{F} \sim \mathcal{N}_k(\mathbf{0}, \mathbf{I}_k)$ is the vector of standardized orthogonal latent factors.
* $\mathbf{U} \sim \mathcal{N}_p(\mathbf{0}, \mathbf{\Psi})$ is the vector of specific variances (uniquenesses), with diagonal covariance matrix $\mathbf{\Psi} = \text{diag}(\psi_1, \dots, \psi_p)$.

Under the orthogonality assumptions ($\text{Cov}(\mathbf{F}, \mathbf{U}) = \mathbf{0}$), the total sample population covariance matrix $\mathbf{\Sigma} = \text{Var}(\mathbf{X})$ decomposes into:

$$\mathbf{\Sigma} = \mathbf{\Lambda} \mathbf{\Lambda}^T + \mathbf{\Psi}$$

---

## Modular Pipeline & Implementation

The repository is organized into three progressive R modules in `R/`:

* **`R/01_data_cleaning_and_diagnostics.R`**: Resolves exact linear dependency ($\text{total\_claim\_amount} = \text{injury} + \text{property} + \text{vehicle}$) to eliminate correlation matrix singularity ($\det(\mathbf{R}) > 0$) and computes Kaiser-Meyer-Olkin adequacy (**Overall KMO = 0.61**).
* **`R/02_matrix_spectral_decomposition.R`**: Standardizes observations ($\mathbf{Z}$), solves the characteristic polynomial $\det(\mathbf{R} - \lambda \mathbf{I}_p) = 0$, applies the **Kaiser Criterion** ($\lambda_i > 1.0$), and extracts initial unrotated factor loadings ($\hat{\mathbf{\Lambda}}_1$).
* **`R/03_varimax_rotation_and_scoring.R`**: Executes orthogonal **Varimax rotation** ($\mathbf{\Lambda}^* = \hat{\mathbf{\Lambda}}_1 \mathbf{T}$), generates factor loading heatmaps, and estimates latent factor scores ($\hat{\mathbf{F}}$) via **Thomson's Regression Method**:

$$\hat{\mathbf{F}} = \mathbf{Z} \mathbf{R}^{-1} \mathbf{\Lambda}^*$$

---

## Rotated Loading Structure & Variance Decomposition

| Variable | Factor 1 (Loss Severity) | Factor 2 (Policyholder Maturity) | Communality ($h^2$) | Uniqueness ($u^2$) |
| :--- | :---: | :---: | :---: | :---: |
| `vehicle_claim` | **-0.918** | -0.029 | 0.843 | 0.157 |
| `property_claim` | **-0.849** | -0.038 | 0.722 | 0.278 |
| `injury_claim` | **-0.845** | -0.009 | 0.714 | 0.286 |
| `number_of_vehicles_involved` | **-0.428** | -0.033 | 0.184 | 0.816 |
| `months_as_customer` | -0.091 | **0.976** | 0.961 | 0.039 |
| `age` | -0.106 | **0.974** | 0.960 | 0.040 |
| `policy_annual_premium` | 0.018 | 0.025 | 0.001 | 0.999 |
| `policy_deductable` | -0.078 | 0.045 | 0.008 | 0.992 |

---

## Visualizations

![Scree Plot](scree_plot.png)

![Rotated Factor Map](factor_loadings_map.png)

![Factor Heatmap](factor_loadings_heatmap.png)

---

## Author

**Sebastián H. Beltrán**  
B.Sc. in Mathematics  
Master's in Statistics — *Universidad Nacional de Colombia*
