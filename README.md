# Insurance Risk Factor Analysis
### Actuarial Applications of Exploratory Factor Analysis (EFA) & Latent Risk Identification

---

## Author & Academic Context

* **Author:** Sebastián H. Beltrán
* **Academic Background:** B.Sc. in Mathematics | Master's in Statistics — *Universidad Nacional de Colombia*
* **Domain:** Actuarial Science / Multivariate Statistical Analysis / Risk Analytics

---

## 1. Project Overview & Actuarial Context

In insurance risk analytics and loss modeling, financial portfolios typically consist of multi-dimensional policy datasets capturing vehicle claims, driver demographics, and policy terms. A major statistical challenge in this setting is **high dimensionality coupled with severe multicollinearity**, which destabilizes regression coefficients and distorts risk pricing models.

This project implements a statistically rigorous **Exploratory Factor Analysis (EFA)** pipeline in R. The primary objectives are:
1. Identifying underlying unobserved latent drivers (**latent risk factors**) that govern vehicle loss behavior.
2. Resolving exact structural identities among variables that cause matrix singularity ($\det(\mathbf{R}) = 0$).
3. Reducing the dimensionality of an 8-variable insurance portfolio into orthogonal latent factor scores ($\hat{\mathbf{F}}$) suitable for downstream predictive modeling (e.g., Generalized Linear Models).

---

## 2. Theoretical Framework & Mathematical Formulation

### 2.1 The Classical Linear Factor Model
Let $\mathbf{X} = (X_1, X_2, \dots, X_p)^T \in \mathbb{R}^p$ be an observed vector of continuous random variables representing policy attributes. We model $\mathbf{X}$ as a linear combination of $k$ common latent factors $\mathbf{F} = (F_1, \dots, F_k)^T \in \mathbb{R}^k$ (where $k \ll p$) plus unique errors $\mathbf{U} = (U_1, \dots, U_p)^T \in \mathbb{R}^p$:

$$\mathbf{X} - \boldsymbol{\mu} = \mathbf{\Lambda} \mathbf{F} + \mathbf{U}$$

Where:
* $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}] \in \mathbb{R}^p$ is the expectation vector of observed variables.
* $\mathbf{\Lambda} \in \mathbb{R}^{p \times k}$ represents the matrix of **factor loadings**, where $\lambda_{ij}$ quantifies the covariance between variable $i$ and latent factor $j$.
* $\mathbf{F} \sim \mathcal{N}_k(\mathbf{0}, \mathbf{I}_k)$ is the vector of standardized common factors ($\mathbb{E}[\mathbf{F}] = \mathbf{0}$, $\text{Var}(\mathbf{F}) = \mathbf{I}_k$).
* $\mathbf{U} \sim \mathcal{N}_p(\mathbf{0}, \mathbf{\Psi})$ represents unique variation (specific variance + measurement error), where $\mathbf{\Psi} = \text{diag}(\psi_1, \dots, \psi_p)$.

### 2.2 Covariance Matrix Structure & Variance Decomposition
Assuming common factors and specific errors are uncorrelated ($\text{Cov}(\mathbf{F}, \mathbf{U}) = \mathbf{0}$), the population covariance matrix $\mathbf{\Sigma} = \text{Var}(\mathbf{X})$ decomposes into:

$$\mathbf{\Sigma} = \mathbf{\Lambda} \mathbf{\Lambda}^T + \mathbf{\Psi}$$

For any individual variable $X_i$, its total variance decomposes into **Communality** ($h_i^2$) and **Uniqueness** ($u_i^2 = \psi_i$):

$$\text{Var}(X_i) = \sigma_{ii} = \underbrace{\sum_{j=1}^k \lambda_{ij}^2}_{\text{Communality } h_i^2} + \underbrace{\psi_i}_{\text{Uniqueness } u_i^2}$$

* **Communality ($h_i^2$):** Proportion of variance in $X_i$ shared with the common latent factors.
* **Uniqueness ($u_i^2$):** Proportion of variance specific to $X_i$ not captured by common factors.

---

## 3. Step-by-Step Analytical Pipeline

### Step 1: Diagnostics & Multicollinearity Resolution
* **File:** `R/01_data_cleaning_and_diagnostics.R`

#### Mathematical Issue
When exact linear identities exist within the dataset, the sample correlation matrix $\mathbf{R}$ becomes singular ($\det(\mathbf{R}) = 0$). This prevents matrix inversion ($\mathbf{R}^{-1}$ does not exist) and causes Kaiser-Meyer-Olkin (KMO) partial correlation metrics to yield undefined ($\text{NaN}$) values.

In the raw claims dataset, an exact identity is present:

$$\text{total\_claim\_amount} = \text{injury\_claim} + \text{property\_claim} + \text{vehicle\_claim}$$

#### Resolution & Factorability Assessment
Removing `total_claim_amount` restores positive-definiteness to the correlation matrix ($\det(\mathbf{R}) > 0$). The overall Kaiser-Meyer-Olkin measure yields **Overall KMO = 0.61**, confirming sample adequacy for factor extraction.

---

### Step 2: Spectral Decomposition & Dimension Selection
* **File:** `R/02_matrix_spectral_decomposition.R`

#### Eigenvalue Decomposition
1. **Standardization:** Observations are standardized: $\mathbf{Z} = (\mathbf{X} - \mathbf{1}\boldsymbol{\mu}^T) \mathbf{D}_{\sigma}^{-1}$
2. **Correlation Matrix Computation:** $\mathbf{R} = \frac{1}{n-1} \mathbf{Z}^T \mathbf{Z}$
3. **Spectral Solution:** Solving the characteristic equation $\det(\mathbf{R} - \lambda \mathbf{I}_p) = 0$ yields eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p$ and orthonormal eigenvectors $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_p]$.

#### Dimension Retention Rule (Kaiser Criterion)
Factors are retained if their eigenvalue satisfies $\lambda_i > 1.0$:
* $\lambda_1 = 2.5076$ (Explains $31.35\%$ of total variance)
* $\lambda_2 = 1.8854$ (Explains $23.57\%$ of total variance)
* **Cumulative Explained Variance ($k = 2$):** $54.91\%$

$$\hat{\mathbf{\Lambda}}_1 = \mathbf{V}_k \mathbf{D}_k^{1/2} = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 \end{bmatrix} \begin{bmatrix} \sqrt{\lambda_1} & 0 \\ 0 & \sqrt{\lambda_2} \end{bmatrix}$$

---

### Step 3: Orthogonal Varimax Rotation & Latent Factor Scoring
* **File:** `R/03_varimax_rotation_and_scoring.R`

#### Varimax Transformation
Unrotated factor loadings often exhibit complex cross-loadings. We apply an orthogonal rotation matrix $\mathbf{T}$ ($\mathbf{T}^T \mathbf{T} = \mathbf{I}$) to obtain rotated loadings $\mathbf{\Lambda}^* = \hat{\mathbf{\Lambda}}_1 \mathbf{T}$, maximizing variance of squared loadings:

$$V = \frac{1}{p} \sum_{j=1}^k \left[ \sum_{i=1}^p \left( \frac{\lambda_{ij}^*}{\sqrt{h_i^2}} \right)^4 - \frac{1}{p} \left( \sum_{i=1}^p \frac{(\lambda_{ij}^*)^2}{h_i^2} \right)^2 \right]$$

#### Thomson's Regression Scoring
Latent factor scores $\hat{\mathbf{F}}$ for individual policyholders are estimated using Thomson's regression method:

$$\hat{\mathbf{F}} = \mathbf{Z} \mathbf{R}^{-1} \mathbf{\Lambda}^*$$

---

## 4. Empirical Results & Actuarial Interpretation

### Rotated Factor Loading Matrix & Variance Decomposition

| Observed Variable | Factor 1 (Loss Severity) | Factor 2 (Policyholder Maturity) | Communality ($h^2$) | Uniqueness ($u^2$) |
| :--- | :---: | :---: | :---: | :---: |
| `vehicle_claim` | **-0.918** | -0.029 | 0.843 | 0.157 |
| `property_claim` | **-0.849** | -0.038 | 0.722 | 0.278 |
| `injury_claim` | **-0.845** | -0.009 | 0.714 | 0.286 |
| `number_of_vehicles_involved` | **-0.428** | -0.033 | 0.184 | 0.816 |
| `months_as_customer` | -0.091 | **0.976** | 0.961 | 0.039 |
| `age` | -0.106 | **0.974** | 0.960 | 0.040 |
| `policy_annual_premium` | 0.018 | 0.025 | 0.001 | 0.999 |
| `policy_deductable` | -0.078 | 0.045 | 0.008 | 0.992 |

### Key Findings
1. **Factor 1 (Economic Loss Severity):** Dominates financial loss attributes (`vehicle_claim`, `property_claim`, `injury_claim`). Reduces claim severity into a unified economic risk metric.
2. **Factor 2 (Policyholder Tenure & Demographic Maturity):** Captures customer longevity (`months_as_customer`) and age (`age`), exhibiting high communality ($h^2 > 0.96$).
3. **Policy Noise:** Policy contract terms (`policy_annual_premium`, `policy_deductable`) display uniqueness $u^2 \approx 0.99$, indicating zero common variance with historical loss amounts or driver age.

---

## 5. Visualizations

### Scree Plot Diagnostic
![Scree Plot](scree_plot.png)

### Rotated Factor Loading Space (Varimax)
![Factor Loading Map](factor_loadings_map.png)

### Factor Loading Heatmap
![Factor Heatmap](factor_loadings_heatmap.png)

---

## 6. Future Extensions: Predictive Risk Analytics (GLM Integration)

The generated latent factor scores ($\hat{\mathbf{F}}$) resolve multicollinearity and serve as orthogonal predictors in downstream actuarial ratemaking models:

$$\mathbb{E}[Y] = g^{-1}\left( \beta_0 + \beta_1 \hat{F}_1 + \beta_2 \hat{F}_2 \right)$$

Where $Y$ represents total loss claim amounts modeled via Gamma or Tweedie distributions within Generalized Linear Model (GLM) frameworks.
