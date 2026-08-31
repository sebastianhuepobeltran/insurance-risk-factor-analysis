# Insurance Risk Factor Analysis: Complete Theoretical & Empirical Report
### Exploratory Factor Analysis (EFA), Spectral Decomposition & Actuarial Risk Modeling

---

## Author & Academic Context

* **Author:** Sebastián H. Beltrán
* **Academic Background:** Master's in Statistics — *Universidad Nacional de Colombia*
* **Domain:** Multivariate Statistical Analysis / Actuarial Science / Latent Variable Modeling

---

## 1. Introduction & Scientific Motivation

In complex analytical fields such as actuarial science, financial econometrics, and psychometrics, researchers frequently encounter critical phenomena driven by underlying mechanisms that cannot be observed directly—such as driver risk profile, economic wellbeing, or underlying claim severity. The foundational statistical question guiding this investigation is:

$$\text{Is it possible to explain the joint relationship among multiple observed variables through a smaller set of unobserved latent variables?}$$

When analyzing high-dimensional insurance portfolios, actuaries routinely record dozens of policyholder and claim attributes (e.g., policyholder age, customer tenure, annual premiums, deductibles, and individual payout amounts for vehicle damage, property destruction, and medical injuries). Feeding highly correlated metrics directly into predictive risk models (such as Generalized Linear Models for loss pricing) triggers severe statistical complications: **extreme multicollinearity, unstable parameter estimation, variance inflation, and matrix singularity ($\det(\mathbf{R}) \approx 0$)**.

This project implements a comprehensive **Exploratory Factor Analysis (EFA)** pipeline grounded in the **Principle of Parsimony**. The fundamental objective is to compress an 8-dimensional space ($p = 8$) into a low-dimensional subspace of orthogonal latent factors ($k = 2$, where $k \ll p$), isolating genuine structural covariance from unshared measurement noise.

---

## 2. Theoretical Framework & Mathematical Formulation

### 2.1 The Classical Linear Factor Model
Let $\mathbf{X} = (X_1, X_2, \dots, X_p)^T \in \mathbb{R}^p$ be an observable random vector with expectation vector $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$ and population covariance matrix $\mathbf{\Sigma}$. The classical linear factor model expresses each observed variable as a linear combination of $k$ unobserved common latent factors $\mathbf{f} = (f_1, f_2, \dots, f_k)^T$ plus a unique specific error term $\mathbf{U} = (U_1, U_2, \dots, U_p)^T$:

$$\mathbf{X} = \boldsymbol{\mu} + \Lambda \mathbf{f} + \mathbf{U}$$

Where:
* $\boldsymbol{\mu} \in \mathbb{R}^p$ is the location parameter vector.
* $\Lambda \in \mathbb{R}^{p \times k}$ represents the **matrix of factor loadings** ($\lambda_{ij}$), where each entry measures the structural linear contribution of common factor $j$ to observed variable $i$.
* $\mathbf{f} \sim \mathcal{N}_k(\mathbf{0}, \Phi)$ represents the standardized common latent factors.
* $\mathbf{U} \sim \mathcal{N}_p(\mathbf{0}, \Psi)$ represents unique errors or specific factors, where $\Psi = \text{diag}(\psi_1, \psi_2, \dots, \psi_p)$ is a diagonal matrix containing unique variances.

### 2.2 Covariance Decomposition & The Concept of Communality
Assuming that common factors and unique errors are mutually uncorrelated ($\text{Cov}(\mathbf{f}, \mathbf{U}) = \mathbf{0}$), the population covariance matrix $\mathbf{\Sigma} = \text{Var}(\mathbf{X})$ decomposes into structural common variance and specific variance:

$$\mathbf{\Sigma} = \Lambda \Phi \Lambda^T + \Psi$$

When latent factors are assumed to be orthogonal ($\Phi = \mathbf{I}_k$), this structure simplifies to:

$$\mathbf{\Sigma} = \Lambda \Lambda^T + \Psi$$

For any observed variable $X_i$, its total variance $\sigma_{ii}$ splits into two mutually exclusive components: **Communality** ($h_i^2$) and **Uniqueness** ($u_i^2 = \psi_i$):

$$\text{Var}(X_i) = \sigma_{ii} = \underbrace{\sum_{j=1}^k \lambda_{ij}^2}_{\text{Communality } h_i^2} + \underbrace{\psi_i}_{\text{Uniqueness } u_i^2}$$

* **What is Communality ($h_i^2$)?** Communality measures the total proportion of a variable's variance that is shared with and explained by the $k$ common latent factors. A high communality ($h_i^2 \to 1.0$) confirms that the variable is a reliable indicator of the underlying latent dimensions.
* **What is Uniqueness / Specific Variance ($u_i^2 = \psi_i$)?** Uniqueness represents the residual variance attributable strictly to that specific variable plus random measurement error. Variables with high uniqueness ($u_i^2 \to 1.0$) do not share covariance with the rest of the system and behave as unshared noise.

---

## 3. Rotations, Geometry, and Indeterminacy

### 3.1 Standardization and Correlation Structure
In actuarial applications, variables operate on different scales (currency amounts, years, vehicle counts). Standardizing variables ($Y_i = \frac{X_i - \mu_i}{\sigma_i}$) transforms the covariance matrix into the sample correlation matrix $\text{Cov}(\mathbf{Y}) = \mathbf{R}$:

$$\mathbf{R} = \Lambda \Lambda^T + \Psi$$

### 3.2 Rotation Indeterminacy & The Varimax Criterion
A core mathematical property of factor models is **rotation indeterminacy**. If $\mathbf{T} \in \mathbb{R}^{k \times k}$ is an orthogonal rotation matrix ($\mathbf{T}^T \mathbf{T} = \mathbf{T} \mathbf{T}^T = \mathbf{I}$), transformed loadings and factors can be defined as:

$$\Lambda^* = \Lambda \mathbf{T}, \quad \mathbf{f}^* = \mathbf{T}^T \mathbf{f} \implies \Lambda^* \Lambda^{*T} = (\Lambda \mathbf{T})(\Lambda \mathbf{T})^T = \Lambda \mathbf{T} \mathbf{T}^T \Lambda^T = \Lambda \Lambda^T$$

Because the population covariance matrix $\mathbf{\Sigma}$ and individual communalities ($h_i^{*2} = h_i^2$) remain completely invariant under orthogonal rotation, infinite mathematically equivalent factor configurations exist.

To resolve cross-loading ambiguity, we apply **Orthogonal Varimax Rotation**, which maximizes the variance of squared loadings within each factor column:

$$V = \sum_{j=1}^k \left[ \frac{1}{p} \sum_{i=1}^p \lambda_{ij}^{*4} - \left( \frac{1}{p} \sum_{i=1}^p \lambda_{ij}^{*2} \right)^2 \right]$$

This optimization drives loadings toward extreme values ($\pm 1$ or $0$), fulfilling Thurstone's **Simple Structure** principle: each variable aligns distinctly with a single factor axis while minimizing cross-loadings.

---

## 4. Communality Estimation, Reduced Matrices & Heywood Cases

### 4.1 The Reduced Correlation Matrix
Subtracting specific variances $\Psi$ from the correlation matrix yields the **reduced correlation matrix** $\mathbf{R}^*$:

$$\mathbf{R}^* = \mathbf{R} - \Psi = \Lambda \Lambda^T$$

In $\mathbf{R}^*$, the main diagonal elements are replaced by the communalities ($h_i^2$) instead of ones.

### 4.2 Estimation Methods & Heywood Cases
Because $\Psi$ is unknown prior to model fitting, communalities are estimated iteratively using methods such as:
1. **Maximum Absolute Correlation:** $h_i^2 = \max_{j \neq i} |r_{ij}|$
2. **Squared Multiple Correlation ($R_i^2$):** Regressing $X_i$ on all remaining $p-1$ variables.

Under severe sample limitations or strong collinearity, iterative algorithms can produce inadmissible numerical solutions where specific variance becomes negative ($\psi_i < 0$), causing communalities to exceed unity ($h_i^2 > 1.0$). Known as **Heywood Cases**, these anomalies are resolved by bounding communalities at $h_i^2 \le 1.0$.

---

## 5. Step-by-Step Methodological Pipeline

The analytical pipeline is organized into three sequential R modules inside `R/`:

1. **`R/01_data_cleaning_and_diagnostics.R` (Multicollinearity & KMO Test):**
   * Identifies and removes exact linear identities (e.g., `total_claim_amount` = `injury_claim` + `property_claim` + `vehicle_claim`) that break matrix inversion ($\det(\mathbf{R}) = 0$).
   * Calculates the **Kaiser-Meyer-Olkin (KMO)** index to verify sample factorability.
2. **`R/02_matrix_spectral_decomposition.R` (Factor Extraction):**
   * Solves the characteristic equation $\det(\mathbf{R} - \lambda \mathbf{I}_p) = 0$ to extract eigenvalues and eigenvectors.
   * Retains $k = 2$ factors based on **Kaiser's Criterion** ($\lambda > 1.0$).
3. **`R/03_varimax_rotation_and_scoring.R` (Rotation & Scoring):**
   * Applies Varimax rotation to align factor axes.
   * Computes **Thomson's Regression Factor Scores** ($\hat{\mathbf{F}} = \mathbf{Z} \mathbf{R}^{-1} \hat{\Lambda}^*$) for downstream actuarial GLM modeling.

---

## 6. Empirical Results & Actuarial Interpretation

After removing exact linear identities, the dataset achieved an **Overall KMO of 0.61**, confirming sampling adequacy. Spectral extraction retained $k = 2$ orthogonal factors explaining **$54.91\%$** of total portfolio variance ($\lambda_1 = 2.51$, $\lambda_2 = 1.89$).

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

### Detailed Interpretation of Latent Factors:
* **Factor 1: Economic Loss Severity (Claims Magnitude):** Strongly loaded by `vehicle_claim` ($-0.918$), `property_claim` ($-0.849$), and `injury_claim` ($-0.845$). This factor measures the overall financial damage of an accident across all claim categories.
* **Factor 2: Policyholder Maturity & Loyalty:** Driven almost entirely by `months_as_customer` ($+0.976$) and driver `age` ($+0.974$). This factor captures driver experience, stability, and customer tenure.

---

## 7. Geometric Analysis of Factor Visualizations

### 7.1 Scree Plot Diagnostic
![Scree Plot](scree_plot.png)
* **Interpretation:** The Scree Plot orders eigenvalues ($\lambda_j$) from largest to smallest. Components above Kaiser's threshold ($\lambda = 1.0$) contain substantial common variance and are retained ($\text{PC}_1$ and $\text{PC}_2$), while subsequent components represent residual noise.

---

### 7.2 Rotated Factor Loading Space (Geometrical Breakdown)
![Factor Loading Map](factor_loadings_map.png)

In classical multivariate textbooks (e.g., Johnson & Wichern), the Factor Loading Map projects observed variables as points within a $k$-dimensional coordinate system. Understanding its geometric properties is critical:

* **What do the Coordinate Axes represent?**
  * **Horizontal Axis ($X$-axis — Factor 1: Economic Loss Severity):** Quantifies financial payout magnitude. Variables positioned far to the left (`vehicle_claim`, `property_claim`, `injury_claim`) exhibit strong negative loadings ($\approx -0.85$ to $-0.92$), meaning the $X$-axis measures financial claim magnitude.
  * **Vertical Axis ($Y$-axis — Factor 2: Policyholder Maturity):** Measures customer longevity and experience. Variables positioned high on the vertical axis (`months_as_customer`, `age`) exhibit positive loadings approaching $+0.98$.
* **What does the Proximity Between Red Points mean?**
  * Points located close together (e.g., `vehicle_claim`, `property_claim`, and `injury_claim` in the lower-left quadrant) share high pairwise correlations and belong to the **same latent risk dimension**. Conversely, points far apart (like `age` vs. `vehicle_claim`) represent orthogonal, independent risk concepts.
* **What does Proximity to the Coordinate Axes mean?**
  * When a red point lies **directly on or immediately adjacent to an axis** (e.g., `age` sitting directly on the vertical axis at $X \approx -0.10, Y \approx 0.97$), it indicates **Simple Structure achieved via Varimax rotation**. This implies the variable is *almost purely explained* by that specific factor and exhibits virtually zero cross-loading on the other factor.
* **Uninformative Variables at the Origin $(0,0)$:**
  * Variables near the origin (`policy_annual_premium`, `policy_deductable`) have communalities near zero ($h^2 \approx 0.001$). They share no variance with claims severity or driver maturity, behaving strictly as unshared noise ($u^2 \approx 0.999$).

---

### 7.3 Factor Loading Heatmap
![Factor Heatmap](factor_loadings_heatmap.png)
* **Interpretation:** The heatmap presents a clean matrix view of rotated loadings $\hat{\Lambda}^*$. The sharp contrast confirms that the 8 variables split cleanly into two non-overlapping latent modules.

---

## 8. Comprehensive Academic Conclusions

1. **Validation of the Principle of Parsimony ($k \ll p$):**
   * *Theoretical Link:* The core objective of factor analysis is optimal data compression without losing underlying covariance structure.
   * *Empirical Evidence:* Spectral extraction compressed $p = 8$ insurance variables into $k = 2$ orthogonal latent factors, capturing **$54.91\%$** of total portfolio variance ($\lambda_1 = 2.51$, $\lambda_2 = 1.89$).

2. **Mitigation of Multicollinearity and Sampling Adequacy:**
   * *Theoretical Link:* Matrix inversion in multivariate models requires positive-definiteness ($\det(\mathbf{R}) > 0$).
   * *Empirical Evidence:* Removing exact structural identities restored matrix non-singularity. The **Overall KMO of 0.61** quantitatively confirmed sufficient common covariance to proceed with factor extraction.

3. **Simple Structure & Geometric Isolation via Varimax Rotation:**
   * *Theoretical Link:* Rotation indeterminacy ($\Lambda^* = \Lambda \mathbf{T}$) guarantees that the population covariance structure $\mathbf{\Sigma} = \Lambda \Lambda^T + \Psi$ remains invariant while maximizing loading interpretability.
   * *Empirical Evidence:* Varimax rotation effectively aligned variables with coordinate axes. Claims indicators (`vehicle_claim`, `property_claim`, `injury_claim`) fell along **Factor 1 (Loss Severity)**, while demographic attributes (`age`, `months_as_customer`) aligned along **Factor 2 (Policyholder Maturity)**.

4. **Methodological Superiority: EFA vs. Principal Component Analysis (PCA):**
   * *Theoretical Link:* While PCA is a deterministic transformation explaining total observed variance ($\mathbf{Y} = \Gamma \mathbf{X}$), Factor Analysis models a generative latent structure separating common variance from unique measurement errors ($\mathbf{X} = \Lambda \mathbf{f} + \Psi$).
   * *Empirical Evidence:* Estimating Thomson's factor scores ($\hat{\mathbf{F}}$) provides uncorrelated risk metrics free of collinear redundancy, serving as an ideal quantitative foundation for downstream actuarial pricing in Generalized Linear Models (GLMs).

---

## 9. Bibliography & Recommended References

1. **Johnson, R. A., & Wichern, D. W. (2007).** *Applied Multivariate Statistical Analysis* (6th ed.). Pearson Prentice Hall.
2. **Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2014).** *Multivariate Data Analysis* (7th ed.). Pearson Education.
3. **Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979).** *Multivariate Analysis*. Academic Press.
4. **Thomson, G. H. (1951).** *The Factorial Analysis of Human Ability*. University of Edinburgh Press.
