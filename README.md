# Insurance Risk Factor Analysis: Complete Theoretical & Empirical Report
### Exploratory Factor Analysis (EFA), Spectral Decomposition & Actuarial Risk Modeling

---

## Author & Academic Context

* **Authors:** Sebastián H. Beltrán & Jose Ochoa
* **Academic Background:** Master's in Statistics — *Universidad Nacional de Colombia*
* **Domain:** Multivariate Statistical Analysis / Actuarial Science / Latent Variable Modeling

---

## 1. Introduction & Scientific Motivation

In diverse fields such as psychology, economics, and actuarial science, complex phenomena involve concepts that cannot be measured directly (e.g., general intelligence, economic wellbeing, or underlying insurance risk). The central statistical question underlying this project is:

$$\text{¿Es posible explicar la relación entre múltiples variables observadas mediante un número reducido de variables latentes?}$$

When dealing with high-dimensional policy portfolios, analysts track numerous attributes simultaneously. Feeding highly correlated variables directly into pricing models creates severe multicollinearity, unstable regression parameters, and matrix singularity. This project implements a rigorous Exploratory Factor Analysis (EFA) pipeline to resolve these challenges through the **Principle of Parsimony**: explaining the maximum observed variance using the minimum number of latent factors ($k \ll p$).

---

## 2. Theoretical Framework & Mathematical Formulation

### 2.1 The Classical Factor Model
Let $\mathbf{X} = (X_1, X_2, \dots, X_p)^T \in \mathbb{R}^p$ be an observable random vector with mean vector $\bm{\mu}$ and covariance matrix $\mathbf{\Sigma}$. The fundamental linear factor model decomposes $\mathbf{X}$ into common latent factors $\mathbf{f} \in \mathbb{R}^k$ and unique error terms $\mathbf{U} \in \mathbb{R}^p$:

$$\mathbf{X} = \bm{\mu} + \Lambda \mathbf{f} + \mathbf{U}$$

Where:
* $\bm{\mu} = \mathbb{E}[\mathbf{X}] \in \mathbb{R}^p$ is the expectation vector of observed variables.
* $\Lambda \in \mathbb{R}^{p \times k}$ represents the matrix of **factor loadings** ($\lambda_{ij}$).
* $\mathbf{f} \sim \mathcal{N}_k(\mathbf{0}, \Phi)$ represents the vector of common factors.
* $\mathbf{U} \sim \mathcal{N}_p(\mathbf{0}, \Psi)$ represents unique errors, where $\Psi = \text{diag}(\psi_1, \dots, \psi_p)$ is a diagonal matrix of specific variances.

### 2.2 Covariance Structure & Variance Decomposition
Assuming $\text{Cov}(\mathbf{f}, \mathbf{U}) = 0$, the population covariance matrix $\mathbf{\Sigma} = \text{Var}(\mathbf{X})$ decomposes into structural common variance and specific variance:

$$\mathbf{\Sigma} = \Lambda \Phi \Lambda^T + \Psi$$

When factors are orthogonal ($\Phi = \mathbf{I}_k$), this simplifies to:

$$\mathbf{\Sigma} = \Lambda \Lambda^T + \Psi$$

For any individual variable $X_i$, its total variance splits into **Communality** ($h_i^2$) and **Uniqueness** ($u_i^2 = \psi_i$):

$$\text{Var}(X_i) = \sigma_{ii} = \underbrace{\sum_{j=1}^k \lambda_{ij}^2}_{\text{Communality } h_i^2} + \underbrace{\psi_i}_{\text{Uniqueness } u_i^2}$$

---

## 3. Properties and Transformations of the Model

### 3.1 Standardization and Correlation Matrix
When variables are standardized ($Y_i = \frac{X_i - \mu_i}{\sigma_i}$), the covariance matrix becomes the correlation matrix $\text{Cov}(\mathbf{Y}) = \mathbf{R}$, and the model is expressed as:

$$\mathbf{R} = \Lambda \Lambda^T + \Psi$$

### 3.2 Non-Uniqueness of Weights & Orthogonal Rotation (Varimax)
Factor loadings are not unique. If $\mathbf{T}$ is an orthogonal matrix ($\mathbf{T}^T \mathbf{T} = \mathbf{T} \mathbf{T}^T = \mathbf{I}$), we can define transformed loadings and factors:

$$\Lambda^* = \Lambda \mathbf{T}, \quad \mathbf{f}^* = \mathbf{T}^T \mathbf{f}$$

Substituting this into the covariance structure yields:

$$\mathbf{\Sigma} = \Lambda^* \Lambda^{*T} + \Psi = (\Lambda \mathbf{T})(\Lambda \mathbf{T})^T + \Psi = \Lambda \mathbf{T} \mathbf{T}^T \Lambda^T + \Psi = \Lambda \Lambda^T + \Psi$$

Because the covariance matrix $\mathbf{\Sigma}$ and the communalities remain completely invariant ($h_i^{*2} = h_i^2$), there exist infinite equivalent mathematical solutions. To eliminate ambiguity and achieve a "simple structure," we apply **Varimax rotation**, maximizing the variance of squared loadings:

$$V = \sum_{j=1}^k \left[ \frac{1}{p} \sum_{i=1}^p \lambda_{ij}^{*4} - \left( \frac{1}{p} \sum_{i=1}^p \lambda_{ij}^{*2} \right)^2 \right]$$

---

## 4. Communality Estimation & Heywood Cases

### 4.1 Reduced Correlation Matrix
Subtracting unique variances from the correlation matrix yields the reduced correlation matrix $\mathbf{R}^*$:

$$\mathbf{R}^* = \mathbf{R} - \Psi = \Lambda \Lambda^T$$

where diagonal elements represent the communalities ($h_i^2$).

### 4.2 Estimation Methods
Communalities can be estimated via:
1. **Maximum Correlation:** $h_i^2 = \max_j |r_{ij}|$
2. **Average Correlation:** $h_i^2 = \frac{\sum_{j \neq i} r_{ij}}{p-1}$
3. **Squared Multiple Correlation ($R_i^2$):** Regressing $X_i$ against all other variables.

### 4.3 Heywood Cases
In certain estimations, sample constraints or multicollinearity can produce inadmissibly large communality estimates where $h_i^2 > 1$, known as **Heywood cases**. These are typically resolved by bounding communalities at unity ($h_i^2 \le 1$).

---

## 5. Estimation Methods & Data Adequacy

### 5.1 Estimation Approaches
* **Principal Component Method:** Uses spectral decomposition of the correlation matrix ($\mathbf{R} = \mathbf{P} \mathbf{D} \mathbf{P}^T$) to approximate $\hat{\Lambda} = \mathbf{P}_1 \mathbf{D}_1^{1/2}$.
* **Principal Factor Method:** Iteratively solves $\mathbf{R} - \Psi = \Lambda \Lambda^T$.
* **Maximum Likelihood (ML):** Assumes multivariate normality to evaluate probabilistic fit.

### 5.2 Kaiser-Meyer-Olkin (KMO) Adequacy Test
The KMO index measures the ratio of squared correlation coefficients to squared partial correlation coefficients, ensuring data factorability:

$$\text{KMO} = \frac{\sum_{i \neq j} r_{ij}^2}{\sum_{i \neq j} r_{ij}^2 + \sum_{i \neq j} a_{ij}^2}$$

Values above $0.60$ indicate acceptable sampling adequacy for factor analysis.

---

## 6. Empirical Results & Actuarial Interpretation

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

---

## 7. Visualizations

### Scree Plot Diagnostic
![Scree Plot](scree_plot.png)

### Rotated Factor Loading Space (Varimax)
![Factor Loading Map](factor_loadings_map.png)

### Factor Loading Heatmap
![Factor Heatmap](factor_loadings_heatmap.png)

---

## 8. Comprehensive Academic Conclusions

1. **Validation of the Principle of Parsimony ($k \ll p$):**
   * *Theoretical Link:* The primary objective of multivariate factor modeling is to explain complex data structures with fewer latent dimensions.
   * *Empirical Evidence:* Spectral decomposition and Kaiser's criterion ($\lambda > 1$) successfully condensed $p = 8$ observed insurance attributes into $k = 2$ orthogonal latent factors, capturing **$54.91\%$** of total portfolio variance ($\lambda_1 = 2.51$, $\lambda_2 = 1.89$).

2. **Resolution of Multicollinearity & Sampling Adequacy:**
   * *Theoretical Link:* Matrix inversion demands non-singularity ($\det(\mathbf{R}) > 0$).
   * *Empirical Evidence:* Removing the exact structural identity ($\text{total\_claim\_amount} = \text{injury} + \text{property} + \text{vehicle}$) restored positive-definiteness. The resulting **Overall KMO MSA of 0.61** empirically validates sample adequacy.

3. **Structural Clarity via Varimax Rotation:**
   * *Theoretical Link:* Rotation indeterminacy ($\Lambda^* = \Lambda \mathbf{T}$) guarantees that the underlying covariance structure $\mathbf{\Sigma} = \Lambda \Lambda^T + \Psi$ remains invariant while maximizing loading variance.
   * *Empirical Evidence:* The Varimax rotation successfully eliminated ambiguous cross-loadings, isolating **Factor 1 (Economic Loss Severity)**—dominated by high communalities in `vehicle_claim` ($h^2 = 0.843$), `property_claim` ($h^2 = 0.722$), and `injury_claim` ($h^2 = 0.714$)—and **Factor 2 (Policyholder Maturity)**, governed by customer longevity (`months_as_customer`, $h^2 = 0.961$) and age (`age`, $h^2 = 0.960$).

4. **Conceptual Distinction: EFA vs. Principal Component Analysis (ACP):**
   * *Theoretical Link:* While ACP performs a deterministic linear transformation ($\mathbf{Y} = \Gamma \mathbf{X}$) explaining total observed variance, Exploratory Factor Analysis models a probabilistic latent structure separating common variance from unique errors ($\mathbf{X} = \Lambda \mathbf{f} + \Psi$).
   * *Empirical Evidence:* Estimating specific variances ($\Psi$) confirms that latent risk constructs operate independently of policy noise terms (`policy_annual_premium`, $u^2 \approx 0.999$), yielding robust orthogonal factor scores ($\hat{\mathbf{F}}$) ideal for downstream actuarial Generalized Linear Models (GLMs).

---

## 9. Bibliography & Recommended References

1. **Johnson, R. A., & Wichern, D. W. (2007).** *Applied Multivariate Statistical Analysis* (6th ed.). Pearson Prentice Hall.
2. **Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2014).** *Multivariate Data Analysis* (7th ed.). Pearson Education.
3. **Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979).** *Multivariate Analysis*. Academic Press.
4. **Thomson, G. H. (1951).** *The Factorial Analysis of Human Ability*. University of Edinburgh Press.
EOF~
cd ~/Documents/insurance-risk-factor-analysis

cat << 'EOF' > README.md
# Insurance Risk Factor Analysis: Complete Theoretical & Empirical Report
### Exploratory Factor Analysis (EFA), Spectral Decomposition & Actuarial Risk Modeling

---

## Author & Academic Context

* **Authors:** Sebastián H. Beltrán & Jose Ochoa
* **Academic Background:** Master's in Statistics — *Universidad Nacional de Colombia*
* **Domain:** Multivariate Statistical Analysis / Actuarial Science / Latent Variable Modeling

---

## 1. Introduction & Scientific Motivation

In diverse fields such as psychology, economics, and actuarial science, complex phenomena involve concepts that cannot be measured directly (e.g., general intelligence, economic wellbeing, or underlying insurance risk). The central statistical question underlying this project is:

$$\text{¿Es posible explicar la relación entre múltiples variables observadas mediante un número reducido de variables latentes?}$$

When dealing with high-dimensional policy portfolios, analysts track numerous attributes simultaneously. Feeding highly correlated variables directly into pricing models creates severe multicollinearity, unstable regression parameters, and matrix singularity. This project implements a rigorous Exploratory Factor Analysis (EFA) pipeline to resolve these challenges through the **Principle of Parsimony**: explaining the maximum observed variance using the minimum number of latent factors ($k \ll p$).

---

## 2. Theoretical Framework & Mathematical Formulation

### 2.1 The Classical Factor Model
Let $\mathbf{X} = (X_1, X_2, \dots, X_p)^T \in \mathbb{R}^p$ be an observable random vector with mean vector $\bm{\mu}$ and covariance matrix $\mathbf{\Sigma}$. The fundamental linear factor model decomposes $\mathbf{X}$ into common latent factors $\mathbf{f} \in \mathbb{R}^k$ and unique error terms $\mathbf{U} \in \mathbb{R}^p$:

$$\mathbf{X} = \bm{\mu} + \Lambda \mathbf{f} + \mathbf{U}$$

Where:
* $\bm{\mu} = \mathbb{E}[\mathbf{X}] \in \mathbb{R}^p$ is the expectation vector of observed variables.
* $\Lambda \in \mathbb{R}^{p \times k}$ represents the matrix of **factor loadings** ($\lambda_{ij}$).
* $\mathbf{f} \sim \mathcal{N}_k(\mathbf{0}, \Phi)$ represents the vector of common factors.
* $\mathbf{U} \sim \mathcal{N}_p(\mathbf{0}, \Psi)$ represents unique errors, where $\Psi = \text{diag}(\psi_1, \dots, \psi_p)$ is a diagonal matrix of specific variances.

### 2.2 Covariance Structure & Variance Decomposition
Assuming $\text{Cov}(\mathbf{f}, \mathbf{U}) = 0$, the population covariance matrix $\mathbf{\Sigma} = \text{Var}(\mathbf{X})$ decomposes into structural common variance and specific variance:

$$\mathbf{\Sigma} = \Lambda \Phi \Lambda^T + \Psi$$

When factors are orthogonal ($\Phi = \mathbf{I}_k$), this simplifies to:

$$\mathbf{\Sigma} = \Lambda \Lambda^T + \Psi$$

For any individual variable $X_i$, its total variance splits into **Communality** ($h_i^2$) and **Uniqueness** ($u_i^2 = \psi_i$):

$$\text{Var}(X_i) = \sigma_{ii} = \underbrace{\sum_{j=1}^k \lambda_{ij}^2}_{\text{Communality } h_i^2} + \underbrace{\psi_i}_{\text{Uniqueness } u_i^2}$$

---

## 3. Properties and Transformations of the Model

### 3.1 Standardization and Correlation Matrix
When variables are standardized ($Y_i = \frac{X_i - \mu_i}{\sigma_i}$), the covariance matrix becomes the correlation matrix $\text{Cov}(\mathbf{Y}) = \mathbf{R}$, and the model is expressed as:

$$\mathbf{R} = \Lambda \Lambda^T + \Psi$$

### 3.2 Non-Uniqueness of Weights & Orthogonal Rotation (Varimax)
Factor loadings are not unique. If $\mathbf{T}$ is an orthogonal matrix ($\mathbf{T}^T \mathbf{T} = \mathbf{T} \mathbf{T}^T = \mathbf{I}$), we can define transformed loadings and factors:

$$\Lambda^* = \Lambda \mathbf{T}, \quad \mathbf{f}^* = \mathbf{T}^T \mathbf{f}$$

Substituting this into the covariance structure yields:

$$\mathbf{\Sigma} = \Lambda^* \Lambda^{*T} + \Psi = (\Lambda \mathbf{T})(\Lambda \mathbf{T})^T + \Psi = \Lambda \mathbf{T} \mathbf{T}^T \Lambda^T + \Psi = \Lambda \Lambda^T + \Psi$$

Because the covariance matrix $\mathbf{\Sigma}$ and the communalities remain completely invariant ($h_i^{*2} = h_i^2$), there exist infinite equivalent mathematical solutions. To eliminate ambiguity and achieve a "simple structure," we apply **Varimax rotation**, maximizing the variance of squared loadings:

$$V = \sum_{j=1}^k \left[ \frac{1}{p} \sum_{i=1}^p \lambda_{ij}^{*4} - \left( \frac{1}{p} \sum_{i=1}^p \lambda_{ij}^{*2} \right)^2 \right]$$

---

## 4. Communality Estimation & Heywood Cases

### 4.1 Reduced Correlation Matrix
Subtracting unique variances from the correlation matrix yields the reduced correlation matrix $\mathbf{R}^*$:

$$\mathbf{R}^* = \mathbf{R} - \Psi = \Lambda \Lambda^T$$

where diagonal elements represent the communalities ($h_i^2$).

### 4.2 Estimation Methods
Communalities can be estimated via:
1. **Maximum Correlation:** $h_i^2 = \max_j |r_{ij}|$
2. **Average Correlation:** $h_i^2 = \frac{\sum_{j \neq i} r_{ij}}{p-1}$
3. **Squared Multiple Correlation ($R_i^2$):** Regressing $X_i$ against all other variables.

### 4.3 Heywood Cases
In certain estimations, sample constraints or multicollinearity can produce inadmissibly large communality estimates where $h_i^2 > 1$, known as **Heywood cases**. These are typically resolved by bounding communalities at unity ($h_i^2 \le 1$).

---

## 5. Estimation Methods & Data Adequacy

### 5.1 Estimation Approaches
* **Principal Component Method:** Uses spectral decomposition of the correlation matrix ($\mathbf{R} = \mathbf{P} \mathbf{D} \mathbf{P}^T$) to approximate $\hat{\Lambda} = \mathbf{P}_1 \mathbf{D}_1^{1/2}$.
* **Principal Factor Method:** Iteratively solves $\mathbf{R} - \Psi = \Lambda \Lambda^T$.
* **Maximum Likelihood (ML):** Assumes multivariate normality to evaluate probabilistic fit.

### 5.2 Kaiser-Meyer-Olkin (KMO) Adequacy Test
The KMO index measures the ratio of squared correlation coefficients to squared partial correlation coefficients, ensuring data factorability:

$$\text{KMO} = \frac{\sum_{i \neq j} r_{ij}^2}{\sum_{i \neq j} r_{ij}^2 + \sum_{i \neq j} a_{ij}^2}$$

Values above $0.60$ indicate acceptable sampling adequacy for factor analysis.

---

## 6. Empirical Results & Actuarial Interpretation

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

---

## 7. Visualizations

### Scree Plot Diagnostic
![Scree Plot](scree_plot.png)

### Rotated Factor Loading Space (Varimax)
![Factor Loading Map](factor_loadings_map.png)

### Factor Loading Heatmap
![Factor Heatmap](factor_loadings_heatmap.png)

---

## 8. Comprehensive Academic Conclusions

1. **Validation of the Principle of Parsimony ($k \ll p$):**
   * *Theoretical Link:* The primary objective of multivariate factor modeling is to explain complex data structures with fewer latent dimensions.
   * *Empirical Evidence:* Spectral decomposition and Kaiser's criterion ($\lambda > 1$) successfully condensed $p = 8$ observed insurance attributes into $k = 2$ orthogonal latent factors, capturing **$54.91\%$** of total portfolio variance ($\lambda_1 = 2.51$, $\lambda_2 = 1.89$).

2. **Resolution of Multicollinearity & Sampling Adequacy:**
   * *Theoretical Link:* Matrix inversion demands non-singularity ($\det(\mathbf{R}) > 0$).
   * *Empirical Evidence:* Removing the exact structural identity ($\text{total\_claim\_amount} = \text{injury} + \text{property} + \text{vehicle}$) restored positive-definiteness. The resulting **Overall KMO MSA of 0.61** empirically validates sample adequacy.

3. **Structural Clarity via Varimax Rotation:**
   * *Theoretical Link:* Rotation indeterminacy ($\Lambda^* = \Lambda \mathbf{T}$) guarantees that the underlying covariance structure $\mathbf{\Sigma} = \Lambda \Lambda^T + \Psi$ remains invariant while maximizing loading variance.
   * *Empirical Evidence:* The Varimax rotation successfully eliminated ambiguous cross-loadings, isolating **Factor 1 (Economic Loss Severity)**—dominated by high communalities in `vehicle_claim` ($h^2 = 0.843$), `property_claim` ($h^2 = 0.722$), and `injury_claim` ($h^2 = 0.714$)—and **Factor 2 (Policyholder Maturity)**, governed by customer longevity (`months_as_customer`, $h^2 = 0.961$) and age (`age`, $h^2 = 0.960$).

4. **Conceptual Distinction: EFA vs. Principal Component Analysis (ACP):**
   * *Theoretical Link:* While ACP performs a deterministic linear transformation ($\mathbf{Y} = \Gamma \mathbf{X}$) explaining total observed variance, Exploratory Factor Analysis models a probabilistic latent structure separating common variance from unique errors ($\mathbf{X} = \Lambda \mathbf{f} + \Psi$).
   * *Empirical Evidence:* Estimating specific variances ($\Psi$) confirms that latent risk constructs operate independently of policy noise terms (`policy_annual_premium`, $u^2 \approx 0.999$), yielding robust orthogonal factor scores ($\hat{\mathbf{F}}$) ideal for downstream actuarial Generalized Linear Models (GLMs).

---

## 9. Bibliography & Recommended References

1. **Johnson, R. A., & Wichern, D. W. (2007).** *Applied Multivariate Statistical Analysis* (6th ed.). Pearson Prentice Hall.
2. **Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2014).** *Multivariate Data Analysis* (7th ed.). Pearson Education.
3. **Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979).** *Multivariate Analysis*. Academic Press.
4. **Thomson, G. H. (1951).** *The Factorial Analysis of Human Ability*. University of Edinburgh Press.
