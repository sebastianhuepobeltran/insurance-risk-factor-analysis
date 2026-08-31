# Insurance Risk Factor Analysis

### Exploratory Factor Analysis (EFA), Spectral Decomposition & Latent Risk Structure

---

## Author & Academic Context

**Author:** Sebastián H. Beltrán  
**Academic Background:** Master's in Statistics — Universidad Nacional de Colombia  
**Domain:** Multivariate Statistical Analysis / Insurance Analytics / Latent Variable Modeling

---

## 1. Project Overview

Insurance datasets often contain multiple variables describing policyholders, policies, and claim outcomes. Several of these variables may exhibit substantial dependence, making it difficult to distinguish the underlying dimensions represented in the data.

This project investigates whether the covariance structure of selected insurance variables can be represented through a smaller number of interpretable latent dimensions using **Exploratory Factor Analysis (EFA)**.

The analysis follows the pipeline:

**Data Cleaning → Exploratory Analysis → Factorability Diagnostics → Factor Selection → Factor Extraction → Varimax Rotation → Factor Interpretation → Factor Scores**

The final empirical analysis reduces an 8-dimensional observable space to a two-factor solution, with the retained factors explaining approximately **54.91% of the total standardized variance**.

The analysis also demonstrates an important data-quality issue: `total_claim_amount` is an exact linear combination of three claim components and therefore introduces deterministic redundancy into the correlation structure.

---

## 2. Scientific Motivation

In insurance analytics, observed variables may represent different aspects of a policyholder or claim:

- claim amounts,
- number of vehicles involved,
- policyholder age,
- customer tenure,
- annual premium,
- deductible.

Some of these variables may share common variation, while others may represent largely variable-specific information.

The central question of this project is:

> **Can the joint covariance structure of observed insurance variables be represented by a smaller number of interpretable latent dimensions?**

Mathematically, the classical factor model represents the observed vector as:

$$\mathbf{X} = \boldsymbol{\mu} + \Lambda \mathbf{f} + \mathbf{U}$$

where:

- $\mathbf{X}$ is the vector of observed variables;
- $\boldsymbol{\mu}$ is the vector of means;
- $\Lambda$ is the factor-loading matrix;
- $\mathbf{f}$ is the vector of common latent factors;
- $\mathbf{U}$ contains variable-specific components.

The corresponding covariance decomposition is:

$$\mathbf{\Sigma} = \Lambda\Lambda^T + \Psi$$

where $\Psi$ represents unique variances.

The objective is therefore not simply dimensionality reduction, but the identification of **common covariance structure** that can be interpreted as latent dimensions.

---

## 3. Dataset & Variables

The analysis uses an automobile insurance dataset containing policyholder, policy, and claim-related variables.

The initial dataset contains an aggregate claim variable together with its component claims.

The following variables are retained after removing the deterministic aggregate:

| Variable | Description |
|---|---|
| `vehicle_claim` | Vehicle-related claim amount |
| `property_claim` | Property-related claim amount |
| `injury_claim` | Injury-related claim amount |
| `number_of_vehicles_involved` | Number of vehicles involved in the incident |
| `months_as_customer` | Customer tenure |
| `age` | Policyholder age |
| `policy_annual_premium` | Annual policy premium |
| `policy_deductable` | Policy deductible |

---

## 4. Data Cleaning & Structural Redundancy

### 4.1 Deterministic Relationship

The original dataset contains `total_claim_amount`, `injury_claim`, `property_claim`, and `vehicle_claim`.

These variables satisfy the exact accounting identity:

$$\text{total\_claim\_amount} = \text{injury\_claim} + \text{property\_claim} + \text{vehicle\_claim}$$

or equivalently:

$$\text{total\_claim\_amount} - \text{injury\_claim} - \text{property\_claim} - \text{vehicle\_claim} = 0$$

Including all four variables introduces deterministic linear dependence. As a consequence, the corresponding correlation structure becomes singular, producing a zero eigenvalue and preventing matrix inversion.

Therefore, `total_claim_amount` is excluded from the factor-analysis input because its information is already contained exactly in the three component claim variables. This is treated as a **structural data redundancy issue**, rather than as an arbitrary feature-selection decision.

---

## 5. Exploratory Factor Analysis Framework

The standardized observed variables are represented by $\mathbf{Z} = \mathbf{D}_{\sigma}^{-1/2} (\mathbf{X} - \boldsymbol{\mu})$ and the analysis is performed using the correlation structure $\mathbf{R} = \Lambda\Lambda^T + \Psi$ under the orthogonal factor-model assumptions.

The main assumptions are:

1. **Common factors are standardized:** $\mathbb{E}(\mathbf{f}) = \mathbf{0}, \text{Var}(\mathbf{f}) = \mathbf{I}$
2. **Unique components are mutually uncorrelated:** $\text{Cov}(U_i, U_j) = 0, \forall i \neq j$
3. **Common factors and unique components are uncorrelated:** $\text{Cov}(\mathbf{f}, \mathbf{U}) = \mathbf{0}$

These assumptions provide the theoretical basis for the covariance decomposition used throughout the analysis.

---

## 6. Factorability Diagnostics

Before extracting factors, the correlation structure is evaluated to determine whether the data contain sufficient common variation.

### 6.1 Bartlett's Test of Sphericity
Bartlett's test evaluates $H_0: \mathbf{R} = \mathbf{I}$ against $H_1: \mathbf{R} \neq \mathbf{I}$. The test is statistically significant, providing evidence that the correlation matrix contains non-zero correlations and that factor analysis is potentially appropriate.

### 6.2 Kaiser-Meyer-Olkin Measure
The overall KMO statistic is:

$$\text{KMO} = \frac{\sum_{i \neq j} r_{ij}^2}{\sum_{i \neq j} r_{ij}^2 + \sum_{i \neq j} a_{ij}^2}$$

where $r_{ij}$ is the observed correlation and $a_{ij}$ is the corresponding partial correlation.

The analysis produces **$\text{KMO} = 0.61$**. This represents **moderate but acceptable sampling adequacy** for exploratory factor analysis. The result should therefore be interpreted as evidence supporting exploratory analysis, rather than as evidence of an exceptionally strong factor structure.

---

## 7. Factor Selection

The number of factors is evaluated using the eigenvalue structure of the correlation matrix. For the retained solution:

$$\lambda_1 = 2.51, \quad \lambda_2 = 1.89$$

Both exceed the Kaiser criterion ($\lambda_j > 1$). The corresponding explained variance is approximately:

- **Factor 1:** 31.42%
- **Factor 2:** 23.49%
- **Cumulative:** 54.91%

Thus, $k/p = 2/8 = 25\%$, meaning that two retained latent dimensions represent 25% of the original number of observed dimensions while accounting for approximately 54.91% of the standardized variance.

### Scree Plot
The scree plot provides additional visual evidence for a low-dimensional structure, with the dominant eigenvalues concentrated in the first two dimensions.

*Note: Parallel analysis should be included as an additional factor-retention diagnostic in a future revision of the pipeline before treating the two-factor solution as fully validated.*

---

## 8. Factor Extraction

Factor extraction is performed using Principal Factor Analysis on the reduced correlation structure. The reduced matrix is represented conceptually as $\mathbf{R}^* = \mathbf{R} - \Psi$ with diagonal elements replaced by initial estimates of common variance.

The spectral decomposition is based on $\det(\mathbf{R}^* - \lambda \mathbf{I}) = 0$, producing eigenvalues and corresponding eigenvectors that provide the basis for constructing the initial factor-loading matrix.

---

## 9. Varimax Rotation

The initial factor solution is subject to rotational indeterminacy. If $\Lambda$ is a valid loading matrix, an orthogonal transformation produces $\Lambda^* = \Lambda \mathbf{T}$ with $\mathbf{T}^T \mathbf{T} = \mathbf{I}$ and therefore $\Lambda^* \Lambda^{*T} = \Lambda \Lambda^T$.

The common covariance structure is preserved. To improve interpretability, the analysis applies **orthogonal Varimax rotation** to obtain a simpler loading pattern in which variables tend to have stronger associations with particular factors and smaller cross-loadings on others.

---

## 10. Empirical Results

The rotated solution produces two interpretable dimensions:

| Observed Variable | Factor 1 | Factor 2 | Communality ($h^2$) | Uniqueness ($u^2$) | Primary Alignment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `vehicle_claim` | **-0.918** | -0.029 | 0.843 | 0.157 | Factor 1 |
| `property_claim` | **-0.849** | -0.038 | 0.722 | 0.278 | Factor 1 |
| `injury_claim` | **-0.845** | -0.009 | 0.714 | 0.286 | Factor 1 |
| `number_of_vehicles_involved` | **-0.428** | -0.033 | 0.184 | 0.816 | Factor 1 |
| `months_as_customer` | -0.091 | **0.976** | 0.961 | 0.039 | Factor 2 |
| `age` | -0.106 | **0.974** | 0.960 | 0.040 | Factor 2 |
| `policy_annual_premium` | 0.018 | 0.025 | 0.001 | 0.999 | Weak common structure |
| `policy_deductable` | -0.078 | 0.045 | 0.008 | 0.992 | Weak common structure |

---

## 11. Interpretation of the Latent Dimensions

### Factor 1 — Claim Loss Magnitude
Factor 1 is strongly associated with `vehicle_claim`, `property_claim`, and `injury_claim`. The three claim components have large loadings in the same direction, suggesting that Factor 1 represents a common dimension associated with **claim loss magnitude**. The loading for `number_of_vehicles_involved` is weaker ($|\lambda| = 0.428$), so this variable should be interpreted as a secondary contributor rather than a defining indicator.

### Factor 2 — Policyholder Profile
Factor 2 is dominated by `months_as_customer` ($0.976$) and `age` ($0.974$). The factor captures a common dimension associated with **policyholder age and customer tenure**. The conservative interpretation is a policyholder profile / tenure dimension.

### Contractual Variables
`policy_annual_premium` and `policy_deductable` have extremely low communalities ($h^2 \approx 0$) and high uniqueness ($u^2 \approx 1$). This indicates that their variance is **not captured by the two retained common factors**. Importantly, this does *not* demonstrate that premiums or deductibles are statistically independent of claims or policyholder characteristics in general; it only indicates that their variation is largely unexplained by the particular latent dimensions identified in this EFA.

---

## 12. Factor Loading Geometry

The rotated loading space provides a geometric representation of the two-factor solution:

- **Factor 1:** Claim-related variables cluster strongly along the first factor axis.
- **Factor 2:** Age and customer tenure cluster strongly along the second factor axis.
- **Near-Origin Variables:** Premium and deductible are located close to the origin because their loadings on both retained factors are negligible.

---

## 13. Factor Loading Heatmap

The heatmap provides a compact representation of the rotated loading matrix. The strongest contrasts confirm that claim variables align with Factor 1, age and tenure align with Factor 2, and contractual variables show weak association with either retained factor.

---

## 14. Communalities and Uniqueness

For each observed variable $\sigma_{ii} = \sum_{j=1}^k \lambda_{ij}^2 + \psi_i$, common variance is $h_i^2 = \sum_{j=1}^k \lambda_{ij}^2$ while unique variance is $u_i^2 = \psi_i$. High communalities indicate that the retained factors explain a substantial portion of a variable's variance (e.g., `vehicle_claim` $h^2 = 0.843$, `age` $h^2 = 0.960$). In contrast, contractual metrics exhibit negligible communalities (`policy_annual_premium` $h^2 = 0.001$).

---

## 15. Factor Scores

Individual observations can be projected into the latent factor space using Thomson's regression scoring approach:

$$\hat{\mathbf{F}} = \mathbf{Z} \mathbf{R}^{-1} \hat{\Lambda}^*$$

where $\mathbf{Z}$ is the standardized observation matrix, $\mathbf{R}^{-1}$ is the inverse correlation matrix, and $\hat{\Lambda}^*$ is the rotated loading matrix. These scores provide observation-level representations of the latent dimensions for downstream modeling.

---

## 16. Potential Downstream Actuarial Application

A natural extension of this analysis is to evaluate whether the latent factor scores provide useful predictors in actuarial models. For example, a generalized linear model could be specified as:

$$\log(\mathbb{E}[Y_i]) = \beta_0 + \beta_1 \hat{F}_{i1} + \beta_2 \hat{F}_{i2} + \sum_{m} \gamma_m W_{im}$$

### Important Distinction
The current project estimates and interprets the factor structure and factor scores. The GLM stage is presented as a **downstream modeling extension**, not as an empirical GLM result of the current analysis.

---

## 17. Limitations

1. **Moderate KMO:** Overall KMO of 0.61 indicates moderate sampling adequacy.
2. **Weakly Represented Variables:** Premium and deductible exhibit almost no common variance with the retained factors.
3. **Factor Retention:** Parallel analysis should be added as an additional factor-retention criterion.
4. **Exploratory Interpretation:** Factor labels are qualitative interpretations of loading patterns.
5. **Downstream Prediction:** Predictive performance of factor scores in GLMs remains to be evaluated empirically in future work.

---

## 18. Reproducible R Pipeline

The project is organized into modular R scripts inside `R/`:

- `R/01_data_cleaning_and_diagnostics.R`: Data cleaning, structural redundancy diagnosis, and KMO diagnostics.
- `R/02_matrix_spectral_decomposition.R`: Reduced correlation matrix construction, spectral decomposition, scree plot generation.
- `R/03_varimax_rotation_and_scoring.R`: Varimax rotation, loading matrix visualization, and Thomson factor scoring.

---

## 19. Key Findings & Conclusion

1. **Structural Redundancy Identified:** `total_claim_amount` was excluded due to deterministic identity.
2. **Two-Factor Structure:** Retained factors explain 54.91% of standardized variance.
3. **Interpretable Dimensions:** Factor 1 represents Claim Loss Magnitude and Factor 2 represents Policyholder Profile.
4. **Variable Specificity:** Contractual variables show unshared specific variance rather than common factor structure.

---

## 20. References

1. Johnson, R. A., & Wichern, D. W. (2007). *Applied Multivariate Statistical Analysis*. Pearson Prentice Hall.
2. Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2014). *Multivariate Data Analysis*. Pearson Education.
3. Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979). *Multivariate Analysis*. Academic Press.
4. Thomson, G. H. (1951). *The Factorial Analysis of Human Ability*. University of Edinburgh Press.
5. Ohlsson, E., & Johansson, B. (2010). *Non-Life Insurance Pricing with Generalized Linear Models*. Springer Finance.
