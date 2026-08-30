# Insurance Risk Factor Analysis: Demystifying Latent Risk

---

## Author & Academic Context

* **Author:** Sebastián H. Beltrán
* **Academic Background:** B.Sc. in Mathematics | Master's in Statistics — *Universidad Nacional de Colombia*
* **Domain:** Actuarial Science / Multivariate Statistical Analysis / Risk Analytics

---

## 1. Introduction: The "Explain It Like I'm 5" (ELI5) Intuition

Imagine you run a giant insurance company and you have to evaluate thousands of drivers. For every single driver, you track **hundreds of different clues**: 
* How old they are, how long they've been a customer, how much they pay for their premium, their deductible amount, how many cars were involved in an accident, and the exact money paid for vehicle damage, property damage, and medical injuries.

If an actuary tries to look at all 8 or 100 variables at the same time, it is like trying to listen to 100 people talking at once—**total chaos (multicollinearity and high dimensionality)**. 

### What does Exploratory Factor Analysis (EFA) do?
Think of EFA as a **super-powered pair of sunglasses** that lets you look past the noise and find the **hidden invisible strings** connecting everything. Instead of dealing with 8 messy variables, EFA groups them into just **2 main hidden drivers (latent factors)**:
1. **Factor 1 (The Accident Severity):** If someone crashes, how bad is the financial damage across vehicles, property, and injuries? 
2. **Factor 2 (The Driver's Maturity):** How experienced and mature is the policyholder based on their age and how long they've stayed with the company?

By doing this, we throw away the confusing clutter and keep only the pure, clean information. That is the true magic and necessity of **dimensionality reduction**.

---

## 2. Why Dimensionality Reduction Matters in Actuarial Science

In real-world insurance ratemaking, feeding too many correlated variables directly into risk models (like Generalized Linear Models or GLMs) creates severe statistical problems:
* **Multicollinearity:** Variables like vehicle claims and property claims move together, which tricks the model and makes regression coefficients bounce wildly or flip signs unpredictably.
* **Overfitting:** Models get distracted by random noise instead of learning true underlying risk patterns.
* **Curse of Dimensionality:** As dimensions grow, the data becomes sparse, making reliable risk pricing nearly impossible.

**The Solution:** Reducing $p$ correlated variables into $k$ orthogonal latent factors ($k \ll p$) gives us uncorrelated, standardized risk scores ($\hat{\mathbf{F}}$) that capture the true economic essence of the portfolio without statistical interference.

---

## 3. Theoretical Framework & Mathematical Formulation

### 3.1 The Classical Linear Factor Model
Let $\mathbf{X} = (X_1, X_2, \dots, X_p)^T \in \mathbb{R}^p$ be an observed vector of continuous random variables representing policy attributes. We model $\mathbf{X}$ as a linear combination of $k$ common latent factors $\mathbf{F} = (F_1, \dots, F_k)^T \in \mathbb{R}^k$ plus unique errors $\mathbf{U} = (U_1, \dots, U_p)^T \in \mathbb{R}^p$:

$$\mathbf{X} - \boldsymbol{\mu} = \mathbf{\Lambda} \mathbf{F} + \mathbf{U}$$

Where:
* $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}] \in \mathbb{R}^p$ is the expectation vector of observed variables.
* $\mathbf{\Lambda} \in \mathbb{R}^{p \times k}$ represents the matrix of **factor loadings**, where $\lambda_{ij}$ measures how strongly variable $i$ is tied to latent factor $j$.
* $\mathbf{F} \sim \mathcal{N}_k(\mathbf{0}, \mathbf{I}_k)$ is the vector of standardized common factors.
* $\mathbf{U} \sim \mathcal{N}_p(\mathbf{0}, \mathbf{\Psi})$ represents unique variance and measurement error, with $\mathbf{\Psi} = \text{diag}(\psi_1, \dots, \psi_p)$.

### 3.2 Variance Decomposition (Communality & Uniqueness)
Assuming common factors and unique errors are uncorrelated, the covariance matrix $\mathbf{\Sigma} = \text{Var}(\mathbf{X})$ decomposes into:

$$\mathbf{\Sigma} = \mathbf{\Lambda} \mathbf{\Lambda}^T + \mathbf{\Psi}$$

For any individual variable $X_i$, its total variance splits into **Communality** ($h_i^2$) and **Uniqueness** ($u_i^2$):

$$\text{Var}(X_i) = \sigma_{ii} = \underbrace{\sum_{j=1}^k \lambda_{ij}^2}_{\text{Communality } h_i^2} + \underbrace{\psi_i}_{\text{Uniqueness } u_i^2}$$

* **Communality ($h_i^2$):** The fraction of a variable's variance explained by the hidden factors.
* **Uniqueness ($u_i^2$):** The leftover unique noise or specific variance.

---

## 4. Step-by-Step Analytical Pipeline

### Step 1: Diagnostics & Multicollinearity Resolution
* **Script:** `R/01_data_cleaning_and_diagnostics.R`

#### The Problem
In raw insurance datasets, exact identities often exist, such as:
$$\text{total\_claim\_amount} = \text{injury\_claim} + \text{property\_claim} + \text{vehicle\_claim}$$
This creates a singular correlation matrix ($\det(\mathbf{R}) = 0$), breaking matrix inversion.

#### The Fix & Adequacy Test
Removing `total_claim_amount` restores positive definiteness. We then evaluate sample factorability using the **Kaiser-Meyer-Olkin (KMO)** test, achieving an **Overall MSA = 0.61**, confirming that factor analysis is viable for this portfolio.

---

### Step 2: Spectral Decomposition & Dimension Selection
* **Script:** `R/02_matrix_spectral_decomposition.R`

1. **Standardization:** Data is scaled to mean zero and unit variance ($\mathbf{Z}$).
2. **Eigenvalue Problem:** Solving $\det(\mathbf{R} - \lambda \mathbf{I}_p) = 0$ extracts orthogonal principal axes.
3. **Kaiser Criterion ($\lambda > 1$):** We retain $k = 2$ factors, capturing **$54.91\%$** of total portfolio variance ($\lambda_1 = 2.51$, $\lambda_2 = 1.89$).

---

### Step 3: Orthogonal Varimax Rotation & Thomson Scoring
* **Script:** `R/03_varimax_rotation_and_scoring.R`

To make the hidden factors easy to interpret, we apply an orthogonal **Varimax rotation** ($\mathbf{\Lambda}^* = \mathbf{\Lambda} \mathbf{T}$), maximizing the variance of squared loadings so that each variable clearly loads onto one primary factor. Finally, **Thomson’s regression method** estimates individual policyholder factor scores ($\hat{\mathbf{F}} = \mathbf{Z} \mathbf{R}^{-1} \mathbf{\Lambda}^*$).

---

## 5. Empirical Results & Actuarial Interpretation

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

## 6. Visualizations

### Scree Plot Diagnostic
![Scree Plot](scree_plot.png)

### Rotated Factor Loading Space (Varimax)
![Factor Loading Map](factor_loadings_map.png)

### Factor Loading Heatmap
![Factor Heatmap](factor_loadings_heatmap.png)

---

## 7. Recommended References for Academic Expansion

If you wish to dive deeper into the theoretical foundations of multivariate statistics, factor analysis, and actuarial modeling, consult the following literature:

1. **Johnson, R. A., & Wichern, D. W. (2007).** *Applied Multivariate Statistical Analysis* (6th ed.). Pearson Prentice Hall. 
   *(Essential reference for matrix spectral decomposition, population covariance structures, and principal component/factor estimation).*
2. **Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2014).** *Multivariate Data Analysis* (7th ed.). Pearson Education.
   *(Comprehensive guide on exploratory factor analysis assumptions, KMO adequacy diagnostics, and orthogonal Varimax rotations).*
3. **Mishra, S. (2015).** *Financial Risk Management and Actuarial Modeling*. Springer.
   *(Explores real-world applications of latent risk factor extraction in insurance claims pricing and risk evaluation).*
4. **Thomson, G. H. (1951).** *The Factorial Analysis of Human Ability*. University of Edinburgh Press.
   *(The foundational text detailing regression-based factor scoring methods).*
