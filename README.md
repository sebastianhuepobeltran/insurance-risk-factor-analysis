# Insurance Risk Factor Analysis: Complete Theoretical & Empirical Report
### Exploratory Factor Analysis (EFA), Spectral Decomposition & Actuarial Risk Modeling

---

## Author & Academic Context

* **Author:** Sebastián H. Beltrán
* **Academic Background:** Master's in Statistics — *Universidad Nacional de Colombia*
* **Domain:** Multivariate Statistical Analysis / Actuarial Science / Latent Variable Modeling

---

## 1. Introduction & Scientific Motivation

In diverse fields such as psychology, economics, and actuarial science, complex phenomena involve underlying concepts that cannot be measured directly (e.g., general intelligence, economic wellbeing, or underlying insurance risk). The central statistical question guiding this investigation is:

$$\text{¿Es posible explicar la relación conjunta entre múltiples variables observadas mediante un número reducido de variables latentes?}$$

Cuando se trabaja con carteras de seguros de alta dimensión, los analistas rastrean simultáneamente decenas o cientos de atributos de las pólizas y de los siniestros. La introducción de variables altamente colineales de forma directa en los modelos predictivos y de tarificación (como los Modelos Lineales Generalizados o GLMs) precipita consecuencias estadísticas graves: multicolinealidad severa, parámetros de regresión inestables, inflación de varianza y singularidad en la matriz de correlaciones. 

Para resolver este desafío, este proyecto implementa un pipeline riguroso de **Análisis de Factores Exploratorios (EFA)** bajo el **Principio de Parsimonia**, cuyo objetivo primordial es explicar la máxima variabilidad observada en el portafolio mediante un conjunto mínimo de dimensiones latentes ortogonales ($k \ll p$).

---

## 2. Theoretical Framework & Mathematical Formulation

### 2.1 The Classical Linear Factor Model
Sea $\mathbf{X} = (X_1, X_2, \dots, X_p)^T \in \mathbb{R}^p$ un vector aleatorio observable con vector de medias $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$ y matriz de covarianzas $\mathbf{\Sigma}$. El modelo factorial lineal postula que cada variable observada es una combinación lineal de un conjunto de $k$ factores comunes no observables $\mathbf{f} = (f_1, f_2, \dots, f_k)^T$ y un término de error específico $\mathbf{U} = (U_1, U_2, \dots, U_p)^T$:

$$\mathbf{X} = \boldsymbol{\mu} + \Lambda \mathbf{f} + \mathbf{U}$$

Donde:
* $\Lambda \in \mathbb{R}^{p \times k}$ es la **matriz de cargas factoriales** (factor loadings), cuyos elementos $\lambda_{ij}$ miden la fuerza de asociación entre la variable $i$ y el factor latente $j$.
* $\mathbf{f} \sim \mathcal{N}_k(\mathbf{0}, \Phi)$ representa el vector de factores comunes estandarizados.
* $\mathbf{U} \sim \mathcal{N}_p(\mathbf{0}, \Psi)$ representa el vector de factores únicos o errores específicos, donde $\Psi = \text{diag}(\psi_1, \psi_2, \dots, \psi_p)$ es una matriz diagonal que contiene las varianzas específicas de cada variable observada.

### 2.3 Structural Covariance Decomposition
Bajo los supuestos clásicos de que los factores comunes y los errores específicos son incorrelacionados entre sí ($\text{Cov}(\mathbf{f}, \mathbf{U}) = \mathbf{0}$), la matriz de covarianzas poblacional $\mathbf{\Sigma} = \text{Var}(\mathbf{X})$ se descompone estructuralmente como:

$$\mathbf{\Sigma} = \Lambda \Phi \Lambda^T + \Psi$$

Cuando los factores latentes son ortogonales entre sí ($\Phi = \mathbf{I}_k$), esta expresión se simplifica elegantemente a:

$$\mathbf{\Sigma} = \Lambda \Lambda^T + \Psi$$

Para cualquier variable individual $X_i$, su varianza total $\sigma_{ii}$ se particiona de manera exacta en dos componentes fundamentales: la **Comunalidad** ($h_i^2$) y la **Uniqueness o Varianza Específica** ($\psi_i$):

$$\text{Var}(X_i) = \sigma_{ii} = \underbrace{\sum_{j=1}^k \lambda_{ij}^2}_{\text{Comunalidad } h_i^2} + \underbrace{\psi_i}_{\text{Uniqueness } u_i^2}$$

* **Comunalidad ($h_i^2$):** Proporción de la varianza de la variable $X_i$ que es explicada conjuntamente por los $k$ factores comunes.
* **Uniqueness ($u_i^2$):** Varianza residual atribuible exclusivamente a la variable $X_i$ y al error de medición aleatorio.

---

## 3. Properties and Transformations of the Model

### 3.1 Standardization and Correlation Structure
En la práctica actuarial, las variables poseen unidades de medida heterogéneas (dólares, meses, conteos de vehículos). Al estandarizar el vector observado mediante la transformación:

$$Y_i = \frac{X_i - \mu_i}{\sigma_i}$$

Se obtiene que la matriz de covarianzas del nuevo vector estandarizado coincide con la matriz de correlaciones muestral, $\text{Cov}(\mathbf{Y}) = \mathbf{R}$, permitiendo formular el modelo factorial sobre dicha matriz:

$$\mathbf{R} = \Lambda \Lambda^T + \Psi$$

### 3.2 Rotation Indeterminacy and the Varimax Criterion
Una propiedad fundamental del modelo factorial es su **falta de unicidad en las cargas**. Si $\mathbf{T} \in \mathbb{R}^{k \times k}$ es una matriz ortogonal tal que $\mathbf{T}^T \mathbf{T} = \mathbf{T} \mathbf{T}^T = \mathbf{I}$, es posible redefinir los factores y las cargas sin alterar el modelo:

$$\Lambda^* = \Lambda \mathbf{T}, \quad \mathbf{f}^* = \mathbf{T}^T \mathbf{f}$$

Sustituyendo en la estructura de covarianza:

$$\mathbf{\Sigma} = \Lambda^* \Lambda^{*T} + \Psi = (\Lambda \mathbf{T})(\Lambda \mathbf{T})^T + \Psi = \Lambda \mathbf{T} \mathbf{T}^T \Lambda^T + \Psi = \Lambda \Lambda^T + \Psi$$

Dado que la matriz de covarianza $\mathbf{\Sigma}$ y las comunalidades permanecen completamente invariantes ante transformaciones ortogonales ($h_i^{*2} = h_i^2$), existen infinitas soluciones matemáticas equivalentes. Para disipar esta ambigüedad y facilitar la interpretación sustantiva, se implementa la **Rotación Ortogonal Varimax**, la cual maximiza la varianza de las cargas al cuadrado dentro de cada factor:

$$V = \sum_{j=1}^k \left[ \frac{1}{p} \sum_{i=1}^p \lambda_{ij}^{*4} - \left( \frac{1}{p} \sum_{i=1}^p \lambda_{ij}^{*2} \right)^2 \right]$$

Este procedimiento fuerza una "estructura simple" donde cada variable observada exhibe una carga alta en un único factor y cargas cercanas a cero en los demás.

---

## 4. Communality Estimation, Reduced Matrices & Heywood Cases

### 4.1 The Reduced Correlation Matrix
Al sustraer la matriz de varianzas específicas $\Psi$ de la matriz de correlaciones $\mathbf{R}$, se obtiene la denominada **matriz de correlación reducida** $\mathbf{R}^*$:

$$\mathbf{R}^* = \mathbf{R} - \Psi = \Lambda \Lambda^T$$

En esta matriz transformada, los elementos de la diagonal principal ya no son unos, sino las comunalidades exactas ($h_i^2$).

### 4.2 Analytical Estimation Methods for Communalities
Dado que $\Psi$ es inicialmente desconocida, las comunalidades se estiman iterativamente mediante diversas estrategias auxiliares:
1. **Máxima Correlación Absoluta:** $h_i^2 = \max_{j \neq i} |r_{ij}|$
2. **Promedio de Correlaciones:** $h_i^2 = \frac{\sum_{j \neq i} r_{ij}}{p-1}$
3. **Coeficiente de Determinación Múltiple ($R_i^2$):** Obtenido al regredir la variable $X_i$ sobre las $p-1$ variables restantes del conjunto de datos.

### 4.3 Heywood Cases and Inadmissible Solutions
Durante el proceso de estimación iterativa, restricciones muestrales severas o problemas de colinealidad extrema pueden arrojar estimaciones donde la varianza específica deviene negativa ($\psi_i < 0$), lo que a su vez genera comunidades inadmisibles superiores a la unidad ($h_i^2 > 1$). Estos escenarios se conocen como **Casos Heywood** y se corrigen comúnmente acotando teóricamente las comunalidades al intervalo cerrado $h_i^2 \le 1$.

---

## 5. Estimation Methods & Sampling Adequacy

### 5.1 Comparative Overview of Estimation Approaches
* **Método de Componentes Principales:** Utiliza la descomposición espectral de la matriz de correlaciones ($\mathbf{R} = \mathbf{P} \mathbf{D} \mathbf{P}^T$) para aproximar la matriz de cargas mediante los $k$ autovalores mayores: $\hat{\Lambda} = \mathbf{P}_1 \mathbf{D}_1^{1/2}$. No requiere supuestos distribucionales estrictos.
* **Método del Factor Principal:** Opera de manera iterativa sobre la matriz de correlación reducida $\mathbf{R}^*$, actualizando las comunalidades hasta alcanzar la convergencia numérica.
* **Máxima Verosimilitud (ML):** Requiere el supuesto de normalidad multivariada sobre las observaciones para maximizar la función de verosimilitud del modelo, permitiendo contrastes formales de bondad de ajuste mediante pruebas de razón de verosimilitud ($\chi^2$).

### 5.2 Kaiser-Meyer-Olkin (KMO) Measure of Sampling Adequacy
Para verificar empíricamente si la matriz de correlaciones posee la estructura de correlación parcial idónea para el análisis factorial, se calcula el índice KMO:

$$\text{KMO} = \frac{\sum_{i \neq j} r_{ij}^2}{\sum_{i \neq j} r_{ij}^2 + \sum_{i \neq j} a_{ij}^2}$$

Donde $r_{ij}$ son los coeficientes de correlación simple y $a_{ij}$ son los coeficientes de correlación parcial entre las variables. Valores superiores a $0.60$ garantizan una adecuación muestral aceptable para proceder con la extracción de factores.

---

## 6. Empirical Results & Actuarial Interpretation

A través del pipeline computacional implementado en R, se procesó el conjunto de datos de siniestros de automóviles. Tras resolver la multicolinealidad estructural (eliminando la identidad exacta `total_claim_amount` = `injury_claim` + `property_claim` + `vehicle_claim`), se obtuvo un **KMO global de 0.61** y una retención óptima de $k = 2$ factores bajo el Criterio de Kaiser ($\lambda > 1$), explicando el **$54.91\%$** de la varianza total del portafolio.

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

1. **Validación Rigurosa del Principio de Parsimonia ($k \ll p$):**
   * *Conexión Teórica:* El objetivo central del análisis multivariado de factores es lograr una compresión óptima de la información reduciendo el espacio dimensional.
   * *Evidencia Empírica:* La descomposición espectral y el criterio de Kaiser permitieron sintetizar $p = 8$ variables complejas del sector asegurador en tan solo $k = 2$ dimensiones latentes ortogonales, capturando de forma parsimoniosa el **$54.91\%$** de la variabilidad total del sistema ($\lambda_1 = 2.51$, $\lambda_2 = 1.89$).

2. **Mitigación Exitosa de la Colinealidad y Garantía de Adecuación:**
   * *Conexión Teórica:* La estabilidad numérica en la inversión de matrices de covarianza exige la no singularidad ($\det(\mathbf{R}) > 0$).
   * *Evidencia Empírica:* La depuración de identidades lineales exactas restableció la condición de definida positiva en la matriz de correlaciones. El índice **KMO Global de 0.61** validó cuantitativamente la presencia de covarianza común suficiente para justificar la extracción de factores.

3. **Estructura Simple e Invarianza Obtenida mediante Rotación Varimax:**
   * *Conexión Teórica:* La indeterminación rotacional ($\Lambda^* = \Lambda \mathbf{T}$) garantiza que la estructura de covarianza poblacional $\mathbf{\Sigma} = \Lambda \Lambda^T + \Psi$ permanezca intacta mientras se maximiza la interpretabilidad de los pesos.
   * *Evidencia Empírica:* La rotación ortogonal Varimax eliminó las cargas cruzadas ambiguas, segmentando con absoluta claridad conceptual el espacio vectorial en dos ejes interpretables: **Factor 1 (Severidad de Pérdida Económica)**, respaldado por elevadas comunalidades en `vehicle_claim` ($h^2 = 0.843$), `property_claim` ($h^2 = 0.722$) y `injury_claim` ($h^2 = 0.714$), y **Factor 2 (Madurez y Antigüedad del Asegurado)**, gobernado por la estabilidad del cliente (`months_as_customer`, $h^2 = 0.961$) y su edad (`age`, $h^2 = 0.960$). En contraste, las variables contractuales (`policy_annual_premium`, `policy_deductable`) mostraron unicidades cercanas a uno ($u^2 \approx 0.999$), comportándose como ruido específico independiente.

4. **Diferenciación Conceptual y Metodológica: EFA vs. ACP:**
   * *Conexión Teórica:* Mientras que el Análisis de Componentes Principales opera como una transformación lineal determinista orientada a explicar la varianza total de las observaciones ($\mathbf{Y} = \Gamma \mathbf{X}$), el Análisis Factorial postula un modelo estadístico generativo que separa explícitamente la varianza compartida de los errores de medición específicos ($\mathbf{X} = \Lambda \mathbf{f} + \Psi$).
   * *Evidencia Empírica:* La estimación rigurosa de las puntuaciones factoriales de Thomson ($\hat{\mathbf{F}}$) proporciona variables sintéticas incorrelacionadas exentas de redundancia colineal, ofreciendo una base cuantitativa óptima para alimentar fases posteriores de tarificación en Modelos Lineales Generalizados (GLMs).

---

## 9. Bibliography & Recommended References

1. **Johnson, R. A., & Wichern, D. W. (2007).** *Applied Multivariate Statistical Analysis* (6th ed.). Pearson Prentice Hall.
2. **Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2014).** *Multivariate Data Analysis* (7th ed.). Pearson Education.
3. **Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979).** *Multivariate Analysis*. Academic Press.
4. **Thomson, G. H. (1951).** *The Factorial Analysis of Human Ability*. University of Edinburgh Press.
