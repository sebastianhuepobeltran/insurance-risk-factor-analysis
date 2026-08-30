# ==============================================================================
# SPECTRAL DECOMPOSITION & UNROTATED LOADINGS
# ==============================================================================
if (!file.exists("data/auto_claims_clean.csv")) {
  stop("No se encuentra 'data/auto_claims_clean.csv'. Corre el Paso 1 primero.")
}

df_clean <- read.csv("data/auto_claims_clean.csv")
n <- nrow(df_clean)
p <- ncol(df_clean)

# Estandarización Z y Matriz de Correlación R
Z <- scale(df_clean)
R <- (1 / (n - 1)) * t(Z) %*% Z

# Descomposición Espectral
spectral_decomp <- eigen(R)
eigenvalues <- spectral_decomp$values
eigenvectors <- spectral_decomp$vectors

rownames(eigenvectors) <- colnames(df_clean)
colnames(eigenvectors) <- paste0("PC", 1:p)

# Criterio de Kaiser (k = 2)
k <- 2
D_k <- diag(eigenvalues[1:k])
Lambda_unrotated <- eigenvectors[, 1:k] %*% sqrt(D_k)
rownames(Lambda_unrotated) <- colnames(df_clean)
colnames(Lambda_unrotated) <- c("Factor1_Unrotated", "Factor2_Unrotated")

cat("==================================================================\n")
cat("MATRIZ DE CARGAS NO ROTADAS (Lambda_1)\n")
cat("==================================================================\n")
print(round(Lambda_unrotated, 3))
