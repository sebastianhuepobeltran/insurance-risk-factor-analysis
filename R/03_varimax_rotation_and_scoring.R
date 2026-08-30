# ==============================================================================
# VARIMAX ROTATION, FACTOR SCORES & GENERACIÓN DE GRÁFICAS (.PNG)
# ==============================================================================
if (!require("psych")) install.packages("psych")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("ggrepel")) install.packages("ggrepel")
library(psych)
library(tidyverse)
library(ggrepel)

df_clean <- read.csv("data/auto_claims_clean.csv")
n <- nrow(df_clean)
p <- ncol(df_clean)
Z <- scale(df_clean)
R <- (1 / (n - 1)) * t(Z) %*% Z

spectral_decomp <- eigen(R)
eigenvalues <- spectral_decomp$values
k <- 2
Lambda_unrotated <- spectral_decomp$vectors[, 1:k] %*% diag(sqrt(spectral_decomp$values[1:k]))
rownames(Lambda_unrotated) <- colnames(df_clean)

# Rotación Varimax
rotation_res <- varimax(Lambda_unrotated)
Lambda_rotated <- rotation_res$loadings[]
colnames(Lambda_rotated) <- c("Factor1_Severity", "Factor2_Tenure")
rownames(Lambda_rotated) <- colnames(df_clean)

# Factor Scores (Método de Thomson)
R_inv <- solve(R)
W <- R_inv %*% Lambda_rotated
F_scores <- Z %*% W
colnames(F_scores) <- c("Score_F1_Severity", "Score_F2_Tenure")
df_with_scores <- cbind(df_clean, F_scores)
write.csv(df_with_scores, "data/auto_claims_with_factor_scores.csv", row.names = FALSE)

# ------------------------------------------------------------------------------
# 1. GENERAR Y GUARDAR SCREE PLOT
# ------------------------------------------------------------------------------
df_scree <- data.frame(PC = factor(paste0("PC", 1:p), levels = paste0("PC", 1:p)), Eigenvalue = eigenvalues)
scree_gg <- ggplot(df_scree, aes(x = PC, y = Eigenvalue, group = 1)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "#d95f02", linewidth = 0.8) +
  geom_line(color = "#2b5c8f", linewidth = 1.2) +
  geom_point(color = "#2b5c8f", fill = "#e0f3f8", shape = 21, size = 3.5, stroke = 1.2) +
  annotate("text", x = 6.5, y = 1.15, label = "Kaiser Criterion (λ = 1)", color = "#d95f02", fontface = "italic") +
  labs(
    title = "Scree Plot: Spectral Decomposition Threshold",
    subtitle = "Dimensionality Reduction: 8 Variables -> 2 Latent Factors",
    x = "Principal Components / Latent Dimensions",
    y = "Eigenvalues (Variance Explained)"
  ) +
  theme_minimal()

ggsave("scree_plot.png", plot = scree_gg, width = 7, height = 4.5, dpi = 300)

# ------------------------------------------------------------------------------
# 2. GENERAR Y GUARDAR FACTOR MAP
# ------------------------------------------------------------------------------
df_loadings <- as.data.frame(Lambda_rotated)
df_loadings$Variable <- rownames(df_loadings)
factor_map_gg <- ggplot(df_loadings, aes(x = Factor1_Severity, y = Factor2_Tenure, label = Variable)) +
  geom_vline(xintercept = 0, color = "gray70", linewidth = 0.5) +
  geom_hline(yintercept = 0, color = "gray70", linewidth = 0.5) +
  geom_point(color = "#d62728", size = 4, alpha = 0.85) +
  geom_text_repel(fontface = "bold", size = 3.8, box.padding = 0.5) +
  xlim(-1.1, 1.1) + ylim(-1.1, 1.1) +
  labs(
    title = "Rotated Latent Factor Loadings Space (Varimax)",
    subtitle = "Orthogonal Mapping of Insurance Claims & Policyholder Characteristics",
    x = "Factor 1: Loss Severity (Economic Impact)",
    y = "Factor 2: Policyholder Maturity & Customer Tenure"
  ) +
  theme_bw()

ggsave("factor_loadings_map.png", plot = factor_map_gg, width = 7, height = 5, dpi = 300)

# ------------------------------------------------------------------------------
# 3. GENERAR Y GUARDAR HEATMAP
# ------------------------------------------------------------------------------
df_heatmap <- df_loadings %>% pivot_longer(cols = starts_with("Factor"), names_to = "Factor", values_to = "Loading")
heatmap_gg <- ggplot(df_heatmap, aes(x = Factor, y = reorder(Variable, abs(Loading)), fill = Loading)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.3f", Loading)), color = "black", fontface = "bold", size = 3.5) +
  scale_fill_gradient2(low = "#4575b4", mid = "#ffffbf", high = "#d73027", midpoint = 0, limits = c(-1, 1)) +
  labs(
    title = "Factor Loading Matrix Heatmap",
    subtitle = "Absolute Structural Associations (Varimax Rotated)",
    x = "Latent Factors",
    y = "Observed Actuarial Variables"
  ) +
  theme_minimal()

ggsave("factor_loadings_heatmap.png", plot = heatmap_gg, width = 6.5, height = 4.5, dpi = 300)

cat("\n>>> ¡PROCESO COMPLETADO! Las 3 imágenes .png se han guardado en la raíz del proyecto. <<<\n")
