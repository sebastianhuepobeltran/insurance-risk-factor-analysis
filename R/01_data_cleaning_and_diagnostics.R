# ==============================================================================
# 1. LOAD REQUIRED LIBRARIES & LOCATE DATASET
# ==============================================================================
if (!require("psych")) install.packages("psych")
if (!require("tidyverse")) install.packages("tidyverse")
library(psych)
library(tidyverse)

# Look for raw file in root or data folder
raw_file <- list.files(pattern = "insurance_claims", full.names = TRUE)
if (length(raw_file) == 0) raw_file <- list.files("data", pattern = "insurance_claims", full.names = TRUE)

if (length(raw_file) == 0) {
  stop("Error: No se encontró 'insurance_claims.csv.xls' en la raíz ni en 'data/'.")
}

cat("Cargando dataset desde:", raw_file[1], "\n")
df_raw <- read.csv(raw_file[1], stringsAsFactors = FALSE)

# ==============================================================================
# 2. CLEAN & FILTER NUMERIC VARIABLES
# ==============================================================================
# Eliminamos multicolinealidad exacta (total_claim_amount)
df_clean <- df_raw %>%
  mutate(across(everything(), ~na_if(as.character(.), "?"))) %>%
  select(
    age, 
    months_as_customer, 
    policy_annual_premium, 
    policy_deductable,
    injury_claim, 
    property_claim, 
    vehicle_claim,
    number_of_vehicles_involved
  ) %>%
  mutate(across(everything(), ~as.numeric(as.character(.)))) %>%
  drop_na()

if (!dir.exists("data")) dir.create("data")
write.csv(df_clean, "data/auto_claims_clean.csv", row.names = FALSE)

cat("==================================================================\n")
cat("DIAGNÓSTICO KMO (Adecuación Muestral)\n")
cat("==================================================================\n")
kmo_diag <- KMO(cor(df_clean))
print(kmo_diag)
