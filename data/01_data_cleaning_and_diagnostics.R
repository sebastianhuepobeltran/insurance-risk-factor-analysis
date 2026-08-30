# ==============================================================================
# STEP 1: DATA CLEANING & MULTICOLLINEARITY DIAGNOSTICS
# Project: Insurance Risk Factor Analysis
# Author: Sebastián H. Beltrán
# ==============================================================================

# 1. LOAD REQUIRED LIBRARIES
if (!require("psych")) install.packages("psych")
if (!require("tidyverse")) install.packages("tidyverse")

library(psych)
library(tidyverse)

# 2. LOCATE & LOAD RAW DATASET
data_raw_path <- "data/auto_claims_clean.csv"

if (!file.exists(data_raw_path)) {
  cat("Please select the raw dataset 'insurance_claims.csv.xls':\n")
  data_raw_path <- file.choose()
}

df_raw <- read.csv(data_raw_path, sep = ",", stringsAsFactors = FALSE)

# 3. FILTER NUMERIC VARIABLES & FIX MULTICOLLINEARITY
# Excluded: 'total_claim_amount' (sum of injury + property + vehicle claims)
# Reason: Prevents matrix singularity det(R) = 0 and resolves KMO NaNs.
df_clean <- df_raw %>%
  mutate(across(everything(), ~na_if(as.character(.), "?"))) %>%
  select(-matches("_c39")) %>%
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

# 4. ENSURE DATA/ DIRECTORY EXISTS & EXPORT PROCESSED DATASET
if (!dir.exists("data")) {
  dir.create("data")
}

write.csv(df_clean, "data/auto_claims_clean.csv", row.names = FALSE)

cat("==================================================================\n")
cat("DATA DIAGNOSTICS SUMMARY\n")
cat("==================================================================\n")
cat("Dataset successfully cleaned & saved to: data/auto_claims_clean.csv\n")
cat("Dimensions:", nrow(df_clean), "rows x", ncol(df_clean), "columns\n\n")

# 5. FACTORABILITY TEST (KMO ADEQUACY)
cat("--- KAISER-MEYER-OLKIN (KMO) ADEQUACY TEST ---\n")
kmo_diag <- KMO(cor(df_clean))
print(kmo_diag)

