import pandas as pd
import numpy as np

# --- Reference dictionaries ---
numerical = {(1,0): -0.95087, (2,0): -0.20355, (2,1): -0.20299,
             (3,0): -0.06866, (3,1): -0.06816, (3,2): -0.06714}
susy = {(1,0): -0.95092, (2,0): -0.20355, (2,1): -0.20299,
        (3,1): -0.06814, (3,2): -0.06713}

# --- Load data ---
df = pd.read_csv("hbc2.csv")

def rmsre(df, ref_dict):
    results = {}
    for (n, l), ref in ref_dict.items():
        energies = df.query("n==@n and l==@l")["energy"].values
        if len(energies) == 0:
            continue
        rel_err = (energies - ref) / ref
        rmsre_val = np.sqrt(np.mean(rel_err**2))
        results[(n, l)] = rmsre_val
    return results

# Compute for both
rmsre_numerical = rmsre(df, numerical)
rmsre_susy = rmsre(df, susy)

print("RMSRE (numerical reference):")
for k, v in rmsre_numerical.items():
    print(f"  State {k}: {v:.3e}")

print("\nRMSRE (SUSY reference):")
for k, v in rmsre_susy.items():
    print(f"  State {k}: {v:.3e}")

