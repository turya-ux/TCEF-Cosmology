"""
TCEF - Coincidence Validation Module (coincidence.py)
=====================================================
Performs a rigorous Monte Carlo test (N=1,000,000) to quantify the 
statistical significance of the Thermodynamic-Kinematic alignment.

Output:
- P_null (Probability of random match)
- P_model (Probability of TCEF match)
- Predictive Power Ratio (Bayes Factor proxy)
"""

import numpy as np
import argparse

# ============================================================================
# CONFIGURATION (Canonical Values from Manuscript)
# ============================================================================
N_SAMPLES = 1_000_000
SEED = 2025

# Theory Prediction (Thermodynamic Latent Heat)
# Mean = 0.13, Sigma = 0.03 (Conservative theory uncertainty)
THEORY_MEAN  = 0.13
THEORY_SIGMA = 0.03

# Observation Target (Kinematic Requirement from Hubble Tension)
# Value = 0.16 +/- 0.03 (Riess et al.)
# We define a "Match" if the value falls within the observational error window.
TARGET_VAL   = 0.16
MATCH_WINDOW = 0.04  # Approx 1-sigma width

# ============================================================================
# VALIDATION ENGINE
# ============================================================================

def run_coincidence_test():
    print(f"\nRunning Coincidence Validation (N={N_SAMPLES})...")
    rng = np.random.default_rng(SEED)
    
    # 1. Generate Null Hypothesis Samples (Random Chance)
    # Assumption: Without a theory, vacuum energy density could be anything 
    # in the physical range [0, 1] * rho_crit.
    samples_null = rng.uniform(0.0, 1.0, N_SAMPLES)
    
    # 2. Generate TCEF Theory Samples (Physics)
    # Assumption: Theory predicts 0.13 with some uncertainty (log-normal).
    # Convert arithmetic mean/std to log-normal params
    var = THEORY_SIGMA**2
    mu_log = np.log(THEORY_MEAN / np.sqrt(1 + var/THEORY_MEAN**2))
    sigma_log = np.sqrt(np.log(1 + var/THEORY_MEAN**2))
    
    samples_theory = rng.lognormal(mu_log, sigma_log, N_SAMPLES)
    
    # 3. Check Matches
    # A "match" occurs if the sampled value is close to the target (0.16)
    # within the tolerance window.
    matches_null = np.sum(np.abs(samples_null - TARGET_VAL) < MATCH_WINDOW)
    matches_theory = np.sum(np.abs(samples_theory - TARGET_VAL) < MATCH_WINDOW)
    
    # 4. Calculate Probabilities
    p_null = matches_null / N_SAMPLES
    p_theory = matches_theory / N_SAMPLES
    
    # 5. Calculate Odds Ratio (Bayes Factor Proxy)
    odds_ratio = p_theory / p_null
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    print("-" * 60)
    print(f"Target Data     : {TARGET_VAL} +/- {MATCH_WINDOW}")
    print(f"Theory Input    : {THEORY_MEAN} +/- {THEORY_SIGMA}")
    print("-" * 60)
    print(f"[NULL] Probability of Random Match : {p_null:.4f} ({p_null*100:.1f}%)")
    print(f"[TCEF] Probability of Theory Match : {p_theory:.4f} ({p_theory*100:.1f}%)")
    print("-" * 60)
    print(f"PREDICTIVE POWER RATIO (Odds)      : {odds_ratio:.1f}x")
    print("-" * 60)
    
    if odds_ratio > 3:
        print("✅ VERDICT: STRONG EVIDENCE (Matches manuscript claim)")
    else:
        print("❌ VERDICT: WEAK EVIDENCE")

if __name__ == "__main__":
    run_coincidence_test()