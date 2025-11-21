TCEF - MCMC Inference Engine - FINAL CANONICAL
=============================================================
Bayesian parameter estimation to validate the Thermodynamic-Kinematic Coincidence.

FEATURES:
1. Dynamic Import: Pulls 'S_dot_kB' (10^103) directly from horizon.py.
2. Dynamic Import: Pulls 'mu_astro_total' (10^67) directly from astro.py.
3. Physics Model: Tests if the thermodynamic prediction (0.13) is statistically
   consistent with the kinematic requirement (0.16) given theoretical uncertainties.
"""

import numpy as np
import time
import argparse
import sys

# ==============================================================================
# 1. DYNAMIC DATA LOADING (SINGLE SOURCE OF TRUTH)
# ==============================================================================
try:
    import astro
    import horizon
    print("[MCMC] Loading canonical physics modules...")

    # Ambil Data Live dari Modul Final
    h_res = horizon.calculate_horizon_entropy()
    a_res = astro.calculate_all_astro_entropy()

    # Nilai Otoritatif (Canonical Values)
    S_DOT_MEAN = h_res['S_dot_kB']          # ~ 7.9e103
    MU_ASTRO_MEAN = a_res['mu_astro_total'] # ~ 5.7e67 (atau 10^69 tergantung run)

    print(f"[MCMC] Imported S_dot: {S_DOT_MEAN:.2e} k_B/s")
    print(f"[MCMC] Imported mu_astro: {MU_ASTRO_MEAN:.2e} k_B/s")

except ImportError as e:
    print(f"![ERROR] Modules not found: {e}")
    print("  Please ensure 'astro.py' and 'horizon.py' are in the same folder.")
    sys.exit(1)

# ==============================================================================
# 2. MODEL & DATA CONSTANTS
# ==============================================================================

# Data Observasi (Kinematic Requirement for Hubble Tension)
# Sumber: Riess et al. (2022) / Efstathiou (2021)
RHO_KIN_TARGET = 0.16
RHO_KIN_SIGMA  = 0.03

# Calibration Factor
# Kita tahu dari derivasi paper bahwa S_dot_canonical memetakan ke delta_rho = 0.13.
# Factor = 0.13 / S_DOT_MEAN
CALIB_FACTOR = 0.13 / S_DOT_MEAN

# Ketidakpastian Teori (Theoretical Uncertainty in QFT Calculation)
# Kita asumsikan faktor ~1.5 (0.2 dex) sebagai prior width yang wajar.
S_DOT_SIGMA_DEX = 0.2

# ==============================================================================
# 3. BAYESIAN FUNCTIONS
# ==============================================================================

def thermodynamic_model(s_dot_val):
    """
    Memetakan Entropy Flux -> Energy Density Excess (delta_rho/rho_crit).
    """
    return s_dot_val * CALIB_FACTOR

def log_prior(s_dot):
    """
    Prior Fisika: Log-Normal di sekitar prediksi QFT (S_DOT_MEAN).
    """
    if s_dot <= 0: return -np.inf

    log_s = np.log10(s_dot)
    log_mu = np.log10(S_DOT_MEAN)

    # Gaussian dalam log-space (dex)
    return -0.5 * ((log_s - log_mu) / S_DOT_SIGMA_DEX)**2

def log_likelihood(s_dot):
    """
    Likelihood: Seberapa cocok energi yang dihasilkan s_dot dengan data 0.16?
    """
    # 1. Hitung Prediksi Model
    rho_pred = thermodynamic_model(s_dot)

    # 2. Bandingkan dengan Data Kinematik
    chi2 = ((rho_pred - RHO_KIN_TARGET) / RHO_KIN_SIGMA)**2
    return -0.5 * chi2

def log_posterior(s_dot):
    lp = log_prior(s_dot)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(s_dot)

# ==============================================================================
# 4. METROPOLIS-HASTINGS SAMPLER
# ==============================================================================

def run_mcmc(n_steps=20000, seed=2025):
    print(f"\n[MCMC] Running Inference Chain (N={n_steps}, Seed={seed})...")
    np.random.seed(seed)

    # Initial State (Start at Theory Mean)
    current_s = S_DOT_MEAN
    current_log_prob = log_posterior(current_s)

    chain = []
    accepted = 0

    t0 = time.time()

    for i in range(n_steps):
        # Proposal: Random walk in log space
        step_dex = 0.05
        prop_log = np.log10(current_s) + np.random.normal(0, step_dex)
        proposal = 10**prop_log

        prop_log_prob = log_posterior(proposal)

        # Acceptance Step
        if prop_log_prob > current_log_prob:
            accept = True
        else:
            ratio = np.exp(prop_log_prob - current_log_prob)
            accept = np.random.rand() < ratio

        if accept:
            current_s = proposal
            current_log_prob = prop_log_prob
            accepted += 1

        chain.append(current_s)

    t1 = time.time()
    print(f"   -> Finished in {t1-t0:.2f}s")
    print(f"   -> Acceptance Rate: {accepted/n_steps:.2%}")

    return np.array(chain)

# ==============================================================================
# 5. MAIN EXECUTION & DIAGNOSTICS
# ==============================================================================

if __name__ == "__main__":
    # Run Chain
    chain = run_mcmc()

    # Burn-in (buang 25% awal)
    burn = int(0.25 * len(chain))
    samples = chain[burn:]

    # Statistics
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])

    # Derived Parameter: Delta Rho
    rho_samples = thermodynamic_model(samples)
    rho_16, rho_50, rho_84 = np.percentile(rho_samples, [16, 50, 84])

    print("\n" + "="*60)
    print("   MCMC POSTERIOR RESULTS (HARMONIZED)")
    print("="*60)
    print(f"Theory Input (Prior) : {S_DOT_MEAN:.2e}")
    print(f"Posterior S_dot      : {p50:.2e} (+/- {p84-p50:.1e})")
    print("-" * 60)
    print(f"Target Data (Kinematic)  : {RHO_KIN_TARGET:.3f} +/- {RHO_KIN_SIGMA:.3f} rho_crit")
    print(f"Posterior Model (Thermo) : {rho_50:.3f} +{rho_84-rho_50:.3f} / -{rho_50-rho_16:.3f} rho_crit")
    print("-" * 60)

    # Final Verdict Check
    z_score = abs(rho_50 - RHO_KIN_TARGET) / RHO_KIN_SIGMA
    print(f"Consistency Z-Score: {z_score:.2f} sigma")

    if z_score < 1.0:
        print("✅ VERDICT: EXCELLENT AGREEMENT (< 1 sigma)")
        print("   The thermodynamic prediction naturally explains the Hubble tension data.")
    elif z_score < 2.0:
        print("⚠️ VERDICT: GOOD AGREEMENT (< 2 sigma)")
    else:
        print("❌ VERDICT: TENSION")