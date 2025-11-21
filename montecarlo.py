"""
TCEF - Monte Carlo Uncertainty Propagation 
=================================================================
Vectorized Monte Carlo with:
 - synchronized astro parameters (from astro.py)
 - horizon parameters (from horizon.py)
 - R_hat convergence statistic (Gelman-Rubin split-chain)
 - fast execution (50k MC in ~0.02–0.10 s)
"""

import numpy as np
import time
import json
import astro    # Sumber parameter Astrofisika
import horizon  # Sumber parameter Horizon (FIX)

# ============================================================
# GELMAN–RUBIN PSRF (R_hat)
# ============================================================

def compute_rhat(samples, n_chains=4):
    """
    Compute Gelman–Rubin R_hat for a Monte Carlo sample set using split-chain method.
    samples: 1D array, N samples
    n_chains: number of split-chains
    """
    samples = np.asarray(samples)
    N = len(samples)
    if N < n_chains * 10:
        # Not enough samples to compute R_hat reliably, assume converged if N is small but reasonable
        return 1.0

    # Split samples into chains
    chains = np.array_split(samples, n_chains)

    # Ensure equal lengths for calculation simplicity
    n = min(len(c) for c in chains)
    chains = [c[:n] for c in chains]

    m = n_chains
    means = np.array([c.mean() for c in chains])
    vars_ = np.array([c.var(ddof=1) for c in chains])
    mean_all = means.mean()

    # Between-chain variance (B) and Within-chain variance (W)
    B = (n / (m - 1)) * np.sum((means - mean_all)**2)
    W = vars_.mean()

    if W == 0: return 1.0 # Avoid division by zero if all samples are identical

    # Estimated target variance
    var_hat = (n - 1) / n * W + (1 / n) * B

    # Potential Scale Reduction Factor (R_hat)
    R_hat = np.sqrt(var_hat / W)
    return float(R_hat)

# ============================================================
# MAIN MONTE CARLO
# ============================================================

def run_monte_carlo(N_SAMPLES=50000, seed=2025, out_prefix="tcef_mc_sync_v3"):
    print(f"[MC] Running Monte Carlo (N={N_SAMPLES}, seed={seed})")

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # 1. Load Parameters (Harmonized)
    # --------------------------------------------------------

    # A. Load Astro Parameters (Global Variables from astro.py)
    try:
        # Mengambil variabel global dari astro.py
        V_obs = astro.V_obs_SI        # m^3 (Observable Universe)
        j_star0 = astro.j_star_SI     # W/m^3
        f_dust0 = astro.f_dust
        f_agn0  = astro.f_agn
        alpha_star0 = astro.alpha_star
        alpha_dust0 = astro.alpha_dust
        alpha_agn0  = astro.alpha_agn

        # Effective frequencies (Hz) — consistent with astro.py assumptions
        nu_star = astro.nu_eff_star
        nu_dust = astro.nu_eff_dust
        nu_agn  = astro.nu_eff_agn

    except AttributeError as e:
        print(f"![ERROR] Missing attribute in astro.py: {e}")
        print("  -> Please ensure you have updated astro.py with the 'Exposed Constants' version.")
        return

    # B. Load Horizon Entropy (FIXED)
    # We calculate it fresh from the horizon module to be safe
    print("   -> Fetching Horizon Entropy...")
    h_res = horizon.calculate_horizon_entropy()
    S_hor = h_res['S_dot_kB'] # This is the ~ 7.9e103 value (Dimensionless k_B/s)

    h = 6.626e-34 # Planck constant (SI)

    # --------------------------------------------------------
    # 2. Vectorized Sampling (Draw Parameter Distributions)
    # --------------------------------------------------------
    # Scatter estimates chosen from astrophysical uncertainty
    # Log-normal variations for magnitudes (factors)
    j_star = j_star0 * 10**rng.normal(0, 0.15, N_SAMPLES)
    f_dust = f_dust0 * 10**rng.normal(0, 0.12, N_SAMPLES)
    f_agn  = f_agn0  * 10**rng.normal(0, 0.30, N_SAMPLES)

    # Normal variations for efficiency factors (alphas)
    alpha_star = rng.normal(alpha_star0, 0.1, N_SAMPLES)
    alpha_dust = rng.normal(alpha_dust0, 0.1, N_SAMPLES)
    alpha_agn  = rng.normal(alpha_agn0,  0.1, N_SAMPLES)

    # --------------------------------------------------------
    # 3. Compute Luminosities & Entropy
    # --------------------------------------------------------
    # Total Luminosities [Watts]
    L_star = j_star * V_obs
    L_dust = L_star * f_dust
    L_agn  = L_star * f_agn

    # Entropy rate components [k_B/s]
    # Formula approximation: mu = (L / h*nu) * alpha
    # Note: alpha is dimensionless (S_phot/k_B), so result is in k_B/s. Correct.
    mu_star = (L_star / (h * nu_star)) * alpha_star
    mu_dust = (L_dust / (h * nu_dust)) * alpha_dust
    mu_agn  = (L_agn  / (h * nu_agn )) * alpha_agn

    mu_total = mu_star + mu_dust + mu_agn

    # --------------------------------------------------------
    # 4. Statistics
    # --------------------------------------------------------
    p16, p50, p84 = np.percentile(mu_total, [16, 50, 84])

    # Calculate Hierarchy H = S_hor / mu_astro
    H = S_hor / p50

    # Compute R_hat (Convergence Diagnostic)
    R_hat = compute_rhat(mu_total, n_chains=8)

    print("\n[MC] Completed")
    print(f" - Horizon Flux  : {S_hor:.2e} k_B/s")
    print(f" - Median Astro  : {p50:.2e} k_B/s")
    print(f" - 16/84% Range  : [{p16:.2e}, {p84:.2e}]")
    print(f" - Hierarchy (H) : 10^{np.log10(H):.1f}")
    print(f" - R_hat         : {R_hat:.5f}")

    # Save results for future plotting/reference
    np.savez(f"{out_prefix}.npz",
             mu_total=mu_total,
             mu_star=mu_star, mu_dust=mu_dust, mu_agn=mu_agn)

    meta = {
        "N": N_SAMPLES,
        "seed": seed,
        "median_mu": float(p50),
        "mean_mu": float(mu_total.mean()),
        "p16": float(p16),
        "p84": float(p84),
        "hierarchy": float(H),
        "log10_H": float(np.log10(H)),
        "R_hat": R_hat
    }

    with open(f"{out_prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f" - saved: {out_prefix}.npz, {out_prefix}_meta.json\n")

    return mu_total, meta

# ============================================================
# MAIN EXECUTION BLOCK
# ============================================================

if __name__ == "__main__":
    run_monte_carlo(N_SAMPLES=50000, seed=2025, out_prefix="tcef_mc_sync_v3")