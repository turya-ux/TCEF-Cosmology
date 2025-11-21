"""
TCEF - Nested Sampling Evidence Calculator (nested.py)
======================================================
Calculates Bayes Factor (ln K) using normalized likelihoods.
"""
import numpy as np
import sys
from scipy.special import erfinv
from scipy.stats import norm

# --- DATA & KONSTANTA ---
DATA_MEAN  = 0.16
DATA_SIGMA = 0.03
LOG_NORM_CONST = -np.log(np.sqrt(2 * np.pi) * DATA_SIGMA)
THEORY_MEAN = 0.13

# --- MODEL DEFINITIONS ---
def log_likelihood_normalized(theta):
    rho_val = theta[0]
    if rho_val < 0: return -np.inf
    chi2 = ((rho_val - DATA_MEAN) / DATA_SIGMA)**2
    return -0.5 * chi2 + LOG_NORM_CONST

def prior_transform_tcef(u, sigma_theory_dex=0.1):
    mu = np.log10(THEORY_MEAN)
    sigma = sigma_theory_dex
    log_rho = mu + sigma * np.sqrt(2) * erfinv(2 * u[0] - 1)
    return np.array([10**log_rho])

def calculate_analytic_null(max_rho=1.0):
    mass_fraction = norm.cdf((max_rho - DATA_MEAN)/DATA_SIGMA) - \
                    norm.cdf((0.0     - DATA_MEAN)/DATA_SIGMA)
    mass_fraction = max(mass_fraction, 1e-300)
    return np.log(mass_fraction / max_rho)

# --- MAIN RUNNER ---
def run_analysis():
    # Check for Dynesty inside function to avoid import errors at top level
    try:
        import dynesty
        from dynesty import NestedSampler
    except ImportError:
        print("[SKIP] Dynesty not installed. Skipping nested sampling.")
        return

    print("\n" + "="*60)
    print("   FINAL AUDIT: NESTED SAMPLING")
    print("="*60)

    # 1. Null Hypothesis
    lnZ0 = calculate_analytic_null(max_rho=1.0)
    print(f"Null Evidence (Z0) : {lnZ0:.4f}")
    
    # 2. TCEF Model (Sigma = 0.1 dex)
    print(f"Running Nested Sampling for Sigma = 0.1 dex...")
    
    # Define prior wrapper
    def prior(u): return prior_transform_tcef(u, sigma_theory_dex=0.1)
    
    # Run Sampler
    sampler = NestedSampler(log_likelihood_normalized, prior, ndim=1, 
                            nlive=500, bound='single', sample='slice')
    sampler.run_nested(dlogz=0.1, print_progress=False)
    
    # Results
    lnZ1 = sampler.results.logz[-1]
    lnK = lnZ1 - lnZ0
    
    print("-" * 60)
    print(f"Model Evidence (Z1) : {lnZ1:.4f}")
    print(f"Bayes Factor (ln K) : {lnK:.4f}")
    print("-" * 60)
    
    if lnK > 1:
        print("✅ VERDICT: SUBSTANTIAL EVIDENCE.")
    else:
        print("❌ VERDICT: INCONCLUSIVE.")
    
    return lnK

if __name__ == "__main__":
    run_analysis()