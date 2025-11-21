"""
TCEF - Astrophysical Entropy Module (astro.py) - FINAL HARMONIZED
=================================================================
Calculation engine for Astrophysical Entropy Production (mu_astro).

AUDIT COMPLIANCE & SYNC UPDATES:
1. Global Constants: Parameters exposed for montecarlo_sync.py.
2. Volume: Observable Universe (Particle Horizon).
3. Units: Dimensionless Entropy Rate [k_B s^-1].
"""

# ============================================================================
# 1. LIBRARY IMPORTS
# ============================================================================
import numpy as np
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
import warnings

# Suppress integration warnings (e.g., trapz future warnings)
warnings.filterwarnings('ignore')

# Attempt to import FSPS (Flexible Stellar Population Synthesis)
try:
    import fsps
    FSPS_AVAILABLE = True
except ImportError:
    FSPS_AVAILABLE = False

# ============================================================================
# 2. PHYSICAL CONSTANTS & COSMOLOGY (GLOBAL)
# ============================================================================
c     = const.c.value       # m/s
h     = const.h.value       # J s
k_B   = const.k_B.value     # J / K
L_sun = const.L_sun.value   # Watts
Mpc_m = 3.08567758e22       # meters

# Cosmology Setup
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)

# --- VITAL SCRIPT 1: OBSERVABLE VOLUME DEFINITION ---
# We calculate the volume to the Particle Horizon (z=1100), not just Hubble Sphere.
# This ensures we capture ALL causal matter entropy (Conservative Upper Bound).
r_comoving_Mpc = cosmo.comoving_distance(1100).value 
V_obs_Mpc3     = (4.0/3.0) * np.pi * r_comoving_Mpc**3
V_obs_SI       = V_obs_Mpc3 * (Mpc_m**3)

# Global Frequency Grid (Log-spaced 10^9 - 10^21 Hz)
nu_grid = np.logspace(9, 21, 5000)

# ============================================================================
# 3. ASTROPHYSICAL PARAMETERS (EXPOSED GLOBAL VARIABLES)
# ============================================================================
# --- VITAL SCRIPT 2: EXPOSED PARAMETERS FOR MONTE CARLO ---
# These variables are placed in the global scope so 'montecarlo.py' 
# can import them directly (e.g., astro.j_star_SI).

# Luminosity Density (Baseline from Driver et al. 2016)
j_stellar_bol_density = 1.8e8 # L_sun / Mpc^3
j_star_SI = j_stellar_bol_density * L_sun / (Mpc_m**3) # W/m^3

# Component Ratios
f_dust = 1.7   # Infrared excess ratio (Dust re-processing)
f_agn  = 0.01  # AGN accretion ratio

# Entropy Factors (Alpha = S_phot / k_B)
alpha_star = 3.7  # Thermal blackbody-like
alpha_dust = 3.9  # Greybody (cooler -> higher entropy/energy)
alpha_agn  = 4.0  # Non-thermal

# Effective Frequencies (For MC Approximation speed-up)
nu_eff_star = 6.0e14  # Optical (~500nm)
nu_eff_dust = 3.0e12  # Far-IR (~100 micron)
nu_eff_agn  = 1.0e16  # UV/X-ray

# ============================================================================
# 4. SED GENERATORS
# ============================================================================

def get_fsps_sed(sp_instance, nu_grid):
    """FSPS Stellar SED with log-log interp and area normalization."""
    sp_instance.params['tau'] = 5.0
    sp_instance.params['zmet'] = 1
    
    wave_Ang, spec_Ang = sp_instance.get_spectrum(tage=10.0, peraa=True)
    wave_m = wave_Ang * 1e-10
    nu_native = c / wave_m
    
    idx = np.argsort(nu_native)
    nu_native = nu_native[idx]
    spec_Hz = spec_Ang[idx] * (wave_m[idx]**2 / c)
    
    # Log-Log Interpolation to preserve power-law tails
    spec_Hz = np.maximum(spec_Hz, 1e-99)
    log_spec_interp = np.interp(
        np.log(nu_grid), np.log(nu_native), np.log(spec_Hz),
        left=-99, right=-99
    )
    sed_final = np.exp(log_spec_interp)
    
    # --- VITAL SCRIPT 3: ENERGY CONSERVATION ---
    # We normalize by the Integral Area (trapz), not the Peak.
    # This ensures total Luminosity is conserved regardless of spectral shape.
    norm = np.trapz(sed_final, nu_grid)
    return sed_final / norm

def stellar_sed_composite(nu):
    """Fallback Template (Blackbody Sum)."""
    components = [{'T': 25000, 'w': 0.1}, {'T': 6500, 'w': 0.3}, {'T': 3500, 'w': 0.2}]
    sed = np.zeros_like(nu)
    for comp in components:
        x = np.minimum((h * nu) / (k_B * comp['T']), 700)
        bb = (nu**3) / np.expm1(x)
        sed += comp['w'] * bb
    return sed / np.trapz(sed, nu)

def dust_sed_greybody(nu):
    """Dust Modified Blackbody (IR dominant entropy source)."""
    T_dust = 25.0; beta = 1.6; nu_0 = 3e12
    x = np.minimum((h * nu) / (k_B * T_dust), 700)
    bb = (nu**3) / np.expm1(x)
    tau = (nu / nu_0)**beta
    sed = tau * bb
    return sed / np.trapz(sed, nu)

def agn_sed_template(nu):
    """AGN Template (Disk + X-ray Power Law)."""
    T_disk = 1e5
    x = np.minimum((h*nu)/(k_B*T_disk), 700)
    disk = (nu**3) / np.expm1(x)
    pl = np.zeros_like(nu)
    mask = nu > 1e16
    pl[mask] = nu[mask]**(-0.9)
    sed = disk + 1e-4 * pl * np.max(disk)
    return sed / np.trapz(sed, nu)

# ============================================================================
# 5. INTEGRATION ENGINE
# ============================================================================

def entropy_integration(L_nu_array, alpha_val):
    """
    --- VITAL SCRIPT 4: UNIT CONSISTENCY ---
    Calculates Dimensionless Entropy Rate [k_B/s].
    Note: We do NOT multiply by k_B here. 
    Formula: S_dot = Integral [ (L_nu / h*nu) * alpha ] d_nu
    """
    # Photon Rate [photons/s/Hz]
    photon_rate = L_nu_array / (h * nu_grid)
    # Entropy Rate Density [k_B/s/Hz]
    entropy_density = photon_rate * alpha_val
    return np.trapz(entropy_density, nu_grid)

# ============================================================================
# 6. MAIN CALCULATOR
# ============================================================================

def calculate_all_astro_entropy(sp_instance=None, use_fsps=True):
    print("Running ASTRO entropy calculation (Harmonized)...")
    
    # 1. Total Power [Watts]
    # Using exposed global variables
    L_tot_star_W = j_star_SI * V_obs_SI
    L_tot_dust_W = f_dust * L_tot_star_W
    L_tot_agn_W  = f_agn * L_tot_star_W
    
    # 2. Shapes
    if use_fsps and FSPS_AVAILABLE and sp_instance is not None:
        print("   -> Using FSPS")
        shape_star = get_fsps_sed(sp_instance, nu_grid)
    else:
        print("   -> Using Fallback Template")
        shape_star = stellar_sed_composite(nu_grid)
        
    shape_dust = dust_sed_greybody(nu_grid)
    shape_agn  = agn_sed_template(nu_grid)
    
    # 3. L_nu (Spectral Luminosity)
    L_nu_star = shape_star * L_tot_star_W
    L_nu_dust = shape_dust * L_tot_dust_W
    L_nu_agn  = shape_agn  * L_tot_agn_W
    
    # 4. Integrate
    mu_star = entropy_integration(L_nu_star, alpha_star)
    mu_dust = entropy_integration(L_nu_dust, alpha_dust)
    mu_agn  = entropy_integration(L_nu_agn,  alpha_agn)
    
    mu_total = mu_star + mu_dust + mu_agn
    
    print(f"   -> Result: {mu_total:.4e} k_B/s")
    return {
        'mu_astro_total': mu_total,
        'mu_components': [mu_star, mu_dust, mu_agn]
    }

if __name__ == "__main__":
    # Self-test block
    sp = None
    if FSPS_AVAILABLE:
        try: sp = fsps.StellarPopulation(zcontinuous=1, sfh=1)
        except: pass
    calculate_all_astro_entropy(sp)