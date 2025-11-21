TCEF PROJECT: REPRODUCIBILITY SUITE
==============================================================================
The Theory of Cosmic Entropy Functional (TCEF):
A Minimum-Entropy Horizon Framework Unifying H0, S8, and w(z) Anomalies.

Author: Putu Arya Wiryawan
Affiliation: Universal Science Initiative
Contact: pt.aryawiryawan@gmail.com
Date: November 2025
==============================================================================

1. OVERVIEW
------------------------------------------------------------------------------
This repository contains the complete Python numerical pipeline used to generate
the results, statistical evidence, and prediction figures presented in the
manuscript submitted to Physical Review D.

The code verifies three core claims:
1. The Horizon Entropy Flux is ~ 10^103 k_B/s (Calculated from QFT).
2. The Astrophysical Entropy Production is ~ 10^69 k_B/s (Upper bound census).
3. The Thermodynamic-Kinematic Coincidence (0.13 approx 0.16) is statistically
   significant (Bayes Factor ln K ~ 2.1).

2. FILE STRUCTURE
------------------------------------------------------------------------------
This suite consists of 6 harmonized modules designed to work together.

[CORE PHYSICS]
- horizon.py        : Calculates S_dot_hor from fundamental constants (GR+QFT).
- astro.py          : Calculates mu_astro from luminosity functions (Census).

[STATISTICAL VALIDATION]
- montecarlo.py     : Performs uncertainty propagation (N=50,000) to test the
                      robustness of the 10^34 hierarchy.
- mcmc.py           : Runs Metropolis-Hastings MCMC to estimate posterior 
                      thermodynamic energy density.
- nested.py         : Runs Dynamic Nested Sampling (Dynesty) to compute the
                      Bayesian Evidence (ln K) against a null hypothesis.

[VISUALIZATION]
- prediction.py     : Generates the falsifiable prediction plots (Figures 1-5)
                      for Hubble, JWST, LISA, Euclid, and PIXIE.

3. INSTALLATION & REQUIREMENTS
------------------------------------------------------------------------------
The code requires Python 3.0 and standard scientific libraries.

Recommended: Create a clean environment (conda or venv).

Install dependencies via pip:
$ pip install numpy scipy matplotlib astropy h5py tqdm dynesty

Optional:
- 'fsps' (python-fsps) for high-precision stellar synthesis.
  If not installed, 'astro.py' will automatically use a built-in fallback
  composite model that approximates FSPS to within 5%.

4. QUICK START (HOW TO RUN)
------------------------------------------------------------------------------
To reproduce the results in the paper, run the modules in the following order:

A. Verify Core Physics Numbers (Hierarchy ~ 10^34)
   $ python horizon.py
   $ python astro.py
   
   Output should confirm:
   - Horizon Flux ~ 7.9e103 k_B/s
   - Astro Flux   ~ 2.6e69 k_B/s (Conservative Upper Bound)

B. Generate Prediction Plots (Figures 1-5)
   $ python prediction.py
   
   This will generate PDF files (hubble.pdf, lisa.pdf, etc.) in the current
   directory.

C. Run Statistical Validation (MCMC & Nested Sampling)
   $ python mcmc.py
   $ python nested.py
   
   Output will display the Bayes Factor and Posterior constraints.

5. REPRODUCIBILITY GUARANTEE
------------------------------------------------------------------------------
All stochastic modules (Monte Carlo, MCMC, Nested) use a fixed random seed
(SEED = 2025) to ensure bit-for-bit reproducibility of the results reported
in the manuscript.

6. LICENSE & CITATION
------------------------------------------------------------------------------
This code is released under the MIT License. If you use this code or the TCEF
framework in your research, please cite the accompanying paper:

Wiryawan, P. A. (2025). "The Theory of Cosmic Entropy Functional...".
Submitted to Physical Review D.