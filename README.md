# FDOA-Only Source Geolocation — Optimisation Methods

**Author:** Lara Longney  
**Project:** Le Yang's Optimisation Project, 2026  
**Topic:** Satellite-based radio source geolocation using Frequency Difference of Arrival (FDOA) measurements

---

## Overview

This repository implements and compares three optimisation approaches for FDOA-only geolocation using four satellites. A radio source emits a signal received by each satellite; the Doppler frequency shift differences between satellite pairs (FDOA measurements) are used to locate the source on the Earth's surface.

The search is parameterised over two variables:
- **θ (theta)** — bearing angle (rad)
- **r₁** — source-to-reference-satellite distance (m)

All distances are internally scaled by `scaling = 1000 km` for numerical conditioning, and the ML cost function is evaluated in log-space for the same reason.

The benchmark method is the **BFGS algorithm with bisection line search**, provided by Le Yang from the course ENEL445. The two implemented methods are **Gauss-Newton** and **Particle Swarm Optimisation (PSO)** which are contained in this git repository

---

## Repository Structure

```
.
├── README.md
│
├── Gauss_Newton/
│   ├── Newton_main.m                  % Main script — run this
│   ├── Newton.m                       % Gauss-Newton optimiser
│   ├── GradDescent.m                  % Gradient descent (fallback/hybrid)
│   ├── LineSearch_Bisection_Count.m   % Line search with MLCost counter
│   ├── Strictly_Gauss_Newton_1/       % Results: L-inf norm, tol=1e-6
│   ├── Strictly_Gauss_Newton_2/       % Results: 2-norm, tol=1e-6
│   ├── Strictly_Gauss_Newton_3/       % Results: L-inf + solution change, tol=1e-6
│   ├── Strictly_Gauss_Newton_4/       % Results: L-inf norm, tol=1e-8
│   ├── Strictly_Gauss_Newton_5/       % Results: L-inf norm, tol=1e-2 (chosen)
│   ├── Strictly_Gauss_Newton_6/       % Results: 40 starts, tol=1e-2
│   └── Strictly_Gauss_Newton_7/       % Results: random init, tol=1e-2
│
├── PSO/
│   ├── PSO_main.m                     % Main script — run this
│   ├── PSO.m                          % PSO optimiser
│   ├── GradDescent.m                  % Gradient descent for PSO+GD variant
│   ├── LineSearch_Bisection_Count.m   % Line search with MLCost counter
│   ├── PSO_only/                      % Results: 1 stopping condition
│   ├── PSO_added_condition/           % Results: 2 stopping conditions (chosen)
│   └── PSO_GradDescent/               % Results: PSO + gradient descent
│
└── Shared/                            % Shared helper functions (used by all methods)
    ├── MLCost.m                       % ML cost function and gradient
    ├── FDOAGen.m                      % True FDOA generation
    ├── grad_computation.m             % Analytical Jacobian
    ├── LineSearch_Bisection.m         % Original BFGS line search (Le's)
    ├── BFGS_Bisection.m               % BFGS benchmark (Le's)
    ├── Bound.m                        % Search bound enforcement
    ├── HammersleySeq.m                % Hammersley low-discrepancy sequence
    ├── sphere_LLA2ECEF.m              % Coordinate conversion: LLA → ECEF
    ├── sphere_ECEF2LLA.m              % Coordinate conversion: ECEF → LLA
    ├── sphere_ENU2ECEF.m              % Coordinate conversion: ENU → ECEF
    └── sphere_r1_range.m              % Compute r1 search bounds
```

---

## How to Run

### Gauss-Newton

1. Ensure the folder is on your MATLAB path
2. Open `main.m`
3. Set the desired configuration at the top of the file under **GLOBAL SETTINGS**:

```matlab
sigma_f     = 4;      % Noise standard deviation (Hz) — try 1, 4, 8, 16
ensembleRun = 1000;   % Number of Monte Carlo runs
```

4. To switch between stopping condition variants, comment/uncomment the relevant blocks inside `Newton.m` (clearly marked with `% --- Option N ---` comments)
5. Run `main.m` — figures are generated automatically and printed results appear in the command window

### PSO

1. Ensure folder is in the correct MATLAB path (same as above)
2. Open `main.m`
3. Set global settings at the top:

```matlab
sigma_f      = 4;                       % Noise standard deviation (Hz)
ensembleRun  = 1000;                    % Number of Monte Carlo runs
gif_filename = 'PSO_animation.gif';    % Output filename for particle animation
```

4. To enable/disable gradient descent refinement, find the `% --- [GD TOGGLE]` comments in `PSO_main.m` and uncomment/comment the relevant lines
5. Run `main.m`

> **Note:** The particle animation (Figure 5) saves a GIF to the working directory. This can take a few minutes for long runs. If you only want the RMSE and convergence figures, comment out the animation section at the bottom of `PSO_main.m`.

---

## Variants Implemented

### Gauss-Newton Variants

| Variant | Starts | Gradient Condition | Tolerance | Notes |
|---------|--------|--------------------|-----------|-------|
| GN 1 | 20 | L-inf norm | 1e-6 | Baseline |
| GN 2 | 20 | 2-norm | 1e-6 | Norm comparison |
| GN 3 | 20 | L-inf + solution change | 1e-6 | Extra stopping condition |
| GN 4 | 20 | L-inf norm | 1e-8 | Tighter tolerance |
| GN 5 | 20 | L-inf norm | 1e-2 | **Chosen configuration for this project** |
| GN 6 | 40 | L-inf norm | 1e-2 | Double the starts |
| GN 7 | 20 (random) | L-inf norm | 1e-2 | Random vs Hammersley |

**Chosen configuration (GN 5):** 20 Hammersley starts, L-inf gradient norm, tolerance 1e-2. The coarser tolerance gives identical RMSE to 1e-8 for all noise levels tested (the noise floor dominates), while providing more consistent computation time.

### PSO Variants

| Variant | Particles | Stopping Conditions | Notes |
|---------|-----------|---------------------|-------|
| PSO only | 25 | Cost stagnation over 20 iters (< 1e-8) | Baseline |
| PSO + condition | 25 | Cost stagnation + position spread < 1e-6 | **Chosen for this project** |
| PSO + GD | 25 | Cost stagnation (< 1e-2) → hand off to GD | Hybrid |

**Chosen configuration:** Two stopping conditions — cost stagnation over 20 iterations and position spread convergence. Achieves the same RMSE as the single-condition baseline in less computation time.

---

## Key Files Explained

### `Newton.m`
The core Gauss-Newton optimiser. Takes a single starting point `(theta, r1)` and returns the converged estimate. The Hessian is approximated as:

```
H ≈ (J' * Q^{-1} * J) / (G + 1/scaling)
```

Two checks determine if the Newton direction is usable:
- `rcond(H) < 1e-12` → Hessian too ill-conditioned (nearly singular)
- `p'*(-d) >= 0` → Hessian indefinite, direction not downhill

Both trigger termination (pure Gauss-Newton, no gradient descent fallback). The stopping conditions are clearly labelled and can be toggled by commenting/uncommenting.

### `LineSearch_Bisection_Count.m`
Wraps the bisection line search and counts every `MLCost` call made — both in the outer bracketing loop and inside the bisection sub-function. Returns `nfev` (number of function evaluations) alongside the step length and boundary flag. Used to populate the line search evaluation histograms.

### `PSO.m`
Implements the full PSO algorithm. Velocity update follows the standard formulation:

```
v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
```

with `w = 0.7`, `c1 = c2 = 1.5`. Absorbing boundaries clip particles to the search bounds and zero their velocity when they escape. Returns particle position history for animation and the global best `(theta_out, r1_out)` for optional gradient descent warm-starting.

### `GradDescent.m`
Simple steepest descent with bisection line search. Used as the refinement stage in the PSO + GD variant. Takes the PSO global best as its starting point and refines to the precise minimum.

---

## Dependencies

All functions are standard MATLAB — no additional toolboxes are required. The shared helper functions (`sphere_*.m`, `MLCost.m`, etc.) must be on the MATLAB path when running any main script.

---

## Reference
A majority of the helper functions were provided by Le Yang, especially system transformations ie. from ECEF to LLA and more. 
J. R. R. A. Martins and A. Ning, *Engineering Design Optimization*. Cambridge, UK: Cambridge University Press, 2021.
