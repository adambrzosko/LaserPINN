# Gain-Switched DFB Laser Simulation Suite

## Overview

This simulation suite models the dynamics of a 1550 nm InGaAsP/InP gain-switched DFB laser for quantum key distribution (QKD) source characterisation. The core physics solves stochastic rate equations (carrier density N, photon density S, optical phase phi) using Euler-Maruyama integration with Langevin noise sources, compiled via Numba JIT for performance.

The primary metric of interest is the **inter-pulse phase correlation r1**, which quantifies residual coherence between consecutive pulses and directly impacts QKD security.

---

## Architecture

```
dfb_laser.py                  Core parameters (DFBLaserParams dataclass) + make_laser() factory
                              Supports DFB (R2=0.95 HR) and Fabry-Perot (R1=R2=0.32) cavities
sld_injection.py              SLD broadband injection model (ASE coupling)
million_pulse_comparison.py   High-throughput stochastic solver (Numba JIT) + waveform builders
├── simulate_pulses()              Standard raised-cosine modulation
├── simulate_pulses_waveform()     Arbitrary I(t) waveform input
├── build_raised_cosine()          Default modulation shape
├── build_square()                 Ideal step pulse
├── build_gaussian()               Gaussian envelope
├── build_fourier()                Fourier-parameterised envelope
└── build_trapezoid()              Asymmetric trapezoid (variable rise/fall)
```

All analysis scripts import from these core modules.

---

## Scripts and Their Purposes

### Core Infrastructure

| File | Purpose |
|------|---------|
| `dfb_laser.py` | `DFBLaserParams` dataclass + `make_laser('dfb'\|'fp')` factory; supports DFB (HR rear facet) and Fabry-Perot (cleaved facets) cavities; derived quantities recomputed automatically |
| `sld_injection.py` | SLD (superluminescent diode) model: `SLDParams`, `InjectionParams`, steady-state solver, ASE-to-injection-field conversion |
| `million_pulse_comparison.py` | Numba-compiled Euler-Maruyama solver for 10^5-10^6 pulse statistics; extracts per-pulse peak phase, peak power, timing |

### Physics Studies (Recommendations #1-6)

| # | File | What It Studies | Output Directory |
|---|------|-----------------|-----------------|
| 1 | `multimode_analysis.py` | Multi-longitudinal-mode competition (5 modes) with shared carrier reservoir; mode partition noise, SMSR, k-factor | `images/multimode/` |
| 3 | `waveform_optimisation.py` | Differential evolution optimisation of modulation waveform for user-selected objectives (jitter, phase randomness, power, balanced) | `images/waveform_opt/` |
| 5 | `carrier_transport_analysis.py` | Carrier transport effects (SCH capture time tau_cap) on gain-switching dynamics; sweep over capture times | `images/carrier_transport/` |
| 6 | `fiber_propagation.py` | Split-step Fourier propagation through SMF-28; chirp compensation in anomalous dispersion; SLD impact on fiber effects | `images/fiber_propagation/` |

### PINN / Machine Learning Studies (Recommendations #7-12)

| # | File | Approach | Output Directory |
|---|------|----------|-----------------|
| 7 | `pinn_inverse_extraction.py` | Inverse PINN: extract (alpha_H, epsilon, beta_sp, a) from intensity-only time series | `images/pinn_inverse/` |
| 8 | `pinn_bandwidth.py` | HistogramNet: map AMZI visibility histograms to (bandwidth, coupling) | `images/pinn_bandwidth/` |
| 9 | `neural_surrogate.py` | MLP surrogate: (f_rep, S_inj, duty, I_bias) -> pulse statistics | `images/neural_surrogate/` |
| 10 | `neural_ode_noise.py` | Neural SDE: learn stochastic noise amplitudes via moment-matching | `images/neural_ode/` |
| 11 | `bayesian_pinn.py` | Bayesian PINN with MC Dropout for uncertainty quantification on extracted parameters | `images/bayesian_pinn/` |
| 12 | `transfer_learning.py` | Pre-train on simulation, fine-tune on sparse experimental data from a different device | `images/transfer_learning/` |

### Other Analysis Scripts

| File | Purpose | Output Directory |
|------|---------|-----------------|
| `phase_transition_analysis.py` | Maps the phase-coherence transition as a function of bias/modulation | `images/phase_transition/` |
| `qkd_source_analysis.py` | Full QKD source characterisation (QBER, key rate estimates) | `images/qkd_source/` |
| `qkd_sinj_sweep.py` | Sweep SLD injection power and measure r1 vs S_inj | `images/qkd_source/` |
| `timing_jitter_analysis.py` | Detailed timing jitter decomposition (turn-on delay statistics) | `images/timing_jitter/` |
| `amzi_pulse_analysis.py` | Simulated asymmetric Mach-Zehnder interferometer measurements | `images/amzi/` |
| `gain_switched_interference.py` | Pulse-to-pulse interference visibility (free-running) | `images/gain_switched/` |
| `gs_injected_interference.py` | Same with SLD injection | `images/gain_switched_injection/` |

---

## Key Parameters (Default Device)

### DFB Laser (default)
```
Wavelength:        1550 nm
Cavity length:     300 um
Active volume:     1.2e-17 m^3
Confinement:       Gamma = 0.3
Differential gain: a = 2.5e-20 m^2
Transparency:      N_tr = 1.5e24 m^-3
Gain compression:  epsilon = 3e-23 m^3
Alpha_H:           3.0
Beta_sp:           1e-4
R1 / R2:           0.32 / 0.95 (HR rear facet)
Mirror loss:       ~1985 m^-1
Threshold current: ~16.7 mA
Photon lifetime:   ~3.1 ps
Front facet frac:  0.93
```

### Fabry-Perot Laser
```
Same material/gain params as DFB, but:
R1 / R2:           0.32 / 0.32 (both cleaved facets)
Mirror loss:       ~3798 m^-1  (1.9x higher)
Threshold current: ~21.1 mA
Photon lifetime:   ~2.1 ps
Front facet frac:  0.50
No grating → all modes compete equally within gain bandwidth
```

Create via: `make_laser('fp')` or `make_laser('fp', L=500e-6)` for overrides.

---

## Usage

All scripts are run from the `Simulations/` root directory using `python3 -m`:

### Core Simulations

```bash
python3 -m core.million_pulse_comparison     # 1M-pulse phase correlation, 1-10 GHz
```

### Physics Studies

```bash
python3 -m studies.multimode_analysis        # Multi-mode competition (5 modes)
python3 -m studies.fiber_propagation         # SSFM fiber propagation
python3 -m studies.carrier_transport_analysis # Carrier transport effects

# Waveform optimisation (CLI with options)
python3 -m studies.waveform_optimisation --objective min_jitter --freq 5
python3 -m studies.waveform_optimisation --objective all --freq 5

python3 -m studies.phase_transition_analysis # Phase transition mapping
python3 -m studies.qkd_source_analysis       # QKD source characterisation
python3 -m studies.qkd_sinj_sweep            # S_inj threshold sweep
python3 -m studies.timing_jitter_analysis    # Jitter decomposition
python3 -m studies.amzi_pulse_analysis       # AMZI visibility analysis
```

### PINN / ML Experiments

```bash
python3 -m pinn.pinn_inverse_extraction      # Inverse parameter extraction
python3 -m pinn.pinn_bandwidth               # Histogram -> bandwidth
python3 -m pinn.neural_surrogate             # MLP surrogate model
python3 -m pinn.neural_ode_noise             # Neural SDE noise learning
python3 -m pinn.bayesian_pinn               # Bayesian PINN (MC Dropout)
python3 -m pinn.transfer_learning            # Transfer learning
```

### Tests

```bash
python3 -m tests.test_coherence_threshold
python3 -m tests.test_threshold_sweep
```

---

## Key Physical Findings

### 1. Multi-mode Competition (#1)
- Multi-mode competition **increases** phase coherence and **reduces** intensity noise compared to single-mode (counter-intuitive)
- SLD injection dramatically worsens mode partition noise (MPN k-factor increases from ~0.15 to ~0.35)
- SMSR degrades from >30 dB (free-running) to ~15 dB (SLD-injected)
- The dominant mode carries most of the useful signal; side modes add noise
- **DFB vs Fabry-Perot**: FP SMSR ~16 dB vs DFB ~34 dB (free-running at 5 GHz) due to absence of grating-based mode selectivity. FP mode partition noise is correspondingly worse.

### 2. Waveform Optimisation (#3)
- Raised-cosine is near-optimal at 5 GHz free-running
- Differential evolution finds ~5-10% improvements in specific objectives but with diminishing returns
- The optimisation landscape is relatively flat near the default operating point
- Fourier parameterisation (3 harmonics, 6 parameters) gives sufficient degrees of freedom

### 3. Carrier Transport (#5)
- NOT the explanation for the experimental phase transition or discrepancies
- Main effect: ~10% power penalty and extra timing jitter at tau_cap > 5 ps
- Negligible effect on phase correlation at typical InGaAsP capture times (~1-3 ps)

### 4. Fiber Propagation (#6)
- Chirp compensation effect: pulses compress before broadening in anomalous dispersion regime
- Optimal compression distance: ~5-10 km for typical chirped pulses at 1550 nm
- SLD injection destroys chirp coherence, eliminating the compression benefit
- After 50+ km, GVD-dominated broadening regardless of initial chirp

### 5. PINN/ML Summary (#7-12)
- **Transfer learning works**: pre-training provides good initialisation for new devices
- **Neural surrogates** useful for "easy" statistics (power, width) but fail on phase metrics
- **Inverse PINNs fail** due to fundamental parameter identifiability issues from intensity-only data
- **Neural SDEs** remain an open research challenge (differentiating through stochastic paths)
- **Bayesian PINNs** give overconfident posteriors with poor coverage

---

## Laser Factory

```python
from core.dfb_laser import DFBLaserParams, make_laser

dfb = make_laser('dfb')                   # Default DFB (R2=0.95)
fp  = make_laser('fp')                    # Fabry-Perot (R1=R2=0.32)
fp2 = make_laser('fp', L=500e-6)          # FP with longer cavity
custom = DFBLaserParams(R2=0.70)          # Direct construction with overrides
```

The `cavity_type` field is carried through to `mode_setup()` in the multi-mode
analysis, which skips the DFB grating penalty for FP cavities.

---

## Shared Utilities (`gsdfb/` Package)

All analysis scripts import from the `gsdfb` package rather than duplicating common code.

### `gsdfb.analysis`
```python
from gsdfb import compute_r1, compute_metrics

r1, dphi = compute_r1(phi)               # Phase correlation + wrapped differences
metrics = compute_metrics(phi, pk_S, pk_k, dt, laser)  # Full QKD metric dict

from gsdfb.analysis import phase_randomisation_quality, absolute_jitter
pq = phase_randomisation_quality(phi)     # r1, KL divergence, KS stat
sigma_t, t_peak = absolute_jitter(pk_k, dt)
```

### `gsdfb.plotting`
```python
from gsdfb import setup_plotting, save_fig
from gsdfb.plotting import img_dir

setup_plotting()                          # Apply consistent matplotlib style
save_fig(fig, 'images/myplot/fig1.png')   # Save + makedirs + print path
out = img_dir('myplot')                   # Create and return 'images/myplot'
```

### `gsdfb.pinn_utils`
```python
from gsdfb.pinn_utils import (
    Scales, LaserPINN, LearnableParams,
    physics_residuals, compute_loss,
    generate_reference, generate_gain_switched_data,
    add_noise_snr, make_collocation, subsample,
    to_tensor, DEVICE, DTYPE,
)
```

---

## Dependencies

```
numpy
scipy
matplotlib
numba
torch (for PINN/ML scripts only)
```

Install:
```bash
pip install numpy scipy matplotlib numba torch
```

---

## File Organisation

```
Simulations/
│
├── core/                               # Core simulation engine
│   ├── dfb_laser.py                    #   DFBLaserParams dataclass + ODE solvers
│   ├── sld_injection.py                #   SLD model, injection coupling, Lang-Kobayashi
│   ├── million_pulse_comparison.py     #   Numba JIT stochastic solver + waveform builders
│   └── gain_switched_interference.py   #   GainSwitchParams, autocorrelation, MZI functions
│
├── gsdfb/                              # Shared utilities package
│   ├── analysis.py                     #   Phase metrics (r1, KL, KS), jitter, AMZI
│   ├── plotting.py                     #   Matplotlib config, save_fig, img_dir
│   └── pinn_utils.py                   #   Scales, LaserPINN, LearnableParams,
│                                       #   physics_residuals, compute_loss, add_noise_snr
│
├── studies/                            # Physics analysis scripts
│   ├── multimode_analysis.py           #   Multi-mode competition (#1)
│   ├── waveform_optimisation.py        #   Waveform optimisation CLI (#3)
│   ├── carrier_transport_analysis.py   #   Carrier transport (#5)
│   ├── fiber_propagation.py            #   Fiber propagation SSFM (#6)
│   ├── phase_transition_analysis.py    #   Phase transition mapping
│   ├── qkd_source_analysis.py          #   QKD source characterisation
│   ├── qkd_sinj_sweep.py              #   S_inj threshold sweep
│   ├── timing_jitter_analysis.py       #   Jitter decomposition
│   ├── amzi_pulse_analysis.py          #   AMZI simulation
│   ├── gs_injected_interference.py     #   Injected interference
│   ├── gs_injected_statistics.py       #   Injection pulse statistics
│   └── gs_phase_sweep.py              #   Phase sweep (PDF report)
│
├── pinn/                               # PINN / ML experiment scripts
│   ├── pinn_inverse_extraction.py      #   Inverse PINN (#7)
│   ├── pinn_bandwidth.py               #   Histogram -> bandwidth (#8)
│   ├── neural_surrogate.py             #   MLP surrogate (#9)
│   ├── neural_ode_noise.py             #   Neural SDE (#10)
│   ├── bayesian_pinn.py                #   Bayesian PINN (#11)
│   └── transfer_learning.py            #   Transfer learning (#12)
│
├── tests/                              # Validation tests
│   ├── test_coherence_threshold.py     #   Coherence destruction threshold
│   └── test_threshold_sweep.py         #   eta_coupling sweep
│
├── archive/                            # Superseded (not imported)
│   ├── laser_pinn_monolithic.py
│   └── sld_standalone_1300nm.py
│
├── images/                             # All output figures (auto-created)
│
└── DOCUMENTATION.md
```

---

## Further Steps

### High-Priority (Directly Actionable)

1. **Experimental validation of chirp compensation**
   - The fiber propagation simulation predicts a compression sweet-spot at ~5-10 km. This is measurable with a streak camera or fast photodiode + sampling scope. Compare pulse widths at 0, 5, 10, 20 km fiber lengths.

2. **Phase-resolved measurements for parameter extraction**
   - The PINN inverse extraction failed because intensity-only data leaves alpha_H and epsilon degenerate. A heterodyne or self-homodyne measurement of the optical field would break this degeneracy. Consider adding a coherent detection arm to the existing AMZI setup.

3. **Multi-mode rate equation with measured gain spectrum**
   - The 5-mode model uses a parabolic gain approximation. Replace with a measured/fitted ASE spectrum shape from the actual device. This will give quantitative MPN predictions rather than qualitative trends.

4. **Transfer learning on real experimental data**
   - The framework in `transfer_learning.py` is ready. Pre-train on simulation, then fine-tune on actual experimental time traces. The key question: does the physics prior in the PINN regularisation actually help when the model is slightly wrong?

5. **Waveform optimisation at higher repetition rates**
   - The current results are at 5 GHz where the laser is well-behaved. At 10+ GHz the dynamics become much more constrained and waveform shaping may yield larger improvements. Run:
   ```bash
   python3 waveform_optimisation.py --objective balanced --freq 10 --n_harmonics 4
   ```

### Medium-Priority (Require More Development)

6. **Incorporate temperature dependence**
   - Add T-dependent gain parameters: a(T), N_tr(T), and thermal roll-off. The current model assumes isothermal operation but real devices heat up during high-duty-cycle operation, shifting threshold and gain.

7. **Correlated noise model for multi-mode**
   - The current multi-mode model uses independent Langevin sources per mode. In reality, carrier noise couples all modes through the shared reservoir. Implement a correlated noise matrix with cross-spectral terms.

8. **Full QKD security analysis pipeline**
   - Connect the pulse statistics (r1, intensity distributions) to an actual QKD security proof. Compute the mutual information leakage from residual coherence and translate to key rate penalty. Compare with the decoy-state BB84 bounds.

9. **GPU-accelerated solver for parameter sweeps**
   - The current Numba solver is CPU-bound. For large parameter sweeps (e.g., 2D maps of r1 vs bias and frequency), port the solver to CuPy or write a custom CUDA kernel. Expected speedup: 10-50x for batch simulations.

10. **Dispersion-managed link optimisation**
    - Extend `fiber_propagation.py` to model DCF (dispersion-compensating fiber) spans. Optimise the DCF length and placement for minimum pulse distortion at the receiver while maintaining phase randomness.

### Lower-Priority (Exploratory)

11. **Reservoir computing with gain-switched dynamics**
    - The nonlinear transient dynamics of the gain-switched laser could serve as a physical reservoir computer. Train a linear readout on the transient ring-down features to classify input waveforms.

12. **Stochastic optimal control formulation**
    - Reformulate waveform optimisation as a proper stochastic optimal control problem (Pontryagin's principle with the rate equations as constraints). This would give a continuous-time optimal drive rather than a discrete parameterisation.

13. **Polarisation-resolved model**
    - Add TE/TM mode splitting for polarisation-multiplexed QKD. The current scalar model ignores polarisation-dependent gain and birefringence.

14. **Feedback effects (external cavity)**
    - Model the effect of residual back-reflections from fiber connectors. Even -40 dB feedback can significantly perturb phase dynamics in gain-switched lasers.

15. **Score-matching for Neural SDE training**
    - The moment-matching loss in `neural_ode_noise.py` failed because gradients through stochastic paths are noisy. Try Stein score matching or denoising score matching as alternative loss functions that avoid differentiating through the SDE sample paths.
