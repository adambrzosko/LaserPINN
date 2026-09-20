# Fabry–Perot phase randomisation: session report

*16–20 September 2026. Simulation only; no FP measurement retake is planned.*

**Question.** Does a multi-longitudinal-mode Fabry–Perot (FP) laser randomise pulse-to-pulse phase more easily than the single-mode DFB of Chapter 5, and can a 19 mW SLD do it at 10 GHz?

> **Two earlier claims are retracted.** A sampling error inflated the FP's apparent randomisation whenever the laser was partly coherent. With it fixed, the long-cavity advantage disappears and the FP no longer needs less injected light than the DFB. See "The sampling artefact" below. Unless stated otherwise, every number here uses fixed-delay sampling.

## Headline findings

1. **The model now reproduces the measured behaviour.** With cross-saturation θ < 1 and instantaneous gain saturation, a nearly-CW FP forms a self-mode-locked comb: every line coherent (r₁ 0.97 / 0.94 / 0.94 / 0.83) and lines 1–2 locked at 0.957. That matches the report of separate lines behaving as if phase-locked.
2. **The cause of the coherence is almost certainly the drive.** Gain-switching deeply enough destroys the lock; at 2 GHz an off-current at or below 0.8 × threshold makes every line random and mutually uncorrelated, with no injection at all. Above ~1.2 × threshold the laser stays coherent.
3. **The SLD doing nothing is a separate problem.** Even a laser held above threshold is randomised by injection in the model. For it to have no effect, the light reaching the cavity must be below roughly 1e16 m⁻³ per mode, a coupling efficiency under 2 × 10⁻⁴. An isolator in the package would explain it.
4. **The FP's advantage over the DFB is modest, not large.** Free-running at 2 GHz: FP 0.145 against DFB 0.270 on the filtered-line metric. At 10 GHz there is no advantage at all — at intermediate injection the FP is worse, and both cross the 0.01 threshold in the same bracket.
5. **What survives for a multi-stream QRNG.** Under deep gain switching the lines are individually random and mutually uncorrelated (cross-mode correlation 0.022, at the noise floor), so several filtered lines could in principle give parallel streams. But that regime is precisely the one the real device is *not* in.

## The sampling artefact

Each mode's stored field carries its frequency offset, E_j ∝ exp(i·Δω_j·t). Sampling each pulse at its **own peak** therefore adds a phase Δω_j·δt, where δt is the peak-timing jitter. With 135 GHz mode spacing and a few picoseconds of jitter, that is radians for the first side mode, and it scrambles every off-centre line. A real AMZI compares E(t) with E(t−T) at a **fixed** delay and never sees this term. Only the centre mode (Δω = 0) is immune, which is why the strongest line always looked coherent while its neighbours looked random.

The bias only matters where the laser is genuinely coherent. Where the phases are truly random, both samplings agree.

Long-cavity sweep at 10 GHz, free-running (N = 800):

| Cavity | Peak-time (previously reported) | Fixed delay | Strongest line |
|---|---|---|---|
| 150 µm | 0.925 | 0.946 | 0.973 |
| 300 µm | 0.743 | 0.926 | 0.982 |
| 600 µm | 0.232 | 0.896 | 0.975 |
| 1200 µm | 0.112 | **0.832** | 0.969 |

The reported 10× improvement from a longer cavity was the artefact growing with mode count. It was reproducible and ~31σ against seed noise, because the bias is systematic rather than random — statistical significance could not catch it.

`gain_switch_fp` now returns `E_fixed` alongside `E_peak`, so old numbers stay reproducible.

## Reproducing the experiment

Two ingredients were missing, and neither alone is enough.

- **Cross-saturation θ < 1.** At θ = 1 each mode suppresses its neighbours exactly as hard as itself, which forces a single dominant line: in CW the model ran at 1.1–1.2 effective modes with 92–96% in one line. There was never a comb to lock. Real FP diodes stay multimode because self-saturation exceeds cross-saturation.
- **Instantaneous gain saturation.** The model compressed gain with the period-averaged photon density. Real gain follows the instantaneous intracavity intensity, and at a 135 GHz beat the sub-picosecond nonlinearities track it.

CW at 2 × threshold, 40 ns (~5400 round trips):

| θ | fast gain | Lasing modes | FM lock 1 ns | 5 ns | 20 ns | Single-mode coherence 10 ns |
|---|---|---|---|---|---|---|
| 1.0 | off | 1.2 | no comb | – | – | – |
| 0.7 | off | 3.4 | 0.435 | 0.134 | 0.122 | 0.258 |
| **0.7** | **on** | **3.5** | **0.891** | **0.897** | **0.894** | 0.141 |

With fast gain the relative phases stay pinned out to 20 ns while the common phase diffuses away — the signature of a mode-locked comb. Without it the relative phase decays like independent diffusion.

Gain-switched at 2 GHz with that locked-comb model, fixed-delay sampling:

| I_off | SLD | Line 1 | Line 2 | Lines 1–2 locked |
|---|---|---|---|---|
| 0.8 × I_th | – | 0.057 | 0.033 | 0.033 |
| 2.0 × I_th | – | 0.972 | 0.940 | **0.957** |
| 2.0 × I_th | 1e19 | 0.026 | 0.056 | 0.021 |

The near-CW row reproduces all of the measured behaviour. The deep-switching row is the proposed fix.

## What the drive does

Free-running at 2 GHz, single filtered line (the centre mode, so unaffected by the artefact):

| I_off / I_th | FP single line | DFB | Jitter | Peak power |
|---|---|---|---|---|
| 1.2 | 0.850 | 0.885 | – | – |
| 0.9 | 0.143 | 0.255 | 2.7 ps | 5.7e21 |
| **0.8** | **0.027** | 0.137 | 3.5 ps | 5.0e21 |
| 0.7 | 0.020 | 0.047 | 4.1 ps | 4.1e21 |
| 0.5 | 0.009 | 0.046 | 6.4 ps | 2.2e21 |

The transition is sharp between 0.9 and 0.8 × threshold, and costs about 12% of peak power and 0.8 ps of jitter. Both lasers randomise when driven deep enough; the FP does so about one bias step earlier.

At 10 GHz the drive barely matters (free-running single line 0.982 at 0.9 × threshold, 0.988 at 1.2 ×) because a 70 ps off-interval is too short for the field to decay either way.

## Injection at 10 GHz

FP 300 µm, I_off = 0.9 × I_th, N = 12000, floor 0.0081. DFB compared at matched total density.

| Per mode (total) | FP peak | FP fixed | FP line | DFB |
|---|---|---|---|---|
| 0 | 0.771 | 0.934 | 0.982 | 0.988 |
| 1e17 (2.1e18) | 0.574 | 0.874 | 0.963 | 0.803 |
| 1e18 (2.1e19) | 0.131 | 0.508 | 0.692 | 0.380 |
| 1e19 (2.1e20) | 0.008 | 0.018 | 0.034 | 0.032 |
| 5e19 (1.05e21) | 0.009 | 0.009 | 0.009 | 0.005 |

The locked-comb model gives nearly the same figures (0.927 / 0.862 / 0.448 / 0.015 / 0.008 fixed), so mode locking does not change the injection requirement at 10 GHz.

**Both lasers need roughly the same total injected density**, crossing 0.01 between 2.1e20 and 1.05e21. What still favours the FP is spectral acceptance, which is an analytic result untouched by the sampling fix: its 21 modes collectively accept 1.57 THz of the SLD's 4.12 THz, against 75 GHz for a single DFB mode. At the same total density the FP needs a coupling efficiency near 1, the DFB about 22. So the FP is borderline feasible from a 19 mW SLD and the DFB is not — a roughly 20× advantage in required coupling, not in injected power.

## Changes to existing code

| File | Change | Consequence |
|---|---|---|
| [core/sld_injection.py:397](../core/sld_injection.py) | Field update `E*(1+g*dt)` → `E*exp(g*dt)` | The first-order update let α_H leak into the amplitude. DFB free-running r₁ at 2 GHz moved 0.310 → 0.254; 5 and 10 GHz stay within 0.02 of Chapter 5. **Chapter 5's absolute numbers should be re-run.** |

`core/dfb_laser.py` and `core/gain_switched_interference.py` are untouched. The carrier-noise term in `sld_injection.py` was investigated and deliberately left alone.

## New code

- **[core/fp_laser.py](../core/fp_laser.py)** — multimode FP model. M modes on the cavity FSR grid under a Gaussian gain envelope, sharing one carrier reservoir; complex field per mode; spontaneous emission and SLD light as random-phase phasors; exponential field update. Returns both `E_fixed` (AMZI-like, use this) and `E_peak`. Optional physics, all off by default: `theta_cross` (cross/self saturation), `fast_gain` (instantaneous gain saturation), `fwm_coupling` (Kerr phase coupling), `carrier_noise_scale`, `spont_gain_weighted`.
- **[core/phase_estimators.py](../core/phase_estimators.py)** — `r1_power_weighted` for multimode output. `r1_summed_field` counts mode-power fluctuations as randomisation; `r1_per_mode_mean` is dominated by dark modes. Use neither.
- **studies/** — `fp_validated_sweeps.py` (resumable sweeps), `fp_diag_drive.py`, `fp_diag_model.py`. `fp_randomisation_sweep.py` and `fp_mode_count_sweep.py` predate the fixes; their FP numbers are superseded.

## Bugs found

| # | Bug | How it showed up | Fix |
|---|---|---|---|
| 1 | FSR stored in metres, printed as nm | FSR shown as 0.000 nm | store in metres |
| 2 | Gain profile given a double unit conversion | every mode saw identical gain | pass metres directly |
| 3 | Threshold formula cancelled to q·V·N_th | I_th shown as 0.0 mA | I_th = q·V·R_sp(N_th) |
| 4 | Injection added to photon density only, never to phase | identical mode-switch counts with and without injection | complex-field model, injection as random phasors |
| 5 | S_inj treated as a rate, not a density | injection ~11 orders too weak | rate = S_inj / τ_p |
| 6 | Spontaneous emission divided by M, though β_sp is per mode | FP looked only 1.2× better than the DFB | full β_sp·B·N² per mode |
| 7 | First-order field update | photon lifetime too long | exponential update, both files |
| 8 | `max()` on a per-mode array | crash with `spont_gain_weighted` | `np.maximum` |
| 9 | First FWM attempt added a phase term directly to the field | photon density inflated to 1e128 | split-step exponential |
| 10 | **Sampling each pulse at its own peak** | side modes falsely random; long-cavity and injection advantages overstated | `E_fixed` at a fixed delay |

## Measurement problems found

- **Summed-field r₁ is invalid for multimode light.** With every mode's phase locked but amplitudes fluctuating it returns 0.033 where the truth is 1. The power-weighted per-mode estimator returns 1.000.
- **Unweighted per-mode mean fails on real output**, reading 0.03–0.08 everywhere because most tracked modes are dark.
- **Peak-time sampling biases coherent regimes** (item 10 above).
- **Single-seed runs over-read noise.** σ = 0.0255 at N = 1500, 0.0068 at N = 6000. An apparent r₁-vs-mode-count trend was 2.9σ, i.e. noise.
- **r₁ cannot reach zero.** Random phases give ≈ 0.886/√N: 0.023 at N = 1500, 0.008 at N = 12000.

## Checked and left unchanged

| Item | Test | Outcome |
|---|---|---|
| Carrier noise missing 1/√V (also in `sld_injection.py`) | scale 1 vs 1/√V at 2, 5, 10 GHz | r₁ moved ≤ 0.0033; immaterial |
| Spontaneous emission not gain-weighted | mode sweep with and without | spread 0.098 vs 0.094; no effect |
| Pure-phase FWM coupling | swept to 10⁴× gain-compression scale | no locking, no coherence — phase-only coupling cannot lock modes, energy exchange is required |

## Model caveats from the literature

Reviewed 20 September 2026. Ranked by how much each threatens our conclusions.

| # | Omission | What the literature says | Effect on our conclusions |
|---|---|---|---|
| 1 | **Longitudinal spatial hole burning** | The standard route to multimode operation. SHB fixes the number of lasing modes well above threshold, while carrier diffusion *strengthens* cross-saturation and restores single-mode emission. Standard travelling-wave FP models carry spectral **and** spatial hole burning explicitly. | We got a multimode comb by *lowering* cross-saturation (θ = 0.7). That is a phenomenological stand-in for SHB, not the mechanism. The comb exists for roughly the right reason but by the wrong route, so θ is a fitted knob and the predicted mode count, and its dependence on current and cavity length, are not trustworthy. |
| 2 | **Cavity group velocity dispersion** | GVD makes the mode spacing non-uniform. FM comb formation rests on a *balance* between GVD and the Kerr/FWM nonlinearity, with FWM pulling modes back onto a uniform grid. Zero dispersion is the ideal comb case. | Our mode grid is exactly uniform, so the phase matching 2ω_m = ω_(m+1) + ω_(m−1) holds by construction. We therefore model the most favourable possible case for locking, and likely overestimate how easily and how broadly the comb locks. Real combs often lock only a subset of modes. |
| 3 | **Bogatov effect (asymmetric nonlinear gain)** | Beating between modes writes carrier gratings, regulated by carrier diffusion, giving an asymmetric nonlinear gain that favours longer wavelengths when α is positive. It is the accepted cause of asymmetric side-mode suppression. | Our model is symmetric about the gain peak. Real FP spectra are not, so mode selection, mode partition statistics, and the equivalence of filtered channels for a multi-line QRNG are all affected. |
| 4 | **Optical feedback** | Pigtailed modules tolerate only about −15 to −17 dB of feedback (at 4% coupling) before coherence collapse. Weak or strong feedback instead *narrows* the linewidth and *increases* side-mode suppression. | Our model has no feedback at all, and the real device is fibre-pigtailed. A stray reflection could either collapse coherence or externally stabilise and lock the modes. This is a live alternative explanation for the locked lines that our model cannot otherwise produce. |
| 5 | **α_H mode dependence** | Mode-resolved measurements on FP lasers show α varying between adjacent longitudinal modes and rising sharply at the gain edges, reaching values near 14 there. | We use one constant α = 3 for every mode. Mode-dependent α gives mode-dependent chirp and acts as an extra effective dispersion, compounding item 2. |
| 6 | **Gain peak shift with carrier density** | Band filling moves the gain peak as carriers change, producing mode hopping. | Our Gaussian envelope is static, yet a gain-switched pulse swings the carrier density hard. Affects which modes dominate within a pulse. |
| 7 | **Separate SHB and carrier-heating time constants** | Normally treated as distinct processes with different time constants and different real/imaginary contributions. | We lump them into one compression factor plus the instantaneous option. Changes the coupling strength needed for locking, not whether locking happens. |

**Choices the literature supports — not worth revisiting:**

- **Our drive conclusion is the standard account.** Residual photons left in the cavity make a new pulse inherit the previous pulse's phase, and the accepted remedy is to set the drive so the cavity empties fully between pulses. This is exactly what we found and why deep switching fixes it.
- **Four-wave mixing alone can mode-lock a semiconductor laser**, with no saturable absorber. Single-section diodes form frequency-modulated combs: near-constant intensity in time with fixed non-zero phase differences between lines. Our locked state is FM-like, which matches.
- **Per-mode β_sp** and the general multimode rate-equation structure are standard.
- **FP lasers as QKD transmitters** already appear in the literature, including a multimode FP injection-locked by a tunable laser.

**Isolators are device-specific.** Isolators are commonly packaged with laser diodes, but SFP TO-can packages are space constrained and some contain no isolator at all. This has to be settled from the part number rather than assumed either way.

Sources: [SHB rate-equation model](https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-7-8143&id=282423), [asymmetric nonlinear gain and FWM](https://pubs.aip.org/aip/apl/article-abstract/59/5/499/59439/Effects-of-nonlinear-gain-on-four-wave-mixing-and), [self-mode-locking without a saturable absorber](https://opg.optica.org/abstract.cfm?URI=CLEO_SI-2020-STh3E.2), [single-section self-mode-locking](https://opg.optica.org/oe/fulltext.cfm?uri=oe-28-4-5317&id=427426), [travelling-wave comb model](https://arxiv.org/pdf/1707.01582), [FM comb laser](https://www.nature.com/articles/s41377-023-01225-z), [feedback noise in pigtailed modules](https://ieeexplore.ieee.org/document/93252), [mode-resolved α_H](https://www.researchgate.net/publication/235767553_Mode-Resolved_Measurements_of_the_Linewidth_Enhancement_Factor_of_a_Fabry-PErot_Laser), [higher-order phase correlations in gain-switched sources](https://link.springer.com/article/10.1140/epjqt/s40507-025-00340-7), [phase randomness SDE analysis](https://arxiv.org/pdf/2011.10401), [phase-correlation-free SLED source](https://arxiv.org/html/2606.04947).

## Open issues

1. Re-run Chapter 5's DFB numbers with the exponential field update.
2. θ = 0.7 and full instantaneous gain saturation are plausible but uncalibrated. The locking threshold in θ, and the real device's value, are unknown.
3. Re-run the crossing brackets and the 2 GHz multi-seed headline with `E_fixed`; the bracket figures elsewhere in the history used peak sampling.
4. Speed and memory: the solvers store every time step for every mode, so an 85-mode 20000-pulse run needs ~11 GB on an 8 GB machine. Peak-only storage plus Numba would make 10⁶-pulse runs feasible, which is what a QRNG independence claim needs (correlations to ~10⁻³, against a floor of 0.02 today).
5. The real FP device is unknown: no raw data, unreadable spectra. All results are for a generic 300 µm InGaAsP device.
6. Unverified audit findings remain: `amzi_splitting_ratio` applies a per-mode delay phase it should not, and `mode_frequencies` uses the opposite sign convention to `mode_wavelengths`.

## Suggested bench tests

1. **Measure the current at the laser.** Lower the DC bias or raise the RF until the single-line histogram goes two-horned. The model predicts a sharp transition near 0.8 × threshold at 2 GHz.
2. **Check that SLD light enters the cavity** — a spectral change with the SLD on and off, and the package part number for an isolator.
3. **Filter two lines onto two detectors and cross-correlate.** Under deep switching the model predicts uncorrelated streams; if they stay correlated, the locking is faster than the model's and multi-stream QRNG is not viable.

## Reproduce

```bash
cd ~/Documents/Work/PhD/Simulations && source .venv/bin/activate
python studies/fp_validated_sweeps.py <headline2g|inj300|injlong|bracket300|bracketlong|bracketdfb>
```

Raw data: `images/fp_randomisation/*.csv`. The rep-rate scans, cavity sweeps, locking tests and sampling comparison were run inline; their results are recorded only in this report.
