# Figure provenance: thesis sections 5.4 and 6.4

Every figure in those two sections, mapped to the script that draws it, the line of the
`savefig` call, and whether the underlying numbers are saved to disk.

Run everything from the `Simulations/` root with `python3 -m studies.<name>`.
All paths below are relative to that root.

Style note: the 5.4 figures use the house style in `gsdfb/plotting.py` (`setup_plotting()`,
DejaVu Serif, top/right spines off, savefig dpi 200). The 6.4 figures deliberately do NOT use
it; they are plain default matplotlib to match the older Chapter 6 figures. Whichever you
standardise on, `gsdfb/plotting.py::STYLE` is the one place to change it for 5.4.

---

## Section 5.4 (Chapter 5) — data saved 2026-09-04

All seven scripts now call `save_run(locals(), ...)` at the end of their `__main__` block
and were re-run, so every figure can be redrawn from disk without re-simulating.

| Thesis figure (`Chapter5/Figs/`) | Script | savefig line | Source PNG |
|---|---|---|---|
| `sim_universal_collapse.png` | `studies/phase_transition_analysis.py` | 373 | `images/phase_transition/universal_collapse.png` |
| `sim_Sc_scaling.png` | `studies/phase_transition_analysis.py` | 422 | `images/phase_transition/Sc_scaling.png` |
| `sim_phase_diagram.png` | `studies/phase_transition_analysis.py` | 513 | `images/phase_transition/phase_diagram.png` |
| `sim_multimode_comparison.png` | `studies/multimode_analysis.py` | 764 | `images/multimode/multimode_vs_singlemode.png` |
| `sim_pulse_train_modes.png` | `studies/multimode_analysis.py` | 826 | `images/multimode/pulse_train_modes.png` |
| `sim_transport_impact.png` | `studies/carrier_transport_analysis.py` | 587 | `images/carrier_transport/transport_impact.png` |
| `sim_jitter_vs_freq.png` | `studies/timing_jitter_analysis.py` | 177 | `images/timing_jitter/jitter_vs_freq.png` |
| `sim_arrival_distributions.png` | `studies/timing_jitter_analysis.py` | 217 | `images/timing_jitter/arrival_distributions.png` |
| `sim_pulse_evolution.png` | `studies/fiber_propagation.py` | 306 | `images/fiber_propagation/pulse_evolution.png` |
| `sim_chirp_evolution.png` | `studies/fiber_propagation.py` | 407 | `images/fiber_propagation/chirp_evolution.png` |
| `sim_phase_randomisation.png` | `studies/qkd_source_analysis.py` | 270 | `images/qkd_source/phase_randomisation.png` |
| `sim_sinj_tradeoff.png` | `studies/qkd_sinj_sweep.py` | 396 | `images/qkd_source/sinj_sweep_tradeoff.png` |

Cost of a re-run, roughly, on this machine:
- `phase_transition_analysis` — 200k pulses x 24 injection levels x 5 frequencies. Minutes.
- `timing_jitter_analysis`, `qkd_source_analysis` — 1M pulses x 10 frequencies x 2 cases. Longest.
- `carrier_transport_analysis` — 500k pulses x 10 frequencies x 2 models.
- `multimode_analysis` — 5 coupled modes, expensive per pulse.
- `fiber_propagation` — split-step, moderate; uses dt = 0.5 ps for chirp resolution.

### The data

| npz | size | contents |
|---|---|---|
| `images/phase_transition/phase_transition_data.npz` | 3.8 MB | r1 vs S_inj for all 5 frequencies, the collapse, S_c fit, exponent |
| `images/timing_jitter/timing_jitter_data.npz` | 0.8 MB | per-frequency jitter metrics, Allan deviation, arrival histograms |
| `images/carrier_transport/carrier_transport_data.npz` | 0.06 MB | both models across frequency, tau_cap sweep |
| `images/multimode/multimode_data.npz` | 14.4 MB | per-mode waveforms and statistics, SMSR, MPN |
| `images/fiber_propagation/fiber_propagation_data.npz` | 1.3 MB | pulse/spectrum/chirp vs distance, TBP |
| `images/qkd_source/qkd_source_data.npz` | 0.6 MB | r1, KL, Fano, CV, key rates per frequency and case |
| `images/qkd_source/qkd_sinj_sweep_data.npz` | 0.3 MB | the S_inj sweep at 3, 5 and 10 GHz |

Keys are the script's own variable names, with nested dicts flattened on `.`, so
`results['10GHz_free']['r1']` is stored as `results.10GHz_free.r1`:

```python
from gsdfb.io import load_run
d = load_run('images/timing_jitter/timing_jitter_data.npz')
[k for k in d if k.endswith('.sig_t')]        # what is in there
d['data.10GHz_free.sig_t']
```

Raw per-pulse ensembles (arrays over 200k elements) are stored as a 512-bin histogram plus
moments rather than verbatim, which is what keeps these files small enough to version:

```
<name>__hist_counts   512 counts
<name>__hist_edges    513 edges
<name>__stats         [mean, std, min, max, n]
```

That is enough to redraw the distribution figures and to re-bin them coarser. If you need the
raw samples for something else, re-run the script with `_MAX_ELEMS` raised in `gsdfb/io.py`.

---

## Section 6.4 (Chapter 6) — data IS saved

`studies/detector_imperfections.py` (513 lines) writes both the figures and the numbers.

| Thesis figure (`Chapter6/Figs/`) | save line | Source PNG (`images/detector_imperfections/`) |
|---|---|---|
| `noisy_noiseless.png` | 404 | `noisy_noiseless.png` |
| `shifted_unshifted.png` | 404 | `shifted_unshifted.png` |
| `noisy_shifted.png` | 404 | `noisy_shifted.png` |
| `SNRmeas_variance.png` | 417 | `SNRmeas_variance.png` |
| `estimatedSNR.png` | 426 | `estimatedSNR.png` |
| `noise_recovery.png` | 451 | `noise_recovery.png` |
| `gm_noise.png` | 461 | `gm_noise.png` |
| `gm_noise_shift.png` | 472 | `gm_noise_shift.png` |
| `correction_efficacy.png` | 495 | `correction_efficacy.png` |
| (not used in thesis) | 484 | `gm_relative_bias.png` |

Plotting block: lines 371 to 497. Everything above line 371 is physics and analysis; you can
replace the whole block without touching it.

### The data

`images/detector_imperfections/detector_imperfections.npz`

```python
import numpy as np
d = np.load('images/detector_imperfections/detector_imperfections.npz')
d['snr_db']          # (45,)   SNR sweep, -4 to 40 dB in 1 dB steps
d['orders']          # (3,)    [2, 3, 4]
d['g_true']          # (3,)    noise-free g^(m)(0) of the simulated source
d['g2_noise'] ...    # (45,)   each of g{2,3,4}_{noise,shift,corr}: mean over realisations
d['excess_bias']     # (3,45)  (g_meas - g_true)/(g_true - 1), uncorrected
d['corr_bias']       # (3,45)  same for the corrected estimator
d['corr_bias_lo/hi'] # (3,45)  95% interval over realisations
d['analytic']        # (45,)   10^(-SNR/10), the closed-form bias law
d['V_meas'], d['V_e_meas'], d['V_e_true'], d['snr_est']   # (45,) variances and inferred SNR
d['vs_mean'], d['vs_lo'], d['vs_hi']                      # (45,) inferred signal-variance error
d['thr_raw'], d['thr_corr']  # (3,) SNR at which bias falls below tol
d['tol'], d['delta'], d['mean_I'], d['var_I']             # scalars
```

`images/detector_imperfections/I_signal_1000000_0.95_1.6.npy`
The raw ensemble: 10^6 integrated pulse intensities, float64, 8 MB. This is the cached
pulse train, keyed by `<n_pulses>_<bias factor>_<peak factor>`. Everything in the npz is
derived from it, so the three distribution histograms (`noisy_*`, `shifted_*`) can be redrawn
from this file alone. Delete it to force a fresh simulation.

Reproducing the histograms from the cache:

```python
import numpy as np
I = np.load('images/detector_imperfections/I_signal_1000000_0.95_1.6.npy')
d = np.load('images/detector_imperfections/detector_imperfections.npz')
V, mean = d['var_I'], d['mean_I']
rng = np.random.default_rng(7)                  # the seed the figures used
V_e  = V / 10**(15.0/10)                        # the 15 dB demo point
noise   = rng.normal(0, np.sqrt(V_e), I.size)
I_noisy = I + noise
I_shift = I - float(d['delta']) * mean
I_both  = I_shift + noise
```

Environment variables that shrink a re-run while you iterate:
`DI_PULSES` (default 1000000), `DI_REAL` (realisations per SNR, default 40),
`DI_STEP` (SNR step in dB, default 1.0), `DI_CACHE_ONLY=1` (simulate and cache, then stop).
The pulse train takes about 60 s at 10^6; the sweep dominates after that.

---

## Where the figures are used in the thesis

`Chapter5/Chapter5.tex` sections 5.4.2 to 5.4.9, and `Chapter6/Chapter6.tex` sections 6.4 and
6.5.4. If you rename a PNG, update the `\includegraphics` path; the labels are `Img:sim_*` in
Chapter 5 and `Img:noise_distr`, `Img:noise_recovery`, `Img:meas_variance`, `Img:estimateSNR`,
`Img:SNRimpactG`, `Img:correction` in Chapter 6.
